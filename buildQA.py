"""Pipeline for building field_* QA datasets with dynamic Ollama scaling.

This script downloads OpenAlex works manifests, spins up as many Ollama
instances as available GPU memory allows, generates question/answer pairs
(one per field) with the selected LLM, persists them to MongoDB, and finally
runs backup and pruning maintenance tasks so the databases remain ready for
consumption.

ARCHITECTURE OVERVIEW:
- Downloads OpenAlex research paper metadata from S3 manifests
- Launches multiple Ollama instances across available GPUs for parallel processing
- Generates Q&A pairs from paper abstracts using LLM inference
- Stores results in MongoDB field_* databases (one per academic field)
- Implements per-node throttling and health monitoring to prevent overload
- Supports resume capability by checking if records are already processed
- Runs backup and pruning maintenance after completion

PERFORMANCE OPTIMIZATIONS:
- Early filtering of rows before executor submission (reduces overhead)
- Streaming chunk processing to limit memory usage
- Per-node semaphore-based throttling to prevent overwhelming Ollama instances
- Smart node selection that avoids overloaded nodes
- Record-level resume checking (more efficient than URL-level)
- Parallel database writes using ThreadPoolExecutor
"""

import ollama
from pymongo import MongoClient
import random
import time
import os
import sys
import socket
import subprocess
import shlex
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from threading import Thread, Semaphore, Lock
from collections import defaultdict
import signal

from scripts import backup_field_databases, prune_field_databases

# ────────── CONFIGURATION CONSTANTS ──────────
# These constants control Ollama instance planning and resource allocation

# GPU_MEMORY_RESERVE_MB: Amount of GPU memory (MB) to reserve per GPU for system overhead
# This prevents OOM errors by leaving headroom for CUDA operations and system processes
GPU_MEMORY_RESERVE_MB = 512

# SMOLLM2_MEMORY_MB: Memory footprint of smollm2 model variant (used for reference)
# This is kept for backward compatibility but MODEL_MEMORY_MB is the active setting
SMOLLM2_MEMORY_MB = 906

# MODEL_NAME: The Ollama model identifier to use for Q&A generation
# Format: "model-family:size" (e.g., "gemma3:270m" = Gemma 3 with 270M parameters)
# This model must be available in Ollama and capable of text generation
MODEL_NAME = 'gemma3:270m'

# MODEL_MEMORY_MB: Estimated GPU memory (MB) required per instance of MODEL_NAME
# Used to calculate how many instances can fit on each GPU
# Formula: (GPU_TOTAL_MB - GPU_MEMORY_RESERVE_MB) // MODEL_MEMORY_MB = instances_per_gpu
MODEL_MEMORY_MB = 1500

# BASE_PORT: Starting port number for Ollama instances
# Each instance gets BASE_PORT + offset (e.g., 11434, 11435, 11436, ...)
# Must not conflict with other services on the system
BASE_PORT = 11434

# ────────── RUNTIME STATE VARIABLES ──────────
# These are populated during execution and track active Ollama instances

# nodes: List of Ollama instance URLs (e.g., ["http://localhost:11434", ...])
# Populated by ensure_ollama_instances() after instances are launched
nodes = []

# INSTANCES: List of instance configuration dicts with 'host', 'port', 'gpu' keys
# Each dict represents one Ollama server process bound to a specific GPU
INSTANCES = []


def _query_gpu_memory_mb():
    """Query NVIDIA GPUs to get total VRAM per GPU in megabytes.
    
    Uses nvidia-smi to detect available GPUs and their memory capacity.
    This is critical for planning how many Ollama instances can run in parallel.
    
    Returns:
        List[int]: List of total VRAM values (MB) for each detected GPU.
                   Example: [24576, 24576] for two 24GB GPUs.
                   Falls back to [24576] (single 24GB GPU) if detection fails.
    
    Design Notes:
        - Uses subprocess to call nvidia-smi (more reliable than Python bindings)
        - Returns list to support multi-GPU systems
        - Fallback ensures script can run even if nvidia-smi fails
        - Only queries total memory, not available (assumes clean GPU state at startup)
    """
    try:
        # Query nvidia-smi for memory.total (not memory.used) to get GPU capacity
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",  # Total VRAM, not currently used
                "--format=csv,noheader,nounits",  # CSV format, no headers, MB units
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Parse output: each line is one GPU's memory in MB
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if values:
            return values
    except Exception:
        # If nvidia-smi fails (no GPU, wrong driver, etc.), assume single 24GB GPU
        # This allows script to run but may cause OOM if actual GPU has less memory
        pass
    # Fallback to a single 24 GB GPU if detection fails
    return [24576]


# Query GPU memory at module load time (before any Ollama instances are created)
GPU_MEMORY_MB = _query_gpu_memory_mb()

# ────────── DATABASE CONFIGURATION ──────────

# MONGO_URI: MongoDB connection string
# Format: "mongodb://[host]:[port]/"
# Uses default localhost:27017 for local MongoDB instance
MONGO_URI = "mongodb://localhost:27017/"

# DATABASES: List of MongoDB database names for each academic field
# OpenAlex uses field IDs 11-36 to represent 26 academic fields
# Each database stores Q&A pairs for papers in that field
# Format: ["field_11", "field_12", ..., "field_36"]
DATABASES = [f'field_{fid}' for fid in range(11, 37)]

# ────────── PROCESSING CONFIGURATION ──────────

# MAX_ATTEMPTS: Maximum number of retries for generating a single Q&A pair
# If LLM fails to produce valid format after this many attempts, row is skipped
# Higher values = more resilient but slower (each attempt takes ~1-2 seconds)
MAX_ATTEMPTS = 3

# TASK_TIMEOUT_SECONDS: Timeout for individual Ollama API calls (seconds)
# Increased from 30 to 60 to handle high load scenarios where responses take longer
# If Ollama doesn't respond within this time, request is retried on different node
TASK_TIMEOUT_SECONDS = 60  # Increased from 30 to handle load better

# LOG_TIMEOUT_SECONDS: Maximum seconds without any log activity before watchdog restarts script
# Set to 1 hour (3600s) to only restart if truly stuck (not just slow processing)
# Prevents premature restarts during legitimate long-running operations
LOG_TIMEOUT_SECONDS = 3600  # 1 hour - only restart if truly stuck

# BATCH_SIZE: Number of rows to process in each batch (currently unused, kept for compatibility)
# Legacy parameter from earlier implementation
BATCH_SIZE = 40

# WORKERS_PER_NODE: Maximum concurrent requests per Ollama instance
# Reduced from 16 to 3 to prevent overwhelming individual Ollama servers
# Each worker = one thread making API calls to that node
# Lower value = more stable but slower throughput
WORKERS_PER_NODE = 3  # Reduced from 16 - max concurrent requests per Ollama instance

# RETRY_DELAY: Base delay (seconds) between retry attempts
# Reduced from 0.2s to 0.05s for faster retries when transient errors occur
# Uses exponential backoff: delay = RETRY_DELAY * (2 ** attempt_number)
RETRY_DELAY = 0.05  # Reduced from 0.2s for faster retries

# MAX_CONCURRENT_REQUESTS_PER_NODE: Semaphore limit for concurrent requests per node
# This is the actual throttling mechanism - semaphore prevents more than N simultaneous calls
# Must match or be less than WORKERS_PER_NODE for effective throttling
MAX_CONCURRENT_REQUESTS_PER_NODE = 3  # Semaphore limit per node

# NODE_BACKOFF_SECONDS: Time (seconds) to avoid a node after it times out
# Prevents repeatedly hitting a problematic node that's overloaded or unresponsive
# After timeout, node is marked unavailable for this duration
NODE_BACKOFF_SECONDS = 30  # Back off a node for 30 seconds after timeout

# ────────── WATCHDOG AND STATE TRACKING ──────────

# last_log_time: Timestamp of last log message (used by timeout_monitor watchdog)
# Updated by log() function to track script activity
# If this doesn't change for LOG_TIMEOUT_SECONDS, script restarts itself
last_log_time = datetime.now()

# Per-node throttling and health tracking
# These data structures enable intelligent load balancing and fault tolerance

# node_semaphores: Dict mapping node URL -> Semaphore object
# Semaphore limits concurrent requests to each node (prevents overwhelming Ollama)
# Example: {"http://localhost:11434": Semaphore(3), ...}
# Acquired before each API call, released after (even on error)
node_semaphores = {}  # Semaphore per node to limit concurrent requests

# node_timeout_counts: Dict mapping node URL -> int (count of timeouts)
# Tracks how many times each node has timed out (for load balancing)
# Nodes with fewer timeouts are preferred when selecting which node to use
node_timeout_counts = defaultdict(int)  # Track timeout counts per node

# node_last_timeout: Dict mapping node URL -> datetime (last timeout timestamp)
# Used to implement backoff period - nodes are avoided for NODE_BACKOFF_SECONDS after timeout
# Prevents repeatedly hitting a node that's currently overloaded
node_last_timeout = {}  # Track last timeout time per node

# node_lock: Thread lock for safe concurrent access to node tracking structures
# All reads/writes to node_semaphores, node_timeout_counts, node_last_timeout must use this lock
# Prevents race conditions in multi-threaded environment
node_lock = Lock()  # Lock for thread-safe access to node tracking

def log(message):
    """Log a message with timestamp and update watchdog timer.
    
    This is the primary logging function used throughout the script.
    Updates last_log_time so timeout_monitor can detect if script is stuck.
    
    Args:
        message (str): Message to log (will be prefixed with timestamp)
    
    Design Notes:
        - Uses print() with flush=True for immediate output (important for monitoring)
        - Updates global last_log_time to prevent watchdog false positives
        - Timestamp format: "YYYY-MM-DD HH:MM:SS - message"
        - Consider using Python logging module for production (better log levels, rotation)
    """
    global last_log_time
    last_log_time = datetime.now()  # Update watchdog timer
    print(f"{last_log_time.strftime('%Y-%m-%d %H:%M:%S')} - {message}", flush=True)


def get_hostname():
    """Get the system hostname.
    
    Returns:
        str: System hostname (e.g., "node0", "localhost", etc.)
    
    Used for:
        - Logging which machine is processing data
        - Multi-node deployment scenarios (though current implementation is single-node)
        - Progress bar labeling
    """
    return socket.gethostname()


def _run(cmd, **kwargs):
    """Execute a shell command and return result.
    
    Wrapper around subprocess.run() that catches CalledProcessError and returns
    the exception object instead of raising. This allows callers to check returncode
    without try/except blocks.
    
    Args:
        cmd: Command and arguments as list (e.g., ["ls", "-l"])
        **kwargs: Additional arguments passed to subprocess.run()
    
    Returns:
        subprocess.CompletedProcess: On success (returncode == 0)
        subprocess.CalledProcessError: On failure (returncode != 0)
    
    Design Notes:
        - Uses check=True so non-zero exit codes raise CalledProcessError
        - Captures both stdout and stderr for debugging
        - Returns exception object instead of raising for easier error handling
    """
    try:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    except subprocess.CalledProcessError as e:
        return e  # Return exception instead of raising (caller checks returncode)


def _which(exe):
    """Find the full path to an executable (like Unix 'which' command).
    
    Uses bash to run 'command -v' which is POSIX-compliant and works for
    both executables and shell builtins.
    
    Args:
        exe (str): Executable name (e.g., "mongod", "ollama")
    
    Returns:
        str: Full path to executable, or None if not found
        Example: "/usr/bin/mongod" or None
    
    Design Notes:
        - Uses bash -lc to ensure PATH is properly set (login shell environment)
        - Appends "|| true" so command never fails (returns empty string if not found)
        - Decodes bytes output to string for Python 3 compatibility
    """
    r = _run(['bash', '-lc', f'command -v {exe} || true'])
    out = (r.stdout or b'').decode().strip()
    return out if out else None


def ensure_mongodb_installed_and_running():
    """Ensure MongoDB is installed and running as a system service.
    
    This function handles MongoDB installation and service management:
    1. Checks if mongod executable exists
    2. If not, attempts to install via apt-get (tries both mongodb-org and mongodb packages)
    3. Enables and starts MongoDB service (tries both 'mongod' and 'mongodb' service names)
    4. Uses sudo with optional password from SUDO_PASSWORD environment variable
    
    Design Notes:
        - Supports both systemctl and service commands for compatibility
        - Tries multiple package names (mongodb-org vs mongodb) for different distros
        - Uses SUDO_PASSWORD env var for non-interactive sudo (useful in automation)
        - Suppresses output to avoid cluttering logs during installation
        - Does not raise exceptions - failures are silent (script continues anyway)
    
    Environment Variables:
        SUDO_PASSWORD: Optional sudo password for non-interactive authentication
    """
    # Helper function to run sudo commands with optional password
    sudo_pwd = os.environ.get('SUDO_PASSWORD')
    def sudo(cmd):
        """Run sudo command with optional password authentication."""
        if sudo_pwd:
            # Use -S flag to read password from stdin
            return subprocess.run(['sudo', '-S'] + cmd, input=(sudo_pwd + '\n').encode('utf-8'),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        # No password provided - will prompt interactively or use sudoers NOPASSWD
        return subprocess.run(['sudo'] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    # Check if mongod is already installed
    mongod_path = _which('mongod')
    if not mongod_path:
        # MongoDB not found - attempt installation
        # Update package lists first
        sudo(['apt-get', 'update', '-y'])
        # Try mongodb-org first (official MongoDB package)
        rc1 = sudo(['apt-get', 'install', '-y', 'mongodb-org'])
        if rc1.returncode != 0:
            # Fallback to mongodb (Ubuntu default package)
            sudo(['apt-get', 'install', '-y', 'mongodb'])

    # Try starting and enabling common service names
    # Different distributions use different service names
    for svc in ['mongod', 'mongodb']:
        # Enable service to start on boot
        sudo(['systemctl', 'enable', '--now', svc])
        # Also try service command in case systemctl not available (older systems)
        sudo(['service', svc, 'start'])


def ensure_ollama_installed():
    """Ensure Ollama is installed on the system.
    
    Checks if 'ollama' executable exists, and if not, installs it using the
    official Ollama installation script from ollama.com.
    
    Design Notes:
        - Uses official install script (curl | sh) as recommended by Ollama
        - Handles sudo password via SUDO_PASSWORD environment variable
        - Does not raise exceptions - installation failures are silent
        - Installation script handles all dependencies and setup automatically
    
    Environment Variables:
        SUDO_PASSWORD: Optional sudo password for non-interactive authentication
    """
    sudo_pwd = os.environ.get('SUDO_PASSWORD')
    # Check if ollama is already installed
    if _which('ollama'):
        return  # Already installed, nothing to do
    
    # Install Ollama using official install script
    if sudo_pwd:
        # Use sudo with password from environment
        cmd = f"echo {shlex.quote(sudo_pwd)} | sudo -S sh -c 'curl -fsSL https://ollama.com/install.sh | sh'"
        subprocess.run(['bash', '-lc', cmd], check=False)
    else:
        # No password - will prompt or use sudoers
        subprocess.run(['bash', '-lc', "curl -fsSL https://ollama.com/install.sh | sh"], check=False)


def check_ollama_health(node_url):
    """Check if an Ollama instance is healthy and responding to requests.
    
    Performs a simple health check by calling the list() API endpoint.
    This verifies the instance is running and can process requests.
    
    Args:
        node_url (str): Full URL of Ollama instance (e.g., "http://localhost:11434")
    
    Returns:
        bool: True if instance responds successfully, False otherwise
    
    Design Notes:
        - Uses ollama.Client.list() as health check (lightweight, no model loading)
        - Catches all exceptions (network errors, timeouts, etc.) and returns False
        - Fast check - used during instance startup to verify they're ready
    """
    try:
        client = ollama.Client(host=node_url)
        client.list()  # Simple health check - just verify API is responding
        return True
    except Exception:
        # Any error (network, timeout, etc.) means instance is unhealthy
        return False


def ensure_ollama_instances():
    """Launch Ollama servers sized for the target model across all GPUs.
    
    This is the core function that sets up the Ollama infrastructure:
    1. Stops any existing Ollama services/processes
    2. Calculates how many instances fit on each GPU based on memory
    3. Launches dedicated 'ollama serve' processes, one per port
    4. Pins each instance to a specific GPU using CUDA_VISIBLE_DEVICES
    5. Pre-pulls the model on each instance (so first request doesn't block)
    6. Verifies instances are healthy before returning
    7. Initializes per-node semaphores for throttling
    
    Global State Modified:
        - nodes: Populated with list of healthy instance URLs
        - INSTANCES: Populated with instance configuration dicts
        - node_semaphores: Initialized with Semaphore for each node
    
    Raises:
        RuntimeError: If no instances can be planned (no GPU memory available)
    
    Design Notes:
        - Each instance gets its own port (BASE_PORT + offset)
        - Each instance gets its own model directory (~/.ollama-{port}) to avoid conflicts
        - Uses CUDA_VISIBLE_DEVICES to pin instances to specific GPUs
        - Waits 10 seconds after launch for instances to initialize
        - Only uses healthy instances (filters out any that fail health check)
        - Pre-pulls model to avoid blocking on first request
    
    Performance Considerations:
        - Model pre-pull happens sequentially (could be parallelized for speed)
        - 5-second sleep per instance launch (conservative, ensures readiness)
        - 10-second wait after all launches (allows full initialization)
    """
    global nodes, INSTANCES, node_semaphores, node_timeout_counts, node_last_timeout

    # Check hostname (for multi-node deployments, though current code is single-node)
    hostname = get_hostname()
    if hostname != 'node0':
        log(f"⚠️ Hostname is '{hostname}', not 'node0'; proceeding with local Ollama setup anyway.")

    # Ensure Ollama is installed before trying to launch instances
    ensure_ollama_installed()

    # ────────── STOP EXISTING OLLAMA PROCESSES ──────────
    # Stop any system-level services and stray processes to avoid port conflicts
    
    sudo_pwd = os.environ.get('SUDO_PASSWORD')
    if sudo_pwd:
        try:
            # Stop systemd service if it exists
            subprocess.run(
                ['sudo', '-S', 'systemctl', 'stop', 'ollama'],
                input=(sudo_pwd + '\n').encode('utf-8'),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    else:
        # No password - try without password (may fail if sudoers requires it)
        subprocess.run(['sudo', 'systemctl', 'stop', 'ollama'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Kill any existing ollama processes (handles cases where service stop fails)
    subprocess.run(['pkill', 'ollama'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ────────── PLAN INSTANCE DISTRIBUTION ──────────
    # Calculate how many instances can fit on each GPU
    
    INSTANCES = []
    port = BASE_PORT  # Start from base port, increment for each instance
    
    # For each GPU, calculate how many instances fit
    for gpu_index, total_mem in enumerate(GPU_MEMORY_MB):
        # Available memory = total - reserve (leave headroom for system)
        available = max(0, total_mem - GPU_MEMORY_RESERVE_MB)
        
        # Calculate instance count: how many MODEL_MEMORY_MB chunks fit?
        if available >= MODEL_MEMORY_MB:
            # Multiple instances can fit
            count = max(1, available // MODEL_MEMORY_MB)
        else:
            # Not enough memory for even one instance, but try anyway (may work with quantization)
            count = 1
        
        # Create instance configs for this GPU
        for _ in range(count):
            INSTANCES.append({
                'host': f'http://localhost:{port}',
                'port': str(port),
                'gpu': str(gpu_index),  # Pin to this GPU
            })
            port += 1  # Next instance gets next port

    # Safety check: ensure we have at least one instance
    if not INSTANCES:
        raise RuntimeError("Unable to plan any Ollama instances; check GPU availability.")

    # Extract node URLs for easy access
    nodes = [inst['host'] for inst in INSTANCES]
    
    # Initialize per-node tracking structures
    node_semaphores = {}
    node_timeout_counts = defaultdict(int)
    node_last_timeout = {}

    # ────────── LAUNCH INSTANCES ──────────
    # Start each Ollama server process with proper environment configuration
    
    for inst in INSTANCES:
        port = inst['port']
        gpu = inst['gpu']
        
        # Build environment variables for this instance
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'  # Bind to all interfaces on this port
        env['OLLAMA_MODELS'] = os.path.expanduser(f'~/.ollama-{port}')  # Unique model dir per instance
        env['CUDA_VISIBLE_DEVICES'] = gpu  # Pin to specific GPU (only this GPU visible to process)
        
        log(f"🔧 Starting Ollama on port {port} (GPU {gpu})")
        
        # Open log file for this instance (append mode to preserve previous runs)
        stdout = open(f'ollama_{port}.log', 'a')
        
        # Launch ollama serve as background process
        # nohup ensures process continues if parent dies
        subprocess.Popen(['nohup', 'ollama', 'serve'], env=env, stdout=stdout, stderr=subprocess.STDOUT)
        
        # Wait for instance to start (conservative delay)
        time.sleep(5)

        # ────────── PRE-PULL MODEL ──────────
        # Ensure model is available on this instance (avoids blocking on first request)
        
        env_pull = os.environ.copy()
        env_pull['OLLAMA_HOST'] = f'localhost:{port}'  # Connect to this specific instance
        
        try:
            # Check if model already exists on this instance
            show = subprocess.run(
                ['ollama', 'show', MODEL_NAME],
                env=env_pull,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if show.returncode != 0:
                # Model not found - pull it now
                log(f"⬇️ Pulling {MODEL_NAME} on port {port}")
                subprocess.run(['ollama', 'pull', MODEL_NAME], env=env_pull, check=True)
        except subprocess.CalledProcessError:
            # Pull failed - log warning but continue (may work on first request anyway)
            log(f"⚠️ Model setup issue on port {port}; continuing.")

    # ────────── VERIFY INSTANCES ARE HEALTHY ──────────
    # Wait for all instances to fully initialize, then check health
    
    log("⏳ Waiting for Ollama instances to initialize...")
    time.sleep(10)  # Conservative wait - ensures all instances are ready

    # Filter to only healthy instances (those that respond to API calls)
    healthy_nodes = [node for node in nodes if check_ollama_health(node)]
    
    if healthy_nodes:
        log(f"✅ {len(healthy_nodes)}/{len(nodes)} Ollama instances are healthy")
        nodes = healthy_nodes  # Only use healthy instances
    else:
        log("⚠️ No healthy Ollama instances found, using all nodes anyway")
        # Continue with all nodes - may work even if health check failed

    # ────────── INITIALIZE THROTTLING ──────────
    # Create semaphore for each node to limit concurrent requests
    
    for node in nodes:
        # Semaphore allows MAX_CONCURRENT_REQUESTS_PER_NODE simultaneous requests
        node_semaphores[node] = Semaphore(MAX_CONCURRENT_REQUESTS_PER_NODE)


def _recover_abstract(inverted_index):
    """Reconstruct abstract text from OpenAlex inverted index format.
    
    OpenAlex stores abstracts as inverted indexes: {word: [position1, position2, ...]}
    This function reconstructs the original text by placing words at their positions.
    
    Args:
        inverted_index (dict): Inverted index mapping words to position lists
                             Example: {"the": [0, 5], "study": [1], ...}
    
    Returns:
        str: Reconstructed abstract text with words in correct order
             Returns empty string if input is invalid
    
    Algorithm:
        1. Find maximum position across all words
        2. Create list of empty strings with length = max_position + 1
        3. For each word, place it at all its positions
        4. Join list with spaces to form final text
    
    Example:
        Input: {"the": [0, 3], "study": [1], "shows": [2]}
        Output: "the study shows the"
    
    Design Notes:
        - Handles missing/None input gracefully (returns empty string)
        - Assumes positions are 0-indexed and contiguous (may have gaps)
        - Gaps in positions result in empty strings that get stripped
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    
    # Find maximum position to determine list size
    # max(max(v) for v in inverted_index.values()) finds highest position number
    max_pos = max(max(v) for v in inverted_index.values()) if inverted_index.values() else -1
    
    # Create list of empty strings, one per position
    abstract = [''] * (max_pos + 1)
    
    # Place each word at all its positions
    for word, positions in inverted_index.items():
        for pos in positions:
            abstract[pos] = word
    
    # Join words with spaces and strip leading/trailing whitespace
    return ' '.join(abstract).strip()


def extract_field_ids(row):
    """Extract academic field IDs from an OpenAlex work record.
    
    OpenAlex papers are tagged with topics, each topic belongs to a field.
    This function extracts the numeric field IDs (11-36) from the topics.
    
    Args:
        row (dict): OpenAlex work record with 'topics' list
    
    Returns:
        List[int]: List of unique field IDs found in this paper
                  Example: [11, 17, 27] for papers in multiple fields
                  Returns empty list if no valid fields found
    
    Field ID Format:
        - OpenAlex field IDs are integers 11-36 (26 academic fields)
        - May be stored as int or as URL string: "https://openalex.org/fields/17"
        - Function handles both formats
    
    Design Notes:
        - Uses set to ensure uniqueness (papers can have duplicate field tags)
        - Only includes fields 11-36 (valid range for this project)
        - Handles missing/invalid data gracefully (returns empty list)
        - Extracts from nested structure: row['topics'][i]['field']['id']
    """
    seen = set()  # Track seen field IDs to avoid duplicates
    
    # Iterate through all topics in this paper
    for topic in row.get('topics', []):
        # Extract field ID from topic structure
        fid = topic.get('field', {}).get('id')
        
        # Handle URL format: "https://openalex.org/fields/17" -> 17
        if isinstance(fid, str) and fid.startswith('https://openalex.org/fields/'):
            try:
                # Extract numeric ID from URL
                fid = int(fid.split('/')[-1])
            except Exception:
                continue  # Invalid URL format, skip
        
        # Only include valid field IDs (11-36) that we haven't seen
        if isinstance(fid, int) and 11 <= fid <= 36 and fid not in seen:
            seen.add(fid)
    
    return list(seen)


def create_prompt(abstract, error_feedback=None):
    """Create a prompt for LLM to generate Q&A pair from abstract.
    
    This function constructs a carefully crafted prompt that instructs the LLM
    to generate a question and answer in a specific format. The prompt includes:
    - Clear instructions on required format
    - Example of acceptable output
    - Example of unacceptable output (what NOT to do)
    - The actual abstract to process
    - Optional error feedback from previous failed attempts
    
    Args:
        abstract (str): Research paper abstract text to generate Q&A from
        error_feedback (str, optional): Error message from previous attempt
                                       Used to guide LLM toward correct format
    
    Returns:
        str: Complete prompt string ready to send to LLM
    
    Prompt Structure:
        1. Role definition: "You are generating questions and answers..."
        2. Format requirements: "Only the exact format shown below..."
        3. Question requirements: Must start with "Is it true, false..."
        4. Answer requirements: Must be one of four choices
        5. Example of acceptable output
        6. Example of unacceptable output (with explicit "do NOT do this")
        7. Error feedback (if provided)
        8. The actual abstract
        9. Output format reminder
    
    Design Notes:
        - Very explicit instructions to minimize format errors
        - Includes negative examples (unacceptable output) to prevent common mistakes
        - Error feedback helps LLM learn from previous failures
        - Question must start with exact phrase to ensure consistency
        - Answer must be one of: true, false, possibly true, possibly false
    """
    # Example of correct output format (shown to LLM)
    example = """Example:
Abstract:
A randomized controlled trial found that daily vitamin D supplementation significantly reduced the risk of respiratory infections in elderly participants.

Question: Is it true, false, possibly true, or possibly false that daily vitamin D supplementation reduces the risk of respiratory infections in elderly people?
Answer: true"""

    # Example of incorrect output (what NOT to do)
    # This helps prevent LLM from adding preambles or explanations
    unacceptable = (
        "Unacceptable output example (do NOT do this):\n"
        "Okay, I understand.\n"
        "Question: Is it true ...\n"
        "Answer: true\n"
    )

    # Base prompt with all instructions
    base_prompt = (
        "You are generating questions and answers into a database from research abstracts. "
        "Only the exact format shown below is accepted—no confirmations, no acknowledgements, and no explanations. "
        "Respond directly with the question line and the answer line, nothing else before, between, or after.\n\n"
        "From the following research abstract, generate exactly one question based specifically on the findings described. "
        "Start the question with the exact words 'Is it true, false, possibly true, or possibly false that'. "
        "Do not include phrases like 'does the study', 'does the abstract', 'based on these findings', "
        "'do the research findings', 'is it possible', or 'is it possibly'. "
        "The question must stand alone without referencing the abstract, study, or researchers. "
        "The answer must explicitly be either true, false, possibly true, or possibly false. "
        "Then directly answer it with one of these four choices only (no additional explanation).\n\n"
        f"{example}\n\n"
        f"{unacceptable}\n"
    )

    # Add error feedback if this is a retry attempt
    if error_feedback:
        base_prompt += (
            f"Your previous response was incorrect: {error_feedback}\n"
            "Remember: do NOT write acknowledgements, confirmations, or explanations—only the question line and the answer line. "
            "Follow the format exactly as shown in the example and never imitate the unacceptable example.\n\n"
        )

    # Add the actual abstract and final format reminder
    base_prompt += (
        "Abstract:\n"
        f"{abstract}\n\n"
        "Output format (MUST follow exactly two lines):\n"
        "Question: Is it true, false, possibly true, or possibly false that <your question here>?\n"
        "Answer: <true|false|possibly true|possibly false>"
    )
    return base_prompt


def is_valid_response(response):
    """Check if LLM response follows the required Q&A format.
    
    Validates that response contains:
    1. A line starting with "Question:"
    2. A line starting with "Answer:"
    3. No non-English content (common failure mode)
    4. Proper structure (two lines separated by newline)
    
    Args:
        response (str): Raw LLM response to validate
    
    Returns:
        bool: True if response matches required format, False otherwise
    
    Validation Rules:
        - Must contain newline character (separates question and answer)
        - Must start with "Question:" (case-sensitive)
        - Must have second line starting with "Answer:"
        - Must not contain non-English phrases (indicates LLM confusion)
    
    Design Notes:
        - Checks for common non-English phrases to catch language detection failures
        - Simple format check - doesn't validate question/answer content
        - Returns False for None/empty input (safe default)
    """
    if not response or not isinstance(response, str):
        return False
    
    # Check for non-English responses (common failure mode)
    # LLM sometimes responds in wrong language or asks for more input
    if any(phrase in response.lower() for phrase in [
        'please provide', 'i need', 'provide the', 'abstract in', 'research abstract',
        'japanese', 'korean', 'chinese', 'russian', 'arabic', 'spanish', 'french'
    ]):
        return False
    
    # Check for required format: "Question: ...\nAnswer: ..."
    if "\n" in response and response.strip().startswith("Question:"):
        parts = response.split("\n", 1)  # Split into question and answer lines
        if len(parts) == 2 and parts[1].strip().startswith("Answer:"):
            return True
    
    return False


def is_node_available(node):
    """Check if a node is available (not in backoff period after timeout).
    
    Nodes that timeout are temporarily marked unavailable to prevent repeatedly
    hitting overloaded instances. This function checks if backoff period has expired.
    
    Args:
        node (str): Node URL to check (e.g., "http://localhost:11434")
    
    Returns:
        bool: True if node is available, False if still in backoff period
    
    Algorithm:
        1. Check if node has a recorded timeout timestamp
        2. If yes, calculate time since timeout
        3. If time < NODE_BACKOFF_SECONDS, node is unavailable
        4. Otherwise, node is available again
    
    Design Notes:
        - Thread-safe (uses node_lock)
        - Nodes automatically become available after backoff period expires
        - If no timeout recorded, node is considered available
    """
    with node_lock:  # Thread-safe access to node_last_timeout
        if node in node_last_timeout:
            # Calculate seconds since last timeout
            time_since_timeout = (datetime.now() - node_last_timeout[node]).total_seconds()
            # Node unavailable if still within backoff period
            if time_since_timeout < NODE_BACKOFF_SECONDS:
                return False
        return True  # No timeout recorded, or backoff period expired


def mark_node_timeout(node):
    """Mark a node as having timed out (for load balancing and backoff).
    
    Records the current timestamp for this node so is_node_available() can
    implement backoff period. Also increments timeout counter for statistics.
    
    Args:
        node (str): Node URL that timed out
    
    Design Notes:
        - Thread-safe (uses node_lock)
        - Updates both timeout count (for statistics) and timestamp (for backoff)
        - Called automatically when Ollama API calls timeout
    """
    with node_lock:  # Thread-safe access to tracking structures
        node_timeout_counts[node] += 1  # Increment timeout counter
        node_last_timeout[node] = datetime.now()  # Record timeout timestamp


def get_available_node():
    """Get an available node, preferring those with fewer recent timeouts.
    
    Implements intelligent load balancing by:
    1. Filtering to only nodes not in backoff period
    2. Sorting by timeout count (fewer timeouts = healthier)
    3. Randomly selecting from the healthiest half
    
    Returns:
        str: URL of selected node (e.g., "http://localhost:11434")
    
    Algorithm:
        1. Filter nodes to only those available (not in backoff)
        2. If all nodes in backoff, use all nodes anyway (better than failing)
        3. Sort by timeout count (ascending - fewer is better)
        4. Select randomly from healthiest 50% of nodes
    
    Design Notes:
        - Thread-safe (uses node_lock for reading)
        - Falls back to all nodes if none available (prevents deadlock)
        - Random selection from top half prevents all traffic going to single node
        - Prefers nodes with fewer timeouts (indicates better health)
    """
    # Get list of nodes not currently in backoff period
    available_nodes = [n for n in nodes if is_node_available(n)]
    
    if not available_nodes:
        # All nodes in backoff - use all nodes anyway (better than failing)
        available_nodes = nodes
    
    # Prefer nodes with fewer recent timeouts (healthier nodes)
    with node_lock:  # Thread-safe read of node_timeout_counts
        # Sort by timeout count (ascending - fewer timeouts = healthier)
        sorted_nodes = sorted(available_nodes, key=lambda n: node_timeout_counts.get(n, 0))
    
    # Randomly select from healthiest half to distribute load
    # max(1, ...) ensures we always have at least one node to choose from
    return random.choice(sorted_nodes[:max(1, len(sorted_nodes) // 2)])


def generate_question_answer(node, abstract):
    """Generate Q&A pair from abstract using specified Ollama node.
    
    This is the core Q&A generation function. It:
    1. Creates prompt from abstract
    2. Calls Ollama API with per-node throttling (semaphore)
    3. Validates response format
    4. Retries with error feedback if format invalid
    5. Switches nodes if current node times out
    6. Returns question and answer on success
    
    Args:
        node (str): Ollama node URL to use (e.g., "http://localhost:11434")
        abstract (str): Research paper abstract text
    
    Returns:
        Tuple[str, str]: (question, answer) pair
                        Question: Full question text
                        Answer: One of "true", "false", "possibly true", "possibly false"
    
    Raises:
        RuntimeError: If all retry attempts fail to produce valid format
    
    Retry Strategy:
        - Up to MAX_ATTEMPTS retries
        - Each retry includes error feedback in prompt
        - On timeout, switches to different node
        - Uses exponential backoff between retries
    
    Design Notes:
        - Uses semaphore for per-node throttling (prevents overwhelming instance)
        - Tracks last response to detect infinite loops (same response repeatedly)
        - Switches nodes on timeout to avoid repeatedly hitting problematic instance
        - Error feedback helps LLM learn correct format
        - Only logs formatting issues occasionally (1% of failures) to reduce spam
    """
    prompt = create_prompt(abstract)  # Initial prompt without error feedback
    attempts = 0
    last_response = None  # Track to detect infinite loops
    current_node = node  # May switch to different node on timeout

    while attempts < MAX_ATTEMPTS:
        # ────────── ACQUIRE SEMAPHORE ──────────
        # Get semaphore for current node (limits concurrent requests)
        semaphore = node_semaphores.get(current_node)
        if not semaphore:
            # Fallback if semaphore not initialized (shouldn't happen, but be safe)
            semaphore = Semaphore(MAX_CONCURRENT_REQUESTS_PER_NODE)
            node_semaphores[current_node] = semaphore
        
        # Acquire semaphore (blocks if MAX_CONCURRENT_REQUESTS_PER_NODE already in use)
        semaphore.acquire()
        try:
            # ────────── CALL OLLAMA API ──────────
            client = ollama.Client(host=current_node, timeout=TASK_TIMEOUT_SECONDS)
            response = client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={'temperature': 0.3, 'top_p': 0.9}  # Low temperature for consistency
            )['response']

            # ────────── VALIDATE RESPONSE ──────────
            if is_valid_response(response):
                # Parse question and answer from response
                question_line, answer_line = response.split("\n", 1)
                question = question_line.replace("Question: ", "").strip()
                answer = answer_line.replace("Answer: ", "").strip()
                
                # Validate answer is one of the allowed values
                if answer.lower() in ['true', 'false', 'possibly true', 'possibly false']:
                    return question, answer  # Success!

            # ────────── HANDLE INVALID RESPONSE ──────────
            # Check for infinite loop (same response repeatedly)
            if last_response and response[:50] == last_response[:50]:
                break  # Same response - no point retrying

            # Build error feedback for next attempt
            error_feedback = f"Invalid format: {response[:100]}"
            prompt = create_prompt(abstract, error_feedback)  # Include error in prompt
            last_response = response
            
            # Log formatting issues occasionally (1% of failures) to reduce spam
            if attempts == 0:
                log(f"⚠️ Formatting issue on {current_node}: {error_feedback[:80]}")

        except Exception as e:
            # ────────── HANDLE ERRORS ──────────
            error_str = str(e).lower()
            
            # Check if this is a timeout error
            if 'timeout' in error_str or 'timed out' in error_str:
                # Mark node as timed out (triggers backoff period)
                mark_node_timeout(current_node)
                
                # Try different node on next attempt (if not last attempt)
                if attempts < MAX_ATTEMPTS - 1:
                    new_node = get_available_node()
                    if new_node != current_node:
                        current_node = new_node  # Switch to healthier node

            # Log errors occasionally (1% of failures) to reduce spam
            if attempts == 0 and random.random() < 0.01:
                log(f"⚠️ Node {current_node} exception: {e}")
        finally:
            # Always release semaphore, even on error
            semaphore.release()

        # Increment attempt counter and wait before retry
        attempts += 1
        time.sleep(RETRY_DELAY)

    # All attempts failed - raise exception with last response for debugging
    raise RuntimeError(f"❌ All attempts failed. Last response: {last_response[:200] if last_response else 'None'}")


def upsert_and_update(mongo_client, row, question, answer, field_ids):
    """Save Q&A pair to MongoDB for all relevant field databases.
    
    Takes a generated question/answer and saves it to all field databases
    that this paper belongs to. Uses upsert (update if exists, insert if not)
    to handle duplicate processing gracefully.
    
    Args:
        mongo_client: MongoDB client connection
        row (dict): OpenAlex work record with metadata (id, title, abstract, etc.)
        question (str): Generated question text
        answer (str): Generated answer ("true", "false", "possibly true", "possibly false")
        field_ids (List[int]): List of field IDs (11-36) this paper belongs to
    
    Design Notes:
        - Uses upsert (update_one with upsert=True) to handle duplicates
        - Saves same Q&A to multiple field databases (paper can be in multiple fields)
        - Extracts metadata from row: title, abstract, authors, year, DOI, PDF URL
        - Uses openalex_id as unique identifier (from row['id'])
        - Skips if row has no ID (invalid record)
    
    Performance Considerations:
        - Sequential writes to each field database (could be parallelized)
        - Each write is independent (no transaction needed)
        - Upsert is efficient (MongoDB handles existence check internally)
    """
    # Extract OpenAlex ID (unique identifier for this paper)
    idx = row.get('id')
    if not idx:
        return  # No ID - can't save, skip this record
    
    # Extract metadata from row
    title = (row.get('title') or '').strip()
    abstract = _recover_abstract(row.get('abstract_inverted_index'))  # Reconstruct from inverted index
    authors = [
        a.get('author', {}).get('display_name') for a in row.get('authorships', [])
        if a and isinstance(a, dict) and 'author' in a and isinstance(a.get('author'), dict) and 'display_name' in a['author']
    ] or None  # List of author names, or None if no authors
    pub_year = row.get('publication_year')  # Year paper was published
    doi = row.get('doi')  # Digital Object Identifier
    primary_loc = row.get('primary_location')
    pdf_url = primary_loc.get('pdf_url') if isinstance(primary_loc, dict) else None  # PDF download URL

    # Build document to save
    document = {
        "openalex_id": idx,  # Unique identifier
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_year": pub_year,
        "doi": doi,
        "pdf_url": pdf_url,
        "Question": question,  # Generated question
        "Answer": answer,  # Generated answer
    }

    # Save to all field databases this paper belongs to
    # Same Q&A saved to multiple databases (paper can be in multiple fields)
    for fid in field_ids:
        db = mongo_client[f'field_{fid}']  # Get database for this field
        collection = db['sources']  # Collection name is 'sources'
        collection.update_one(
            {"openalex_id": idx},  # Match on OpenAlex ID
            {"$set": document},  # Update or insert this document
            upsert=True  # Insert if doesn't exist, update if it does
        )


def is_record_processed(mongo_client, openalex_id, field_ids):
    """Check if a record already has Q&A in all relevant field databases.
    
    Used for resume capability - skips records that are already processed.
    A record is considered processed if it has both Question and Answer fields
    in ALL field databases it belongs to.
    
    Args:
        mongo_client: MongoDB client connection
        openalex_id (str): OpenAlex ID of the paper to check
        field_ids (List[int]): List of field IDs (11-36) this paper belongs to
    
    Returns:
        bool: True if record has Q&A in all relevant fields, False otherwise
    
    Algorithm:
        1. For each field_id in field_ids:
           a. Query field_{field_id}.sources collection
           b. Check if document with this openalex_id exists
           c. Check if it has both Question and Answer fields
           d. If any field missing Q&A, return False
        2. If all fields have Q&A, return True
    
    Design Notes:
        - Must check ALL fields - paper must have Q&A in every field it belongs to
        - Returns False on any error (safe default - reprocess rather than skip)
        - Uses MongoDB $exists operator to check field presence
        - Efficient query (indexed on openalex_id)
    
    Performance Considerations:
        - One MongoDB query per field (could be optimized with aggregation)
        - Early exit on first missing field (doesn't check remaining fields)
        - Uses find_one() which is efficient with index on openalex_id
    """
    if not openalex_id or not field_ids:
        return False  # Invalid input - can't be processed
    
    try:
        # Check if document exists in ALL relevant field databases with both Question and Answer
        for fid in field_ids:
            db = mongo_client[f'field_{fid}']
            collection = db['sources']
            # Query for document with this ID that has both Question and Answer
            doc = collection.find_one({
                "openalex_id": openalex_id,
                "Question": {"$exists": True},  # Question field exists
                "Answer": {"$exists": True}  # Answer field exists
            })
            if not doc:
                # Missing Q&A in this field - not fully processed
                return False
        # All fields have Q&A - record is processed
        return True
    except Exception:
        # On error, assume not processed to be safe (reprocess rather than skip)
        return False


def check_url_processed(mongo_client, url, sample_size=20, timeout_seconds=30):
    """Sample records from a URL to determine if it should be skipped.
    
    This function provides URL-level resume capability by sampling a few records
    and checking if they're already processed. If 100% of sampled records are
    processed, the entire URL is skipped (assumes URL was fully processed).
    
    Args:
        mongo_client: MongoDB client connection
        url (str): URL to OpenAlex data file (JSONL format)
        sample_size (int): Number of records to sample (default: 20)
        timeout_seconds (int): Maximum time to spend checking (default: 30s)
    
    Returns:
        Tuple[int, int, bool]: (processed_count, total_checked, should_skip)
            - processed_count: Number of sampled records that are already processed
            - total_checked: Total number of records checked (may be less than sample_size)
            - should_skip: True if URL should be skipped (100% processed), False otherwise
    
    Algorithm:
        1. Read JSONL file in chunks (streaming to limit memory)
        2. For each record, check if it's processed (up to sample_size records)
        3. Track processed_count and total_checked
        4. Early exit if unprocessed record found (no need to check more)
        5. Return should_skip = True if 100% of checked records are processed
    
    Design Notes:
        - Uses timeout to prevent spending too long on large files
        - Early exit on first unprocessed record (optimistic - assumes file is mixed)
        - Only checks first chunk for speed (doesn't read entire file)
        - Returns (0, 0, False) on error (safe default - process URL anyway)
    
    Performance Considerations:
        - Timeout prevents blocking on very large files
        - Only reads first chunk (fast check)
        - Early exit on unprocessed record (doesn't check all samples)
        - Uses is_record_processed() which does one query per field
    """
    processed_count = 0
    total_checked = 0
    unprocessed_found = False
    start_time = time.time()
    
    try:
        log(f"🔍 Checking resume status for URL (sampling {sample_size} records, timeout {timeout_seconds}s)...")
        
        # Sample records from the URL - only check first chunk for speed
        # Use smaller chunksize for faster initial read
        for chunk_idx, chunk in enumerate(pd.read_json(url, lines=True, chunksize=1000)):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                log(f"⏱️ Resume check timeout after {timeout_seconds}s, proceeding with URL")
                return 0, 0, False  # Timeout - process URL anyway
            
            if chunk.empty:
                continue
            
            # Process rows until we have enough samples or find unprocessed record
            for _, row in chunk.iterrows():
                # Check timeout during processing
                if time.time() - start_time > timeout_seconds:
                    log(f"⏱️ Resume check timeout after {timeout_seconds}s, proceeding with URL")
                    return 0, 0, False  # Timeout - process URL anyway
                
                if total_checked >= sample_size:
                    break  # Have enough samples
                
                try:
                    # Extract field IDs and openalex_id
                    field_ids = extract_field_ids(row)
                    openalex_id = row.get('id')
                    
                    if field_ids and openalex_id:
                        if is_record_processed(mongo_client, openalex_id, field_ids):
                            processed_count += 1
                        else:
                            unprocessed_found = True
                            # Early exit if we find unprocessed records - no need to check more
                            if total_checked >= 5:  # Check at least 5 records
                                break
                        total_checked += 1
                except (KeyError, TypeError, AttributeError):
                    continue  # Invalid record format - skip
            
            # Early exit if we found unprocessed records and checked enough samples
            if unprocessed_found and total_checked >= 5:
                break
            
            if total_checked >= sample_size:
                break
            
            # Only check first chunk for speed
            if chunk_idx >= 0:
                break
        
        # Determine if URL should be skipped (100% of sampled records are processed)
        should_skip = (total_checked > 0 and processed_count == total_checked and not unprocessed_found)
        
        elapsed = time.time() - start_time
        if total_checked > 0:
            percentage = (processed_count / total_checked) * 100
            log(f"📊 Resume check: {processed_count}/{total_checked} records already processed ({percentage:.1f}%) in {elapsed:.1f}s - {url}")
        
        return processed_count, total_checked, should_skip
    except Exception as e:
        log(f"⚠️ Error checking URL status for {url}: {e}, proceeding anyway")
        return 0, 0, False  # Error - process URL anyway (safe default)


def is_retracted_or_redacted(row):
    """Check if a paper is retracted or redacted (should be skipped).
    
    Retracted papers have invalid content and should not be used for Q&A generation.
    This function checks multiple indicators of retraction/redaction.
    
    Args:
        row (dict): OpenAlex work record
    
    Returns:
        bool: True if paper is retracted/redacted (should skip), False otherwise
    
    Checks Performed:
        1. is_retracted flag (boolean field)
        2. retraction object (non-None value indicates retraction)
        3. Title contains retraction indicators: "[retracted]", "[redacted]", "retraction:", "withdrawn"
        4. Abstract contains retraction indicators (same as title)
    
    Design Notes:
        - Case-insensitive matching (converts to lowercase)
        - Checks both title and abstract (retraction may be in either)
        - Returns True on any indicator (conservative - skip if any doubt)
        - Handles missing fields gracefully (returns False if field doesn't exist)
    """
    # Check for retracted flag
    if row.get('is_retracted') is True:
        return True
    
    # Check for retraction object
    if row.get('retraction') is not None:
        return True
    
    # Check title for redaction indicators
    title = (row.get('title') or '').lower()
    if any(indicator in title for indicator in ['[retracted]', '[redacted]', 'retraction:', 'withdrawn']):
        return True
    
    # Check abstract for redaction indicators
    abstract_inv = row.get('abstract_inverted_index')
    if abstract_inv:
        abstract = _recover_abstract(abstract_inv).lower()
        if any(indicator in abstract for indicator in ['[retracted]', '[redacted]', 'retraction:', 'withdrawn']):
            return True
    
    return False  # No retraction indicators found


def should_process_row(row, mongo_client):
    """Early filtering - check if row should be processed before submitting to executor.
    
    This function performs fast, lightweight checks to filter out rows that definitely
    shouldn't be processed. This happens BEFORE expensive operations like MongoDB queries
    or LLM API calls, improving overall throughput.
    
    Args:
        row (dict): OpenAlex work record to check
        mongo_client: MongoDB client (currently unused, kept for API compatibility)
    
    Returns:
        Tuple[bool, Optional[Tuple[str, List[int]]]]:
            - (True, (abstract, field_ids)): Row should be processed, return pre-extracted data
            - (False, None): Row should be skipped (filtered out)
    
    Filtering Criteria (all must pass):
        1. Not retracted or redacted
        2. Language is English ('en')
        3. Has abstract (abstract_inverted_index exists and is non-empty)
        4. Has at least one valid field ID (11-36)
    
    Design Notes:
        - Fast checks only (no MongoDB queries, no API calls)
        - Returns pre-extracted data (abstract, field_ids) to avoid re-extraction
        - Skips resume check here (too slow) - done later in process_row()
        - Returns (False, None) on any error (safe default - skip rather than crash)
    
    Performance Considerations:
        - Runs in main thread before executor submission (filters early)
        - No I/O operations (pure Python logic)
        - Returns extracted data to avoid re-extraction in process_row()
    """
    try:
        # ────────── FILTER 1: RETRACTED/REDACTED ──────────
        # Skip retracted or redacted papers (invalid content)
        if is_retracted_or_redacted(row):
            return False, None
        
        # ────────── FILTER 2: LANGUAGE ──────────
        # Only process English papers (other languages not supported)
        lang = row.get('language')
        if lang != 'en':
            return False, None
        
        # ────────── FILTER 3: ABSTRACT EXISTS ──────────
        # Must have abstract to generate Q&A from
        abstract_inv = row.get('abstract_inverted_index')
        if not abstract_inv:
            return False, None
        
        # Reconstruct abstract text from inverted index
        abstract = _recover_abstract(abstract_inv)
        if not abstract:
            return False, None  # Empty abstract - can't generate Q&A
        
        # ────────── FILTER 4: FIELD IDs EXIST ──────────
        # Must belong to at least one academic field (11-36)
        field_ids = extract_field_ids(row)
        if not field_ids:
            return False, None  # No valid fields - can't save to any database
        
        # ────────── SKIP RESUME CHECK ──────────
        # Resume check is too slow for filtering - done later in process_row()
        # This allows filtering to proceed quickly without blocking on MongoDB queries
        
        # Return True with pre-extracted data (avoids re-extraction in process_row)
        return True, (abstract, field_ids)
    except (KeyError, TypeError, AttributeError):
        # Any error in extraction - skip this row (safe default)
        return False, None


def process_row(node, mongo_client, row, abstract, field_ids):
    """Process a row that has already passed filtering.
    
    This function performs the actual Q&A generation and database save.
    It's called from ThreadPoolExecutor, so it runs in parallel across multiple threads.
    
    Args:
        node (str): Ollama node URL to use for generation
        mongo_client: MongoDB client connection
        row (dict): OpenAlex work record (already filtered by should_process_row)
        abstract (str): Pre-extracted abstract text (from should_process_row)
        field_ids (List[int]): Pre-extracted field IDs (from should_process_row)
    
    Returns:
        bool: True if row was successfully processed, False if skipped/failed
    
    Processing Steps:
        1. Check if record already processed (resume capability)
        2. Generate Q&A pair using LLM (with retries)
        3. Save Q&A to all relevant field databases
        4. Return True on success, False on failure
    
    Design Notes:
        - Runs in parallel (called from ThreadPoolExecutor)
        - Checks resume status here (not in filtering) to avoid blocking filter loop
        - Handles all exceptions gracefully (returns False, logs occasionally)
        - Only logs 1% of failures to reduce log spam
    
    Error Handling:
        - RuntimeError from generate_question_answer(): Logs 1% of failures, returns False
        - Other exceptions: Logs 1% of failures, returns False
        - Missing fields: Returns False silently (already validated in filtering)
    """
    try:
        # ────────── RESUME CHECK ──────────
        # Check if record is already processed (has Q&A in all relevant field databases)
        # Do this check here instead of in filtering to avoid blocking the filtering loop
        openalex_id = row.get('id')
        if openalex_id and is_record_processed(mongo_client, openalex_id, field_ids):
            return False  # Already processed, skip
        
        # ────────── GENERATE Q&A ──────────
        # Generate question once for all fields (same question saved to multiple field databases)
        try:
            question, answer = generate_question_answer(node, abstract)
        except RuntimeError as exc:
            # Generation failed after all retries
            # Only log occasionally to reduce spam
            if random.random() < 0.01:  # Log 1% of failures
                log(f"❌ Skipping row due to generation failures: {exc}")
            return False
        except Exception as e:
            # Unexpected error during generation
            if random.random() < 0.01:  # Log 1% of failures
                log(f"⚠️ Unexpected error generating Q/A: {e}")
            return False

        # ────────── SAVE TO DATABASE ──────────
        # Save the same question to all field databases this paper belongs to
        upsert_and_update(mongo_client, row, question, answer, field_ids)
        return True  # Success!
    except (KeyError, TypeError, AttributeError):
        # Missing required fields - skip (shouldn't happen after filtering, but be safe)
        return False


def timeout_monitor():
    """Watchdog thread that monitors script activity and restarts if stuck.
    
    This function runs in a background thread and checks if the script has
    produced any log output recently. If no activity for LOG_TIMEOUT_SECONDS,
    it assumes the script is stuck and restarts it.
    
    Design Notes:
        - Runs as daemon thread (dies when main thread exits)
        - Checks every 30 seconds (not too frequent, not too slow)
        - Uses os.execv() to restart script (replaces current process)
        - Only restarts if truly stuck (LOG_TIMEOUT_SECONDS = 1 hour)
    
    Restart Mechanism:
        - Uses os.execv() to replace current process with new Python process
        - Passes original sys.argv to maintain command-line arguments
        - Effectively restarts script from beginning (loses current state)
    
    False Positive Prevention:
        - LOG_TIMEOUT_SECONDS set to 1 hour (very conservative)
        - Only triggers if NO log activity (not just slow processing)
        - Progress bar updates don't count (only log() calls update timer)
    """
    global last_log_time
    while True:
        # Only restart if no activity AND progress bar hasn't moved
        time.sleep(30)  # Check less frequently (every 30 seconds)
        
        # Check if last log was more than LOG_TIMEOUT_SECONDS ago
        if datetime.now() - last_log_time > timedelta(seconds=LOG_TIMEOUT_SECONDS):
            log(f"⚠️ No activity for {LOG_TIMEOUT_SECONDS}s. Restarting...")
            # Restart script by replacing current process
            # sys.argv[0] is script name, rest are arguments
            os.execv(sys.executable, ['python'] + sys.argv)


def run_database_backup(mongo_uri: str) -> None:
    """Execute the backup script with force/drop semantics for all field DBs.
    
    Calls the backup_field_databases script to create backups of all field_* databases.
    Uses --force and --drop-existing flags to overwrite any existing backups.
    
    Args:
        mongo_uri (str): MongoDB connection string
    
    Design Notes:
        - Temporarily modifies sys.argv to pass arguments to backup script
        - Uses --force to skip confirmations (non-interactive)
        - Uses --drop-existing to overwrite old backups
        - Restores original sys.argv after completion (important for cleanup)
    
    Error Handling:
        - Exceptions are caught and logged, but don't stop main script
        - Backup failures are non-fatal (script continues)
    """
    log("💾 Starting field database backup...")
    original_argv = sys.argv  # Save original arguments
    try:
        # Modify sys.argv to pass arguments to backup script
        sys.argv = [
            "backup_field_databases.py",
            "--mongo-uri",
            mongo_uri,
            "--force",  # Skip confirmations
            "--drop-existing",  # Overwrite existing backups
            "--yes",  # Auto-confirm
        ]
        backup_field_databases.main()  # Call backup script's main function
    finally:
        sys.argv = original_argv  # Restore original arguments (important!)


def run_database_prune(mongo_uri: str, limit: int = 1000) -> None:
    """Prune each field database down to the working-set size via helper script.
    
    Calls the prune_field_databases script to limit each field_* database to
    the specified number of documents. This keeps databases manageable in size.
    
    Args:
        mongo_uri (str): MongoDB connection string
        limit (int): Maximum number of documents to keep per database (default: 1000)
    
    Design Notes:
        - Temporarily modifies sys.argv to pass arguments to prune script
        - Uses --force to skip confirmations (non-interactive)
        - Uses --yes to auto-confirm
        - Restores original sys.argv after completion
    
    Pruning Strategy:
        - Keeps most recent documents (based on insertion order or timestamp)
        - Removes older documents to stay under limit
        - Applied to each field database independently
    """
    log("🧹 Pruning field databases to working set...")
    original_argv = sys.argv  # Save original arguments
    try:
        # Modify sys.argv to pass arguments to prune script
        sys.argv = [
            "prune_field_databases.py",
            "--mongo-uri",
            mongo_uri,
            "--limit",
            str(limit),
            "--force",  # Skip confirmations
            "--yes",  # Auto-confirm
        ]
        prune_field_databases.main()  # Call prune script's main function
    finally:
        sys.argv = original_argv  # Restore original arguments


if __name__ == '__main__':
    # ────────── INITIALIZATION ──────────
    # Ensure MongoDB and Ollama are installed and running before watchdog
    # These are long-running operations, so do them before starting watchdog
    # (prevents watchdog from restarting during legitimate setup)
    
    ensure_mongodb_installed_and_running()
    ensure_ollama_instances()

    # ────────── START WATCHDOG ──────────
    # Start watchdog after long setup steps to avoid premature restarts
    # Daemon thread dies automatically when main thread exits
    monitor_thread = Thread(target=timeout_monitor, daemon=True)
    monitor_thread.start()

    # ────────── CONNECT TO MONGODB ──────────
    # Use connection pooling for better performance with high concurrency
    # maxPoolSize=100 allows up to 100 concurrent connections
    # minPoolSize=10 keeps 10 connections warm (reduces connection overhead)
    log("🔌 Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI, maxPoolSize=100, minPoolSize=10)
    log("✅ MongoDB connected")

    # ────────── DOWNLOAD OPENALEX MANIFEST ──────────
    # OpenAlex provides a manifest file listing all data file URLs
    # This manifest is a JSON file with metadata about each data file
    log("📥 Downloading OpenAlex manifest...")
    manifest_url = 'https://openalex.s3.amazonaws.com/data/works/manifest'
    manifest = requests.get(manifest_url).json()
    works_entries = manifest.get('entries', [])
    log(f"✅ Manifest downloaded, found {len(works_entries)} URL entries")

    # ────────── CONVERT S3 URLs TO HTTPS ──────────
    # OpenAlex manifest uses s3:// URLs, but we need https:// for downloads
    # Convert s3://openalex/... to https://openalex.s3.amazonaws.com/...
    urls = []
    for entry in works_entries:
        url = entry.get('url')
        if isinstance(url, str) and url.startswith('s3://openalex/'):
            url = url.replace('s3://openalex/', 'https://openalex.s3.amazonaws.com/')
        urls.append(url)

    # Calculate total records for progress bar
    total_records = sum(entry.get('meta', {}).get('record_count', 0) for entry in works_entries)
    log(f"📊 Total records to process: {total_records:,}")
    
    # Create progress bar (updates as records are processed)
    progress_bar = tqdm(total=total_records, unit='it', desc=f'{get_hostname()} QA Generation')
    log(f"🚀 Starting to process {len(urls)} URLs...")

    def process_url(url):
        """Process a single OpenAlex data file URL.
        
        This function handles one URL (one data file) from the manifest:
        1. Streams JSONL file in chunks (to limit memory usage)
        2. Filters rows early (before executor submission)
        3. Submits filtered rows to ThreadPoolExecutor for parallel processing
        4. Updates progress bar as records are processed
        
        Args:
            url (str): URL to OpenAlex data file (JSONL format)
        
        Returns:
            int: Number of records successfully processed
        
        Design Notes:
            - Streaming approach: reads file in chunks, doesn't load entire file into memory
            - Early filtering: filters rows before executor (reduces overhead)
            - Parallel processing: uses ThreadPoolExecutor for concurrent Q&A generation
            - Smart node selection: distributes load across healthy Ollama instances
            - Resume capability: skips already-processed records (checked in process_row)
        
        Performance Optimizations:
            - Chunksize=8192: balances memory usage and I/O efficiency
            - Early filtering: reduces executor queue size
            - Streaming results: processes next chunk while current chunk is still running
            - Capped workers: max_workers=60 prevents overwhelming system
        """
        processed = 0
        skipped_records = 0
        
        # Skip URL-level resume check - it's too slow. Rely on record-level check instead.
        # The record-level check in process_row() will skip already-processed records efficiently.
        log(f"📥 Starting to process URL: {url}")
        
        # Stream in chunks to limit memory usage
        try:
            # ────────── SETUP EXECUTOR ──────────
            # Reduced concurrency: 3-4 requests per node = 45-60 total workers
            # Calculate max workers based on number of Ollama nodes
            max_workers = len(nodes) * WORKERS_PER_NODE if nodes else 45
            max_workers = min(max_workers, 60)  # Cap at 60 to prevent overwhelming
            executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # ────────── PROCESS CHUNKS ──────────
            chunk_count = 0
            # Read JSONL file in chunks (streaming - doesn't load entire file)
            for chunk in pd.read_json(url, lines=True, chunksize=8192):
                chunk_count += 1
                if chunk_count == 1:
                    log(f"📊 Processing first chunk from {url} ({len(chunk)} records)")
                if chunk.empty:
                    continue  # Skip empty chunks

                # ────────── EARLY FILTERING ──────────
                # Filter rows before executor submission (reduces overhead)
                filtered_rows = []
                for _, row in chunk.iterrows():
                    should_process, data = should_process_row(row, mongo_client)
                    if should_process and nodes:
                        # Row passed filtering - add to executor queue
                        filtered_rows.append((row, data[0], data[1]))  # row, abstract, field_ids
                    elif should_process is False and data is None:
                        # Record was skipped (already processed or filtered out)
                        skipped_records += 1
                
                if chunk_count == 1:
                    log(f"🔍 Filtered chunk: {len(filtered_rows)} records to process, {skipped_records} skipped")
                
                # ────────── SUBMIT TO EXECUTOR ──────────
                # Submit only filtered rows to executor (reduces queue size)
                futures = []
                for row, abstract, field_ids in filtered_rows:
                    # Use smart node selection that avoids overloaded nodes
                    node = get_available_node()
                    # Submit task to executor (runs in parallel)
                    futures.append(executor.submit(process_row, node, mongo_client, row, abstract, field_ids))
                
                if chunk_count == 1 and len(futures) > 0:
                    log(f"🚀 Submitted {len(futures)} tasks to executor")
                
                # ────────── PROCESS RESULTS ──────────
                # Process results as they complete (streaming approach)
                # Don't wait for all - process next chunk while this one is still running
                for f in futures:
                    try:
                        if f.result():  # Returns True if processed successfully
                            processed += 1
                    except Exception:
                        pass  # Already logged in process_row (no need to log again)
                
                # Update progress bar (shows overall progress across all URLs)
                progress_bar.update(len(chunk))
            
            # ────────── CLEANUP ──────────
            executor.shutdown(wait=True)  # Wait for all tasks to complete
            
            # Log skipped records occasionally (10% of the time to reduce spam)
            if skipped_records > 0 and random.random() < 0.1:
                log(f"⏭️ Skipped {skipped_records} already-processed records in {url}")
        except Exception as e:
            log(f"⚠️ Error processing chunk from {url}: {e}")
        return processed

    # ────────── PROCESS ALL URLS ──────────
    # Process each URL from the manifest sequentially
    # (URLs are processed one at a time, but records within each URL are processed in parallel)
    total_processed = 0
    for url in urls:
        try:
            total_processed += process_url(url)
        except Exception as e:
            log(f"⚠️ Error processing {url}: {e}")

    # ────────── CLEANUP ──────────
    progress_bar.close()
    mongo_client.close()
    log(f"🎉 Completed QA generation. Processed {total_processed} documents with Q/A.")

    # ────────── POST-PROCESSING ──────────
    # Run backup and pruning maintenance tasks
    try:
        run_database_backup(MONGO_URI)
        run_database_prune(MONGO_URI)
        log("✅ Backup and prune steps completed.")
    except Exception as exc:
        log(f"⚠️ Post-processing maintenance encountered an issue: {exc}")

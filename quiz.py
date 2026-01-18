#!/usr/bin/env python3
"""Adaptive quiz runner that evaluates many LLMs with dynamic Ollama scaling.

This script evaluates a large list of LLM models by:
1. Loading Q&A pairs from MongoDB field_* databases
2. Dynamically scaling Ollama instances based on model size and GPU memory
3. Testing each model's ability to answer questions correctly
4. Scoring responses and writing results to CSV

ARCHITECTURE OVERVIEW:
- Dynamically calculates optimal Ollama instance count based on model memory requirements
- Distributes questions across multiple Ollama instances for parallel processing
- Implements intelligent load balancing and node selection
- Handles model-specific memory requirements and GPU distribution
- Large models (67B+) are spread across all GPUs (one instance per GPU)
- Smaller models can have multiple instances per GPU for higher throughput

PERFORMANCE OPTIMIZATIONS:
- Dynamic worker count based on CPU cores, RAM, and GPU count
- Per-node batch processing to minimize overhead
- Smart node selection to avoid overloaded instances
- Model cleanup after each evaluation (frees GPU memory)
- Connection pooling for MongoDB (scaled to worker count)
"""

from __future__ import annotations
import csv, logging, sys, re, signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import suppress
import subprocess
import shutil
import time
import os
import atexit
import ollama
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ────────── CONFIGURATION AND STATE ──────────

# INSTANCES: List of Ollama instance configuration dicts
# Each dict contains: {"host": "http://127.0.0.1:11434", "port": "11434", "gpu": "0"}
# Populated by plan_instances_for_model() based on model memory requirements
# Used to launch and manage Ollama server processes
INSTANCES: List[Dict[str, str]] = []

# OLLAMA_NODES: List of Ollama instance URLs (e.g., ["http://127.0.0.1:11434", ...])
# Extracted from INSTANCES for easy access
# Used for load balancing and distributing questions across instances
OLLAMA_NODES: List[str] = []
# MODEL_MEMORY_OVERRIDES_MB: Manual memory estimates for specific models (MB)
# Some models have non-standard memory footprints that don't follow the standard formula
# These overrides ensure accurate instance planning
# Format: "model-name:size" -> memory_MB
MODEL_MEMORY_OVERRIDES_MB = {
    "smollm:135m": 906,      # Small model, low memory footprint
    "smollm2:135m": 906,     # Small model variant
    "smollm:360m": 1200,     # Medium-small model
    "smollm2:360m": 1200,    # Medium-small model variant
    "gemma3:1b": 2000,       # 1B parameter model
    "gemma3:270m": 1200,     # 270M parameter model
}

# Dynamic values will be calculated after GPU_MEMORY_MB is set (see bottom of file)

# QUESTIONS_PER_FIELD: Number of questions to test per academic field
# Higher values = more accurate evaluation but longer runtime
# 1000 questions per field = 26,000 total questions per model
QUESTIONS_PER_FIELD = 1_000

# MAX_REPROMPTS: Maximum number of retry attempts if LLM gives invalid answer
# Each reprompt includes error feedback to guide LLM toward correct format
# Higher values = more resilient but slower (each reprompt takes ~1-2 seconds)
MAX_REPROMPTS = 2

# ────────── ANSWER VALIDATION CONSTANTS ──────────
# These constants define valid answer formats and normalization rules

# VALID_ANSWERS: Set of all valid answer strings (case-insensitive matching)
# LLM responses must normalize to one of these values
VALID_ANSWERS     = {"true", "false", "possibly true", "possibly false", "unknown"}

# TRUE_SET, FALSE_SET: Groupings for scoring purposes
# Answers in TRUE_SET are considered equivalent (both indicate "true" answer)
# Answers in FALSE_SET are considered equivalent (both indicate "false" answer)
# Used by score() function to determine if prediction matches ground truth
TRUE_SET, FALSE_SET = {"true", "possibly true"}, {"false", "possibly false"}

# ALIAS: Common misspellings and typos that should be normalized
# Maps misspelled variants to correct answer strings
# Example: "possbilytrue" (typo) -> "possibly true" (correct)
ALIAS = {"possbilytrue": "possibly true", "possiblyfalse": "possibly false"}

# NUMERIC_CHOICES: Maps numeric responses (1-5) to answer strings
# Some LLMs respond with numbers instead of words
# Format: "1" -> "true", "2" -> "false", etc.
NUMERIC_CHOICES = {
    "1": "true",
    "2": "false",
    "3": "possibly true",
    "4": "possibly false",
    "5": "unknown",
}
# FILLER_PHRASES: Maps common verbose phrases to valid answers
# LLMs sometimes respond with full sentences instead of single words
# This dictionary extracts the answer from verbose responses
# Example: "it is false" -> "false", "not sure" -> "unknown"
FILLER_PHRASES = {
    "possible truth": "possibly true",
    "possibly truth": "possibly true",
    "possible true": "possibly true",
    "it is false": "false",
    "the answer is no": "false",
    "answer is no": "false",
    "answer is false": "false",
    "possible false": "possibly false",
    "possibly false": "possibly false",
    "not sure": "unknown",
    "can't tell": "unknown",
    "cannot tell": "unknown",
    "no idea": "unknown",
}

# FILLER_SYNONYMS: Maps synonyms to valid answer strings
# LLMs sometimes use synonyms instead of exact words
# This dictionary normalizes synonyms to standard answers
# Example: "yes" -> "true", "maybe" -> "possibly true"
FILLER_SYNONYMS = {
    "true": {"yes", "yeah", "yep"},
    "false": {"no", "nope", "nah", "negative", "incorrect"},
    "possibly true": {"maybe", "probably", "likely", "perhaps"},
    "possibly false": {"unlikely"},
    "unknown": {"unknown", "unsure", "uncertain", "indeterminate"},
}

# FILLER_HINTS: Phrases that indicate LLM is adding unnecessary preamble
# These phrases suggest the response is not in the required format
# Example: "here is the answer" indicates filler text before the actual answer
FILLER_HINTS = (
    "the paper",
    "here is",
    "here's",
    "the question is",
    "to answer this question",
    "what a fascinating question",
    "the answer to this question",
    "the answer to the above question",
    "this paper",
    "the abstract",
)

def looks_like_filler_response(text: str) -> bool:
    """Check if response contains filler phrases (indicates invalid format).
    
    LLMs sometimes add preambles like "here is the answer" which violates
    the required format. This function detects such responses.
    
    Args:
        text (str): LLM response to check
    
    Returns:
        bool: True if response contains filler phrases, False otherwise
    
    Design Notes:
        - Case-insensitive matching (converts to lowercase)
        - Checks if any FILLER_HINTS phrase appears in response
        - Used to detect responses that need reprompting
    """
    lowered = text.lower()
    return any(hint in lowered for hint in FILLER_HINTS)
# REPROMPT_TEMPLATE: Template for error feedback when LLM gives invalid answer
# Appended to original prompt when reprompting after invalid response
# Includes the invalid answer and reason, plus clear instructions for correct format
# Format: Uses Python string formatting with {answer} and {reason} placeholders
REPROMPT_TEMPLATE = (
    "\n\nYour previous answer \"{answer}\" was invalid because {reason}. "
    "Respond with exactly one lowercase word: true, false, possibly true, possibly false, unknown. "
    "Do not start with filler phrases such as \"here is\" or \"the answer is\", and do not write any sentences. "
    "Copy the word exactly as spelled above with no punctuation or extras. "
    "If you are unsure, reply with unknown. Never leave the reply blank."
)

# FIELD_MAP: Maps MongoDB database names to human-readable field names
# OpenAlex uses field IDs 11-36 to represent 26 academic fields
# Format: "field_11" -> "agricultural-and-biological-sciences"
# Used for logging, CSV headers, and user-facing output
FIELD_MAP = {
    "field_11":"agricultural-and-biological-sciences",
    "field_12":"arts-and-humanities",
    "field_13":"biochemistry-genetics-and-molecular-biology",
    "field_14":"business-management-and-accounting",
    "field_15":"chemical-engineering",
    "field_16":"chemistry",
    "field_17":"computer-science",
    "field_18":"decision-sciences",
    "field_19":"earth-and-planetary-sciences",
    "field_20":"economics-econometrics-and-finance",
    "field_21":"energy",
    "field_22":"engineering",
    "field_23":"environmental-science",
    "field_24":"immunology-and-microbiology",
    "field_25":"materials-science",
    "field_26":"mathematics",
    "field_27":"medicine",
    "field_28":"neuroscience",
    "field_29":"nursing",
    "field_30":"pharmacology-toxicology-and-pharmaceutics",
    "field_31":"physics-and-astronomy",
    "field_32":"psychology",
    "field_33":"social-sciences",
    "field_34":"veterinary",
    "field_35":"dentistry",
    "field_36":"health-professions",
}

# DATABASES: List of MongoDB database names (extracted from FIELD_MAP keys)
# Used to iterate through all field databases when loading questions
DATABASES = list(FIELD_MAP.keys())

# ────────── MODEL LIST ──────────
# List of LLM models to evaluate, ordered from smallest to largest
# Models are tested sequentially (one at a time)
# Each model gets its own Ollama instance configuration based on memory requirements
# Models starting from "deepseek-llm:67b" are spread across all GPUs (one per GPU)
# Smaller models can have multiple instances per GPU for higher throughput
# --- model list trimmed to chat-capable models that fit on a 4090 ---
MODEL_LIST = [
    "smollm2:135m",
    "smollm:135m",
    "gemma3:270m",
    "smollm2:360m",
    "smollm:360m",
    "qwen:0.5b",
    "qwen2:0.5b",
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    "gemma3:1b",
    "tinyllama:1.1b",
    "tinydolphin:1.1b",
    "deepscaler:1.5b",
    "qwen2:1.5b",
    "qwen2.5:1.5b",
    "smollm2:1.7b",
    "smollm:1.7b",
    "qwen3:1.7b",
    "qwen:1.8b",
    "gemma:2b",
    "gemma2:2b",
    "granite3.3:2b",
    "granite3.2:2b",
    "exaone-deep:2.4b",
    "phi-2:2.7b",
    "dolphin-phi:2.7b",
    "qwen2.5:3b",
    "orca-mini:3b",
    "cogito:3b",
    "phi3:3.8b",
    "phi-4-mini:3.8b",
    "qwen:4b",
    "qwen3:4b",
    "gemma3:4b",
    "mistral:7b",
    "qwen:7b",
    "qwen2:7b",
    "qwen2.5:7b",
    "llama2:7b",
    "openchat:7b",
    "orca-mini:7b",
    "wizard-vicuna-uncensored:7b",
    "nous-hermes:7b",
    "olmo2:7b",
    "vicuna:7b",
    "dolphin-mistral:7b",
    "wizardlm2:7b",
    "openthinker:7b",
    "llama3:8b",
    "llama3.1:8b",
    "deepseek-r1:8b",
    "qwen3:8b",
    "dolphin3:8b",
    "aya:8b",
    "hermes3:8b",
    "gemma2:9b",
    "glm4:9b",
    "falcon3:10b",
    "nous-hermes2:10.7b",
    "mistral-nemo:12b",
    "llama2:13b",
    "orca-mini:13b",
    "wizard-vicuna-uncensored:13b",
    "vicuna:13b",
    "qwen:14b",
    "qwen3:14b",
    "qwen2.5:14b",
    "deepseek-r1:14b",
    "phi3:14b",
    "phi-4:14b",
    "cogito:14b",
    "gemma2:27b",
    "qwen3:30b",
    "wizard-vicuna-uncensored:30b",
    "qwen:32b",
    "qwen2.5:32b",
    "qwen3:32b",
    "deepseek-r1:32b",
    "vicuna:33b",
    "nous-hermes2:34b",
    "command-r:35b",
    "aya:35b",
    "falcon:40b",

]

# ────────── FILE PATHS AND CONFIGURATION ──────────

# MONGO_URI: MongoDB connection string (default: localhost:27017)
MONGO_URI = "mongodb://localhost:27017/"

# LOG_PATH: Path to log file (stores all log output for debugging)
LOG_PATH = Path("quiz_llms.log")

# CSV_PATH: Path to results CSV file (stores model scores per field)
# Format: model, overall_score, field_11_score, field_12_score, ..., timestamp
CSV_PATH = Path("results.csv")

# ────────── LOGGING CONFIGURATION ──────────
# Configure Python logging to output to both console and file
# INFO level captures all important events (not DEBUG/TRACE noise)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler(LOG_PATH, encoding="utf-8")  # File output
    ]
)

# Reduce verbosity of third-party libraries (they're too chatty)
logging.getLogger("httpx").setLevel(logging.WARNING)  # HTTP library
logging.getLogger("ollama").setLevel(logging.WARNING)  # Ollama client
logging.getLogger("ollama._client").setLevel(logging.WARNING)  # Ollama internal

# Create logger for this module
logger = logging.getLogger(__name__)

# Convenience aliases for common log operations
log = logger.info      # Info messages (normal operation)
log_warn = logger.warning  # Warning messages (non-fatal issues)
log_err = logger.error  # Error messages (failures, invalid responses)

#if not MODEL_LIST:
#    MODEL_LIST = ["gemma:2b","gemma:7b","llama3:8b","mistral","phi3-mini","tinyllama",
#                  "qwen2:7b","mixtral-8x7b","gemma2b-it","mistral-openorca"]#
#    log_err(f"models.txt not found or empty; using fallback MODEL_LIST ({len(MODEL_LIST)} entries)")

# ────────── SCORING CONSTANTS ──────────
# Scoring system: each question is worth CORRECT_SCORE points if correct
# QUESTIONS_PER_FIELD = 1000, so max score per field = 1000 * 0.1 = 100 points
# Max total score = 26 fields * 100 = 2600 points (if all correct)

# CORRECT_SCORE: Points awarded for correct answer (matches ground truth)
# 1000 correct answers → 100 points per field
CORRECT_SCORE = 0.1

# INCORRECT_SCORE: Points deducted for incorrect answer (doesn't match ground truth)
# Negative score penalizes wrong answers
INCORRECT_SCORE = -0.1

# ────────── GLOBAL STATE TRACKING ──────────

# OLLAMA_PROCESSES: List of subprocess.Popen objects for Ollama server processes
# Used to track and terminate Ollama instances on shutdown
OLLAMA_PROCESSES: List[subprocess.Popen] = []

# OLLAMA_MANAGED_PIDS: Set of process IDs for Ollama instances we manage
# Used to ensure all instances are cleaned up on exit
OLLAMA_MANAGED_PIDS: set[int] = set()

# GLOBAL_SALVAGE_TOTAL: Total count of verbose answers that were "salvaged"
# Salvaging = extracting valid answer from verbose/invalid response
# Example: "The answer is true" -> salvaged to "true"
GLOBAL_SALVAGE_TOTAL: int = 0

# GLOBAL_SALVAGE_BY_MODEL: Dict mapping model name -> count of salvaged answers
# Tracks which models tend to give verbose responses that need salvaging
GLOBAL_SALVAGE_BY_MODEL: Dict[str, int] = {}

# GLOBAL_SALVAGE_BY_FIELD: Dict mapping field name -> count of salvaged answers
# Tracks which fields tend to produce verbose responses
GLOBAL_SALVAGE_BY_FIELD: Dict[str, int] = defaultdict(int)


def _register_pid(pid: int) -> None:
    """Register a process ID as managed by this script.
    
    Tracks PIDs so we can clean them up on exit. Only registers valid PIDs (> 0).
    
    Args:
        pid (int): Process ID to register
    
    Design Notes:
        - Only registers positive PIDs (0 and negative are invalid)
        - Used to track Ollama server processes we launch
        - Ensures cleanup on script exit (via atexit handler)
    """
    if pid > 0:
        OLLAMA_MANAGED_PIDS.add(pid)


def _detect_pids_for_ports(ports: list[str]) -> set[int]:
    """Detect process IDs listening on specified ports.
    
    Uses 'ss' command (socket statistics) to find which processes are bound
    to the given ports. This helps identify existing Ollama instances.
    
    Args:
        ports (list[str]): List of port numbers to check (e.g., ["11434", "11435"])
    
    Returns:
        set[int]: Set of process IDs listening on those ports
                 Returns empty set if detection fails or no processes found
    
    Algorithm:
        1. Run 'ss -ltnp' to list all listening TCP ports with process info
        2. Search output for lines containing our port numbers
        3. Extract PID from "pid=12345" pattern in output
        4. Return set of all found PIDs
    
    Design Notes:
        - Uses 'ss' command (modern replacement for 'netstat')
        - Handles errors gracefully (returns empty set on failure)
        - Port patterns include colon prefix (":11434") to avoid false matches
    """
    if not ports:
        return set()
    try:
        # Run 'ss' to list listening TCP ports with process info
        # -l: listening sockets only
        # -t: TCP only
        # -n: numeric (don't resolve hostnames)
        # -p: show process info (includes PID)
        output = subprocess.check_output(
            ["ss", "-ltnp"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return set()  # Command failed - return empty set

    # Build port patterns with colon prefix (e.g., ":11434")
    # Colon ensures we match port number, not part of IP address
    port_patterns = tuple(f":{port}" for port in ports)
    candidates: set[int] = set()
    
    # Search each line for our port patterns
    for line in output.splitlines():
        if any(pattern in line for pattern in port_patterns):
            # Extract PID from "pid=12345" pattern using regex
            candidates.update(int(match) for match in re.findall(r"pid=(\d+)", line))
    return candidates


def _pid_is_alive(pid: int) -> bool:
    """Check if a process ID is still running.
    
    Uses os.kill(pid, 0) which sends signal 0 (no-op) to check if process exists.
    Signal 0 doesn't actually kill the process, it just checks if it's alive.
    
    Args:
        pid (int): Process ID to check
    
    Returns:
        bool: True if process is alive, False if dead or doesn't exist
    
    Design Notes:
        - os.kill(pid, 0) raises OSError if process doesn't exist
        - Signal 0 is a no-op (doesn't affect the process)
        - Fast check (no subprocess overhead)
    """
    try:
        os.kill(pid, 0)  # Signal 0 = check if process exists (doesn't kill)
    except OSError:
        return False  # Process doesn't exist or permission denied
    return True


def _terminate_pid(pid: int, timeout: float = 5.0) -> None:
    """Terminate a process gracefully, then forcefully if needed.
    
    Attempts graceful shutdown first (SIGTERM), then forceful kill (SIGKILL) if
    process doesn't exit within timeout. This ensures processes are cleaned up.
    
    Args:
        pid (int): Process ID to terminate
        timeout (float): Seconds to wait for graceful shutdown before force kill (default: 5.0)
    
    Algorithm:
        1. Check if process is alive (skip if already dead)
        2. Send SIGTERM (graceful shutdown request)
        3. Wait up to timeout seconds for process to exit
        4. If still alive, send SIGKILL (forceful kill)
    
    Design Notes:
        - SIGTERM allows process to clean up (graceful)
        - SIGKILL cannot be caught/ignored (forceful)
        - Uses suppress() to handle ProcessLookupError (process already dead)
        - Polls every 0.1 seconds to check if process exited
    """
    if pid <= 0:
        return  # Invalid PID
    if not _pid_is_alive(pid):
        return  # Already dead, nothing to do

    # Step 1: Try graceful shutdown (SIGTERM)
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)  # Request graceful shutdown

    # Step 2: Wait for process to exit (up to timeout seconds)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            return  # Process exited gracefully
        time.sleep(0.1)  # Poll every 100ms

    # Step 3: Process still alive - force kill (SIGKILL)
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)  # Forceful kill (cannot be ignored)


def shutdown_ollama_instances(force_all: bool = False) -> None:
    """Terminate tracked Ollama processes and any strays bound to our ports.
    
    Comprehensive cleanup function that terminates all Ollama instances:
    1. Terminates processes we launched (tracked in OLLAMA_PROCESSES)
    2. Terminates processes we registered (tracked in OLLAMA_MANAGED_PIDS)
    3. If force_all=True, also detects and terminates any processes on our ports
    4. Finally, kills any remaining "ollama serve" processes system-wide
    
    Args:
        force_all (bool): If True, also terminate processes detected on our ports
                         (catches stray processes we didn't launch)
    
    Design Notes:
        - Tries graceful shutdown first (terminate), then force kill if needed
        - Uses pkill as final fallback to catch any remaining processes
        - Clears OLLAMA_MANAGED_PIDS after termination (prevents double-kill)
        - Suppresses exceptions to ensure cleanup continues even if some fail
    """
    # Step 1: Terminate processes we launched via subprocess.Popen
    for proc in list(OLLAMA_PROCESSES):
        if proc.poll() is None:  # Process still running
            with suppress(Exception):
                proc.terminate()  # Request graceful shutdown
                proc.wait(timeout=5)  # Wait up to 5 seconds
        if proc.poll() is None:  # Still running after terminate
            with suppress(Exception):
                proc.kill()  # Force kill
        OLLAMA_PROCESSES.remove(proc)

    # Step 2: Terminate processes we registered (by PID)
    managed_pids = set(OLLAMA_MANAGED_PIDS)
    if force_all:
        # Also detect processes on our ports (catches strays)
        managed_pids |= _detect_pids_for_ports([inst["port"] for inst in INSTANCES])

    # Terminate each registered PID
    for pid in managed_pids:
        _terminate_pid(pid)

    # Clear tracking set (processes are dead, no need to track anymore)
    OLLAMA_MANAGED_PIDS.clear()
    
    # Step 3: Final fallback - kill any remaining "ollama serve" processes
    # This catches processes we didn't track or that were launched externally
    with suppress(Exception):
        subprocess.run(
            ["pkill", "-f", "ollama serve"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


# Register cleanup handler to run on script exit (normal or abnormal)
# Ensures Ollama instances are always cleaned up, even if script crashes
atexit.register(lambda: shutdown_ollama_instances(force_all=True))

# ────────── ENVIRONMENT HELPERS ──────────

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Execute a shell command and return result.
    
    Wrapper around subprocess.run() that doesn't raise on non-zero exit codes.
    Used for commands where failure is acceptable (e.g., checking if service exists).
    
    Args:
        cmd (list[str]): Command and arguments as list
    
    Returns:
        subprocess.CompletedProcess: Result object with returncode, stdout, stderr
    
    Design Notes:
        - check=False means non-zero exit codes don't raise exceptions
        - Captures both stdout and stderr for debugging
        - Caller should check returncode to determine success/failure
    """
    return subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ensure_ollama_installed() -> None:
    """Ensure Ollama is installed on the system.
    
    Checks if 'ollama' executable exists in PATH. If not, attempts to install
    it using the official Ollama installation script.
    
    Raises:
        RuntimeError: If installation fails (script should not continue)
    
    Design Notes:
        - Uses shutil.which() to check if ollama is in PATH
        - Installation script handles all dependencies automatically
        - Raises exception on failure (installation is required for script to work)
    """
    if shutil.which("ollama"):
        return  # Already installed
    
    # Install using official script
    install_cmd = ["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"]
    result = subprocess.run(install_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to install Ollama automatically. Please install it manually.")


def _service_action(args: list[str]) -> bool:
    """Execute a systemctl/service command and return success status.
    
    Wrapper for systemctl/service commands (start, stop, enable, etc.).
    
    Args:
        args (list[str]): Command arguments (e.g., ["systemctl", "stop", "ollama"])
    
    Returns:
        bool: True if command succeeded (returncode == 0), False otherwise
    """
    return _run(args).returncode == 0


def ensure_ollama_running() -> None:
    """Ensure Ollama instances are running for all planned instances.
    
    This function launches Ollama server processes for each instance in INSTANCES:
    1. Stops any existing Ollama system service
    2. Detects and registers any existing processes on our ports
    3. Shuts down existing instances if found
    4. Launches new instances with proper GPU pinning
    5. Verifies each instance is healthy before returning
    
    Raises:
        RuntimeError: If INSTANCES is empty or if instance fails to start
    
    Design Notes:
        - Each instance gets its own port and GPU (via CUDA_VISIBLE_DEVICES)
        - Uses ping() function to verify instances are responding
        - Waits 2 seconds after launch for instance to initialize
        - Raises exception if instance doesn't respond (indicates failure)
    
    GPU Pinning:
        - CUDA_VISIBLE_DEVICES restricts which GPU the process can see
        - Instance on GPU 0 only sees GPU 0 (even if system has multiple GPUs)
        - This ensures proper GPU distribution across instances
    """
    if not INSTANCES:
        raise RuntimeError("No Ollama instances planned. Call restart_ollama_instances(model) first.")
    
    # Ensure Ollama is installed before trying to launch
    ensure_ollama_installed()
    
    # Detect any existing processes on our ports
    ports = [inst["port"] for inst in INSTANCES]
    existing_pids = _detect_pids_for_ports(ports)
    
    # Register existing PIDs so we can clean them up
    for pid in existing_pids:
        _register_pid(pid)
    
    # If instances already exist, shut them down first
    if existing_pids:
        shutdown_ollama_instances(force_all=True)
        time.sleep(0.5)  # Brief wait for processes to fully terminate

    # Helper function to check if instance is responding
    def ping(host: str) -> bool:
        """Check if Ollama instance at host is responding to API calls."""
        with suppress(Exception):
            ollama.Client(host=host).list()  # Simple health check
            return True
        return False

    # Stop system-level Ollama service if it exists (may conflict with our instances)
    if shutil.which("systemctl"):
        _service_action(["systemctl", "stop", "ollama"])
        time.sleep(1)

    # Launch each planned instance
    for inst in INSTANCES:
        # Check if instance is already running (may have been launched externally)
        if ping(inst["host"]):
            # Already running - register its PID and skip launch
            for pid in _detect_pids_for_ports([inst["port"]]):
                _register_pid(pid)
            continue  # Skip to next instance

        # Build environment variables for this instance
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = inst["gpu"]  # Pin to specific GPU
        env["OLLAMA_HOST"] = f"0.0.0.0:{inst['port']}"  # Bind to this port

        log(f"Starting Ollama instance on port {inst['port']} using GPU {inst['gpu']}")
        
        # Launch ollama serve as background process
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,  # Suppress output (noise)
            stderr=subprocess.DEVNULL,  # Suppress errors (noise)
            env=env,
        )
        
        # Track this process
        OLLAMA_PROCESSES.append(proc)
        _register_pid(proc.pid)
        
        # Wait for instance to initialize
        time.sleep(2)
        
        # Verify instance is responding
        if not ping(inst["host"]):
            raise RuntimeError(f"Unable to reach Ollama at {inst['host']}. Please ensure the service is running.")

    # Final pass: register any PIDs we might have missed
    for pid in _detect_pids_for_ports(ports):
        _register_pid(pid)


def restart_ollama_instances(model: str) -> None:
    """Stop any running Ollama servers and relaunch workers sized for *model*.
    
    This is the main function called before testing each model. It:
    1. Plans instance distribution based on model memory requirements
    2. Shuts down any existing Ollama instances
    3. Launches new instances configured for this specific model
    
    Args:
        model (str): Model name to configure instances for (e.g., "llama3:8b")
    
    Global State Modified:
        - INSTANCES: Replanned for this model
        - OLLAMA_NODES: Updated with new instance URLs
    
    Design Notes:
        - Called once per model (before testing that model)
        - Instance count depends on model size and GPU memory
        - Large models (67B+) get one instance per GPU
        - Smaller models can have multiple instances per GPU
    """
    global INSTANCES, OLLAMA_NODES
    
    # Plan instance distribution for this model
    INSTANCES = plan_instances_for_model(model)
    OLLAMA_NODES = [inst["host"] for inst in INSTANCES]
    
    # Log instance plan
    mem_per_instance = estimate_model_memory_mb(model)
    log(f"Planning {len(INSTANCES)} Ollama workers for {model} (≈{mem_per_instance} MB each)")
    
    # Shutdown existing instances
    shutdown_ollama_instances(force_all=True)
    time.sleep(0.5)  # Brief wait for cleanup
    
    # Launch new instances
    ensure_ollama_running()


# ────────── MODEL MANAGEMENT HELPERS ──────────

# EMBEDMODELS: Set of model names that are embedding-only (don't support generation)
# These models are skipped to avoid wasting time trying to use them
EMBEDMODELS: set[str] = set()

def pull(model: str) -> bool:
    """Pull model to all Ollama nodes.
    
    Downloads the model to each Ollama instance so it's ready for inference.
    This is called once per model before testing begins.
    
    Args:
        model (str): Model name to pull (e.g., "llama3:8b")
    
    Returns:
        bool: True if model pulled successfully on all nodes, False otherwise
    
    Design Notes:
        - Pulls model on ALL nodes (each instance needs its own copy)
        - Detects embedding-only models and skips them
        - Returns False on any failure (model won't work if any node fails)
        - stream=False means wait for full download (not streaming progress)
    """
    # Skip if we already know this is an embedding-only model
    if model in EMBEDMODELS:
        return False
    
    # Pull model on each node
    for n in OLLAMA_NODES:
        try:
            ollama.Client(host=n).pull(model, stream=False)
        except ollama.ResponseError as e:
            # Check if model is embedding-only (doesn't support text generation)
            if "does not support generate" in str(e).lower():
                EMBEDMODELS.add(model)  # Remember this model is embedding-only
                log_err(f"embed-only {model}")
                return False
            log_err(f"pull error {n} {model}: {e}")
            return False
        except Exception as e:
            log_err(f"pull exc {n} {model}: {e}")
            return False
    return True  # Successfully pulled on all nodes

def delete(model: str):
    """Delete model from all Ollama nodes to free GPU memory.
    
    Called after testing a model to free up GPU memory for the next model.
    Uses force=True to delete even if model is in use.
    
    Args:
        model (str): Model name to delete
    
    Design Notes:
        - Deletes from all nodes (each instance has its own copy)
        - Uses force=True to ensure deletion even if model is loaded
        - Suppresses exceptions (deletion failure is non-fatal)
        - Called after each model test to free memory
    """
    for n in OLLAMA_NODES:
        try:
            ollama.Client(host=n).delete(model, force=True)
        except Exception:
            pass  # Deletion failure is non-fatal (model may not exist)


def stop_model_on_all_nodes(model: str) -> None:
    """Stop/unload model from all Ollama nodes.
    
    Unloads the model from GPU memory without deleting it from disk.
    Called before deleting model to ensure it's not in use.
    
    Args:
        model (str): Model name to stop
    
    Design Notes:
        - Stops model on all nodes (each instance may have it loaded)
        - Suppresses exceptions (stop failure is non-fatal)
        - Called before delete() to ensure clean removal
    """
    for node in OLLAMA_NODES:
        with suppress(Exception):
            ollama.Client(host=node).stop(model)

def ask(model: str, prompt: str, node: str) -> str:
    """Query an Ollama instance to generate a response.
    
    Sends a prompt to the specified Ollama instance and returns the generated response.
    This is the core function for getting LLM answers to quiz questions.
    
    Args:
        model (str): Model name to use (e.g., "llama3:8b")
        prompt (str): Prompt text to send to LLM
        node (str): Ollama instance URL (e.g., "http://127.0.0.1:11434")
    
    Returns:
        str: Generated response text (stripped of whitespace)
             Returns "unknown" on any error
    
    Generation Parameters:
        - temperature: 0.0 (deterministic, no randomness)
        - top_p: 0.9 (nucleus sampling, allows some variety)
        - num_predict: 10 (short responses, just enough for answer)
        - keep_alive: 0 (unload model after request to free memory)
    
    Design Notes:
        - Timeout is calculated dynamically based on worker count
        - Detects embedding-only models and marks them for skipping
        - Returns "unknown" on any error (safe default for scoring)
        - stream=False means wait for complete response (not streaming)
    """
    # Calculate timeout dynamically based on current worker count
    # More workers = higher load = longer response times = longer timeout needed
    num_nodes = len(OLLAMA_NODES) if OLLAMA_NODES else 1
    timeout_sec = _calculate_timeout_sec(WORKERS_PER_NODE, num_nodes)
    
    try:
        # Create client for this specific node
        client = ollama.Client(host=node, timeout=timeout_sec)
        
        # Generate response
        r = client.generate(
            model=model,
            prompt=prompt,
            stream=False,  # Wait for complete response (not streaming)
            options={
                "temperature": 0.0,  # Deterministic (no randomness)
                "top_p": 0.9,  # Nucleus sampling (allows some variety)
                "num_predict": 10,  # Short responses (enough for answer, prevents rambling)
                "keep_alive": 0,  # Unload model after request (free GPU memory)
            },
        )
        return r["response"].strip()  # Return stripped response
    except ollama.ResponseError as e:
        # Check if model is embedding-only
        if "does not support generate" in str(e).lower():
            EMBEDMODELS.add(model)  # Remember for future
        log_err(f"gen error {node} {model}: {e}")
        return "unknown"
    except Exception as e:
        log_err(f"gen exc {node} {model}: {e}")
        return "unknown"


def normalize_and_validate(raw: str) -> Tuple[str, bool, str]:
    """Normalize and validate LLM response, extracting valid answer if possible.
    
    This function implements a multi-stage validation and normalization process:
    1. Check for empty/None responses
    2. Normalize case and whitespace
    3. Check for exact matches
    4. Check for numeric choices (1-5)
    5. Check for answers with punctuation (strip and retry)
    6. Check for filler phrases (extract answer from verbose text)
    7. Check for synonyms (normalize common alternatives)
    8. Salvage valid tokens from verbose responses (last resort)
    
    Args:
        raw (str): Raw LLM response to validate
    
    Returns:
        Tuple[str, bool, str]: (normalized_answer, is_valid, reason)
            - normalized_answer: One of VALID_ANSWERS or "unknown"
            - is_valid: True if answer is acceptable, False if needs reprompt
            - reason: Human-readable explanation of validation result
    
    Validation Stages (in order):
        1. Empty check: None or empty string -> "unknown", invalid
        2. Numeric choice: "1", "2", etc. -> mapped to answer, valid
        3. Exact match: Matches VALID_ANSWERS exactly -> valid
        4. Punctuation strip: Remove trailing punctuation, retry -> invalid (needs reprompt)
        5. Filler phrases: Extract from verbose text -> valid (salvaged)
        6. Synonyms: Normalize common alternatives -> valid (salvaged)
        7. Token salvage: Find valid token in verbose response -> valid (salvaged)
        8. Default: No valid answer found -> "unknown", invalid
    
    Design Notes:
        - Progressive validation (tries strictest first, then lenient)
        - Salvaging allows extracting answers from verbose responses
        - Returns is_valid=False for punctuation issues (can be fixed with reprompt)
        - Returns is_valid=True for salvaged answers (acceptable even if not perfect)
    """
    # Stage 1: Empty/None check
    if raw is None:
        return "unknown", False, "Empty response"

    stripped = raw.strip()
    if not stripped:
        return "unknown", False, "Empty response"

    # Stage 2: Normalize case and whitespace
    # Collapse multiple spaces to single space, convert to lowercase
    normalized = " ".join(stripped.lower().split())
    # Apply alias corrections (fix common typos)
    normalized = ALIAS.get(normalized, normalized)

    # Stage 3: Check for numeric choices (1-5)
    # Some LLMs respond with numbers instead of words
    if re.fullmatch(r"[1-5](?:[).])?", normalized):
        mapped = NUMERIC_CHOICES[normalized[0]]  # Map "1" -> "true", etc.
        return mapped, True, "Numeric choice"

    # Stage 4: Check for exact match (perfect response)
    if normalized in VALID_ANSWERS and stripped.lower() == normalized:
        return normalized, True, "Exact match"

    # Stage 5: Try removing trailing punctuation
    # LLMs sometimes add punctuation: "true." -> should be "true"
    stripped_no_punct = stripped.rstrip(".!,?:;")
    collapsed_no_punct = " ".join(stripped_no_punct.lower().split())
    collapsed_no_punct = ALIAS.get(collapsed_no_punct, collapsed_no_punct)

    if collapsed_no_punct in VALID_ANSWERS:
        # Found valid answer after removing punctuation
        # Mark as invalid (needs reprompt) to encourage correct format
        return collapsed_no_punct, False, "Answer must be exactly one word/phrase with no punctuation."

    # Stage 6: Check for filler phrases (extract answer from verbose text)
    # Example: "it is false" -> extract "false"
    for phrase, mapped in FILLER_PHRASES.items():
        if phrase in normalized:
            return mapped, True, f"Alias phrase '{phrase}'"

    # Stage 7: Check for synonyms (normalize common alternatives)
    # Extract word tokens and check for synonyms
    word_tokens = [token for token in re.split(r"[^a-z]+", normalized) if token]
    if word_tokens:
        for target, synonyms in FILLER_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in word_tokens:
                    return target, True, f"Alias keyword '{synonym}'"

    # Stage 8: Salvage valid token from verbose response (last resort)
    # Example: "I think the answer is true" -> salvage "true"
    for token in VALID_ANSWERS:
        if token in normalized:
            return token, True, "Salvaged token from verbose response"

    # Stage 9: No valid answer found
    return "unknown", False, "Answer must be exactly one of: true, false, possibly true, possibly false, unknown."


def build_reprompt_prompt(base_prompt: str, reason: str, previous: str) -> str:
    """Build a reprompt with error feedback from previous invalid response.
    
    When LLM gives an invalid answer, we reprompt with the original prompt plus
    error feedback. This helps the LLM understand what went wrong and correct it.
    
    Args:
        base_prompt (str): Original prompt (question + instructions)
        reason (str): Explanation of why previous answer was invalid
        previous (str): The invalid answer that was rejected
    
    Returns:
        str: Complete prompt with error feedback appended
    
    Design Notes:
        - Normalizes previous answer for display (removes extra whitespace)
        - Replaces double quotes with single quotes (avoids JSON issues)
        - Uses REPROMPT_TEMPLATE to format error feedback consistently
        - Empty previous answer is shown as "∅" (null symbol)
    """
    reason_text = reason or "the answer was not in the valid list"
    # Normalize previous answer for display
    normalized_previous = " ".join(previous.strip().split()) or "∅"
    # Replace quotes to avoid JSON/formatting issues
    normalized_previous = normalized_previous.replace('"', "'")
    # Append error feedback to original prompt
    return f"{base_prompt}{REPROMPT_TEMPLATE.format(answer=normalized_previous, reason=reason_text)}"


def get_validated_answer(model: str, base_prompt: str, node: str) -> Tuple[str, str, int, bool, str]:
    """Get a validated answer from LLM with retry logic.
    
    This function implements the retry loop for getting valid answers:
    1. Ask LLM for response
    2. Validate response format
    3. If valid, return it
    4. If invalid, reprompt with error feedback (up to MAX_REPROMPTS times)
    5. Handle filler responses specially (abort early if repeated)
    
    Args:
        model (str): Model name (e.g., "llama3:8b")
        base_prompt (str): Original prompt (question + instructions)
        node (str): Ollama instance URL to use
    
    Returns:
        Tuple[str, str, int, bool, str]: (normalized_answer, raw_answer, attempts_used, is_valid, reason_tag)
            - normalized_answer: Validated answer (one of VALID_ANSWERS or "unknown")
            - raw_answer: Original LLM response (for logging/debugging)
            - attempts_used: Number of API calls made (1 to MAX_REPROMPTS+1)
            - is_valid: True if answer is acceptable, False if all retries failed
            - reason_tag: Category of validation result ("exact", "alias", "salvaged_verbose", etc.)
    
    Retry Strategy:
        - Up to MAX_REPROMPTS retries (default: 2)
        - Each retry includes error feedback in prompt
        - Filler responses abort early (no point retrying if LLM keeps adding filler)
        - Returns best available answer even if validation fails
    
    Reason Tags:
        - "exact": Perfect match (exactly one of VALID_ANSWERS)
        - "alias": Normalized via alias/synonym
        - "numeric": Converted from numeric choice (1-5)
        - "salvaged_verbose": Extracted from verbose response
        - "filler_skipped": Filler response detected, aborted early
        - "filler_invalid": Filler response after all retries
        - "invalid": Invalid format after all retries
    """
    attempts = 0
    prompt = base_prompt  # Start with original prompt
    last_raw = ""  # Track last response to detect infinite loops

    while attempts < MAX_REPROMPTS:
        # Ask LLM for response
        raw_response = ask(model, prompt, node)
        
        # Validate response
        normalized, valid, reason = normalize_and_validate(raw_response)
        
        if valid:
            # Valid answer found - categorize and return
            reason_tag = "other"
            if reason:
                if reason.startswith("Alias"):
                    log(f"Alias normalization -> {normalized} | {reason} | raw='{raw_response}'")
                    reason_tag = "alias"
                elif reason == "Numeric choice":
                    log(f"Numeric choice -> {normalized} | raw='{raw_response}'")
                    reason_tag = "numeric"
                elif reason == "Salvaged token from verbose response":
                    log(f"Salvaged verbose -> {normalized} | raw='{raw_response}'")
                    reason_tag = "salvaged_verbose"
                elif reason == "Exact match":
                    log(f"Exact match -> {normalized}")
                    reason_tag = "exact"
                else:
                    log(f"{reason} -> {normalized} | raw='{raw_response}'")
            else:
                reason_tag = "exact"
            return normalized, raw_response, attempts + 1, True, reason_tag

        # Invalid answer - check for filler text
        filler_hit = looks_like_filler_response(raw_response) if raw_response else False
        reprompt_reason = reason
        
        if filler_hit:
            # Filler text detected - handle specially
            preview = (raw_response or "").strip().replace("\n", " ")[:80]
            log(f"Filler invalid (model={model}, attempt={attempts + 1}) -> '{preview}'")
            
            if attempts == 0:
                # First attempt with filler - reprompt with specific feedback
                reprompt_reason = "your reply began with filler text instead of one of the allowed words"
            else:
                # Filler repeated - abort early (no point retrying)
                normalized = normalized if normalized in VALID_ANSWERS else "unknown"
                log(f"Filler repeated (model={model}, attempt={attempts + 1}) -> '{preview}' | aborting")
                return normalized, raw_response, attempts + 1, False, "filler_skipped"

        # Increment attempt counter
        attempts += 1
        last_raw = raw_response
        
        # Check if we've exhausted retries
        if attempts >= MAX_REPROMPTS:
            final_tag = "filler_invalid" if filler_hit else "invalid"
            return normalized, last_raw, attempts, False, final_tag

        # Build reprompt with error feedback
        prompt = build_reprompt_prompt(base_prompt, reprompt_reason, last_raw)

    # Fallback (shouldn't reach here, but be safe)
    return "unknown", last_raw, attempts, False, "invalid"

def build_prompt(field: str, q: str, year: int | None) -> str:
    """Build the prompt sent to LLM for answering a quiz question.
    
    Constructs a prompt with clear instructions, context, and the question.
    The prompt is designed to elicit a single-word answer in the correct format.
    
    Args:
        field (str): Academic field name (e.g., "computer-science")
        q (str): Question text to answer
        year (int | None): Publication year of the paper (optional context)
    
    Returns:
        str: Complete prompt ready to send to LLM
    
    Prompt Structure:
        1. Instructions: Format requirements (single word, no filler, etc.)
        2. Context: Field name and publication year (if available)
        3. Question: The actual question to answer
    
    Design Notes:
        - Very explicit instructions to minimize format errors
        - Includes field context (helps LLM understand domain)
        - Includes publication year (temporal context may be relevant)
        - Clear format requirements (single word, lowercase, no punctuation)
    """
    # Add publication year if available (temporal context)
    yr = f"The paper was published in {year}. " if year else ""
    
    # Instructions for LLM (format requirements)
    instructions = (
        "Respond with exactly one lowercase word: true, false, possibly true, possibly false, unknown.\n"
        "Do not add sentences, introductions, or punctuation—phrases like \"here is\" or \"the answer is\" are invalid.\n"
        "Copy the word exactly as spelled above. If you cannot decide, reply unknown. Never leave the reply blank.\n"
    )
    
    # Context: field name and publication year
    context = f"You are being quizzed in {field}. {yr}"
    
    # Combine: instructions + context + question
    return f"{instructions}{context}Question:\n{q}"

def score(gt: str, pred: str) -> float:
    """Score a prediction against ground truth.
    
    Compares LLM's predicted answer to the correct answer and returns points.
    Uses set-based matching to handle "possibly true"/"possibly false" correctly.
    
    Args:
        gt (str): Ground truth answer (correct answer from database)
        pred (str): Predicted answer from LLM
    
    Returns:
        float: Points awarded (CORRECT_SCORE, INCORRECT_SCORE, or 0.0)
            - CORRECT_SCORE (0.1): Prediction matches ground truth
            - INCORRECT_SCORE (-0.1): Prediction doesn't match
            - 0.0: Prediction is "unknown" (no points, no penalty)
    
    Scoring Logic:
        - "unknown" predictions score 0.0 (neutral, no penalty)
        - TRUE_SET matches: "true" matches "possibly true" (both indicate true)
        - FALSE_SET matches: "false" matches "possibly false" (both indicate false)
        - Cross-set matches: "true" vs "false" = incorrect (penalty)
    
    Examples:
        - gt="true", pred="true" -> CORRECT_SCORE (0.1)
        - gt="true", pred="possibly true" -> CORRECT_SCORE (0.1) [both in TRUE_SET]
        - gt="true", pred="false" -> INCORRECT_SCORE (-0.1) [different sets]
        - gt="true", pred="unknown" -> 0.0 (no points, no penalty)
    """
    # "unknown" predictions get no points (neutral, not penalized)
    if pred == "unknown":
        return 0.0
    
    # Check if prediction matches ground truth (using set-based matching)
    # TRUE_SET: {"true", "possibly true"} - both indicate "true" answer
    # FALSE_SET: {"false", "possibly false"} - both indicate "false" answer
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return CORRECT_SCORE  # Match! Award points
    return INCORRECT_SCORE  # Mismatch! Penalize


def process_batch(node: str, batch: List[Tuple[int, Dict]], model: str, field_name: str) -> Tuple[float, int]:
    """Process a batch of questions for a specific model and field.
    
    This function processes multiple questions in sequence, calling the LLM
    for each question and scoring the responses. Runs in parallel via ThreadPoolExecutor.
    
    Args:
        node (str): Ollama instance URL to use for this batch
        batch (List[Tuple[int, Dict]]): List of (index, document) tuples
            - index: Question number (1-based, for logging)
            - document: MongoDB document with "Question", "Answer", "publication_year"
        model (str): Model name being tested (e.g., "llama3:8b")
        field_name (str): Academic field name (e.g., "computer-science")
    
    Returns:
        Tuple[float, int]: (total_score, salvaged_verbose_count)
            - total_score: Sum of all question scores (positive and negative)
            - salvaged_verbose_count: Number of answers extracted from verbose responses
    
    Processing Flow:
        1. For each question in batch:
           a. Extract question, ground truth answer, and publication year
           b. Build prompt with instructions and context
           c. Get validated answer from LLM (with retries)
           d. Score prediction against ground truth
           e. Log result with detailed information
        2. Track salvaged answers (extracted from verbose responses)
        3. Return total score and salvage count
    
    Logging:
        - Logs every question with: field, question number, year, model, ground truth, prediction, score
        - Uses emoji indicators: ✅ (correct), ❌ (incorrect), ☐ (unknown)
        - Logs invalid responses as errors (helps identify problematic models)
    """
    total = 0.0  # Running total of scores
    salvaged_verbose_count = 0  # Count of salvaged answers
    
    # Process each question in the batch
    for idx, doc in batch:
        # Extract question, ground truth answer, and publication year
        q = doc["Question"]
        gt = doc["Answer"].lower()  # Normalize ground truth to lowercase
        year = doc.get("publication_year")  # Optional: publication year for context
        
        # Build prompt with instructions, context, and question
        prompt = build_prompt(field_name, q, year)

        # Get validated answer from LLM (with retry logic)
        normalized, raw_answer, attempts_used, valid_answer, reason_tag = get_validated_answer(model, prompt, node)
        
        # Track salvaged answers (extracted from verbose responses)
        if reason_tag == "salvaged_verbose":
            salvaged_verbose_count += 1
        
        # Handle invalid answers (log with appropriate level)
        if not valid_answer:
            # Ensure normalized is a valid answer (default to "unknown")
            if normalized not in VALID_ANSWERS:
                normalized = "unknown"
            
            # Check if this was a filler response
            filler_notice = reason_tag.startswith("filler_")
            if filler_notice:
                if reason_tag == "filler_skipped":
                    # Filler detected, aborted early (no retries)
                    message = (
                        f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
                        f"filler response detected; no retry issued; raw='{raw_answer}' -> using '{normalized}'"
                    )
                else:
                    # Filler after all retries
                    message = (
                        f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
                        f"filler response ignored after {attempts_used} attempts; raw='{raw_answer}' -> using '{normalized}'"
                    )
                log_err(message)  # Filler responses are errors
            else:
                # Invalid format (not filler)
                message = (
                    f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
                    f"invalid response after {attempts_used} attempts; raw='{raw_answer}' -> using '{normalized}'"
                )
                if normalized == "unknown" and attempts_used > 1:
                    log_err(message)  # Multiple retries failed = error
                elif normalized == "unknown":
                    log(message)  # Single attempt unknown = info
                else:
                    log(message)  # Invalid but salvaged = info

        # Score prediction against ground truth
        sc = score(gt, normalized)
        
        # Determine result emoji for logging
        res = "✅" if sc > 0 else "❌" if sc < 0 else "☐"  # Correct, incorrect, or unknown
        attempt_info = f"attempts:{attempts_used}"
        
        # Log result with full details
        log(
            f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
            f"GT:{gt} | LLM:{raw_answer} -> {normalized} | {res} | {attempt_info}"
        )
        
        # Add to running total
        total += sc
    
    return total, salvaged_verbose_count

def header()->List[str]: return ["model","overall",*FIELD_MAP.values(),"timestamp"]
def done()->set[str]:
    if not CSV_PATH.exists(): return set()
    with CSV_PATH.open(newline="",encoding="utf-8")as f:
        r=csv.reader(f); next(r,None); return {row[0] for row in r}

def _query_gpu_memory_mb() -> List[int]:
    """Return a list of total VRAM values (MB) for each detected GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if values:
            return values
    except Exception:
        pass
    # Fallback to 8 GB GPU if detection fails (more conservative)
    return [8192]


def _detect_cpu_cores() -> int:
    """Detect CPU core count, including hyperthreading."""
    try:
        # Try os.cpu_count() first (Python 3.4+)
        cores = os.cpu_count()
        if cores:
            return cores
    except Exception:
        pass
    
    try:
        # Try multiprocessing as fallback
        import multiprocessing
        cores = multiprocessing.cpu_count()
        if cores:
            return cores
    except Exception:
        pass
    
    # Try reading from /proc/cpuinfo on Linux
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cores = len([line for line in f if line.startswith('processor')])
            if cores > 0:
                return cores
    except Exception:
        pass
    
    # Fallback to 4 cores if detection fails (more conservative)
    return 4


def _detect_available_ram_gb() -> float:
    """Detect available system RAM in gigabytes."""
    try:
        # Try /proc/meminfo on Linux
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)  # Convert KB to GB
                elif line.startswith('MemTotal:'):
                    # Fallback to total if available not found
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)  # Convert KB to GB
    except Exception:
        pass
    
    try:
        # Try psutil if available
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        return ram_gb
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to 16 GB if detection fails (more conservative)
    return 16.0


def calculate_workers_per_node() -> int:
    """Calculate optimal workers per node based on CPU cores, GPU count, and RAM."""
    cpu_cores = _detect_cpu_cores()
    num_gpus = len(GPU_MEMORY_MB)
    ram_gb = _detect_available_ram_gb()
    
    # Formula: workers = min(max(2, cpu_cores // num_gpus), min(16, ram_gb // 8))
    # Minimum 2, scale with CPU cores divided by GPUs, cap at 16 and RAM-based limit
    if num_gpus > 0:
        workers = min(max(2, cpu_cores // num_gpus), min(16, int(ram_gb // 8)))
    else:
        # If no GPUs detected, use a conservative value
        workers = min(max(2, cpu_cores // 4), min(16, int(ram_gb // 8)))
    
    return workers


def _calculate_gpu_reserve_mb(gpu_memory_mb: List[int]) -> int:
    """Calculate GPU memory reserve as 2% of average GPU memory."""
    if not gpu_memory_mb:
        return 512
    avg = sum(gpu_memory_mb) // len(gpu_memory_mb)
    reserve = max(256, min(1024, int(avg * 0.02)))
    return reserve


def _get_base_port() -> int:
    """Get base port from env var or use default."""
    return int(os.environ.get('OLLAMA_BASE_PORT', '11434'))


def _calculate_default_instance_memory(gpu_memory_mb: List[int]) -> int:
    """Calculate default instance memory as 8% of average GPU memory."""
    if not gpu_memory_mb:
        return 2048
    avg = sum(gpu_memory_mb) // len(gpu_memory_mb)
    instance_mem = int(avg * 0.08)
    return max(1024, min(4096, instance_mem))


def _calculate_timeout_sec(workers_per_node: int, num_nodes: int) -> int:
    """Calculate timeout based on worker count."""
    total_workers = workers_per_node * num_nodes if num_nodes > 0 else workers_per_node
    timeout = 20 + (total_workers // 1000) * 2
    return min(timeout, 60)


def _calculate_max_instances_per_gpu(gpu_memory_mb: int) -> int:
    """Calculate max instances per GPU based on GPU memory."""
    return min(8, max(1, gpu_memory_mb // 2000))


GPU_MEMORY_MB = _query_gpu_memory_mb()
CPU_CORES = _detect_cpu_cores()
RAM_GB = _detect_available_ram_gb()
GPU_MEMORY_RESERVE_MB = _calculate_gpu_reserve_mb(GPU_MEMORY_MB)
DEFAULT_INSTANCE_MEMORY_MB = _calculate_default_instance_memory(GPU_MEMORY_MB)
BASE_PORT = _get_base_port()
WORKERS_PER_NODE = calculate_workers_per_node()

def estimate_model_memory_mb(model: str) -> int:
    """Estimate VRAM per instance required to serve *model* via Ollama."""
    normalized = model.lower()
    if normalized in MODEL_MEMORY_OVERRIDES_MB:
        return MODEL_MEMORY_OVERRIDES_MB[normalized]
    base = normalized.split(":")[0]
    if base in MODEL_MEMORY_OVERRIDES_MB:
        return MODEL_MEMORY_OVERRIDES_MB[base]
    match = re.search(r"(\d+(?:\.\d+)?)([mb])", normalized)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        params_millions = value * (1000 if unit == 'b' else 1)
        estimated = int(params_millions * 7)
        return max(DEFAULT_INSTANCE_MEMORY_MB, estimated)
    return DEFAULT_INSTANCE_MEMORY_MB


def _should_spread_across_gpus(model: str) -> bool:
    """Check if model should be spread across all GPUs (one instance per GPU).
    
    Models starting from deepseek-llm:67b and later in MODEL_LIST are spread
    across all GPUs to avoid trying to fit multiple large model instances on a single GPU.
    """
    try:
        model_index = MODEL_LIST.index(model)
        deepseek_index = MODEL_LIST.index("deepseek-llm:67b")
        return model_index >= deepseek_index
    except ValueError:
        # If model not found in list, don't spread (use default behavior)
        return False


def plan_instances_for_model(model: str) -> List[Dict[str, str]]:
    """Produce an Ollama instance plan sized for the requested model."""
    mem_needed = max(estimate_model_memory_mb(model), 1)
    plan: List[Dict[str, str]] = []
    port = _get_base_port()
    
    # For large models (deepseek-llm:67b and later), spread across all GPUs
    if _should_spread_across_gpus(model):
        # Distribute one instance per GPU across all available GPUs
        for gpu_index, total_mem in enumerate(GPU_MEMORY_MB):
            available = max(0, total_mem - GPU_MEMORY_RESERVE_MB)
            # Only add instance if GPU has enough memory for the model
            if available >= mem_needed:
                plan.append({
                    "host": f"http://127.0.0.1:{port}",
                    "port": str(port),
                    "gpu": str(gpu_index),
                })
                port += 1
        
        # Fallback: if no GPUs have enough memory, place on GPU 0 anyway
        if not plan:
            base_port = _get_base_port()
            plan.append({"host": f"http://127.0.0.1:{base_port}", "port": str(base_port), "gpu": "0"})
    else:
        # Original logic: try to fit as many instances as possible per GPU
        for gpu_index, total_mem in enumerate(GPU_MEMORY_MB):
            available = max(0, total_mem - GPU_MEMORY_RESERVE_MB)
            if available >= mem_needed:
                count = max(1, available // mem_needed)
            else:
                count = 1
            # Dynamic cap based on GPU memory
            max_instances = _calculate_max_instances_per_gpu(total_mem)
            count = max(1, min(count, max_instances))
            for _ in range(count):
                plan.append({
                    "host": f"http://127.0.0.1:{port}",
                    "port": str(port),
                    "gpu": str(gpu_index),
                })
                port += 1
        if not plan:
            base_port = _get_base_port()
            plan.append({"host": f"http://127.0.0.1:{base_port}", "port": str(base_port), "gpu": "0"})
    
    return plan


# ────────── main ──────────
def main():
    global GLOBAL_SALVAGE_TOTAL, GLOBAL_SALVAGE_BY_MODEL, GLOBAL_SALVAGE_BY_FIELD
    GLOBAL_SALVAGE_TOTAL = 0
    GLOBAL_SALVAGE_BY_MODEL = {}
    GLOBAL_SALVAGE_BY_FIELD = defaultdict(int)
    
    # Log system specs and calculated concurrency
    num_gpus = len(GPU_MEMORY_MB)
    timeout_sec = _calculate_timeout_sec(WORKERS_PER_NODE, num_gpus)  # Estimate before nodes are known
    log(f"💻 System specs: {CPU_CORES} CPU cores, {RAM_GB:.1f} GB RAM, {num_gpus} GPU(s)")
    log(f"⚙️  Calculated concurrency: {WORKERS_PER_NODE} workers per node")
    log(f"⚙️  Dynamic config: GPU reserve={GPU_MEMORY_RESERVE_MB}MB, default instance={DEFAULT_INSTANCE_MEMORY_MB}MB, timeout={timeout_sec}s")
    
    completed=done(); hdr=CSV_PATH.exists()
    # Calculate pool size based on worker count
    num_nodes = len(OLLAMA_NODES) if OLLAMA_NODES else 1
    pool_size = max(5, min(50, WORKERS_PER_NODE * num_nodes))
    mongo=MongoClient(MONGO_URI, maxPoolSize=pool_size, minPoolSize=min(5, pool_size // 2))
    cols={db:mongo[db]["sources"] for db in DATABASES}

    for model in MODEL_LIST:
        if model in completed: log(f"skip {model}"); continue
        restart_ollama_instances(model)
        if not pull(model):
            shutdown_ollama_instances(force_all=True)
            continue
        log(f"=== MODEL {model} ===")

        field_scores:Dict[str,float]={}
        field_salvage_counts:Dict[str,int]={}
        model_salvage_total = 0
        for db in DATABASES:
            fname=FIELD_MAP[db]; col=cols[db]
            docs=list(col.aggregate([
                {"$match":{"Question":{"$ne":None},"Answer":{"$in":list(VALID_ANSWERS)}}},
                {"$sample":{"size":QUESTIONS_PER_FIELD}}
            ]))
            indexed_docs=list(enumerate(docs,1))
            node_batches={node:[] for node in OLLAMA_NODES}
            node_count=len(OLLAMA_NODES)
            for idx, doc in indexed_docs:
                node = OLLAMA_NODES[(idx-1) % node_count]
                node_batches[node].append((idx, doc))

            total=0.0
            field_salvaged = 0
            max_workers = node_count * WORKERS_PER_NODE
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures=[
                    executor.submit(process_batch, node, batch, model, fname)
                    for node, batch in node_batches.items() if batch
                ]
                for future in as_completed(futures):
                    batch_total, batch_salvaged = future.result()
                    total += batch_total
                    field_salvaged += batch_salvaged
            model_salvage_total += field_salvaged
            field_salvage_counts[fname] = field_salvaged
            GLOBAL_SALVAGE_BY_FIELD[fname] += field_salvaged
            field_scores[fname]=round(total,4)

        overall=round(sum(field_scores.values())/len(field_scores),4)
        row=[model,str(overall),*(str(field_scores.get(f,0.0)) for f in FIELD_MAP.values()),datetime.now().isoformat()]
        mode="a" if hdr else "w"
        with CSV_PATH.open(mode,newline="",encoding="utf-8")as f:
            w=csv.writer(f); 
            if not hdr: w.writerow(header()); hdr=True
            w.writerow(row)

        stop_model_on_all_nodes(model)
        GLOBAL_SALVAGE_TOTAL += model_salvage_total
        GLOBAL_SALVAGE_BY_MODEL[model] = model_salvage_total
        if model_salvage_total:
            per_field = ", ".join(
                f"{field}:{count}"
                for field, count in field_salvage_counts.items()
                if count
            )
            log(f"{model} salvaged verbose answers: total={model_salvage_total}"
                f"{' | '+per_field if per_field else ''}")
        delete(model); log(f"done {model}")
        shutdown_ollama_instances(force_all=True)

    mongo.close(); log("🎉 all models done")
    shutdown_ollama_instances(force_all=True)
    if GLOBAL_SALVAGE_TOTAL:
        log(
            "Verbose answer salvage summary — "
            f"total:{GLOBAL_SALVAGE_TOTAL} | "
            f"models:{ {m:c for m,c in GLOBAL_SALVAGE_BY_MODEL.items() if c} } | "
            f"fields:{ {f:c for f,c in GLOBAL_SALVAGE_BY_FIELD.items() if c} }"
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("bye")
    finally:
        shutdown_ollama_instances(force_all=True)

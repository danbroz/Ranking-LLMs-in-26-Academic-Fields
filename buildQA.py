"""Pipeline for building field_* QA datasets with dynamic Ollama scaling.

This script downloads OpenAlex works manifests, spins up as many Ollama
instances as available GPU memory allows, generates question/answer pairs
(one per field) with the selected LLM, persists them to MongoDB, and finally
runs backup and pruning maintenance tasks so the databases remain ready for
consumption.
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
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
from threading import Thread, Semaphore, Lock
from collections import defaultdict
import signal

from scripts import backup_field_databases, prune_field_databases

# Dynamic Ollama configuration will be computed at runtime
GPU_MEMORY_RESERVE_MB = 512
SMOLLM2_MEMORY_MB = 906
MODEL_NAME = 'gemma3:270m'
MODEL_MEMORY_MB = 1500
BASE_PORT = 11434

nodes = []
INSTANCES = []


def _query_gpu_memory_mb():
    """Return total VRAM per visible GPU in megabytes (fallback to 24 GB)."""
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
    # Fallback to a single 24 GB GPU if detection fails
    return [24576]


def _detect_cpu_cores():
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
    
    # Fallback to 8 cores if detection fails
    return 8


def _detect_available_ram_gb():
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
    
    # Fallback to 32 GB if detection fails
    return 32.0


GPU_MEMORY_MB = _query_gpu_memory_mb()


def calculate_workers_per_node():
    """Calculate optimal workers per node based on CPU cores, GPU count, and model size."""
    cpu_cores = _detect_cpu_cores()
    num_gpus = len(GPU_MEMORY_MB)
    
    # For small models (like gemma3:270m), we can be more aggressive with concurrency
    # Base calculation: workers = cpu_cores // num_gpus, but scale up for small models
    if num_gpus > 0:
        base_workers = cpu_cores // num_gpus
        # For small models (< 2GB), allow much higher concurrency (up to 50 workers per node)
        # This helps saturate GPU utilization when model inference is fast
        if MODEL_MEMORY_MB < 2000:
            # Small model: allow 3-4x base workers, cap at 50
            workers = min(max(10, base_workers * 3), 50)
        else:
            # Larger models: more conservative, cap at 20
            workers = min(max(5, base_workers * 2), 20)
    else:
        # If no GPUs detected, use a conservative value
        workers = min(max(5, cpu_cores // 4), 20)
    
    return workers


# Calculate dynamic concurrency values at module initialization
WORKERS_PER_NODE = calculate_workers_per_node()
MAX_CONCURRENT_REQUESTS_PER_NODE = WORKERS_PER_NODE  # Match workers per node

MONGO_URI = "mongodb://localhost:27017/"
DATABASES = [f'field_{fid}' for fid in range(11, 37)]

MAX_ATTEMPTS = 3
TASK_TIMEOUT_SECONDS = 45  # Reduced from 60s for faster failure detection
LOG_TIMEOUT_SECONDS = 3600  # 1 hour - only restart if truly stuck
BATCH_SIZE = 40
RETRY_DELAY = 0.01  # Reduced from 0.05s for faster retries
NODE_BACKOFF_SECONDS = 30  # Back off a node for 30 seconds after timeout

last_log_time = datetime.now()

# Per-node throttling and health tracking
node_semaphores = {}  # Semaphore per node to limit concurrent requests
node_timeout_counts = defaultdict(int)  # Track timeout counts per node
node_last_timeout = {}  # Track last timeout time per node
node_lock = Lock()  # Lock for thread-safe access to node tracking
node_round_robin_index = 0  # Round-robin index for simple node selection
node_round_robin_lock = Lock()  # Lock for thread-safe round-robin access

def log(message):
    global last_log_time
    last_log_time = datetime.now()
    print(f"{last_log_time.strftime('%Y-%m-%d %H:%M:%S')} - {message}", flush=True)


def get_hostname():
    return socket.gethostname()


def _run(cmd, **kwargs):
    try:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    except subprocess.CalledProcessError as e:
        return e


def _which(exe):
    r = _run(['bash', '-lc', f'command -v {exe} || true'])
    out = (r.stdout or b'').decode().strip()
    return out if out else None


def ensure_mongodb_installed_and_running():
    sudo_pwd = os.environ.get('SUDO_PASSWORD')
    def sudo(cmd):
        if sudo_pwd:
            return subprocess.run(['sudo', '-S'] + cmd, input=(sudo_pwd + '\n').encode('utf-8'),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return subprocess.run(['sudo'] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    mongod_path = _which('mongod')
    if not mongod_path:
        # Try installing MongoDB via apt; attempt both mongodb-org and mongodb
        sudo(['apt-get', 'update', '-y'])
        rc1 = sudo(['apt-get', 'install', '-y', 'mongodb-org'])
        if rc1.returncode != 0:
            sudo(['apt-get', 'install', '-y', 'mongodb'])

    # Try starting and enabling common service names
    for svc in ['mongod', 'mongodb']:
        sudo(['systemctl', 'enable', '--now', svc])
        # Also try service command in case systemctl not available
        sudo(['service', svc, 'start'])


def ensure_ollama_installed():
    sudo_pwd = os.environ.get('SUDO_PASSWORD')
    if _which('ollama'):
        return
    # Install Ollama using official install script
    if sudo_pwd:
        cmd = f"echo {shlex.quote(sudo_pwd)} | sudo -S sh -c 'curl -fsSL https://ollama.com/install.sh | sh'"
        subprocess.run(['bash', '-lc', cmd], check=False)
    else:
        subprocess.run(['bash', '-lc', "curl -fsSL https://ollama.com/install.sh | sh"], check=False)


def check_ollama_health(node_url):
    """Check if an Ollama instance is healthy and responding."""
    try:
        client = ollama.Client(host=node_url)
        client.list()  # Simple health check
        return True
    except Exception:
        return False


def ensure_ollama_instances():
    """Launch Ollama servers sized for the target model across all GPUs.

    The routine stops any existing servers, computes how many instances of the
    configured model fit into each GPU after leaving a small cushion, and then
    starts dedicated `ollama serve` processes per port. Each server is also
    primed with the model so subsequent generate calls do not block on pulls.
    """
    global nodes, INSTANCES, node_semaphores, node_timeout_counts, node_last_timeout

    hostname = get_hostname()
    if hostname != 'node0':
        log(f"⚠️ Hostname is '{hostname}', not 'node0'; proceeding with local Ollama setup anyway.")

    # Log system specs and calculated concurrency
    cpu_cores = _detect_cpu_cores()
    ram_gb = _detect_available_ram_gb()
    num_gpus = len(GPU_MEMORY_MB)
    log(f"💻 System specs: {cpu_cores} CPU cores, {ram_gb:.1f} GB RAM, {num_gpus} GPU(s)")
    log(f"⚙️  Calculated concurrency: {WORKERS_PER_NODE} workers per node, {MAX_CONCURRENT_REQUESTS_PER_NODE} max concurrent requests per node")

    ensure_ollama_installed()

    # Stop any system-level services and stray processes
    sudo_pwd = os.environ.get('SUDO_PASSWORD')
    if sudo_pwd:
        try:
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
        subprocess.run(['sudo', 'systemctl', 'stop', 'ollama'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', 'ollama'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Build instance plan based on GPU memory
    INSTANCES = []
    port = BASE_PORT
    gpu_instance_counts = defaultdict(int)
    for gpu_index, total_mem in enumerate(GPU_MEMORY_MB):
        available = max(0, total_mem - GPU_MEMORY_RESERVE_MB)
        if available >= MODEL_MEMORY_MB:
            count = max(1, available // MODEL_MEMORY_MB)
        else:
            count = 1
        log(f"📊 GPU {gpu_index}: {total_mem}MB total, {available}MB available -> planning {count} instances")
        for _ in range(count):
            INSTANCES.append({
                'host': f'http://localhost:{port}',
                'port': str(port),
                'gpu': str(gpu_index),
            })
            gpu_instance_counts[gpu_index] += 1
            port += 1

    if not INSTANCES:
        raise RuntimeError("Unable to plan any Ollama instances; check GPU availability.")

    log(f"📋 Planned {len(INSTANCES)} total instances across {len(GPU_MEMORY_MB)} GPUs")
    for gpu_idx, count in sorted(gpu_instance_counts.items()):
        log(f"   GPU {gpu_idx}: {count} instances")

    nodes = [inst['host'] for inst in INSTANCES]
    node_semaphores = {}
    node_timeout_counts = defaultdict(int)
    node_last_timeout = {}

    # Track which instances start successfully per GPU
    gpu_started_counts = defaultdict(int)
    gpu_healthy_counts = defaultdict(int)

    for inst in INSTANCES:
        port = inst['port']
        gpu = inst['gpu']
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
        env['OLLAMA_MODELS'] = os.path.expanduser(f'~/.ollama-{port}')
        env['CUDA_VISIBLE_DEVICES'] = gpu
        log(f"🔧 Starting Ollama on port {port} (GPU {gpu})")
        stdout = open(f'ollama_{port}.log', 'a')
        process = subprocess.Popen(['nohup', 'ollama', 'serve'], env=env, stdout=stdout, stderr=subprocess.STDOUT)
        gpu_started_counts[int(gpu)] += 1
        time.sleep(2)  # Reduced from 5s for faster startup

        # Ensure model is available for this instance
        env_pull = os.environ.copy()
        env_pull['OLLAMA_HOST'] = f'localhost:{port}'
        try:
            show = subprocess.run(
                ['ollama', 'show', MODEL_NAME],
                env=env_pull,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if show.returncode != 0:
                log(f"⬇️ Pulling {MODEL_NAME} on port {port} (GPU {gpu})")
                subprocess.run(['ollama', 'pull', MODEL_NAME], env=env_pull, check=True)
        except subprocess.CalledProcessError:
            log(f"⚠️ Model setup issue on port {port} (GPU {gpu}); continuing.")

    log("⏳ Waiting for Ollama instances to initialize...")
    time.sleep(5)  # Reduced from 10s for faster initialization

    # Check health and track per GPU
    healthy_nodes = []
    for inst in INSTANCES:
        node = inst['host']
        gpu = int(inst['gpu'])
        if check_ollama_health(node):
            healthy_nodes.append(node)
            gpu_healthy_counts[gpu] += 1
        else:
            log(f"⚠️ Instance on port {inst['port']} (GPU {gpu}) failed health check")

    if healthy_nodes:
        log(f"✅ {len(healthy_nodes)}/{len(nodes)} Ollama instances are healthy")
        for gpu_idx in sorted(gpu_healthy_counts.keys()):
            started = gpu_started_counts.get(gpu_idx, 0)
            healthy = gpu_healthy_counts.get(gpu_idx, 0)
            log(f"   GPU {gpu_idx}: {healthy}/{started} instances healthy")
        nodes = healthy_nodes
    else:
        log("⚠️ No healthy Ollama instances found, using all nodes anyway")

    for node in nodes:
        node_semaphores[node] = Semaphore(MAX_CONCURRENT_REQUESTS_PER_NODE)


def _recover_abstract(inverted_index):
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    abstract = [''] * (max(max(v) for v in inverted_index.values()) + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            abstract[pos] = word
    return ' '.join(abstract).strip()


def extract_field_ids(row):
    seen = set()
    for topic in row.get('topics', []):
        fid = topic.get('field', {}).get('id')
        if isinstance(fid, str) and fid.startswith('https://openalex.org/fields/'):
            try:
                fid = int(fid.split('/')[-1])
            except Exception:
                continue
        if isinstance(fid, int) and 11 <= fid <= 36 and fid not in seen:
            seen.add(fid)
    return list(seen)

def create_prompt(abstract, error_feedback=None):
    example = """Example:
Abstract:
A randomized controlled trial found that daily vitamin D supplementation significantly reduced the risk of respiratory infections in elderly participants.

Question: Is it true, false, possibly true, or possibly false that daily vitamin D supplementation reduces the risk of respiratory infections in elderly people?
Answer: true"""

    unacceptable = (
        "Unacceptable output example (do NOT do this):\n"
        "Okay, I understand.\n"
        "Question: Is it true ...\n"
        "Answer: true\n"
    )

    never_say = (
        "Never write any of the following phrases (before, between, or after the required lines):\n"
        "- Okay, I understand.\n"
        "- I'm ready.\n"
        "- Please provide the research abstract.\n"
        "- I will follow the instructions.\n"
        "- Any explanation of what you are doing.\n"
        "- Anything after the answer line.\n"
    )

    question_rules = (
        "Format requirements:\n"
        "1. First line: must start with either 'Question:' or the abbreviated 'Q:' immediately followed by ' Is it true, false, possibly true, or possibly false that' and must end with a question mark '?'.\n"
        "2. Second line: must start with 'Answer:' followed by exactly one of true, false, possibly true, possibly false.\n"
        "3. No other text is allowed—no extra sentences, no blank lines, no truncation.\n"
    )

    base_prompt = (
        "You are generating questions and answers into a database from research abstracts. "
        "Only the exact format shown below is accepted—no confirmations, no acknowledgements, and no explanations. "
        "The very first character you output must be 'Q' (either 'Question:' or 'Q:'), and the line must end with '?'. Respond directly with the question line and the answer line, nothing else before, between, or after.\n\n"
        f"{question_rules}\n"
        "From the following research abstract, generate exactly one question based specifically on the findings described. "
        "Start the question with the exact words 'Is it true, false, possibly true, or possibly false that'. "
        "Do not include phrases like 'does the study', 'does the abstract', 'based on these findings', "
        "'do the research findings', 'is it possible', or 'is it possibly'. "
        "The question must stand alone without referencing the abstract, study, or researchers. "
        "After that line, immediately write the answer line starting with 'Answer:' followed by exactly one of true, false, possibly true, or possibly false. "
        "The answer must explicitly be either true, false, possibly true, or possibly false. "
        "Then directly answer it with one of these four choices only (no additional explanation).\n\n"
        f"{example}\n\n"
        f"{unacceptable}\n"
        f"{never_say}\n"
    )

    if error_feedback:
        base_prompt += (
            f"Your previous response was incorrect: {error_feedback}\n"
            "You must begin your reply with 'Question:' or 'Q:' as the first token, followed immediately by the required question that ends with '?' as shown above. "
            "The second line must start with 'Answer:' exactly and contain only one of the allowed words. "
            "Do NOT write acknowledgements, confirmations, or any extra phrases. "
            "Output exactly two lines—see the example—and never imitate the unacceptable example.\n\n"
        )

    base_prompt += (
        "Abstract:\n"
        f"{abstract}\n\n"
        "Output format (MUST follow exactly two lines):\n"
        "Question: Is it true, false, possibly true, or possibly false that <your question here>?\n"
        "Answer: <true|false|possibly true|possibly false>"
    )
    return base_prompt

def is_valid_response(response):
    """Check if response follows the required format."""
    if not response or not isinstance(response, str):
        return False
    # Check for non-English responses (common failure mode)
    if any(phrase in response.lower() for phrase in [
        'please provide', 'i need', 'provide the', 'abstract in', 'research abstract',
        'japanese', 'korean', 'chinese', 'russian', 'arabic', 'spanish', 'french'
    ]):
        return False
    
    response_stripped = response.strip()
    
    # Accept responses that start with "Q: Is it true..." pattern (even if incomplete)
    if response_stripped.startswith("Q: Is it true") or response_stripped.startswith("Question: Is it true"):
        # Check if it has both lines (preferred)
        if "\n" in response:
            first_line, second_line = response.split("\n", 1)
            first_line_stripped = first_line.strip()
            if first_line_stripped.startswith(("Question:", "Q:")) and second_line.strip().startswith("Answer:"):
                return True
        # Also accept if it just starts with the question pattern (will try to extract)
        return True
    
    # Fallback to original strict validation
    if "\n" in response:
        first_line, second_line = response.split("\n", 1)
        first_line_stripped = first_line.strip()
        if first_line_stripped.startswith(("Question:", "Q:")) and second_line.strip().startswith("Answer:"):
            return True
    return False


def is_node_available(node):
    """Check if node is available (not in backoff period)."""
    with node_lock:
        if node in node_last_timeout:
            time_since_timeout = (datetime.now() - node_last_timeout[node]).total_seconds()
            if time_since_timeout < NODE_BACKOFF_SECONDS:
                return False
        return True


def mark_node_timeout(node):
    """Mark a node as having timed out."""
    with node_lock:
        node_timeout_counts[node] += 1
        node_last_timeout[node] = datetime.now()


def get_available_node():
    """Get an available node using round-robin for speed, with smart fallback on timeout."""
    # Fast path: simple round-robin selection
    with node_round_robin_lock:
        global node_round_robin_index
        if not nodes:
            return None
        selected = nodes[node_round_robin_index % len(nodes)]
        node_round_robin_index += 1
    
    # Check if selected node is available (not in backoff)
    if is_node_available(selected):
        return selected
    
    # Fallback: find any available node
    available_nodes = [n for n in nodes if is_node_available(n)]
    if available_nodes:
        # Use round-robin on available nodes
        with node_round_robin_lock:
            idx = node_round_robin_index % len(available_nodes)
            return available_nodes[idx]
    
    # All nodes in backoff, use round-robin anyway
    return selected


def generate_question_answer(node, abstract):
    """Generate Q&A with per-node throttling."""
    prompt = create_prompt(abstract)
    attempts = 0
    last_response = None
    current_node = node

    while attempts < MAX_ATTEMPTS:
        # Get semaphore for current node (limits concurrent requests)
        semaphore = node_semaphores.get(current_node)
        if not semaphore:
            # Fallback if semaphore not initialized
            semaphore = Semaphore(MAX_CONCURRENT_REQUESTS_PER_NODE)
            node_semaphores[current_node] = semaphore
        
        # Acquire semaphore for this attempt
        semaphore.acquire()
        try:
            client = ollama.Client(host=current_node, timeout=TASK_TIMEOUT_SECONDS)
            response = client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={'temperature': 0.3, 'top_p': 0.9}
            )['response']

            if is_valid_response(response):
                # Handle responses with newline (complete format)
                if "\n" in response:
                    question_line, answer_line = response.split("\n", 1)
                    question_line = question_line.strip()
                    if question_line.startswith("Question:"):
                        question = question_line[len("Question:"):].strip()
                    elif question_line.startswith("Q:"):
                        question = question_line[len("Q:"):].strip()
                    else:
                        question = question_line
                    answer_line = answer_line.strip()
                    if answer_linenswer_line.startswith("Answer:"):
                        answer = answer_line[len("Answer:"):].strip()
                    else:
                        answer = answer_line
                    if answer.lower() in ['true', 'false', 'possibly true', 'possibly false']:
                        return question, answer
                # Handle incomplete responses that start with "Q: Is it true..." pattern
                elif response.strip().startswith("Q: Is it true") or response.strip().startswith("Question: Is it true"):
                    question_line = response.strip()
                    if question_line.startswith("Question:"):
                        question = question_line[len("Question:"):].strip()
                    elif question_line.startswith("Q:"):
                        question = question_line[len("Q:"):].strip()
                    else:
                        question = question_line
                    # Try to extract answer from the question line if it contains one
                    # Otherwise, default to "unknown" for incomplete responses
                    answer = "unknown"
                    # Check if answer appears in the question line (sometimes models put it there)
                    for ans in ['true', 'false', 'possibly true', 'possibly false']:
                        if ans in question_line.lower():
                            answer = ans
                            break
                    # Only return if we have a valid question (not just "Q" or empty)
                    if question and len(question) > 10:  # Basic sanity check
                        return question, answer

            if last_response and response[:50] == last_response[:50]:
                break

            error_feedback = f"Invalid format: {response[:100]}"
            prompt = create_prompt(abstract, error_feedback)
            last_response = response
            if attempts == 0:
                log(f"⚠️ Formatting issue on {current_node}: {error_feedback[:80]}")

        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                mark_node_timeout(current_node)
                if attempts < MAX_ATTEMPTS - 1:
                    new_node = get_available_node()
                    if new_node != current_node:
                        current_node = new_node

            if attempts == 0 and random.random() < 0.01:
                log(f"⚠️ Node {current_node} exception: {e}")
        finally:
            semaphore.release()

        attempts += 1
        time.sleep(RETRY_DELAY)

    raise RuntimeError(f"❌ All attempts failed. Last response: {last_response[:200] if last_response else 'None'}")

def upsert_and_update(mongo_client, row, question, answer, field_ids):
    """Batch write to multiple field databases."""
    idx = row.get('id')
    if not idx:
        return
    title = (row.get('title') or '').strip()
    abstract = _recover_abstract(row.get('abstract_inverted_index'))
    authors = [
        a.get('author', {}).get('display_name') for a in row.get('authorships', [])
        if a and isinstance(a, dict) and 'author' in a and isinstance(a.get('author'), dict) and 'display_name' in a['author']
    ] or None
    pub_year = row.get('publication_year')
    doi = row.get('doi')
    primary_loc = row.get('primary_location')
    pdf_url = primary_loc.get('pdf_url') if isinstance(primary_loc, dict) else None

    document = {
        "openalex_id": idx,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_year": pub_year,
        "doi": doi,
        "pdf_url": pdf_url,
        "Question": question,
        "Answer": answer,
    }

    # Batch write to all field databases in parallel using threads for speed
    # This reduces database write latency when writing to multiple field databases
    def write_to_field(fid):
        db = mongo_client[f'field_{fid}']
        collection = db['sources']
        collection.update_one(
            {"openalex_id": idx},
            {"$set": document},
            upsert=True
        )
    
    # Use ThreadPoolExecutor for parallel writes if multiple fields
    if len(field_ids) > 1:
        with ThreadPoolExecutor(max_workers=min(len(field_ids), 8)) as write_executor:
            write_executor.map(write_to_field, field_ids)
    else:
        # Single field, no need for threading overhead
        write_to_field(field_ids[0])


def is_record_processed(mongo_client, openalex_id, field_ids):
    """Check if a record has Q&A in all relevant field databases."""
    if not openalex_id or not field_ids:
        return False
    
    try:
        # Check if document exists in ALL relevant field databases with both Question and Answer
        for fid in field_ids:
            db = mongo_client[f'field_{fid}']
            collection = db['sources']
            doc = collection.find_one({
                "openalex_id": openalex_id,
                "Question": {"$exists": True},
                "Answer": {"$exists": True}
            })
            if not doc:
                return False
        return True
    except Exception:
        # On error, assume not processed to be safe
        return False


def check_url_processed(mongo_client, url, sample_size=20, timeout_seconds=30):
    """Sample a URL and determine if it should be skipped."""
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
                return 0, 0, False
            
            if chunk.empty:
                continue
            
            # Process rows until we have enough samples or find unprocessed record
            for _, row in chunk.iterrows():
                # Check timeout during processing
                if time.time() - start_time > timeout_seconds:
                    log(f"⏱️ Resume check timeout after {timeout_seconds}s, proceeding with URL")
                    return 0, 0, False
                
                if total_checked >= sample_size:
                    break
                
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
                    continue
            
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
        return 0, 0, False


def is_retracted_or_redacted(row):
    """Check if a paper is retracted or redacted."""
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
    
    return False


def should_process_row(row, mongo_client):
    """Early filtering - check if row should be processed before submitting to executor."""
    try:
        # Skip retracted or redacted papers
        if is_retracted_or_redacted(row):
            return False, None
        
        # Check language
        lang = row.get('language')
        if lang != 'en':
            return False, None
        
        # Check abstract exists
        abstract_inv = row.get('abstract_inverted_index')
        if not abstract_inv:
            return False, None
        
        abstract = _recover_abstract(abstract_inv)
        if not abstract:
            return False, None
        
        # Check field IDs exist
        field_ids = extract_field_ids(row)
        if not field_ids:
            return False, None
        
        # Skip resume check in filtering - it's too slow. Check during processing instead.
        # This allows filtering to proceed quickly without blocking on MongoDB queries.
        
        return True, (abstract, field_ids)
    except (KeyError, TypeError, AttributeError):
        return False, None


def process_row(node, mongo_client, row, abstract, field_ids):
    """Process a row that has already passed filtering."""
    try:
        # Check if record is already processed (has Q&A in all relevant field databases)
        # Do this check here instead of in filtering to avoid blocking the filtering loop
        openalex_id = row.get('id')
        if openalex_id and is_record_processed(mongo_client, openalex_id, field_ids):
            return False  # Already processed, skip
        
        # Generate question once for all fields (same question saved to multiple field databases)
        try:
            question, answer = generate_question_answer(node, abstract)
        except RuntimeError as exc:
            # Only log occasionally to reduce spam
            if random.random() < 0.01:  # Log 1% of failures
                log(f"❌ Skipping row due to generation failures: {exc}")
            return False
        except Exception as e:
            if random.random() < 0.01:  # Log 1% of failures
                log(f"⚠️ Unexpected error generating Q/A: {e}")
            return False

        # Save the same question to all field databases
        upsert_and_update(mongo_client, row, question, answer, field_ids)
        return True
    except (KeyError, TypeError, AttributeError):
        return False

def monitor_gpu_utilization():
    """Periodically log GPU utilization and instance distribution."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_idx, mem_used, mem_total, util = parts[0], parts[1], parts[2], parts[3]
                    gpu_info.append(f"GPU {gpu_idx}: {util}% util, {mem_used}/{mem_total}MB")
            if gpu_info:
                log(f"📊 GPU Status: {' | '.join(gpu_info)}")
        
        # Count Ollama processes per GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            ollama_count = result.stdout.count('ollama')
            if ollama_count > 0:
                log(f"🔧 Active Ollama processes: {ollama_count}")
    except Exception as e:
        # Silently fail - GPU monitoring is optional
        pass

def timeout_monitor():
    global last_log_time
    gpu_monitor_counter = 0
    while True:
        # Only restart if no activity AND progress bar hasn't moved
        time.sleep(30)  # Check less frequently
        if datetime.now() - last_log_time > timedelta(seconds=LOG_TIMEOUT_SECONDS):
            log(f"⚠️ No activity for {LOG_TIMEOUT_SECONDS}s. Restarting...")
            os.execv(sys.executable, ['python'] + sys.argv)
        
        # Monitor GPU utilization every 5 minutes (10 iterations × 30s = 5min)
        gpu_monitor_counter += 1
        if gpu_monitor_counter >= 10:
            monitor_gpu_utilization()
            gpu_monitor_counter = 0

def run_database_backup(mongo_uri: str) -> None:
    """Execute the backup script with force/drop semantics for all field DBs."""
    log("💾 Starting field database backup...")
    original_argv = sys.argv
    try:
        sys.argv = [
            "backup_field_databases.py",
            "--mongo-uri",
            mongo_uri,
            "--force",
            "--drop-existing",
            "--yes",
        ]
        backup_field_databases.main()
    finally:
        sys.argv = original_argv


def run_database_prune(mongo_uri: str, limit: int = 1000) -> None:
    """Prune each field database down to the working-set size via helper script."""
    log("🧹 Pruning field databases to working set...")
    original_argv = sys.argv
    try:
        sys.argv = [
            "prune_field_databases.py",
            "--mongo-uri",
            mongo_uri,
            "--limit",
            str(limit),
            "--force",
            "--yes",
        ]
        prune_field_databases.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    # Ensure MongoDB and Ollama are installed and running before watchdog
    ensure_mongodb_installed_and_running()
    ensure_ollama_instances()

    # Start watchdog after long setup steps to avoid premature restarts
    monitor_thread = Thread(target=timeout_monitor, daemon=True)
    monitor_thread.start()

    # Use connection pooling for better performance with high concurrency
    log("🔌 Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI, maxPoolSize=100, minPoolSize=10)
    log("✅ MongoDB connected")

    log("📥 Downloading OpenAlex manifest...")
    manifest_url = 'https://openalex.s3.amazonaws.com/data/works/manifest'
    manifest = requests.get(manifest_url).json()
    works_entries = manifest.get('entries', [])
    log(f"✅ Manifest downloaded, found {len(works_entries)} URL entries")

    # Convert S3 scheme to HTTPS
    urls = []
    for entry in works_entries:
        url = entry.get('url')
        if isinstance(url, str) and url.startswith('s3://openalex/'):
            url = url.replace('s3://openalex/', 'https://openalex.s3.amazonaws.com/')
        urls.append(url)

    total_records = sum(entry.get('meta', {}).get('record_count', 0) for entry in works_entries)
    log(f"📊 Total records to process: {total_records:,}")
    progress_bar = tqdm(total=total_records, unit='it', desc=f'{get_hostname()} QA Generation')
    log(f"🚀 Starting to process {len(urls)} URLs...")

    def process_url(url):
        processed = 0
        skipped_records = 0
        
        # Skip URL-level resume check - it's too slow. Rely on record-level check instead.
        # The record-level check in should_process_row() will skip already-processed records efficiently.
        log(f"📥 Starting to process URL: {url}")
        
        # Stream in chunks to limit memory usage
        try:
            # Full concurrency: allow all nodes to be fully utilized
            max_workers = len(nodes) * WORKERS_PER_NODE if nodes else 45
            log(f"🚀 Using {max_workers} concurrent workers ({len(nodes)} nodes × {WORKERS_PER_NODE} workers/node)")
            executor = ThreadPoolExecutor(max_workers=max_workers)
            
            chunk_count = 0
            for chunk in pd.read_json(url, lines=True, chunksize=16384):  # Increased from 8192 for faster I/O
                chunk_count += 1
                if chunk_count == 1:
                    log(f"📊 Processing first chunk from {url} ({len(chunk)} records)")
                if chunk.empty:
                    continue

                # Early filtering before executor submission
                filtered_rows = []
                for _, row in chunk.iterrows():
                    should_process, data = should_process_row(row, mongo_client)
                    if should_process and nodes:
                        filtered_rows.append((row, data[0], data[1]))  # row, abstract, field_ids
                    elif should_process is False and data is None:
                        # Record was skipped (already processed or filtered out)
                        skipped_records += 1
                
                if chunk_count == 1:
                    log(f"🔍 Filtered chunk: {len(filtered_rows)} records to process, {skipped_records} skipped")
                
                # Submit only filtered rows to executor
                futures = []
                for row, abstract, field_ids in filtered_rows:
                    # Use smart node selection that avoids overloaded nodes
                    node = get_available_node()
                    futures.append(executor.submit(process_row, node, mongo_client, row, abstract, field_ids))
                
                if chunk_count == 1 and len(futures) > 0:
                    log(f"🚀 Submitted {len(futures)} tasks to executor")
                
                # Process results as they complete using as_completed() for better parallelism
                # Don't wait for all - process next chunk while this one is still running
                for f in as_completed(futures):
                    try:
                        if f.result():
                            processed += 1
                    except Exception:
                        pass  # Already logged in process_row
                
                progress_bar.update(len(chunk))
            
            executor.shutdown(wait=True)
            
            # Log skipped records occasionally
            if skipped_records > 0 and random.random() < 0.1:  # Log 10% of the time
                log(f"⏭️ Skipped {skipped_records} already-processed records in {url}")
        except Exception as e:
            log(f"⚠️ Error processing chunk from {url}: {e}")
        return processed

    total_processed = 0
    for url in urls:
        try:
            total_processed += process_url(url)
        except Exception as e:
            log(f"⚠️ Error processing {url}: {e}")

    progress_bar.close()
    mongo_client.close()
    log(f"🎉 Completed QA generation. Processed {total_processed} documents with Q/A.")

    try:
        run_database_backup(MONGO_URI)
        run_database_prune(MONGO_URI)
        log("✅ Backup and prune steps completed.")
    except Exception as exc:
        log(f"⚠️ Post-processing maintenance encountered an issue: {exc}")


#!/usr/bin/env python3
"""Adaptive quiz runner that evaluates many LLMs with dynamic Ollama scaling."""

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

# ────────── config ──────────
INSTANCES: List[Dict[str, str]] = []
OLLAMA_NODES: List[str] = []
GPU_MEMORY_RESERVE_MB = 512
DEFAULT_INSTANCE_MEMORY_MB = 2048
MODEL_MEMORY_OVERRIDES_MB = {
    "smollm:135m": 906,
    "smollm2:135m": 906,
    "smollm:360m": 1200,
    "smollm2:360m": 1200,
    "gemma3:1b": 2000,
    "gemma3:270m": 1200,
}
BASE_PORT = 11434
TIMEOUT_SEC  = 30
QUESTIONS_PER_FIELD = 1_000
MAX_REPROMPTS = 2

VALID_ANSWERS     = {"true", "false", "possibly true", "possibly false", "unknown"}
TRUE_SET, FALSE_SET = {"true", "possibly true"}, {"false", "possibly false"}
ALIAS = {"possbilytrue": "possibly true", "possiblyfalse": "possibly false"}
NUMERIC_CHOICES = {
    "1": "true",
    "2": "false",
    "3": "possibly true",
    "4": "possibly false",
    "5": "unknown",
}
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
FILLER_SYNONYMS = {
    "true": {"yes", "yeah", "yep"},
    "false": {"no", "nope", "nah", "negative", "incorrect"},
    "possibly true": {"maybe", "probably", "likely", "perhaps"},
    "possibly false": {"unlikely"},
    "unknown": {"unknown", "unsure", "uncertain", "indeterminate"},
}
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
    lowered = text.lower()
    return any(hint in lowered for hint in FILLER_HINTS)
REPROMPT_TEMPLATE = (
    "\n\nYour previous answer \"{answer}\" was invalid because {reason}. "
    "Respond with exactly one lowercase word: true, false, possibly true, possibly false, unknown. "
    "Do not start with filler phrases such as \"here is\" or \"the answer is\", and do not write any sentences. "
    "Copy the word exactly as spelled above with no punctuation or extras. "
    "If you are unsure, reply with unknown. Never leave the reply blank."
)

FIELD_MAP = {  # … unchanged … (same mapping dictionary) ...
    "field_11":"agricultural-and-biological-sciences","field_12":"arts-and-humanities",
    "field_13":"biochemistry-genetics-and-molecular-biology","field_14":"business-management-and-accounting",
    "field_15":"chemical-engineering","field_16":"chemistry","field_17":"computer-science",
    "field_18":"decision-sciences","field_19":"earth-and-planetary-sciences",
    "field_20":"economics-econometrics-and-finance","field_21":"energy","field_22":"engineering",
    "field_23":"environmental-science","field_24":"immunology-and-microbiology",
    "field_25":"materials-science","field_26":"mathematics","field_27":"medicine",
    "field_28":"neuroscience","field_29":"nursing","field_30":"pharmacology-toxicology-and-pharmaceutics",
    "field_31":"physics-and-astronomy","field_32":"psychology","field_33":"social-sciences",
    "field_34":"veterinary","field_35":"dentistry","field_36":"health-professions",
}
DATABASES = list(FIELD_MAP.keys())

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
    "deepseek-llm:67b",
    "llama3:70b",
    "llama3.1:70b",
    "llama3.3:70b",
    "deepseek-r1:70b",
    "dolphin-llama3:70b",
    "orca-mini:70b",
    "hermes3:70b",
    "cogito:70b",
    "qwen:72b",
    "qwen2:72b",
    "qwen2.5:72b",
    "llama4:16×17b",
    "qwen:110b",
]

MONGO_URI, LOG_PATH, CSV_PATH = "mongodb://localhost:27017/", Path("quiz_llms.log"), Path("results.csv")

# ────────── logging ──────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH,encoding="utf-8")])
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("ollama._client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
log, log_warn, log_err = logger.info, logger.warning, logger.error

#if not MODEL_LIST:
#    MODEL_LIST = ["gemma:2b","gemma:7b","llama3:8b","mistral","phi3-mini","tinyllama",
#                  "qwen2:7b","mixtral-8x7b","gemma2b-it","mistral-openorca"]#
#    log_err(f"models.txt not found or empty; using fallback MODEL_LIST ({len(MODEL_LIST)} entries)")

# ────────── scoring constants ──────────
CORRECT_SCORE = 0.1   # 1000 correct answers → 100 points
INCORRECT_SCORE = -0.1

OLLAMA_PROCESSES: List[subprocess.Popen] = []
OLLAMA_MANAGED_PIDS: set[int] = set()
GLOBAL_SALVAGE_TOTAL: int = 0
GLOBAL_SALVAGE_BY_MODEL: Dict[str, int] = {}
GLOBAL_SALVAGE_BY_FIELD: Dict[str, int] = defaultdict(int)


def _register_pid(pid: int) -> None:
    if pid > 0:
        OLLAMA_MANAGED_PIDS.add(pid)


def _detect_pids_for_ports(ports: list[str]) -> set[int]:
    if not ports:
        return set()
    try:
        output = subprocess.check_output(
            ["ss", "-ltnp"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return set()

    port_patterns = tuple(f":{port}" for port in ports)
    candidates: set[int] = set()
    for line in output.splitlines():
        if any(pattern in line for pattern in port_patterns):
            candidates.update(int(match) for match in re.findall(r"pid=(\d+)", line))
    return candidates


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_pid(pid: int, timeout: float = 5.0) -> None:
    if pid <= 0:
        return
    if not _pid_is_alive(pid):
        return

    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            return
        time.sleep(0.1)

    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def shutdown_ollama_instances(force_all: bool = False) -> None:
    """Terminate tracked Ollama processes and any strays bound to our ports."""
    for proc in list(OLLAMA_PROCESSES):
        if proc.poll() is None:
            with suppress(Exception):
                proc.terminate()
                proc.wait(timeout=5)
        if proc.poll() is None:
            with suppress(Exception):
                proc.kill()
        OLLAMA_PROCESSES.remove(proc)

    managed_pids = set(OLLAMA_MANAGED_PIDS)
    if force_all:
        managed_pids |= _detect_pids_for_ports([inst["port"] for inst in INSTANCES])

    for pid in managed_pids:
        _terminate_pid(pid)

    OLLAMA_MANAGED_PIDS.clear()
    with suppress(Exception):
        subprocess.run(["pkill", "-f", "ollama serve"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


atexit.register(lambda: shutdown_ollama_instances(force_all=True))

# ────────── environment helpers ──────────

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ensure_ollama_installed() -> None:
    if shutil.which("ollama"):
        return
    install_cmd = ["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"]
    result = subprocess.run(install_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to install Ollama automatically. Please install it manually.")


def _service_action(args: list[str]) -> bool:
    return _run(args).returncode == 0


def ensure_ollama_running() -> None:
    if not INSTANCES:
        raise RuntimeError("No Ollama instances planned. Call restart_ollama_instances(model) first.")
    ensure_ollama_installed()
    ports = [inst["port"] for inst in INSTANCES]
    existing_pids = _detect_pids_for_ports(ports)
    for pid in existing_pids:
        _register_pid(pid)
    if existing_pids:
        shutdown_ollama_instances(force_all=True)
        time.sleep(0.5)

    primary_hosts = [inst["host"] for inst in INSTANCES]

    def ping(host: str) -> bool:
        with suppress(Exception):
            ollama.Client(host=host).list()
            return True
        return False

    if shutil.which("systemctl"):
        _service_action(["systemctl", "stop", "ollama"])
        time.sleep(1)

    for inst in INSTANCES:
        if ping(inst["host"]):
            for pid in _detect_pids_for_ports([inst["port"]]):
                _register_pid(pid)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = inst["gpu"]
        env["OLLAMA_HOST"] = f"0.0.0.0:{inst['port']}"

        log(f"Starting Ollama instance on port {inst['port']} using GPU {inst['gpu']}")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        OLLAMA_PROCESSES.append(proc)
        _register_pid(proc.pid)
        time.sleep(2)
        if not ping(inst["host"]):
            raise RuntimeError(f"Unable to reach Ollama at {inst['host']}. Please ensure the service is running.")

    for pid in _detect_pids_for_ports(ports):
        _register_pid(pid)


def restart_ollama_instances(model: str) -> None:
    """Stop any running Ollama servers and relaunch workers sized for *model*."""
    global INSTANCES, OLLAMA_NODES
    INSTANCES = plan_instances_for_model(model)
    OLLAMA_NODES = [inst["host"] for inst in INSTANCES]
    log(f"Planning {len(INSTANCES)} Ollama workers for {model} (≈{estimate_model_memory_mb(model)} MB each)")
    shutdown_ollama_instances(force_all=True)
    time.sleep(0.5)
    ensure_ollama_running()


# ────────── helpers ──────────
EMBEDMODELS:set[str]=set()
def pull(model:str)->bool:
    if model in EMBEDMODELS: return False
    for n in OLLAMA_NODES:
        try: ollama.Client(host=n).pull(model,stream=False)
        except ollama.ResponseError as e:
            if "does not support generate" in str(e).lower():
                EMBEDMODELS.add(model); log_err(f"embed-only {model}"); return False
            log_err(f"pull error {n} {model}: {e}"); return False
        except Exception as e: log_err(f"pull exc {n} {model}: {e}"); return False
    return True

def delete(model:str):
    for n in OLLAMA_NODES:
        try: ollama.Client(host=n).delete(model,force=True)
        except Exception: pass


def stop_model_on_all_nodes(model: str) -> None:
    for node in OLLAMA_NODES:
        with suppress(Exception):
            ollama.Client(host=node).stop(model)

def ask(model:str,prompt:str,node:str)->str:
    try:
        client = ollama.Client(host=node, timeout=TIMEOUT_SEC)
        r = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 10,  # enough tokens for full guidance while preventing rambling
                "keep_alive": 0,
            },
        )
        return r["response"].strip()
    except ollama.ResponseError as e:
        if "does not support generate" in str(e).lower():
            EMBEDMODELS.add(model)
        log_err(f"gen error {node} {model}: {e}")
        return "unknown"
    except Exception as e:
        log_err(f"gen exc {node} {model}: {e}")
        return "unknown"


def normalize_and_validate(raw: str) -> Tuple[str, bool, str]:
    if raw is None:
        return "unknown", False, "Empty response"

    stripped = raw.strip()
    if not stripped:
        return "unknown", False, "Empty response"

    normalized = " ".join(stripped.lower().split())
    normalized = ALIAS.get(normalized, normalized)

    if re.fullmatch(r"[1-5](?:[).])?", normalized):
        mapped = NUMERIC_CHOICES[normalized[0]]
        return mapped, True, "Numeric choice"

    if normalized in VALID_ANSWERS and stripped.lower() == normalized:
        return normalized, True, "Exact match"

    stripped_no_punct = stripped.rstrip(".!,?:;")
    collapsed_no_punct = " ".join(stripped_no_punct.lower().split())
    collapsed_no_punct = ALIAS.get(collapsed_no_punct, collapsed_no_punct)

    if collapsed_no_punct in VALID_ANSWERS:
        return collapsed_no_punct, False, "Answer must be exactly one word/phrase with no punctuation."

    for phrase, mapped in FILLER_PHRASES.items():
        if phrase in normalized:
            return mapped, True, f"Alias phrase '{phrase}'"

    word_tokens = [token for token in re.split(r"[^a-z]+", normalized) if token]
    if word_tokens:
        for target, synonyms in FILLER_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in word_tokens:
                    return target, True, f"Alias keyword '{synonym}'"

    for token in VALID_ANSWERS:
        if token in normalized:
            return token, True, "Salvaged token from verbose response"

    return "unknown", False, "Answer must be exactly one of: true, false, possibly true, possibly false, unknown."


def build_reprompt_prompt(base_prompt: str, reason: str, previous: str) -> str:
    reason_text = reason or "the answer was not in the valid list"
    normalized_previous = " ".join(previous.strip().split()) or "∅"
    normalized_previous = normalized_previous.replace('"', "'")
    return f"{base_prompt}{REPROMPT_TEMPLATE.format(answer=normalized_previous, reason=reason_text)}"


def get_validated_answer(model: str, base_prompt: str, node: str) -> Tuple[str, str, int, bool, str]:
    attempts = 0
    prompt = base_prompt
    last_raw = ""

    while attempts < MAX_REPROMPTS:
        raw_response = ask(model, prompt, node)
        normalized, valid, reason = normalize_and_validate(raw_response)
        if valid:
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

        filler_hit = looks_like_filler_response(raw_response) if raw_response else False
        reprompt_reason = reason
        if filler_hit:
            preview = (raw_response or "").strip().replace("\n", " ")[:80]
            log(
                f"Filler invalid (model={model}, attempt={attempts + 1}) -> '{preview}'"
            )
            if attempts == 0:
                reprompt_reason = (
                    "your reply began with filler text instead of one of the allowed words"
                )
            else:
                normalized = normalized if normalized in VALID_ANSWERS else "unknown"
                log(
                    f"Filler repeated (model={model}, attempt={attempts + 1}) -> '{preview}' | aborting"
                )
                return normalized, raw_response, attempts + 1, False, "filler_skipped"

        attempts += 1
        last_raw = raw_response
        if attempts >= MAX_REPROMPTS:
            final_tag = "filler_invalid" if filler_hit else "invalid"
            return normalized, last_raw, attempts, False, final_tag

        prompt = build_reprompt_prompt(base_prompt, reprompt_reason, last_raw)

    return "unknown", last_raw, attempts, False, "invalid"

def build_prompt(field:str,q:str,year:int|None)->str:
    yr = f"The paper was published in {year}. " if year else ""
    instructions = (
        "Respond with exactly one lowercase word: true, false, possibly true, possibly false, unknown.\n"
        "Do not add sentences, introductions, or punctuation—phrases like \"here is\" or \"the answer is\" are invalid.\n"
        "Copy the word exactly as spelled above. If you cannot decide, reply unknown. Never leave the reply blank.\n"
    )
    context = f"You are being quizzed in {field}. {yr}"
    return f"{instructions}{context}Question:\n{q}"

def score(gt:str,pred:str)->float:
    if pred=="unknown": return 0.0
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return CORRECT_SCORE
    return INCORRECT_SCORE


def process_batch(node: str, batch: List[Tuple[int, Dict]], model: str, field_name: str) -> Tuple[float, int]:
    total = 0.0
    salvaged_verbose_count = 0
    for idx, doc in batch:
        q, gt = doc["Question"], doc["Answer"].lower()
        year = doc.get("publication_year")
        prompt = build_prompt(field_name, q, year)

        normalized, raw_answer, attempts_used, valid_answer, reason_tag = get_validated_answer(model, prompt, node)
        if reason_tag == "salvaged_verbose":
            salvaged_verbose_count += 1
        if not valid_answer:
            if normalized not in VALID_ANSWERS:
                normalized = "unknown"
            filler_notice = reason_tag.startswith("filler_")
            if filler_notice:
                if reason_tag == "filler_skipped":
                    message = (
                        f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
                        f"filler response detected; no retry issued; raw='{raw_answer}' -> using '{normalized}'"
                    )
                else:
                    message = (
                        f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
                        f"filler response ignored after {attempts_used} attempts; raw='{raw_answer}' -> using '{normalized}'"
                    )
                log_err(message)
            else:
                message = (
                    f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
                    f"invalid response after {attempts_used} attempts; raw='{raw_answer}' -> using '{normalized}'"
                )
                if normalized == "unknown" and attempts_used > 1:
                    log_err(message)
                elif normalized == "unknown":
                    log(message)
                else:
                    log(message)

        sc = score(gt, normalized)
        res = "✅" if sc > 0 else "❌" if sc < 0 else "☐"
        attempt_info = f"attempts:{attempts_used}"
        log(
            f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | "
            f"GT:{gt} | LLM:{raw_answer} -> {normalized} | {res} | {attempt_info}"
        )
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
    return [24576]


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
    
    # Fallback to 8 cores if detection fails
    return 8


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
    
    # Fallback to 32 GB if detection fails
    return 32.0


def calculate_workers_per_node() -> int:
    """Calculate optimal workers per node based on CPU cores and GPU count."""
    cpu_cores = _detect_cpu_cores()
    num_gpus = len(GPU_MEMORY_MB)
    
    # Formula: workers = min(max(2, cpu_cores // num_gpus), 16)
    # Minimum 2, scale with CPU cores divided by GPUs, cap at 16
    if num_gpus > 0:
        workers = min(max(2, cpu_cores // num_gpus), 16)
    else:
        # If no GPUs detected, use a conservative value
        workers = min(max(2, cpu_cores // 4), 16)
    
    return workers


GPU_MEMORY_MB = _query_gpu_memory_mb()
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


def plan_instances_for_model(model: str) -> List[Dict[str, str]]:
    """Produce an Ollama instance plan sized for the requested model."""
    mem_needed = max(estimate_model_memory_mb(model), 1)
    plan: List[Dict[str, str]] = []
    port = BASE_PORT
    for gpu_index, total_mem in enumerate(GPU_MEMORY_MB):
        available = max(0, total_mem - GPU_MEMORY_RESERVE_MB)
        if available >= mem_needed:
            count = max(1, available // mem_needed)
        else:
            count = 1
        count = max(1, min(count, 8))
        for _ in range(count):
            plan.append({
                "host": f"http://127.0.0.1:{port}",
                "port": str(port),
                "gpu": str(gpu_index),
            })
            port += 1
    if not plan:
        plan.append({"host": f"http://127.0.0.1:{BASE_PORT}", "port": str(BASE_PORT), "gpu": "0"})
    return plan


# ────────── main ──────────
def main():
    global GLOBAL_SALVAGE_TOTAL, GLOBAL_SALVAGE_BY_MODEL, GLOBAL_SALVAGE_BY_FIELD
    GLOBAL_SALVAGE_TOTAL = 0
    GLOBAL_SALVAGE_BY_MODEL = {}
    GLOBAL_SALVAGE_BY_FIELD = defaultdict(int)
    
    # Log system specs and calculated concurrency
    cpu_cores = _detect_cpu_cores()
    ram_gb = _detect_available_ram_gb()
    num_gpus = len(GPU_MEMORY_MB)
    log(f"💻 System specs: {cpu_cores} CPU cores, {ram_gb:.1f} GB RAM, {num_gpus} GPU(s)")
    log(f"⚙️  Calculated concurrency: {WORKERS_PER_NODE} workers per node")
    
    completed=done(); hdr=CSV_PATH.exists()
    mongo=MongoClient(MONGO_URI)
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

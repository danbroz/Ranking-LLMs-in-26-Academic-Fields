#!/usr/bin/env python3
"""
quiz_llms.py — resilient, resumable, round‑robin quiz‑runner
===========================================================

2025‑04‑22 — scoring + prompt + timeout rev
-----------------------------------------
* Prompt now allows **unknown** as a valid answer.
* New scoring:
    • correct (true⇄possibly true or false⇄possibly false) → **+0.01**
    • unknown → **0.00**
    • incorrect → **‑0.01** (keeps random guessing ≈ 0).
* Questions per field raised to **10 000**; log shows counter "n/10 000".
* Year pulled from MongoDB key `publication_year`.
* 30 s timeout on generation; on expiry answer defaults to unknown.
* Model list
    – pruned to chat‑sized models runnable on a single RTX 4090 (≤15 GB VRAM).

"""
from __future__ import annotations

import csv, itertools, logging, sys, signal, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import ollama
from pymongo import MongoClient

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
OLLAMA_NODES: List[str] = [
    "http://laptop0:11434",
    "http://laptop1:11434",
    "http://laptop2:11434",
]

FIELD_MAP: Dict[str, str] = {
    "field_11": "agricultural-and-biological-sciences",
    "field_12": "arts-and-humanities",
    "field_13": "biochemistry-genetics-and-molecular-biology",
    "field_14": "business-management-and-accounting",
    "field_15": "chemical-engineering",
    "field_16": "chemistry",
    "field_17": "computer-science",
    "field_18": "decision-sciences",
    "field_19": "earth-and-planetary-sciences",
    "field_20": "economics-econometrics-and-finance",
    "field_21": "energy",
    "field_22": "engineering",
    "field_23": "environmental-science",
    "field_24": "immunology-and-microbiology",
    "field_25": "materials-science",
    "field_26": "mathematics",
    "field_27": "medicine",
    "field_28": "neuroscience",
    "field_29": "nursing",
    "field_30": "pharmacology-toxicology-and-pharmaceutics",
    "field_31": "physics-and-astronomy",
    "field_32": "psychology",
    "field_33": "social-sciences",
    "field_34": "veterinary",
    "field_35": "dentistry",
    "field_36": "health-professions",
}
DATABASES = list(FIELD_MAP.keys())

VALID_ANSWERS = {"true", "false", "possibly true", "possibly false", "unknown"}
TRUE_SET = {"true", "possibly true"}
FALSE_SET = {"false", "possibly false"}
QUESTIONS_PER_FIELD = 10_000
MAX_REPROMPTS = 3
TIMEOUT_SECONDS = 30

# Pruned list — chat LLMs that run comfortably on 24 GB   (<=15 GB VRAM)
MODEL_LIST: List[str] = [
    "gemma2", "gemma3", "mistral", "mistral-openorca", "mixtral",
    "llama3.2", "tinyllama", "tinydolphin", "starcoder2", "hermes3",
    "phi3", "phi3.5", "phi4-mini", "zephyr", "xwinlm"
]

MONGO_URI = "mongodb://localhost:27017/"
LOG_PATH = Path("quiz_llms.log")
CSV_PATH = Path("results.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)
log = logging.getLogger(__name__).info
log_err = logging.getLogger(__name__).error

EMBEDDING_MODELS: set[str] = set()

# ──────────────────────────────────────────────────────────
# Timeout helper
# ──────────────────────────────────────────────────────────
class TimeoutError(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutError

signal.signal(signal.SIGALRM, _handler)

# ──────────────────────────────────────────────────────────
# Ollama helpers
# ──────────────────────────────────────────────────────────

def pull_or_skip(model: str) -> bool:
    if model in EMBEDDING_MODELS:
        return False
    for node in OLLAMA_NODES:
        try:
            ollama.Client(host=node).pull(model, stream=False)
        except ollama.ResponseError as e:
            if "does not support generate" in str(e).lower():
                EMBEDDING_MODELS.add(model)
                log_err(f"❌ {model} embedding-only — skipped")
                return False
            log_err(f"Pull error {node} ({model}): {e}")
            return False
        except Exception as e:
            log_err(f"Pull exception {node} ({model}): {e}")
            return False
    return True

def delete_everywhere(model: str):
    for node in OLLAMA_NODES:
        try:
            ollama.Client(host=node).delete(model, force=True)
        except Exception:
            pass

def ask(model: str, prompt: str, node: str) -> str | None:
    try:
        signal.alarm(TIMEOUT_SECONDS)
        resp = ollama.Client(host=node).generate(model=model, prompt=prompt, stream=False)
        return resp["response"].strip()
    except TimeoutError:
        log_err(f"⏰ Timeout {model} on {node}")
        return "unknown"
    except ollama.ResponseError as e:
        if "does not support generate" in str(e).lower():
            EMBEDDING_MODELS.add(model)
        log_err(f"Gen error {node} ({model}): {e}")
        return None
    except Exception as e:
        log_err(f"Gen exception {node} ({model}): {e}")
        return None
    finally:
        signal.alarm(0)

# ──────────────────────────────────────────────────────────
# Prompt & scoring
# ──────────────────────────────────────────────────────────

def build_prompt(field: str, question: str, year: int | None) -> str:
    yr = f"The paper was published in {year}. " if year else ""
    return (
        f"You are being quizzed in {field}. {yr}"
        "You are being quizzed. Only the single word true, false, possibly true, "
        "possibly false, or unknown is valid as your answer. No explanation, no punctuation.\n\n"
        f"Question:\n{question}"
    )

def score(gt: str, pred: str) -> float:
    if pred == "unknown":
        return 0.0
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return 0.01
    return -0.01

# CSV helpers

def csv_header() -> List[str]:
    return ["model", "overall", *FIELD_MAP.values(), "timestamp"]

def completed() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        return {row[0] for row in r if row}

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    done = completed()
    header_written = CSV_PATH.exists()

    mongo = MongoClient(MONGO_URI)
    col_map = {d: mongo[d]["sources"] for d in DATABASES}
    node_cycle = itertools.cycle(OLLAMA_NODES)

    for model in MODEL_LIST:
        if model in done:
            log(f"🔄 {model} already done — skipping")
            continue
        log(f"=== MODEL {model} ===")
        if not pull_or_skip(model):
            continue

        field_scores: Dict[str, float] = {}

        for db in DATABASES:
            fname = FIELD_MAP[db]
            col = col_map[db]
            docs = list(
                col.aggregate([
                    {"$match": {"Question": {"$ne": None}, "Answer": {"$in": list(VALID_ANSWERS - {"unknown"})}}},
                    {"$sample": {"size": QUESTIONS_PER_FIELD}},
                ])
            )
            if not docs:
                field_scores[fname] = 0.0
                continue

            total = 0.0
            for idx, d in enumerate(docs, 1):
                q, gt = d["Question"], d["Answer"].lower()
                year = d.get("publication_year")
                prompt = build_prompt(fname, q, year)
                node = next(node_cycle)

                raw = None
                for _ in range(MAX_REPROMPTS):
                    raw = ask(model, prompt, node)
                    if raw is None:
                        pred = "unknown"
                        break
                    pred = raw.lower().strip().rstrip(".")
                    if pred in VALID_ANSWERS:
                        break
                else:
                    pred = "unknown"

                sc = score(gt, pred)
                res = "✅" if sc > 0 else "❌" if sc < 0 else "� neutral"
                log(f"{fname} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | GT:{gt} | LLM:{raw} | {res}")
                total += sc

            field_scores[fname] = round(total, 4)

        overall = round(sum(field_scores.values()) / len(FIELD_MAP), 4)
        row = [model, str(overall)] + [str(field_scores.get(f, 0.0)) for f in FIELD_MAP.values()] + [datetime.now().isoformat()]

        mode = "a" if header_written else "w"
        with CSV_PATH.open(mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not header_written:
                w.writerow(csv_header())
                header_written = True
            w.writerow(row)

        delete_everywhere(model)
        log(f"🏁 Finished {model}\n")

    mongo.close()
    log("🎉 All models processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted — exiting.")


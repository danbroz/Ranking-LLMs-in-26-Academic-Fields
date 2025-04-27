#!/usr/bin/env python3
"""
quiz_llms.py — resilient, resumable, round‑robin quiz‑runner
===========================================================

Patch 2025‑04‑22‑typo‑aliases
---------------------------
Adds support for answer typos **"possbilytrue"** and **"possiblyfalse"** – they
are normalised to "possibly true" and "possibly false" respectively for scoring
purposes.

Scoring (unchanged from previous rev):
    correct class ➜ +0.01  |  unknown ➜ 0.00  |  wrong ➜ -0.01
"""
from __future__ import annotations

import csv
import itertools
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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

# canonical answers
CANONICAL = {
    "true": "true",
    "possibly true": "possibly true",
    "possbilytrue": "possibly true",   # alias
    "false": "false",
    "possibly false": "possibly false",
    "possiblyfalse": "possibly false", # alias
    "unknown": "unknown",
}
VALID_ANSWERS = set(CANONICAL.keys())
TRUE_SET = {"true", "possibly true"}
FALSE_SET = {"false", "possibly false"}

QUESTIONS_PER_FIELD = 10_000
MAX_REPROMPTS = 3
TIMEOUT_SECONDS = 30

# Shrunken chat‑capable model list (RTX 4090‑friendly examples)
MODEL_LIST = [
    "gemma:2b", "mistral:7b", "llama3:8b", "phi3:mini",
    "tinyllama", "mixtral:8x7b", "starcoder2:15b", "deepseek-llm:7b"
]

MONGO_URI = "mongodb://localhost:27017/"
LOG_PATH = Path("quiz_llms.log")
CSV_PATH = Path("results.csv")

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)
log = logging.getLogger(__name__).info
log_err = logging.getLogger(__name__).error

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
EMBEDDING_MODELS: set[str] = set()

def normalise(ans: str) -> str:
    return CANONICAL.get(ans.lower().strip().rstrip("."), "invalid")


def pull_model(model: str) -> bool:
    if model in EMBEDDING_MODELS:
        return False
    for node in OLLAMA_NODES:
        try:
            ollama.Client(host=node).pull(model, stream=False)
        except ollama.ResponseError as e:
            if "does not support generate" in str(e).lower():
                EMBEDDING_MODELS.add(model)
                log_err(f"❌ {model} embed‑only — skip")
                return False
            return False
        except Exception:
            return False
    return True


def delete_model(model: str):
    for n in OLLAMA_NODES:
        try:
            ollama.Client(host=n).delete(model, force=True)
        except Exception:
            pass


def ask(model: str, prompt: str, node: str) -> Optional[str]:
    try:
        resp = ollama.Client(host=node).generate(model=model, prompt=prompt, stream=False, timeout=TIMEOUT_SECONDS)
        return resp["response"].strip()
    except Exception:
        return None


def build_prompt(field: str, q: str, year: int | None) -> str:
    ytxt = f"The paper was published in {year}. " if year else ""
    return (
        f"You are being quizzed in {field}. {ytxt}"
        "You are being quizzed. Only the single word true, false, possibly true, "
        "possibly false, or unknown is valid as your answer. No explanation, no punctuation.\n\n"
        f"Question:\n{q}"
    )


def score(gt: str, pred: str) -> float:
    if pred == "unknown":
        return 0.0
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return 0.01
    return -0.01


def csv_header() -> List[str]:
    return ["model", "overall", *FIELD_MAP.values(), "timestamp"]


def done_models() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        nxt = csv.reader(f)
        next(nxt, None)
        return {r[0] for r in nxt if r}

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    finished = done_models()
    hdr_written = CSV_PATH.exists()

    mongo = MongoClient(MONGO_URI)
    col = {db: mongo[db]["sources"] for db in DATABASES}
    node_cycle = itertools.cycle(OLLAMA_NODES)

    for model in MODEL_LIST:
        if model in finished:
            log(f"Skip {model} — already done")
            continue
        if not pull_model(model):
            continue
        log(f"=== MODEL {model} ===")
        field_scores: Dict[str, float] = {}

        for db in DATABASES:
            fname = FIELD_MAP[db]
            docs = list(
                col[db].aggregate(
                    [
                        {"$match": {"Question": {"$ne": None}, "Answer": {"$in": list(TRUE_SET | FALSE_SET)}}},
                        {"$sample": {"size": QUESTIONS_PER_FIELD}},
                    ]
                )
            )
            total = 0.0
            for idx, d in enumerate(docs, 1):
                gt = normalise(d["Answer"])
                year = d.get("publication_year")
                prompt = build_prompt(fname, d["Question"], year)
                node = next(node_cycle)

                raw = None
                for _ in range(MAX_REPROMPTS):
                    raw = ask(model, prompt, node)
                    if raw is None:
                        raw = "unknown"
                        break
                    pred_norm = normalise(raw)
                    if pred_norm != "invalid":
                        break
                else:
                    pred_norm = "unknown"

                sc = score(gt, pred_norm)
                res = "✅" if sc > 0 else ("" if sc == 0 else "❌")
                log(f"{fname} | {year or '—'} | {model} | {idx}/{QUESTIONS_PER_FIELD} | GT:{gt} | LLM:{raw} | {res}")
                total += sc
            field_scores[fname] = round(total, 3)

        overall = round(sum(field_scores.values()) / len(FIELD_MAP), 3)
        row = [model, str(overall)] + [str(field_scores.get(f, 0.0)) for f in FIELD_MAP.values()] + [datetime.now().isoformat()]

        mode = "a" if hdr_written else "w"
        with CSV_PATH.open(mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not hdr_written:
                w.writerow(csv_header())
                hdr_written = True
            w.writerow(row)

        delete_model(model)
        log(f"Finished {model}\n")

    mongo.close()
    log("All models complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted — exiting")

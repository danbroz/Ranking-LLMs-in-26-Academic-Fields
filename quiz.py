#!/usr/bin/env python3
"""
quiz_llms.py — resilient, resumable, round-robin quiz-runner
===========================================================

Key features
------------
* Three Ollama nodes (laptop0/1/2:11434) used round-robin.
* Pulls each model on all nodes before use; deletes when finished.
* Detects / skips embedding-only models that reject `generate`.
* Strict per-question prompt:

      You are being quizzed in <field-name>.
      The paper was published in <year>.
      You are being quizzed. Only the single word true, false, possibly true,
      or possibly false is valid as your answer. No explanation, no punctuation.

* Logs: **field | year | model | question | GT | LLM | result**  
  ✅ = +0.10 ❌ = -0.10
* New scoring:
      – GT ∈{true, possibly true} & Pred ∈{true, possibly true}  → +0.10  
      – GT ∈{false, possibly false} & Pred ∈{false, possibly false} → +0.10  
      – Otherwise → -0.10 (so random guessing ≈ 0).
* Writes `results.csv` with human field names, overall, timestamp.
* Resumable — skips any LLM already present in `results.csv`.
"""

from __future__ import annotations

import csv
import itertools
import logging
import sys
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

VALID_ANSWERS = {"true", "false", "possibly true", "possibly false"}
TRUE_SET = {"true", "possibly true"}
FALSE_SET = {"false", "possibly false"}
QUESTIONS_PER_FIELD = 1_000
MAX_REPROMPTS = 3

# Full model list (duplicates kept exactly as supplied)
MODEL_LIST: List[str] = [
    "internlm2","all-minilm","nomic-embed-text","snowflake-arctic-embed",
    "granite-embedding","smollm","smollm2","paraphrase-multilingual",
    "bge-large","mxbai-embed-large","qwen","qwen2","qwen2.5","qwen2.5-coder",
    "reader-lm","bge-m3","snowflake-arctic-embed2","falcon3","gemma3",
    "granite3-moe","granite3.1-moe","granite3.1-dense","granite3.2",
    "granite3.2-vision","granite3.3","llama-guard3","llama3.2","sailor2",
    "starcoder","tinydolphin","tinyllama","deepseek-coder","deepcoder",
    "deepscaler","phi","dolphin-phi","qwen2-math","yi-coder",
    "deepseek-coder-v2","exaone-deep","exaone3.5","granite3-guardian",
    "gemma","gemma2","codegemma","granite3.2-vision","llava-phi3",
    "llava-llama3","deepseek-r1","deepseek-llm","starcoder2","hermes3",
    "phi3.5","phi3","phi4-mini","phi4","smollm","smollm2","bge-large",
    "codestral","command-r7b","command-r7b-arabic","wizardlm","wizardlm2",
    "wizardlm-uncensored","wizardcoder","wizard-vicuna",
    "wizard-vicuna-uncensored","wizard-vicuna-uncensored","wizard-math",
    "wizard-math","wizard-vicuna-uncensored","xwinlm","zephyr","granite-code",
    "granite3.1-dense","granite3-dense","granite3.2","granite3.3",
    "granite3-moe","hermes3","internlm2","tinyllama","phi4-mini","smollm",
    "smollm2","deepseek-coder","deepseek-coder-v2","deepseek-r1",
    "deepseek-v2","deepseek-v3","dolphincoder","dolphin-mixtral",
    "dolphin-mistral","dolphin-llama3","dolphin3","moondream","mixtral",
    "mistral","mistral-nemo","mistral-openorca","mistral-small",
    "mistral-small3.1","mistrallite","minicpm-v","mxbai-embed-large",
    "neural-chat","notus","notux","nous-hermes","nous-hermes2",
    "nous-hermes2-mixtral","olmo2","openchat","openhermes","openthinker",
    "orca2","orca-mini","samantha-mistral","solar","solar-pro","sqlcoder",
    "stable-code","stablelm-zephyr","stablelm2","stable-beluga","starling-lm",
    "bge-large","bakllava","aya","aya-expanse","athene-v2","codebooga",
    "codegeex4","codeqwen","codestral","cogito","duckdb-nsql","falcon2",
    "falcon","gemma3","glm4","granite3.1-dense","granite3-moe","internlm2",
    "internlm2","llama3-chatqa","llama3-gradient","llama3-groq-tool-use",
    "mistral","mistral-openorca","mixtral","magicoder","mathstral",
    "medllama2","meditron","nemotron-mini","openhermes","openchat",
    "openthinker","paraphrase-multilingual","phind-codellama","reflection",
    "r1-1776","sailor2","smallthinker","snowflake-arctic-embed",
    "snowflake-arctic-embed2","tulu3","yarn-llama2","yarn-mistral","yi",
    "vicuna",
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


def pull_or_skip(model: str) -> bool:
    """Pull model on all nodes; return False if model is embedding-only or failed."""
    if model in EMBEDDING_MODELS:
        return False
    for node in OLLAMA_NODES:
        try:
            ollama.Client(host=node).pull(model, stream=False)
        except ollama.ResponseError as e:
            if "does not support generate" in str(e).lower():
                EMBEDDING_MODELS.add(model)
                log_err(f"❌ {model} is embedding-only — skipping")
                return False
            log_err(f"Pull error {node} ({model}): {e}")
            return False
        except Exception as e:
            log_err(f"Pull exception {node} ({model}): {e}")
            return False
    return True


def delete_everywhere(model: str) -> None:
    for node in OLLAMA_NODES:
        try:
            ollama.Client(host=node).delete(model, force=True)
        except Exception:
            pass


def ask(model: str, prompt: str, node: str) -> str | None:
    try:
        resp = ollama.Client(host=node).generate(model=model, prompt=prompt, stream=False)
        return resp["response"].strip()
    except ollama.ResponseError as e:
        if "does not support generate" in str(e).lower():
            EMBEDDING_MODELS.add(model)
        log_err(f"Gen error {node} ({model}): {e}")
        return None
    except Exception as e:
        log_err(f"Gen exception {node} ({model}): {e}")
        return None


def build_prompt(field: str, question: str, year: int | None) -> str:
    yr = f"The paper was published in {year}. " if year else ""
    return (
        f"You are being quizzed in {field}. {yr}"
        "You are being quizzed. Only the single word true, false, possibly true, "
        "or possibly false is valid as your answer. No explanation, no punctuation.\n\n"
        f"Question:\n{question}"
    )


def csv_header() -> List[str]:
    return ["model", *FIELD_MAP.values(), "overall", "timestamp"]


def completed_models() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        return {row[0] for row in r if row}


def score(gt: str, pred: str) -> float:
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return 0.10
    return -0.10


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main() -> None:
    done = completed_models()
    header_written = CSV_PATH.exists()

    mongo = MongoClient(MONGO_URI)
    collections = {db: mongo[db]["sources"] for db in DATABASES}
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
            col = collections[db]

            docs = list(
                col.aggregate(
                    [
                        {"$match": {"Question": {"$ne": None}, "Answer": {"$in": list(VALID_ANSWERS)}}},
                        {"$sample": {"size": QUESTIONS_PER_FIELD}},
                    ]
                )
            )
            if not docs:
                field_scores[fname] = 0.0
                continue

            field_total = 0.0
            for d in docs:
                q, gt = d["Question"], d["Answer"].lower()
                year = d.get("year")
                prompt = build_prompt(fname, q, year)
                node = next(node_cycle)

                raw = None
                for _ in range(MAX_REPROMPTS):
                    raw = ask(model, prompt, node)
                    if raw is None:
                        break
                    pred = raw.lower().strip().rstrip(".")
                    if pred in VALID_ANSWERS:
                        break
                else:
                    pred = "skipped"

                sc = score(gt, pred) if pred in VALID_ANSWERS else -0.10
                res = "✅" if sc > 0 else "❌"
                log(f"{fname} | {year or '—'} | {model} | {q} | GT:{gt} | LLM:{raw} | {res}")
                field_total += sc

            field_scores[fname] = round(field_total, 3)

        # write CSV
        row = [model] + [str(field_scores.get(f, 0.0)) for f in FIELD_MAP.values()]
        overall = round(sum(field_scores.values()) / len(FIELD_MAP), 3)
        row += [str(overall), datetime.now().isoformat()]

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


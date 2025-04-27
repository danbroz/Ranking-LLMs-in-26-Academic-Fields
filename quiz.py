#!/usr/bin/env python3
"""
quiz_llms.py — resilient, resumable, round-robin quiz-runner
===========================================================

Changes in this patch
---------------------
* **pred_norm always defined** – even when a reply is invalid or timed-out.
* Normalisation map now covers: “possbilytrue” → possibly true,
  “possiblyfalse” → possibly false.
* Scoring unchanged (+0.01 correct class, 0 unknown, -0.01 wrong).

Everything else (prompt with “unknown”, 30 s timeout, 10 000 Q/A, CSV order
model|overall|fields|timestamp, trimmed model list for RTX 4090) is intact.
"""

from __future__ import annotations
import csv, itertools, logging, sys, signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import ollama
from pymongo import MongoClient

# ────────── config ──────────
OLLAMA_NODES = ["http://laptop0:11434", "http://laptop1:11434", "http://laptop2:11434"]
TIMEOUT_SEC  = 30
QUESTIONS_PER_FIELD = 10_000
MAX_REPROMPTS = 3

VALID_ANSWERS     = {"true", "false", "possibly true", "possibly false", "unknown"}
TRUE_SET, FALSE_SET = {"true", "possibly true"}, {"false", "possibly false"}
ALIAS = {"possbilytrue": "possibly true", "possiblyfalse": "possibly false"}

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
MODEL_LIST = ["gemma:2b","gemma:7b","llama3:8b","mistral","phi3-mini","tinyllama",
              "qwen2:7b","mixtral-8x7b","gemma2b-it","mistral-openorca"]

MONGO_URI, LOG_PATH, CSV_PATH = "mongodb://localhost:27017/", Path("quiz_llms.log"), Path("results.csv")

# ────────── logging ──────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH,encoding="utf-8")])
log, log_err = logging.getLogger(__name__).info, logging.getLogger(__name__).error

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

def ask(model:str,prompt:str,node:str)->str:
    def handler(signum,frame): raise TimeoutError
    signal.signal(signal.SIGALRM,handler); signal.alarm(TIMEOUT_SEC)
    try:
        r=ollama.Client(host=node).generate(model=model,prompt=prompt,stream=False)
        return r["response"].strip()
    except TimeoutError: return "unknown"
    except ollama.ResponseError as e:
        if "does not support generate" in str(e).lower(): EMBEDMODELS.add(model)
        log_err(f"gen error {node} {model}: {e}"); return "unknown"
    except Exception as e:
        log_err(f"gen exc {node} {model}: {e}"); return "unknown"
    finally: signal.alarm(0)

def build_prompt(field:str,q:str,year:int|None)->str:
    yr=f"The paper was published in {year}. " if year else ""
    return (f"You are being quizzed in {field}. {yr}"
            "You are being quizzed. Only the single word true, false, possibly true, "
            "possibly false, or unknown is valid as your answer. No explanation, no punctuation.\n\n"
            f"Question:\n{q}")

def norm(ans:str)->str:
    a=ans.lower().strip().rstrip(".")
    return ALIAS.get(a,a)

def score(gt:str,pred:str)->float:
    if pred=="unknown": return 0.0
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return 0.01
    return -0.01

def header()->List[str]: return ["model","overall",*FIELD_MAP.values(),"timestamp"]
def done()->set[str]:
    if not CSV_PATH.exists(): return set()
    with CSV_PATH.open(newline="",encoding="utf-8")as f:
        r=csv.reader(f); next(r,None); return {row[0] for row in r}

# ────────── main ──────────
def main():
    completed=done(); hdr=CSV_PATH.exists()
    mongo=MongoClient(MONGO_URI)
    cols={db:mongo[db]["sources"] for db in DATABASES}
    nodes=itertools.cycle(OLLAMA_NODES)

    for model in MODEL_LIST:
        if model in completed: log(f"skip {model}"); continue
        if not pull(model): continue
        log(f"=== MODEL {model} ===")

        field_scores:Dict[str,float]={}
        for db in DATABASES:
            fname=FIELD_MAP[db]; col=cols[db]
            docs=list(col.aggregate([
                {"$match":{"Question":{"$ne":None},"Answer":{"$in":list(VALID_ANSWERS)}}},
                {"$sample":{"size":QUESTIONS_PER_FIELD}}
            ]))
            total=0.0
            for i,d in enumerate(docs,1):
                q,gt=d["Question"],d["Answer"].lower()
                year=d.get("publication_year")
                prompt=build_prompt(fname,q,year); node=next(nodes)
                raw=ask(model,prompt,node); pred_norm=norm(raw)
                sc=score(gt,pred_norm)
                res="✅" if sc>0 else "❌" if sc<0 else "☐"
                log(f"{fname} | {i}/{QUESTIONS_PER_FIELD} | {year or '—'} | {model} | GT:{gt} | LLM:{raw} | {res}")
                total+=sc
            field_scores[fname]=round(total,4)

        overall=round(sum(field_scores.values())/len(field_scores),4)
        row=[model,str(overall),*(str(field_scores.get(f,0.0)) for f in FIELD_MAP.values()),datetime.now().isoformat()]
        mode="a" if hdr else "w"
        with CSV_PATH.open(mode,newline="",encoding="utf-8")as f:
            w=csv.writer(f); 
            if not hdr: w.writerow(header()); hdr=True
            w.writerow(row)

        delete(model); log(f"done {model}")

    mongo.close(); log("🎉 all models done")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: log("bye")

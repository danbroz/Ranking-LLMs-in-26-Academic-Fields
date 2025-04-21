import ollama
from pymongo import MongoClient
import random
import time
import os
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from threading import Thread

# Five Ollama instances per processing node (laptop1 & laptop2)
nodes = [
    'http://laptop1:11434', 'http://laptop1:11435', 'http://laptop1:11436', 'http://laptop1:11437', 'http://laptop1:11438',
    'http://laptop2:11434', 'http://laptop2:11435', 'http://laptop2:11436', 'http://laptop2:11437', 'http://laptop2:11438'
]

MONGO_URI = "mongodb://localhost:27017/"
DATABASES = [f'field_{fid}' for fid in range(11, 37)]

MAX_ATTEMPTS = 3
TASK_TIMEOUT_SECONDS = 30
LOG_TIMEOUT_SECONDS = 60
BATCH_SIZE = 40  # Increased for higher concurrency
WORKERS_PER_NODE = 16  # Matches the available threads (8 cores x 2 threads)

last_log_time = datetime.now()

def log(message):
    global last_log_time
    last_log_time = datetime.now()
    print(f"{last_log_time.strftime('%Y-%m-%d %H:%M:%S')} - {message}", flush=True)

def create_prompt(abstract, error_feedback=None):
    base_prompt = (
        "You are generating questions and answers into a database from research abstracts. "
        "The database will not get updated unless you follow these instructions precisely.\n\n"
        "From the following research abstract, generate exactly one question based specifically on the findings described. "
        "Start the question with the exact words 'Is it true, false, possibly true, or possibly false that'. "
        "Do not include phrases like 'does the study', 'does the abstract', 'based on these findings', "
        "'do the research findings', 'is it possible', or 'is it possibly'. "
        "The question must stand alone without referencing the abstract, study, or researchers. "
        "The answer must explicitly be either true, false, possibly true, or possibly false. "
        "Then directly answer it with one of these four choices only (no additional explanation).\n\n"
    )

    if error_feedback:
        base_prompt += f"Your previous response was incorrect: {error_feedback}\nPlease correct and try again carefully.\n\n"

    base_prompt += f"Abstract:\n{abstract}\n\nOutput format:\nQuestion: <question>\nAnswer: <true|false|possibly true|possibly false>"
    return base_prompt

def generate_question_answer(node, abstract):
    prompt = create_prompt(abstract)
    attempts = 0
    last_response = None

    while attempts < MAX_ATTEMPTS:
        try:
            client = ollama.Client(host=node)
            response = client.generate(model='gemma3:4b', prompt=prompt)['response']
            if "\n" in response and response.startswith("Question:"):
                question_line, answer_line = response.split("\n", 1)
                if answer_line.startswith("Answer:"):
                    question = question_line.replace("Question: ", "").strip()
                    answer = answer_line.replace("Answer: ", "").strip()
                    return question, answer

            error_feedback = f"Invalid format: {response[:100]}"
            prompt = create_prompt(abstract, error_feedback)
            last_response = response
            log(f"⚠️ Formatting issue on {node}: {error_feedback}")

        except Exception as e:
            log(f"⚠️ Node {node} exception: {e}")

        attempts += 1
        time.sleep(0.2)

    raise RuntimeError(f"❌ All attempts failed. Last response: {last_response[:200]}")

def process_document(node, collection, doc, db_name):
    abstract = doc['abstract']
    try:
        question, answer = generate_question_answer(node, abstract)
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"Question": question, "Answer": answer}}
        )
        log(f"✅ Updated {doc['_id']} in {db_name} successfully on {node}.")
    except RuntimeError as exc:
        collection.delete_one({"_id": doc["_id"]})
        log(f"❌ Deleted {doc['_id']} from {db_name} after failures on {node}: {exc}")

def timeout_monitor():
    global last_log_time
    while True:
        if datetime.now() - last_log_time > timedelta(seconds=LOG_TIMEOUT_SECONDS):
            log(f"⚠️ No activity for {LOG_TIMEOUT_SECONDS}s. Restarting...")
            os.execv(sys.executable, ['python'] + sys.argv)
        time.sleep(5)

if __name__ == '__main__':
    monitor_thread = Thread(target=timeout_monitor, daemon=True)
    monitor_thread.start()

    client = MongoClient(MONGO_URI)
    db_collections = {db: client[db]['sources'] for db in DATABASES}
    completed_dbs = set()

    log("🚀 Starting processing on laptop1 & laptop2 with 5 Ollama instances each (laptop0 hosting databases).")

    while len(completed_dbs) < len(DATABASES):
        futures = []
        with ThreadPoolExecutor(max_workers=len(nodes) * WORKERS_PER_NODE) as executor:
            for db_name in DATABASES:
                if db_name in completed_dbs:
                    continue

                collection = db_collections[db_name]
                docs = list(collection.find(
                    {"abstract": {"$ne": None}, "Question": None, "Answer": None}
                ).limit(BATCH_SIZE))

                if not docs:
                    log(f"🏁 Completed all abstracts in {db_name}.")
                    completed_dbs.add(db_name)
                    continue

                for doc in docs:
                    node = random.choice(nodes)
                    futures.append(executor.submit(process_document, node, collection, doc, db_name))

            done, not_done = wait(futures, timeout=TASK_TIMEOUT_SECONDS, return_when=ALL_COMPLETED)

            if not_done:
                log(f"⚠️ {len(not_done)} tasks exceeded {TASK_TIMEOUT_SECONDS}s timeout; restarting script...")
                os.execv(sys.executable, ['python'] + sys.argv)

        time.sleep(0.1)

    client.close()
    log("🎉 Completed processing all databases.")


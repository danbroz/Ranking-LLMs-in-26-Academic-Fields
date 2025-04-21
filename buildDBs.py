import os
import socket
import requests
import pandas as pd
from multiprocessing import Process, Manager, Lock
from concurrent.futures import ThreadPoolExecutor
from smart_open import open as smart_open
from tqdm import tqdm
from queue import Empty
from threading import Thread
from pymongo import MongoClient, ASCENDING
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

MONGO_URI = "mongodb://localhost:27017/"
MAX_THREADS_PER_PROCESS = 2
NODE_LIST = ['laptop0', 'laptop1', 'laptop2']
MASTER_NODE = 'laptop0'

def get_hostname():
    return socket.gethostname()

def is_master_node():
    return get_hostname() == MASTER_NODE

def initialize_databases():
    client = MongoClient(MONGO_URI)
    for fid in range(11, 37):
        db = client[f'field_{fid}']
        collection = db['sources']
        collection.create_index([("openalex_id", ASCENDING)], unique=True)
        collection.create_index([("doi", ASCENDING)])
        logging.info(f"Initialized database and indexes for field_{fid}")
    client.close()

def _recover_abstract(inverted_index):
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    abstract = [''] * (max(max(v) for v in inverted_index.values()) + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            abstract[pos] = word
    return ' '.join(abstract).strip()

def _build_document(row):
    title = (row.get('title') or '').strip()
    abstract = _recover_abstract(row.get('abstract_inverted_index'))
    return title, abstract

def process_row(row, mongo_client):
    idx = row['id']
    title, abstract = _build_document(row)

    if not abstract or row.get('language') != 'en':
        return

    authors = [
        a['author']['display_name'] for a in row.get('authorships', [])
        if 'author' in a and 'display_name' in a['author']
    ] or None
    pub_year = row.get('publication_year')
    doi = row.get('doi')
    pdf_url = (row.get('primary_location') or {}).get('pdf_url')

    document = {
        "openalex_id": idx,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_year": pub_year,
        "doi": doi,
        "pdf_url": pdf_url,
        "Question": None,
        "Answer": None
    }

    fields_inserted = set()

    for topic in row.get('topics', []):
        fid = topic.get('field', {}).get('id')
        if isinstance(fid, str) and fid.startswith('https://openalex.org/fields/'):
            try:
                fid = int(fid.split('/')[-1])
            except:
                continue
        if isinstance(fid, int) and 11 <= fid <= 36 and fid not in fields_inserted:
            db = mongo_client[f'field_{fid}']
            collection = db['sources']
            collection.update_one(
                {"openalex_id": idx},
                {"$setOnInsert": document},
                upsert=True
            )
            fields_inserted.add(fid)

def works_url_routine(i_task, url, counter, lock, progress_queue):
    logging.info(f"Task {i_task} started for URL: {url}")
    mongo_client = MongoClient(MONGO_URI)

    try:
        chunks = pd.read_json(smart_open(url), lines=True, chunksize=8192)
        with ThreadPoolExecutor(max_workers=MAX_THREADS_PER_PROCESS) as executor:
            for chunk in chunks:
                if 'topics' not in chunk.columns:
                    continue
                futures = []
                for _, row in chunk.iterrows():
                    futures.append(executor.submit(process_row, row, mongo_client))
                for future in futures:
                    future.result()
                with lock:
                    counter.value += len(chunk)
                progress_queue.put(len(chunk))
    finally:
        mongo_client.close()

    logging.info(f"Task {i_task} completed")

if __name__ == '__main__':
    os.makedirs('partial_works', exist_ok=True)

    if is_master_node():
        initialize_databases()

    manifest_url = 'https://openalex.s3.amazonaws.com/data/works/manifest'
    manifest = requests.get(manifest_url).json()
    works_entries = manifest.get('entries', [])

    node_idx = NODE_LIST.index(get_hostname())
    urls_for_node = [
        entry['url'] for i, entry in enumerate(works_entries) if i % len(NODE_LIST) == node_idx
    ]

    total_entries = sum(
        entry['meta']['record_count'] for i, entry in enumerate(works_entries)
        if i % len(NODE_LIST) == node_idx
    )

    manager = Manager()
    counter = manager.Value('i', 0)
    lock = Lock()
    progress_queue = manager.Queue()

    progress_bar = tqdm(total=total_entries, unit='it', desc=f'{get_hostname()} Processing')

    def update_progress():
        while True:
            try:
                progress_bar.update(progress_queue.get(timeout=1))
            except Empty:
                if counter.value >= total_entries:
                    break

    progress_thread = Thread(target=update_progress)
    progress_thread.start()

    processes = []
    MAX_PROCESSES = 8
    for i, url in enumerate(urls_for_node):
        proc = Process(target=works_url_routine, args=(i, url, counter, lock, progress_queue))
        proc.start()
        processes.append(proc)
        if len(processes) >= MAX_PROCESSES:
            for p in processes:
                p.join()
            processes.clear()

    for p in processes:
        p.join()

    progress_thread.join()
    progress_bar.close()

    logging.info(f"{get_hostname()} done processing {counter.value} rows.")


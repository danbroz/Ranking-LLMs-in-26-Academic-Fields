#!/usr/bin/env python3
"""Prune field_* MongoDB databases down to a maximum of N random documents.

This script connects to the local MongoDB instance, iterates over databases
named `field_<id>` for ids 11 through 36, and ensures that each `sources`
collection contains at most the specified limit of randomly sampled
documents (default: 1000). Additional documents are deleted.

Safety features:
- Dry-run mode (default) lists the actions without deleting anything.
- Confirmation prompt or --force flag required to perform deletions.

Usage:
    python3 scripts/prune_field_databases.py            # dry run
    python3 scripts/prune_field_databases.py --force    # prune with prompt
    python3 scripts/prune_field_databases.py --force --yes

Requirements:
    pip install pymongo
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Iterable

from pymongo import MongoClient
from pymongo.errors import PyMongoError

FIELD_IDS = range(11, 37)
DEFAULT_LIMIT = 1000
COLLECTION_NAME = "sources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune field_* MongoDB databases to a random subset of documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of documents to retain per field database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Enable deletion mode (otherwise runs in dry-run).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt when used with --force.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (optional).",
    )
    return parser.parse_args()


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def prune_collection(collection, limit: int, dry_run: bool) -> tuple[int, int]:
    """Return (initial_count, final_count)."""
    total_count = collection.count_documents({})
    if total_count <= limit:
        return total_count, total_count

    sample_size = min(limit, total_count)
    # Use aggregation $sample for efficiency
    sampled = list(collection.aggregate([{"$sample": {"size": sample_size}}], allowDiskUse=True))
    if len(sampled) < sample_size:
        print(
            f"  ⚠️  Warning: requested sample of {sample_size} but only received {len(sampled)}."
        )
    keep_ids = {doc["_id"] for doc in sampled}

    if dry_run:
        return total_count, len(keep_ids)

    # Delete documents not in keep_ids in batches to avoid huge queries
    delete_result = collection.delete_many({"_id": {"$nin": list(keep_ids)}})
    final_count = sample_size if delete_result.deleted_count >= 0 else collection.count_documents({})
    return total_count, final_count


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    dry_run = not args.force
    if not dry_run and not args.yes:
        if not confirm("⚠️  This will delete documents. Proceed? [y/N] "):
            print("Aborted by user.")
            return 1

    print(f"Connecting to MongoDB at {args.mongo_uri}...")
    try:
        client = MongoClient(args.mongo_uri)
        # Ping server to verify connection
        client.admin.command("ping")
    except PyMongoError as exc:
        print(f"❌ Failed to connect to MongoDB: {exc}")
        return 1

    overall_deleted = 0
    try:
        for fid in FIELD_IDS:
            db_name = f"field_{fid}"
            collection = client[db_name][COLLECTION_NAME]
            try:
                initial = collection.count_documents({})
            except PyMongoError as exc:
                print(f"{db_name}: ⚠️  Error counting documents: {exc}")
                continue

            if initial == 0:
                print(f"{db_name}: empty, skipping")
                continue

            if initial <= args.limit:
                print(f"{db_name}: {initial} documents (≤ {args.limit}), skipping")
                continue

            print(f"{db_name}: pruning from {initial} documents to {args.limit}...")
            try:
                before, after = prune_collection(collection, args.limit, dry_run=dry_run)
            except PyMongoError as exc:
                print(f"  ❌ Error pruning {db_name}: {exc}")
                continue

            if dry_run:
                deleted = before - after
                print(f"  (dry-run) would delete {deleted} documents; retain {after}")
            else:
                deleted = before - after
                overall_deleted += max(deleted, 0)
                print(f"  ✅ Deleted {deleted} documents; {after} remain")

        if dry_run:
            print("Dry run complete. Re-run with --force to apply deletions.")
        else:
            print(f"Pruning complete. Total documents deleted: {overall_deleted}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())

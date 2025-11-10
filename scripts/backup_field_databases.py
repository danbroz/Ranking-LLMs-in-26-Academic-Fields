#!/usr/bin/env python3
"""Create backup copies of field_* MongoDB databases.

For every database named ``field_<id>`` (where id is between 11 and 36
inclusive) this script copies all collections and their indexes into a new
database named ``field_<id>_backup``. Existing backup databases can be
optionally dropped before copying to guarantee a clean snapshot.

Usage examples::

    # Dry run — show what would happen without copying anything
    python3 scripts/backup_field_databases.py

    # Perform the backup (asks for confirmation)
    python3 scripts/backup_field_databases.py --force

    # Perform the backup without prompting
    python3 scripts/backup_field_databases.py --force --yes

Requirements::

    pip install pymongo
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from pymongo import MongoClient
from pymongo.errors import PyMongoError

FIELD_IDS = range(11, 37)
COLLECTION_NAME = "sources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create backup copies of field_* MongoDB databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute copy (otherwise dry-run).",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop backup collections before copying (recommended for fresh backups).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt when used with --force.",
    )
    return parser.parse_args()


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def copy_indexes(source_collection, target_collection, dry_run: bool) -> None:
    try:
        index_info = source_collection.index_information()
    except PyMongoError as exc:
        print(f"    ⚠️  Could not fetch indexes: {exc}")
        return

    for name, spec in index_info.items():
        if name == "_id_":
            continue  # default index
        keys = spec["key"]
        kwargs = {
            k: v for k, v in spec.items() if k not in {"key", "v"}
        }
        if dry_run:
            print(f"    (dry-run) would create index {name} on {keys}")
            continue
        try:
            target_collection.create_index(keys, name=name, **kwargs)
            print(f"    ✅ Created index {name}")
        except PyMongoError as exc:
            print(f"    ⚠️  Failed to create index {name}: {exc}")


def copy_collection(db_name: str, coll_name: str, client: MongoClient, *, dry_run: bool, drop_existing: bool) -> None:
    source = client[db_name][coll_name]
    target_db_name = f"{db_name}_backup"
    target = client[target_db_name][coll_name]

    try:
        count = source.count_documents({})
    except PyMongoError as exc:
        print(f"{db_name}.{coll_name}: ⚠️  Could not count documents ({exc}), skipping")
        return

    if count == 0:
        print(f"{db_name}.{coll_name}: empty, skipping")
        return

    print(f"{db_name}.{coll_name}: {count} documents → {target_db_name}.{coll_name}")

    if drop_existing and not dry_run:
        try:
            target.drop()
            print("    Existing backup collection dropped")
        except PyMongoError as exc:
            print(f"    ⚠️  Could not drop backup collection: {exc}")

    if dry_run:
        print("    (dry-run) would copy documents via $merge")
    else:
        try:
            source.aggregate(
                [
                    {
                        "$merge": {
                            "into": {"db": target_db_name, "coll": coll_name},
                            "whenMatched": "replace",
                            "whenNotMatched": "insert",
                        }
                    }
                ],
                allowDiskUse=True,
            )
            print("    ✅ Documents copied")
        except PyMongoError as exc:
            print(f"    ❌ Failed to copy documents: {exc}")
            return

    copy_indexes(source, target, dry_run=dry_run)


def main() -> int:
    args = parse_args()
    dry_run = not args.force

    if not dry_run and not args.yes:
        if not confirm("⚠️  This will create/overwrite backup databases. Proceed? [y/N] "):
            print("Aborted by user.")
            return 1

    print(f"Connecting to MongoDB at {args.mongo_uri}...")
    try:
        client = MongoClient(args.mongo_uri)
        client.admin.command("ping")
    except PyMongoError as exc:
        print(f"❌ Failed to connect to MongoDB: {exc}")
        return 1

    try:
        for fid in FIELD_IDS:
            db_name = f"field_{fid}"
            try:
                collections = client[db_name].list_collection_names()
            except PyMongoError as exc:
                print(f"{db_name}: ⚠️  Could not list collections ({exc}), skipping")
                continue

            if not collections:
                print(f"{db_name}: no collections found, skipping")
                continue

            print(f"Processing {db_name} → {db_name}_backup")
            for coll in collections:
                copy_collection(
                    db_name,
                    coll,
                    client,
                    dry_run=dry_run,
                    drop_existing=args.drop_existing,
                )

        if dry_run:
            print("Dry run complete. Re-run with --force to create backups.")
        else:
            print("Backup complete.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())

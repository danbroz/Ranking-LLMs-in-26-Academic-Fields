#!/usr/bin/env python3
"""Restore field_* MongoDB databases from backup copies.

For every backup database named ``field_<id>_backup`` (where id is between 11 and 36
inclusive) this script copies all collections and their indexes back into the
original database named ``field_<id>``. Existing databases can be optionally
dropped before restoring to guarantee a clean restore.

Usage examples::

    # Dry run — show what would happen without copying anything
    python3 scripts/restore_field_databases.py

    # Perform the restore (asks for confirmation)
    python3 scripts/restore_field_databases.py --force

    # Perform the restore without prompting
    python3 scripts/restore_field_databases.py --force --yes

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
        description="Restore field_* MongoDB databases from backup copies.",
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
        help="Execute restore (otherwise dry-run).",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing collections before restoring (recommended for clean restore).",
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
    """Copy indexes from source to target collection."""
    if dry_run:
        indexes = list(source_collection.list_indexes())
        if indexes:
            print(f"    (dry-run) would copy {len(indexes)} index(es)")
        return

    try:
        indexes = list(source_collection.list_indexes())
        for idx in indexes:
            name = idx.get("name", "")
            if name == "_id_":
                continue
            try:
                keys = idx.get("key", {})
                target_collection.create_index(
                    list(keys.items()),
                    name=name,
                    background=True,
                )
            except PyMongoError as exc:
                print(f"    ⚠️  Failed to create index {name}: {exc}")
    except PyMongoError as exc:
        print(f"    ⚠️  Failed to list indexes: {exc}")


def restore_collection(db_name: str, coll_name: str, client: MongoClient, *, dry_run: bool, drop_existing: bool) -> None:
    source_db_name = f"{db_name}_backup"
    source = client[source_db_name][coll_name]
    target = client[db_name][coll_name]

    try:
        count = source.count_documents({})
    except PyMongoError as exc:
        print(f"{source_db_name}.{coll_name}: ⚠️  Could not count documents ({exc}), skipping")
        return

    if count == 0:
        print(f"{source_db_name}.{coll_name}: empty, skipping")
        return

    print(f"{source_db_name}.{coll_name}: {count} documents → {db_name}.{coll_name}")

    if drop_existing and not dry_run:
        try:
            target.drop()
            print("    Existing collection dropped")
        except PyMongoError as exc:
            print(f"    ⚠️  Could not drop collection: {exc}")

    if dry_run:
        print("    (dry-run) would copy documents via $merge")
    else:
        try:
            source.aggregate(
                [
                    {
                        "$merge": {
                            "into": {"db": db_name, "coll": coll_name},
                            "whenMatched": "replace",
                            "whenNotMatched": "insert",
                        }
                    }
                ],
                allowDiskUse=True,
            )
            print("    ✅ Documents restored")
        except PyMongoError as exc:
            print(f"    ❌ Failed to restore documents: {exc}")

    copy_indexes(source, target, dry_run=dry_run)


def main() -> int:
    args = parse_args()
    dry_run = not args.force

    if not dry_run and not args.yes:
        if not confirm("⚠️  This will restore/overwrite databases from backups. Proceed? [y/N] "):
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
            backup_db_name = f"{db_name}_backup"
            try:
                collections = client[backup_db_name].list_collection_names()
            except PyMongoError as exc:
                print(f"{backup_db_name}: ⚠️  Could not list collections ({exc}), skipping")
                continue

            if not collections:
                print(f"{backup_db_name}: no collections found, skipping")
                continue

            print(f"Processing {backup_db_name} → {db_name}")
            for coll in collections:
                restore_collection(
                    db_name,
                    coll,
                    client,
                    dry_run=dry_run,
                    drop_existing=args.drop_existing,
                )

        if dry_run:
            print("Dry run complete. Re-run with --force to restore databases.")
        else:
            print("Restore complete.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())






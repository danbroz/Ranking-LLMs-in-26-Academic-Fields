#!/usr/bin/env python3
"""Count documents in each field_* MongoDB database.

This script connects to MongoDB and counts the number of documents in the
"sources" collection for each field database (field_11 through field_36).

Usage examples::

    # Count documents using default MongoDB URI
    python3 scripts/count_field_documents.py

    # Count documents using custom MongoDB URI
    python3 scripts/count_field_documents.py --mongo-uri mongodb://localhost:27017/

Requirements::

    pip install pymongo
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

from pymongo import MongoClient
from pymongo.errors import PyMongoError

FIELD_IDS = range(11, 37)
COLLECTION_NAME = "sources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count documents in each field_* MongoDB database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI",
    )
    return parser.parse_args()


def count_documents(client: MongoClient, db_name: str, collection_name: str) -> int:
    """Count documents in a collection, returning 0 on error."""
    try:
        collection = client[db_name][collection_name]
        return collection.count_documents({})
    except PyMongoError as exc:
        print(f"    ⚠️  Error counting documents in {db_name}.{collection_name}: {exc}", file=sys.stderr)
        return 0


def main() -> None:
    args = parse_args()

    try:
        client = MongoClient(args.mongo_uri)
        # Test connection
        client.admin.command("ping")
    except PyMongoError as exc:
        print(f"❌ Failed to connect to MongoDB at {args.mongo_uri}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"📊 Counting documents in field databases (using {args.mongo_uri})...")
    print()

    counts: Dict[str, int] = {}
    total = 0

    for field_id in FIELD_IDS:
        db_name = f"field_{field_id}"
        count = count_documents(client, db_name, COLLECTION_NAME)
        counts[db_name] = count
        total += count
        print(f"  {db_name:15} {count:>10,} documents")

    print()
    print(f"  {'Total':15} {total:>10,} documents")
    print()

    # Summary statistics
    if counts:
        non_zero = [c for c in counts.values() if c > 0]
        if non_zero:
            print(f"  Databases with documents: {len(non_zero)}/{len(counts)}")
            print(f"  Average per database: {sum(non_zero) / len(non_zero):,.0f}")
            print(f"  Min: {min(non_zero):,}, Max: {max(non_zero):,}")

    client.close()


if __name__ == "__main__":
    main()






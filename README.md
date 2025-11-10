# Ranking-LLMs-in-26-Academic-Fields

## Prune field_* databases

Use `scripts/prune_field_databases.py` to cap each `field_*` MongoDB database at 1,000 random documents in the `sources` collection.

```bash
# Dry run (default): shows what would be deleted
python3 scripts/prune_field_databases.py

# Perform deletions (prompts for confirmation)
python3 scripts/prune_field_databases.py --force

# Perform deletions without prompt
python3 scripts/prune_field_databases.py --force --yes
```

Flags:
- `--limit N` sets a different cap per database.
- `--mongo-uri URI` targets a non-default MongoDB instance.
- `--seed VALUE` enables reproducible sampling.
- Run without `--force` to preview deletions safely.

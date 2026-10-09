"""Count DECIMER cache validity."""
import json
from pathlib import Path

cache = json.loads(Path("data/processed/decimer_cache.json").read_text(encoding="utf-8"))
valid = sum(1 for v in cache.values() if v.get("raw"))
empty = sum(1 for v in cache.values() if not v.get("raw"))
print(f"total: {len(cache)}  valid: {valid}  empty: {empty}")

# Show empty ones
empties = [k for k, v in cache.items() if not v.get("raw")]
print(f"empty keys: {empties}")
print()
print("First 3 results (cmpd_id | raw[:60] | canon[:60]):")
for k, v in list(cache.items())[:3]:
    print(f"  {k}: raw={v['raw'][:60]!r}  canon={v['canon'][:60]!r}")

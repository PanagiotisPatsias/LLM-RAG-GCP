# evaluation/quality_gate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Fail CI if evaluation quality is below threshold.")
    p.add_argument("--results", required=True, help="Path to results.jsonl")
    p.add_argument("--min-overall", type=float, default=0.80)
    args = p.parse_args()

    path = Path(args.results)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No results found.")

    mean_overall = sum(r["scores"]["overall"] for r in rows) / len(rows)
    print(f"Mean overall: {mean_overall:.3f} (min required: {args.min_overall:.3f})")

    if mean_overall < args.min_overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

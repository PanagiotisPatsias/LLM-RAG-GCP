# evaluation/refusal_gate.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REFUSAL_PATTERNS = [
    r"the provided context does not contain enough information to answer this question",
    r"not enough information",
    r"insufficient information",
    r"insufficient context",
    r"does not contain enough information",
    r"cannot determine from the provided context",
    r"cannot answer from the provided context",
]

def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)

def main():
    p = argparse.ArgumentParser(description="Fail if refusal rate is below threshold.")
    p.add_argument("--results", required=True, help="Path to results.jsonl")
    p.add_argument("--min-refusal-rate", type=float, default=0.90)
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.results).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No results found.")

    total = len(rows)
    refused = sum(1 for r in rows if is_refusal(r.get("answer", "")))
    rate = refused / total

    print(f"Total: {total}")
    print(f"Refusals: {refused}")
    print(f"Refusal rate: {rate:.3f} (min required: {args.min_refusal_rate:.3f})")

    if rate < args.min_refusal_rate:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

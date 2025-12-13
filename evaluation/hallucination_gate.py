# evaluation/hallucination_gate.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REFUSAL_PATTERNS = [
    r"not enough information",
    r"insufficient information",
    r"insufficient context",
    r"does not contain enough information",
    r"cannot determine from the provided context",
    r"cannot answer from the provided context",
]

# Matches citations like [1], [2], [12]
CITATION_RE = re.compile(r"\[\d+\]")


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def has_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text or ""))


def main():
    p = argparse.ArgumentParser(description="Fail if hallucination rate is above threshold.")
    p.add_argument("--results", required=True, help="Path to results.jsonl")
    p.add_argument("--max-hallucination-rate", type=float, default=0.10)
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("No results found.")

    total = len(rows)

    good = 0
    refused = 0
    cited = 0

    for r in rows:
        ans = (r.get("answer") or "").strip()
        if is_refusal(ans):
            refused += 1
            good += 1
        elif has_citation(ans):
            cited += 1
            good += 1

    hallucinated = total - good
    rate = hallucinated / total

    print(f"Total: {total}")
    print(f"Refusals: {refused}")
    print(f"Cited non-refusals: {cited}")
    print(f"Hallucinations (no refusal + no citations): {hallucinated}")
    print(f"Hallucination rate: {rate:.3f} (max allowed: {args.max_hallucination_rate:.3f})")

    if rate > args.max_hallucination_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

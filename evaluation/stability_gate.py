# evaluation/stability_gate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Fail if judge reliability is unstable.")
    p.add_argument("--summary", required=True, help="Path to reliability_summary.json")
    p.add_argument("--max-std", type=float, default=0.10)
    args = p.parse_args()

    d = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    overall_std = float(d.get("overall_std", 0.0))
    print(f"Overall std: {overall_std:.3f} (max allowed: {args.max_std:.3f})")

    if overall_std > args.max_std:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

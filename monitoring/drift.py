from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Baseline:
    question_len_mean: float
    mean_distance_mean: float


def load_baseline(path: str = "monitoring/baseline.json") -> Optional[Baseline]:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return Baseline(
        question_len_mean=float(data["question_len_mean"]),
        mean_distance_mean=float(data["mean_distance_mean"]),
    )


def compute_drift_score(current: Baseline, baseline: Baseline) -> dict:
    """
    Very simple drift score: absolute deltas.
    You can make it fancier later (z-scores, PSI, KL, etc.).
    """
    return {
        "delta_question_len_mean": abs(current.question_len_mean - baseline.question_len_mean),
        "delta_mean_distance_mean": abs(current.mean_distance_mean - baseline.mean_distance_mean),
    }

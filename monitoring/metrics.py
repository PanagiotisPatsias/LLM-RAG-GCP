from __future__ import annotations
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def _now_iso() -> str:
    # ISO-like without needing datetime imports; good enough for logs
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


@dataclass
class InferenceMetric:
    timestamp: str
    request_id: str

    # privacy-preserving fields
    question_len: int
    question_sha1: str

    top_k: int
    num_chunks: int
    mean_distance: Optional[float]
    min_distance: Optional[float]
    max_distance: Optional[float]

    cited: bool
    refusal: bool

    latency_ms: int

    # optional extra metadata
    source: str = "rag"            # e.g. "rag", "agent", "api"
    model: str = ""
    collection: str = ""
    extra: Dict[str, Any] = None


class MetricsLogger:
    """
    Logs JSON lines to stdout (Cloud Run logs) and optionally to a file.
    """
    def __init__(self) -> None:
        self.sink = os.getenv("METRICS_SINK", "stdout").lower()  # stdout | file | both
        self.filepath = os.getenv("METRICS_FILE", "metrics.jsonl")
        self.enabled = os.getenv("METRICS_ENABLED", "1") == "1"

    def log(self, metric: InferenceMetric) -> None:
        if not self.enabled:
            return

        payload = asdict(metric)
        if payload.get("extra") is None:
            payload["extra"] = {}

        line = json.dumps(payload, ensure_ascii=False)

        if self.sink in ("stdout", "both"):
            print(line, file=sys.stdout, flush=True)

        if self.sink in ("file", "both"):
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def make_metric(
    *,
    request_id: str,
    question: str,
    top_k: int,
    distances: list[float],
    cited: bool,
    refusal: bool,
    latency_ms: int,
    source: str = "rag",
    model: str = "",
    collection: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> InferenceMetric:
    d_sorted = sorted(distances) if distances else []
    mean_d = (sum(d_sorted) / len(d_sorted)) if d_sorted else None

    return InferenceMetric(
        timestamp=_now_iso(),
        request_id=request_id,
        question_len=len(question or ""),
        question_sha1=_sha1(question or ""),
        top_k=top_k,
        num_chunks=len(d_sorted),
        mean_distance=mean_d,
        min_distance=d_sorted[0] if d_sorted else None,
        max_distance=d_sorted[-1] if d_sorted else None,
        cited=cited,
        refusal=refusal,
        latency_ms=int(latency_ms),
        source=source,
        model=model,
        collection=collection,
        extra=extra or {},
    )

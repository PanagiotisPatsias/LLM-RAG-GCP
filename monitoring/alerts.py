from __future__ import annotations

def check_alerts(*, refusal_rate: float, mean_distance: float | None) -> list[str]:
    alerts: list[str] = []

    # Rule 1: too many refusals (could indicate bad retrieval / wrong domain / broken index)
    if refusal_rate > 0.30:
        alerts.append(f"High refusal rate: {refusal_rate:.2f} (>0.30)")

    # Rule 2: retrieval seems "far" (quality proxy) 
    if mean_distance is not None and mean_distance > 0.55:
        alerts.append(f"High mean retrieval distance: {mean_distance:.3f} (>0.55)")

    return alerts

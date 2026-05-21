from __future__ import annotations

import numpy as np


def dominance(target_score: float, competitor_scores: np.ndarray) -> float:
    """Return target score minus the strongest non-target score."""
    if competitor_scores.size == 0:
        raise ValueError("competitor_scores must not be empty")
    return float(target_score - np.max(competitor_scores))


def recovered(target_score: float, competitor_scores: np.ndarray) -> int:
    """Return 1 if the target strictly beats all non-target candidates."""
    return int(dominance(target_score, competitor_scores) > 0.0)


def recovery_rate(recovered_values: np.ndarray) -> float:
    """Return empirical mean recovery rate."""
    if recovered_values.size == 0:
        raise ValueError("recovered_values must not be empty")
    return float(np.mean(recovered_values))

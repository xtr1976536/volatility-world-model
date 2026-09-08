"""Common direct multi-horizon baselines."""
from __future__ import annotations

import numpy as np

from .data import direct_forecast


def predict_baselines(rv: np.ndarray, iv: np.ndarray, train_end: int, origin: int, horizon: int) -> dict[str, np.ndarray]:
    """Return daily paths for all baselines at one forecast origin."""
    last = np.maximum(rv[origin], 1e-8)
    return {
        "Naive": np.repeat(last[None, :], horizon, axis=0),
        "HAR": np.maximum(direct_forecast(rv, iv, train_end, origin, horizon, False, False), 1e-8),
        "HAR-IV": np.maximum(direct_forecast(rv, iv, train_end, origin, horizon, True, False), 1e-8),
        "GHAR": np.maximum(direct_forecast(rv, iv, train_end, origin, horizon, False, True), 1e-8),
        "GHAR-IV": np.maximum(direct_forecast(rv, iv, train_end, origin, horizon, True, True), 1e-8),
    }

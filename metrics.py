"""Point, probabilistic, path, and cross-sectional evaluation metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def qlike(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.maximum(np.asarray(p, float), 1e-8)
    ratio = np.maximum(y, 1e-8) / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> float:
    """Energy-form CRPS in O(S log S) per observation."""
    x = np.asarray(samples, float)
    y = np.asarray(y, float)
    if x.shape[0] < 2:
        return float(np.mean(np.abs(x - y)))
    term1 = np.mean(np.abs(x - y), axis=0)
    ordered = np.sort(x, axis=0)
    n = ordered.shape[0]
    weights = (2.0 * np.arange(1, n + 1) - n - 1.0).reshape((n,) + (1,) * (ordered.ndim - 1))
    term2 = np.sum(weights * ordered, axis=0) / (n * n)
    return float(np.mean(term1 - term2))


def quantile(samples: np.ndarray, level: float) -> np.ndarray:
    return np.quantile(samples, level, axis=0)


def probabilistic_metrics(samples: np.ndarray, target: np.ndarray) -> dict:
    mean = np.mean(samples, axis=0)
    q05, q25, q50, q75, q95 = [quantile(samples, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
    y = np.asarray(target, float)
    inside90 = (y >= q05) & (y <= q95)
    inside50 = (y >= q25) & (y <= q75)
    width90 = q95 - q05
    winkler = width90 + 20.0 * np.where(y < q05, q05 - y, 0.0) + 20.0 * np.where(y > q95, y - q95, 0.0)
    # Rank PIT is appropriate for an ensemble and avoids assuming Gaussianity.
    pit = (np.sum(samples < y[None, ...], axis=0) + 0.5 * np.sum(samples == y[None, ...], axis=0)) / samples.shape[0]
    return {
        "MSE": float(np.mean((mean - y) ** 2)),
        "QLIKE": qlike(y, mean),
        "MAE": float(np.mean(np.abs(mean - y))),
        "CRPS": crps_ensemble(samples, y),
        "coverage90": float(np.mean(inside90)),
        "coverage50": float(np.mean(inside50)),
        "width90": float(np.mean(width90)),
        "width50": float(np.mean(q75 - q25)),
        "winkler90": float(np.mean(winkler)),
        "pit_mean": float(np.mean(pit)),
        "pit_var": float(np.var(pit)),
        "q05": q05,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q95": q95,
        "mean": mean,
        "pit": pit,
    }


def path_metrics(mean_paths: np.ndarray, sampled_paths: np.ndarray, targets: np.ndarray) -> dict:
    """Metrics for [origin,horizon,asset] arrays and [origin,sample,horizon,asset]."""
    mean_paths = np.asarray(mean_paths, float)
    targets = np.asarray(targets, float)
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.reshape(-1), b.reshape(-1)
        if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    pred_delta, true_delta = np.diff(mean_paths, axis=1), np.diff(targets, axis=1)
    result = {
        "mean_increment_corr": corr(pred_delta, true_delta),
        "pred_increment_sd": float(np.std(pred_delta)),
        "true_increment_sd": float(np.std(true_delta)),
        "increment_sd_ratio": float(np.std(pred_delta) / max(np.std(true_delta), 1e-12)),
        "level_bias": float(np.mean(mean_paths - targets)),
    }
    if sampled_paths is not None and sampled_paths.size:
        sample_delta = np.diff(sampled_paths, axis=2)
        result["sample_increment_sd"] = float(np.std(sample_delta))
        result["sample_increment_corr_to_truth"] = corr(sample_delta.mean(axis=1), true_delta)
        result["sample_path_increment_lag_corr"] = corr(sample_delta[:, :, :-1], sample_delta[:, :, 1:]) if sample_delta.shape[2] > 1 else float("nan")
    return result


def cross_sectional_corr_error(paths: np.ndarray, targets: np.ndarray) -> float:
    errors = []
    for pred, true in zip(np.asarray(paths), np.asarray(targets)):
        if pred.shape[0] < 2:
            continue
        if np.any(np.std(pred, axis=0) < 1e-12) or np.any(np.std(true, axis=0) < 1e-12):
            continue
        pc = np.corrcoef(pred.T)
        tc = np.corrcoef(true.T)
        if np.isfinite(pc).all() and np.isfinite(tc).all():
            errors.append(np.mean((pc - tc) ** 2) ** 0.5)
    return float(np.mean(errors)) if errors else float("nan")


def common_extreme_event_score(samples: np.ndarray, targets: np.ndarray, quantile_level: float = 0.9) -> dict:
    """Compare market-wide high-volatility events under the sampled joint law.

    Each date-horizon pair is an event when the cross-sectional mean exceeds
    its in-sample empirical quantile.  The score is deliberately joint: it
    cannot be improved merely by widening independent marginal intervals.
    """
    draws = np.asarray(samples, float)
    truth = np.asarray(targets, float)
    draw_market = draws.mean(axis=-1)
    truth_market = truth.mean(axis=-1)
    threshold = float(np.quantile(truth_market, quantile_level))
    realized = truth_market >= threshold
    predicted_probability = (draw_market >= threshold).mean(axis=1)
    brier = float(np.mean((predicted_probability - realized) ** 2))
    return {"common_extreme_rate": float(realized.mean()),
            "common_extreme_probability": float(predicted_probability.mean()),
            "common_extreme_brier": brier}


def summarize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in frame.groupby(["model", "horizon"], sort=True):
        y, p = group["target"].to_numpy(float), group["mean"].to_numpy(float)
        rows.append({"model": model, "horizon": int(horizon), "MSE": float(np.mean((y - p) ** 2)),
                     "QLIKE": qlike(y, p), "MAE": float(np.mean(np.abs(y - p))), "N": len(group)})
    return pd.DataFrame(rows)


def dm_comparisons(frame: pd.DataFrame, reference: str = "HAR-IV") -> pd.DataFrame:
    rows = []
    for horizon in sorted(frame.horizon.unique()):
        sub = frame[frame.horizon == horizon]
        ref = sub[sub.model == reference].set_index(["origin_index", "ticker"]).sort_index()
        for model in sorted(set(sub.model) - {reference}):
            other = sub[sub.model == model].set_index(["origin_index", "ticker"]).sort_index()
            common = ref.index.intersection(other.index)
            if len(common) < 3:
                continue
            y = ref.loc[common, "target"].to_numpy(float)
            p_ref = np.maximum(ref.loc[common, "mean"].to_numpy(float), 1e-8)
            p_other = np.maximum(other.loc[common, "mean"].to_numpy(float), 1e-8)
            d = (y / p_ref - np.log(y / p_ref) - 1.0) - (y / p_other - np.log(y / p_other) - 1.0)
            by_date = pd.Series(d, index=[idx[0] for idx in common]).groupby(level=0).mean().to_numpy()
            sd = np.std(by_date, ddof=1)
            stat = float(np.mean(by_date) / (sd / math.sqrt(len(by_date)))) if sd > 0 else float("nan")
            rows.append({"horizon": int(horizon), "model": model, "reference": reference,
                         "loss": "QLIKE", "mean_loss_difference_ref_minus_model": float(np.mean(by_date)),
                         "DM_t_approx": stat, "dates": int(len(by_date))})
    return pd.DataFrame(rows)

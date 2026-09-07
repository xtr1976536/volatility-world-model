"""Causal data loading, normalization, anchors, and forecast windows.

The module is deliberately independent of the model.  Every model and
baseline consumes the same aligned panel and the same point-in-time windows.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


EPS = 1e-8
FEATURE_NAMES = ("log_rv", "log_iv", "return")


@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    rv: np.ndarray
    iv: np.ndarray
    returns: np.ndarray

    @property
    def n_dates(self) -> int:
        return int(self.rv.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.rv.shape[1])


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, panel: Panel) -> np.ndarray:
        x = np.stack(
            [np.log(np.maximum(panel.rv, EPS)),
             np.log(np.maximum(panel.iv, EPS)),
             np.clip(panel.returns, -0.25, 0.25)], axis=-1
        )
        return (x - self.mean) / self.scale

    def transform_log_rv(self, rv: np.ndarray) -> np.ndarray:
        return (np.log(np.maximum(rv, EPS)) - self.mean[..., 0]) / self.scale[..., 0]

    def inverse_log_rv(self, value: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(value) * self.scale[..., 0] + self.mean[..., 0])

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class Block:
    block_id: int
    start: int
    end: int


class AnchorForecaster:
    """Minimal interface for a causal multi-horizon level forecaster."""

    name = "abstract"

    def fit(self, panel: Panel, train_end: int, horizon: int) -> "AnchorForecaster":
        raise NotImplementedError

    def predict(self, panel: Panel, origin: int) -> np.ndarray:
        raise NotImplementedError


class HarIvAnchor(AnchorForecaster):
    """Default causal anchor used by the formal experiment."""

    name = "har_iv"

    def __init__(self) -> None:
        self.betas: list[np.ndarray | None] = []
        self.horizon = 0

    def fit(self, panel: Panel, train_end: int, horizon: int) -> "HarIvAnchor":
        self.horizon = int(horizon)
        self.betas = []
        for h in range(1, self.horizon + 1):
            if train_end - 22 - h < 30:
                self.betas.append(None)
            else:
                self.betas.append(_fit_point(panel.rv, panel.iv, train_end, h, True, False))
        return self

    def predict(self, panel: Panel, origin: int) -> np.ndarray:
        if not self.betas:
            raise RuntimeError("anchor must be fitted before prediction")
        out = []
        for beta in self.betas:
            if beta is None or origin < 22:
                value = np.maximum(panel.rv[origin - 1], EPS)
            else:
                x = np.column_stack([np.ones(panel.n_assets), _har_design(panel.rv, panel.iv, origin, True, False)])
                value = np.maximum(np.sum(x * beta.T, axis=1), EPS)
            out.append(value)
        return np.stack(out, axis=0)


def make_anchor(anchor_type: str = "har_iv") -> AnchorForecaster:
    if anchor_type == "har_iv":
        return HarIvAnchor()
    raise ValueError(f"unknown anchor_type: {anchor_type}; available: har_iv")


def _read_csv(raw: bytes, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), parse_dates=["Date"]).set_index("Date").sort_index()


def _read_panel_files(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = Path(source)
    names = {
        "rv": ("merged_rv_data_filled.csv",),
        "iv": ("merged_iv_data_filled.csv",),
        "returns": ("dow30_daily_returns_2021_2026.csv", "daily_returns.csv"),
    }
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            files = archive.namelist()
            selected = {}
            for key, candidates in names.items():
                path = next((n for n in files if any(n.endswith(c) for c in candidates)), None)
                if path is None:
                    raise FileNotFoundError(f"{source} does not contain {candidates}")
                selected[key] = _read_csv(archive.read(path), path)
    elif source.is_dir():
        selected = {}
        for key, candidates in names.items():
            path = next((p for candidate in candidates for p in source.rglob(candidate)), None)
            if path is None:
                raise FileNotFoundError(f"{source} does not contain {candidates}")
            selected[key] = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    else:
        raise FileNotFoundError(f"data source does not exist: {source}")
    return selected["rv"], selected["iv"], selected["returns"]


def load_panel(source: Path, first_test: int, min_coverage: float = 0.98) -> tuple[Panel, dict]:
    """Load the common panel and freeze the asset universe before first_test.

    Existing ``*_filled.csv`` files are treated as upstream inputs.  Runtime
    filling is forward-only and is recorded separately in the audit metadata.
    No backward fill is permitted.
    """
    rv_df, iv_df, ret_df = _read_panel_files(Path(source))
    for label, frame in (("RV", rv_df), ("IV", iv_df), ("returns", ret_df)):
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{label} dates must be unique and increasing")
    dates = rv_df.index.intersection(iv_df.index).intersection(ret_df.index).sort_values()
    columns = sorted(set(rv_df.columns) & set(iv_df.columns) & set(ret_df.columns))
    if len(dates) == 0 or len(columns) == 0:
        raise ValueError("no common dates or assets")
    rv_df, iv_df, ret_df = (frame.loc[dates, columns].astype(float) for frame in (rv_df, iv_df, ret_df))
    # Forward-only runtime fill.  Leading gaps remain missing and are excluded.
    before = {name: int(frame.isna().sum().sum()) for name, frame in (("RV", rv_df), ("IV", iv_df), ("returns", ret_df))}
    rv_df, iv_df, ret_df = rv_df.ffill(), iv_df.ffill(), ret_df.ffill()
    coverage_end = min(first_test, len(dates))
    valid = []
    for ticker in columns:
        rv_ok = np.isfinite(rv_df[ticker].to_numpy()) & (rv_df[ticker].to_numpy() > 0)
        iv_ok = np.isfinite(iv_df[ticker].to_numpy()) & (iv_df[ticker].to_numpy() > 0)
        ret_ok = np.isfinite(ret_df[ticker].to_numpy())
        coverage = float((rv_ok[:coverage_end] & iv_ok[:coverage_end] & ret_ok[:coverage_end]).mean())
        if coverage >= min_coverage and (rv_ok & iv_ok & ret_ok).all():
            valid.append(ticker)
    if not valid:
        raise ValueError("no asset meets the pre-test coverage requirement")
    rv = rv_df.loc[:, valid].to_numpy(float)
    iv = iv_df.loc[:, valid].to_numpy(float)
    returns = ret_df.loc[:, valid].to_numpy(float)
    if not (np.isfinite(rv).all() and np.isfinite(iv).all() and np.isfinite(returns).all()):
        raise ValueError("panel contains non-finite values after forward-only fill")
    if not ((rv > 0).all() and (iv > 0).all()):
        raise ValueError("RV and IV must be strictly positive")
    panel = Panel(pd.DatetimeIndex(dates), tuple(valid), rv, iv, returns)
    audit = {
        "source": str(source),
        "common_dates": panel.n_dates,
        "date_start": str(panel.dates[0].date()),
        "date_end": str(panel.dates[-1].date()),
        "asset_count": panel.n_assets,
        "tickers": list(panel.tickers),
        "pretest_first_test": int(first_test),
        "minimum_pretest_coverage": float(min_coverage),
        "upstream_missing_counts_before_runtime_fill": before,
        "runtime_fill": "forward_only",
        "upstream_filled_file_causal_audit": "not_performed; source files are treated as upstream inputs",
        "dropped_assets": sorted(set(columns) - set(valid)),
    }
    return panel, audit


def fit_standardizer(panel: Panel, train_end: int) -> Standardizer:
    if not 1 <= train_end <= panel.n_dates:
        raise ValueError("train_end must be inside the panel")
    raw = np.stack(
        [np.log(np.maximum(panel.rv[:train_end], EPS)),
         np.log(np.maximum(panel.iv[:train_end], EPS)),
         np.clip(panel.returns[:train_end], -0.25, 0.25)], axis=-1
    )
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return Standardizer(mean, scale)


def make_blocks(first_test: int, last_test: int, block_size: int = 22) -> list[Block]:
    if first_test < 1 or last_test < first_test or block_size < 1:
        raise ValueError("invalid block protocol")
    blocks = []
    start = first_test
    while start <= last_test:
        end = min(start + block_size, last_test + 1)
        blocks.append(Block(len(blocks), start, end))
        start = end
    return blocks


def _har_design(rv: np.ndarray, iv: np.ndarray, origin: int, use_iv: bool, cross_section: bool) -> np.ndarray:
    if origin < 22:
        raise ValueError("HAR design needs 22 historical observations")
    x = [rv[origin - 1], rv[origin - 5:origin].mean(axis=0), rv[origin - 22:origin - 5].mean(axis=0)]
    if use_iv:
        x.append(iv[origin - 1])
    if cross_section:
        market = rv[origin - 1]
        x.extend([np.full(rv.shape[1], market.mean()), np.full(rv.shape[1], market.std())])
    return np.stack(x, axis=1)


def _fit_direct(rv: np.ndarray, iv: np.ndarray, train_end: int, horizon: int, use_iv: bool, cross_section: bool) -> np.ndarray:
    origins = np.arange(22, train_end - horizon + 1, dtype=int)
    if len(origins) < 30:
        raise ValueError("too few training origins for HAR regression")
    design = np.stack([np.column_stack([np.ones(rv.shape[1]), _har_design(rv, iv, int(t), use_iv, cross_section)]) for t in origins])
    target = np.stack([rv[t:t + horizon].mean(axis=0) for t in origins])
    # Fit one small regression per asset.  This keeps the benchmark and the
    # anchor transparent while retaining cross-sectional heterogeneity.
    beta = np.stack([np.linalg.lstsq(design[:, asset, :], target[:, asset], rcond=1e-8)[0]
                     for asset in range(rv.shape[1])], axis=1)
    return beta


def _fit_point(rv: np.ndarray, iv: np.ndarray, train_end: int, step: int, use_iv: bool, cross_section: bool) -> np.ndarray:
    """Fit a direct forecast for the single daily value t+step."""
    origins = np.arange(22, train_end - step, dtype=int)
    if len(origins) < 30:
        raise ValueError("too few training origins for direct regression")
    design = np.stack([np.column_stack([np.ones(rv.shape[1]), _har_design(rv, iv, int(t), use_iv, cross_section)]) for t in origins])
    target = np.stack([rv[t + step] for t in origins])
    return np.stack([np.linalg.lstsq(design[:, asset, :], target[:, asset], rcond=1e-8)[0]
                     for asset in range(rv.shape[1])], axis=1)


def direct_forecast(rv: np.ndarray, iv: np.ndarray, train_end: int, origin: int, horizon: int, use_iv: bool, cross_section: bool) -> np.ndarray:
    """Return a direct h-step path of daily forecasts.

    Each horizon is fitted to the corresponding future daily RV, so all
    forecasts use the same information set and no predicted values are fed
    back into the regressors.
    """
    out = []
    for h in range(1, horizon + 1):
        beta = _fit_point(rv, iv, train_end, h, use_iv, cross_section)
        x = np.column_stack([np.ones(rv.shape[1]), _har_design(rv, iv, origin, use_iv, cross_section)])
        out.append(np.sum(x * beta.T, axis=1))
    return np.stack(out, axis=0)


def build_causal_anchor_cube(panel: Panel, max_origin: int, horizon: int) -> np.ndarray:
    """Create per-origin HAR-IV forecasts for residual training targets.

    The coefficients at origin t are estimated using observations strictly
    before t.  This is intentionally slower than fitting one global model but
    preserves the causal meaning of the residual innovation target.
    """
    anchors = np.full((max_origin + horizon + 1, horizon, panel.n_assets), np.nan, dtype=float)
    for origin in range(22, max_origin + 1):
        for h in range(1, horizon + 1):
            if origin < 52 or (origin - 22 - h) < 30:
                anchors[origin, h - 1] = np.maximum(panel.rv[origin - 1], EPS)
                continue
            # Direct prediction of the future h-th daily observation.
            beta = _fit_point(panel.rv, panel.iv, origin, h, True, False)
            x = np.column_stack([np.ones(panel.n_assets), _har_design(panel.rv, panel.iv, origin, True, False)])
            anchors[origin, h - 1] = np.maximum(np.sum(x * beta.T, axis=1), EPS)
    # A causal naive fallback only covers very early context points.
    for origin in range(0, min(22, len(anchors))):
        if origin > 0:
            anchors[origin] = np.maximum(panel.rv[origin - 1], EPS)[None, :]
    return anchors


def build_block_anchor_cube(panel: Panel, train_end: int, max_origin: int, horizon: int,
                            anchor: AnchorForecaster | None = None) -> np.ndarray:
    """Build a fast block-frozen HAR-IV anchor cube.

    Coefficients are estimated once from observations before the rolling
    block's training boundary and then held fixed for every training and test
    origin in that block.  This mirrors deployment: a model is retrained at a
    block boundary, and no later observation can change its anchor.
    """
    anchors = np.full((max_origin + horizon + 1, horizon, panel.n_assets), np.nan, dtype=float)
    anchor = anchor or HarIvAnchor()
    anchor.fit(panel, train_end, horizon)
    for origin in range(max_origin + 1):
        if origin > 0:
            anchors[origin] = anchor.predict(panel, origin)
    return anchors


def feature_summary() -> dict:
    return {"features": list(FEATURE_NAMES), "return_clip": [-0.25, 0.25], "log_floor": EPS}

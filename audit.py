"""Fail-fast checks for a completed world-model run."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def audit_run(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root / "predictions.csv.gz")
    required = {"origin_index", "origin_date", "date", "ticker", "horizon", "model", "target", "mean", "median", "q05", "q25", "q75", "q95", "block_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"prediction columns missing: {missing}")
    numeric = frame[["target", "mean", "median", "q05", "q25", "q75", "q95"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("predictions contain NaN or Inf")
    if (numeric[:, [0, 1, 2, 3, 4, 5, 6]] < 0).any():
        raise AssertionError("RV predictions or targets are negative")
    if not ((frame["q05"] <= frame["q25"]) & (frame["q25"] <= frame["median"]) &
            (frame["median"] <= frame["q75"]) & (frame["q75"] <= frame["q95"])).all():
        raise AssertionError("quantiles are not monotone")
    key_cols = ["origin_index", "ticker", "horizon", "model"]
    if frame.duplicated(key_cols).any():
        raise AssertionError("duplicate prediction keys")
    if not (pd.to_datetime(frame["date"]) > pd.to_datetime(frame["origin_date"])).all():
        raise AssertionError("a prediction target date is not strictly after its origin")
    expected_horizon = (pd.to_datetime(frame["date"]) - pd.to_datetime(frame["origin_date"])).dt.days
    if (expected_horizon < frame["horizon"]).any():
        raise AssertionError("calendar target dates violate the requested forecast horizon")
    block_files = sorted(root.glob("block_*/status.json"))
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in block_files]
    if any(item.get("status") != "complete" for item in statuses):
        raise AssertionError("one or more blocks are incomplete")
    return {"status": "pass", "rows": int(len(frame)), "models": sorted(frame.model.unique()),
            "blocks": len(statuses), "assets": int(frame.ticker.nunique()),
            "manifest_status": manifest.get("status")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit_run(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

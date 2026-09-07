"""Run a frozen evaluation configuration under several independent seeds."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd
import yaml

from .audit import audit_run
from .evaluate import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260721, 20260722, 20260723])
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(base["output_dir"])
    summaries = []
    for seed in args.seeds:
        config = dict(base)
        config["seed"] = seed
        config["output_dir"] = str(root / f"seed_{seed}")
        path = root / f"config_seed_{seed}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output = run(path)
        audit_run(output)
        table = pd.read_csv(output / "summary.csv")
        table["seed"] = seed
        summaries.append(table)
    all_rows = pd.concat(summaries, ignore_index=True)
    all_rows.to_csv(root / "summary_by_seed.csv", index=False)
    aggregate = (all_rows.groupby(["model", "horizon"], as_index=False)
                 .agg(MSE_mean=("MSE", "mean"), MSE_sd=("MSE", "std"),
                      QLIKE_mean=("QLIKE", "mean"), QLIKE_sd=("QLIKE", "std"),
                      MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"), runs=("seed", "nunique")))
    aggregate.to_csv(root / "summary_multiseed.csv", index=False)


if __name__ == "__main__":
    main()

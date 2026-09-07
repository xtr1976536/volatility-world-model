"""Strict rolling evaluation for the Dow30 anchor-conditioned RSSM."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .baselines import predict_baselines
from .data import (build_block_anchor_cube, fit_standardizer, load_panel, make_anchor, make_blocks)
from .metrics import (common_extreme_event_score, cross_sectional_corr_error, dm_comparisons, path_metrics,
                      probabilistic_metrics, summarize_prediction_frame)
from .train import train_block


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _progress(path: Path, log_path: Path, payload: dict, message: str) -> None:
    _write_json(path, payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    progress_path, log_path = output / "progress.json", output / "progress.log"
    progress = {"status": "initializing", "phase": "data", "completed_blocks": [],
                "total_blocks": 0, "current_block": None, "current_step": 0,
                "total_steps": 0, "last_metrics": {}, "elapsed_seconds": 0.0}
    _progress(progress_path, log_path, progress, "initializing")
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    panel, data_audit = load_panel(Path(config["data_source"]), int(config["first_test"]), float(config["min_coverage"]))
    blocks = make_blocks(int(config["first_test"]), int(config["last_test"]), int(config["block_size"]))
    progress["total_blocks"] = len(blocks)
    progress["total_steps"] = len(blocks) * int(config["steps"])
    manifest = {"status": "running", "config": config, "data_audit": data_audit,
                "blocks": [], "device": str(device), "seed": seed,
                "code_sha256": _sha256_tree(Path(__file__).parent)}
    _write_json(output / "manifest.json", manifest)
    _progress(progress_path, log_path, progress, f"loaded {panel.n_dates} dates and {panel.n_assets} assets")

    full_rows: list[dict] = []
    block_samples: dict[str, list[np.ndarray]] = {name: [] for name in ("Naive", "HAR", "HAR-IV", "GHAR", "GHAR-IV", "Hybrid-RSSM")}
    block_means: dict[str, list[np.ndarray]] = {name: [] for name in block_samples}
    block_targets: list[np.ndarray] = []
    block_gates: list[np.ndarray] = []
    block_loadings: list[np.ndarray] = []
    all_started = time.time()
    try:
        for block in blocks:
            block_started = time.time()
            block_dir = output / f"block_{block.block_id:02d}"
            block_dir.mkdir(parents=True, exist_ok=True)
            progress.update({"status": "running", "phase": "training", "current_block": block.block_id,
                             "current_step": 0, "block_start": block.start, "block_end": block.end})
            _progress(progress_path, log_path, progress, f"block {block.block_id + 1}/{len(blocks)} training")
            standardizer = fit_standardizer(panel, block.start)
            anchor = make_anchor(str(config.get("anchor_type", "har_iv")))
            anchors = build_block_anchor_cube(panel, block.start, block.end - 1, int(config["horizon"]), anchor)

            def on_step(record: dict) -> None:
                progress["current_step"] = int(record["step"])
                progress["last_metrics"] = record
                progress["elapsed_seconds"] = time.time() - all_started
                _progress(progress_path, log_path, progress,
                          f"block {block.block_id + 1}/{len(blocks)} step {record['step']}/{config['steps']} loss={record['loss']:.5f}")

            model, history = train_block(panel, standardizer, anchors, block.start, config, seed + block.block_id, device, block_dir, on_step)
            scaled = standardizer.transform(panel).astype("float32")
            horizon = int(config["horizon"])
            origins = np.arange(block.start, block.end, dtype=int)
            target_block = np.stack([panel.rv[t + 1:t + horizon + 1] for t in origins])
            block_targets.append(target_block)
            model_block_samples: dict[str, list[np.ndarray]] = {name: [] for name in block_samples}
            model_block_means: dict[str, list[np.ndarray]] = {name: [] for name in block_samples}
            gate_block, loading_block = [], []
            for origin in origins:
                context = torch.as_tensor(scaled[origin - int(config["context_length"]) + 1:origin + 1][None], device=device)
                anchor_raw = anchors[origin]
                anchor_std = standardizer.transform_log_rv(anchor_raw).astype("float32")
                imagined = model.imagine(context, horizon=horizon, n_samples=int(config["n_samples"]))
                residual_samples = imagined["samples"].detach().cpu().numpy()
                hybrid_samples = standardizer.inverse_log_rv(anchor_std[None, :, :] + residual_samples)
                gate_block.append(imagined["gate"].detach().cpu().numpy()[0])
                loading_block.append(imagined["loadings"].detach().cpu().numpy()[0])
                baselines = predict_baselines(panel.rv, panel.iv, block.start, int(origin), horizon)
                paths = {**baselines, "Hybrid-RSSM": hybrid_samples.mean(axis=0)}
                samples = {**{name: np.repeat(value[None], int(config["n_samples"]), axis=0) for name, value in baselines.items()},
                           "Hybrid-RSSM": hybrid_samples}
                for name, value in paths.items():
                    model_block_samples[name].append(samples[name])
                    model_block_means[name].append(value)
                    for h in range(horizon):
                        for asset, ticker in enumerate(panel.tickers):
                            smp = samples[name][:, h, asset]
                            full_rows.append({"origin_index": int(origin), "origin_date": str(panel.dates[origin].date()),
                                              "date": str(panel.dates[origin + h + 1].date()), "ticker": ticker,
                                              "horizon": h + 1, "model": name, "target": float(target_block[origin - block.start, h, asset]),
                                              "mean": float(np.mean(smp)), "median": float(np.quantile(smp, .50)),
                                              "q05": float(np.quantile(smp, .05)), "q25": float(np.quantile(smp, .25)),
                                              "q75": float(np.quantile(smp, .75)), "q95": float(np.quantile(smp, .95)),
                                              "block_id": int(block.block_id)})
            for name in block_samples:
                block_samples[name].append(np.stack(model_block_samples[name], axis=0))
                block_means[name].append(np.stack(model_block_means[name], axis=0))
            block_gates.append(np.stack(gate_block))
            block_loadings.append(np.stack(loading_block))
            np.savez_compressed(block_dir / "samples.npz", hybrid=np.stack(model_block_samples["Hybrid-RSSM"], axis=0),
                                target=target_block, gate=np.stack(gate_block), loadings=np.stack(loading_block))
            block_frame = pd.DataFrame([r for r in full_rows if r["block_id"] == block.block_id])
            block_frame.to_csv(block_dir / "predictions.csv.gz", index=False, compression="gzip")
            _write_json(block_dir / "status.json", {"status": "complete", "block_id": block.block_id,
                                                      "origins": len(origins), "rows": len(block_frame),
                                                      "elapsed_seconds": time.time() - block_started,
                                                      "history_points": len(history)})
            manifest["blocks"].append({"block_id": block.block_id, "start": block.start, "end": block.end,
                                       "status": "complete", "elapsed_seconds": time.time() - block_started})
            manifest["completed_blocks"] = [item["block_id"] for item in manifest["blocks"]]
            _write_json(output / "manifest.json", manifest)
            progress["completed_blocks"] = manifest["completed_blocks"]
            progress["phase"] = "block_complete"
            progress["current_step"] = int(config["steps"])
            progress["elapsed_seconds"] = time.time() - all_started
            _progress(progress_path, log_path, progress, f"block {block.block_id + 1}/{len(blocks)} complete in {time.time() - block_started:.1f}s")
    except Exception as exc:
        progress.update({"status": "failed", "phase": "error", "error": repr(exc), "elapsed_seconds": time.time() - all_started})
        manifest.update({"status": "failed", "error": repr(exc)})
        _progress(progress_path, log_path, progress, f"failed: {exc!r}")
        _write_json(output / "manifest.json", manifest)
        raise

    frame = pd.DataFrame(full_rows)
    frame.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    metric = summarize_prediction_frame(frame)
    metric.to_csv(output / "summary.csv", index=False)
    dm = dm_comparisons(frame)
    dm.to_csv(output / "forecast_comparisons.csv", index=False)
    diag_rows = []
    all_targets = np.concatenate(block_targets, axis=0)
    for name in block_samples:
        mean_paths = np.concatenate(block_means[name], axis=0)
        sample_paths = np.concatenate(block_samples[name], axis=0)
        path = path_metrics(mean_paths, sample_paths, all_targets)
        path.update({"model": name, "cross_sectional_corr_rmse": cross_sectional_corr_error(mean_paths, all_targets)})
        path.update(common_extreme_event_score(sample_paths, all_targets))
        for h in range(int(config["horizon"])):
            sample_h = sample_paths[:, :, h, :].transpose(1, 0, 2)
            prob = probabilistic_metrics(sample_h, all_targets[:, h, :])
            diag_rows.append({"model": name, "horizon": h + 1, **{k: v for k, v in prob.items() if np.isscalar(v)}})
        diag_rows.append({"model": name, "horizon": "path", **path})
    pd.DataFrame(diag_rows).to_csv(output / "diagnostics.csv", index=False)
    np.savez_compressed(output / "samples_all.npz", hybrid=np.concatenate(block_samples["Hybrid-RSSM"], axis=0),
                        target=all_targets, gate=np.concatenate(block_gates, axis=0),
                        loadings=np.concatenate(block_loadings, axis=0))
    manifest.update({"status": "complete", "completed_blocks": [b.block_id for b in blocks],
                     "prediction_rows": len(frame), "elapsed_seconds": time.time() - all_started})
    _write_json(output / "manifest.json", manifest)
    progress.update({"status": "complete", "phase": "statistics", "current_block": None,
                     "current_step": int(config["steps"]), "elapsed_seconds": time.time() - all_started})
    _progress(progress_path, log_path, progress, f"complete: {len(frame)} prediction rows")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(result)


if __name__ == "__main__":
    main()

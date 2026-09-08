"""Training utilities for one causal rolling block."""
from __future__ import annotations

from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch import nn

from .data import Panel, Standardizer
from .model import DualStateRSSM, RSSMConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(n_assets: int, config: dict, device: torch.device) -> DualStateRSSM:
    cfg = RSSMConfig(
        n_assets=n_assets,
        market_deterministic_dim=int(config["market_deterministic_dim"]),
        market_stochastic_dim=int(config["market_stochastic_dim"]),
        asset_deterministic_dim=int(config["asset_deterministic_dim"]),
        asset_stochastic_dim=int(config["asset_stochastic_dim"]),
        asset_embedding_dim=int(config["asset_embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        consistency_weight=float(config["consistency_weight"]),
        kl_weight=float(config["kl_weight"]),
        free_nats=float(config["free_nats"]),
        rollout_gamma=float(config["rollout_gamma"]),
        public_shock_dim=int(config.get("public_shock_dim", 2)),
        public_shock_scale=float(config.get("public_shock_scale", 0.05)),
        observation_loss_weight=float(config.get("observation_loss_weight", 0.25)),
    )
    return DualStateRSSM(cfg).to(device)


def _batch(panel: Panel, standardizer: Standardizer, anchors: np.ndarray, origins: np.ndarray,
           context_length: int, horizon: int, batch_size: int, rng: np.random.Generator):
    chosen = rng.choice(origins, size=batch_size, replace=True)
    full = standardizer.transform(panel)
    contexts = np.stack([full[t - context_length + 1:t + 1] for t in chosen])
    future = np.stack([full[t + 1:t + horizon + 1] for t in chosen])
    anchor_std = np.stack([standardizer.transform_log_rv(anchors[t]) for t in chosen])
    return contexts.astype("float32"), future.astype("float32"), anchor_std.astype("float32")


def train_block(panel: Panel, standardizer: Standardizer, anchors: np.ndarray,
                train_end: int, config: dict, seed: int, device: torch.device,
                output_dir: Path, progress_callback=None) -> tuple[DualStateRSSM, list[dict]]:
    seed_everything(seed)
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(panel.n_assets, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 1e-5)))
    context_length = int(config["context_length"])
    horizon = int(config["horizon"])
    first_origin = max(context_length - 1, 22)
    last_origin = train_end - horizon - 1
    if last_origin < first_origin:
        raise ValueError(f"training boundary {train_end} leaves no valid origins")
    origins = np.arange(first_origin, last_origin + 1, dtype=int)
    rng = np.random.default_rng(seed + 9176)
    history: list[dict] = []
    steps = int(config["steps"])
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        context, future, anchor = _batch(panel, standardizer, anchors, origins, context_length, horizon, int(config["batch_size"]), rng)
        context_t = torch.as_tensor(context, device=device)
        future_t = torch.as_tensor(future, device=device)
        anchor_t = torch.as_tensor(anchor, device=device)
        loss, parts = model.loss(context_t, future_t, anchor_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite loss or gradient at step {step}")
        optimizer.step()
        interval = max(1, steps // 60)
        if step == 1 or step % interval == 0 or step == steps:
            row = {"step": step, "elapsed_seconds": time.time() - started, "grad_norm": float(grad_norm.detach().cpu())}
            row.update({key: float(value.detach().cpu()) for key, value in parts.items()})
            history.append(row)
            if progress_callback:
                progress_callback(row)
    torch.save({"state_dict": model.state_dict(), "rssm_config": model.cfg.__dict__,
                "scaler": standardizer.to_dict(), "seed": seed, "train_end": train_end}, output_dir / "checkpoint.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def load_model(checkpoint: Path, device: torch.device) -> DualStateRSSM:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DualStateRSSM(RSSMConfig(**payload["rssm_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

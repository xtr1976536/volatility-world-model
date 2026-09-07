"""Small deterministic checks used before the long Kaggle run."""
from pathlib import Path
import tempfile
import torch
import yaml

from .audit import audit_run
from .data import HarIvAnchor, build_block_anchor_cube, build_causal_anchor_cube, fit_standardizer, load_panel
from .model import DualStateRSSM
from .train import make_model


def run_smoke(config_path: Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    panel, _ = load_panel(Path(cfg["data_source"]), int(cfg["first_test"]), float(cfg["min_coverage"]))
    scaler = fit_standardizer(panel, int(cfg["first_test"]))
    assert scaler.mean.shape == (panel.n_assets, 3)
    assert (scaler.scale > 0).all()
    anchors = build_causal_anchor_cube(panel, int(cfg["first_test"]), int(cfg["horizon"]))
    assert anchors.shape[1:] == (int(cfg["horizon"]), panel.n_assets)
    anchor_model = HarIvAnchor().fit(panel, int(cfg["first_test"]), int(cfg["horizon"]))
    block_anchors = build_block_anchor_cube(panel, int(cfg["first_test"]), int(cfg["first_test"]), int(cfg["horizon"]), anchor_model)
    assert block_anchors[int(cfg["first_test"])].shape == (int(cfg["horizon"]), panel.n_assets)
    model = make_model(panel.n_assets, cfg, torch.device("cpu"))
    context = torch.randn(2, int(cfg["context_length"]), panel.n_assets, 3)
    future = torch.randn(2, int(cfg["horizon"]), panel.n_assets, 3)
    anchor = torch.randn(2, int(cfg["horizon"]), panel.n_assets)
    loss, _ = model.loss(context, future, anchor)
    assert torch.isfinite(loss)
    imagined = model.imagine(context[:1], int(cfg["horizon"]), 7)
    assert imagined["samples"].shape == (7, int(cfg["horizon"]), panel.n_assets)
    assert (imagined["gate"] >= 0).all() and (imagined["gate"] <= 1).all()
    assert imagined["loadings"].shape == (7, int(cfg["horizon"]), panel.n_assets, int(cfg["public_shock_dim"]))
    assert torch.isfinite(imagined["loadings"]).all()
    cov = torch.cov(imagined["samples"][:, 0, :].T)
    assert torch.isfinite(cov).all()
    off_cfg = dict(cfg)
    off_cfg.update({"public_shock_dim": 0, "public_shock_scale": 0.0})
    off_model = make_model(panel.n_assets, off_cfg, torch.device("cpu"))
    off_loss, _ = off_model.loss(context, future, anchor)
    assert torch.isfinite(off_loss)
    return {"status": "pass", "assets": panel.n_assets, "dates": panel.n_dates}


if __name__ == "__main__":
    print(run_smoke(Path(__file__).with_name("config_smoke.yaml")))

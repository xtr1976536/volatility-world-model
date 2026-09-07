"""Create paper figures from the audited Dow30 world-model output."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "dow30_serial" / "seed_20260721"
FIG = Path(__file__).resolve().parent

COLORS = {
    "Naive": "#8c8c8c", "HAR": "#7b9acc", "HAR-IV": "#496a8f",
    "GHAR": "#b28c60", "GHAR-IV": "#7f6040", "Hybrid-RSSM": "#1f6f78",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.55,
    "grid.alpha": 0.7, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.tight_layout(pad=0.8)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_block_diagnostics():
    files = sorted(OUT.glob("block_*/diagnostics.csv"))
    if not files:
        raise FileNotFoundError("no block diagnostics found")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def qlike_figure():
    df = pd.read_csv(OUT / "summary.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for model, g in df.groupby("model", sort=False):
        g = g.sort_values("horizon")
        ax.plot(g.horizon, g.QLIKE, marker="o", markersize=3.5, linewidth=1.7,
                color=COLORS.get(model, "#555555"), label=model)
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("QLIKE")
    ax.set_xticks(range(1, 6))
    ax.set_title("Out-of-sample QLIKE by forecast horizon")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save(fig, "fig1_qlike_by_horizon")


def probabilistic_figure():
    df = read_block_diagnostics()
    g = df[(df.model == "Hybrid-RSSM") & (df.horizon != "path")].copy()
    g["horizon"] = g["horizon"].astype(int)
    g = g.groupby("horizon", as_index=False).mean(numeric_only=True).sort_values("horizon")
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharex=True)
    specs = [("CRPS", "CRPS", "#1f6f78"), ("coverage90", "90% coverage", "#496a8f"),
             ("coverage50", "50% coverage", "#b28c60"), ("width90", "90% interval width", "#7f6040")]
    for ax, (col, label, color) in zip(axes.flat, specs):
        ax.plot(g.horizon, g[col], marker="o", linewidth=1.8, color=color)
        ax.set_title(label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if col == "coverage90": ax.axhline(0.90, color="#333333", ls="--", lw=0.9)
        if col == "coverage50": ax.axhline(0.50, color="#333333", ls="--", lw=0.9)
        ax.grid(True)
    axes[1, 0].set_xlabel("Forecast horizon (trading days)")
    axes[1, 1].set_xlabel("Forecast horizon (trading days)")
    save(fig, "fig2_probabilistic_diagnostics")


def path_figure():
    df = read_block_diagnostics()
    path = df[df.horizon.astype(str) == "path"].copy()
    path = path.groupby("model", as_index=False).mean(numeric_only=True)
    metrics = [("mean_increment_corr", "Mean-path increment correlation"),
               ("increment_sd_ratio", "Mean-path increment SD ratio"),
               ("sample_increment_sd", "Sample-path increment SD"),
               ("cross_sectional_corr_rmse", "Cross-sectional correlation RMSE")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8))
    for ax, (col, label) in zip(axes.flat, metrics):
        rows = path.dropna(subset=[col]).sort_values("model")
        labels = rows.model.tolist()
        values = rows[col].to_numpy(float)
        colors = [COLORS.get(x, "#777777") for x in labels]
        ax.bar(np.arange(len(values)), values, color=colors, width=0.72)
        ax.set_title(label)
        ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
        ax.grid(axis="y")
    save(fig, "fig3_path_diagnostics")


def training_figure():
    rows = []
    for path in sorted(OUT.glob("block_*/block_00/training_history.json")):
        block = int(path.parts[-3].split("_")[-1])
        history = json.loads(path.read_text())
        last = history[-1]
        rows.append({"block": block + 1, **last})
    df = pd.DataFrame(rows).sort_values("block")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.8))
    for ax, col, label, color in [(axes[0], "prior_rollout_nll", "Prior rollout NLL", "#1f6f78"),
                                  (axes[1], "latent_gap", "Latent mean gap", "#b28c60"),
                                  (axes[2], "gate_mean", "Mean innovation gate", "#496a8f")]:
        ax.plot(df.block, df[col], marker="o", linewidth=1.8, color=color)
        ax.set_xlabel("Rolling block")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "fig4_training_diagnostics")


def extreme_figure():
    bundles = [np.load(path) for path in sorted(OUT.glob("block_*/samples_all.npz"))]
    if not bundles:
        bundles = [np.load(path) for path in sorted(OUT.glob("block_*/block_00/samples.npz"))]
    samples = np.concatenate([b["hybrid"] if "hybrid" in b else b["hybrid_samples"] for b in bundles], axis=0)
    target = np.concatenate([b["target"] for b in bundles], axis=0)
    truth_market = target.mean(axis=-1)
    draw_market = samples.mean(axis=-1)
    threshold = np.quantile(truth_market, 0.9)
    realized = (truth_market >= threshold).mean(axis=0)
    predicted = (draw_market >= threshold).mean(axis=(0, 1))
    horizons = np.arange(1, target.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    width = 0.34
    ax.bar(horizons - width / 2, realized, width=width, label="Realized rate", color="#496a8f")
    ax.bar(horizons + width / 2, predicted, width=width, label="Sample probability", color="#b28c60")
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Common high-volatility event rate")
    ax.set_xticks(horizons)
    ax.legend(frameon=False)
    save(fig, "fig5_common_extreme_events")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    qlike_figure()
    probabilistic_figure()
    path_figure()
    training_figure()
    extreme_figure()

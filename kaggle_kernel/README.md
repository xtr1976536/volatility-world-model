# Anchor-Conditioned Volatility World Model v1

This directory contains the reproducible Dow30 experiment for the paper
`paper/main.tex`.  The model is a passive probabilistic latent-dynamics model,
not an automated trading system.

## Model

For an origin `t`, a causal level anchor is estimated from the rolling
information set. `HarIvAnchor` is the default used by the reported experiment;
the `AnchorForecaster` interface permits a different causal level model only
when residual targets and the RSSM are retrained together. The RSSM receives a 44-day context of standardized
`log(RV)`, `log(IV)`, and clipped returns.  It maintains one market state and
one state per asset.  The market deterministic state is shared by all assets;
asset embeddings preserve heterogeneous responses.  Future observations are
never supplied during imagination.

The decoder predicts the standardized residual

`log(RV[t+h]) - log(HAR-IV-anchor[t+h])`

with a Gaussian mean and positive scale.  A learned market gate multiplies
the innovation mean only.  The scale is not inflated by the gate.  Training
uses a free-running prior NLL, teacher-forced NLL, KL balancing with free nats,
posterior-to-prior consistency, and gradient clipping.
The joint observation additionally uses a two-dimensional public shock and
asset-specific noise. Setting `public_shock_dim: 0` gives the independent-noise
ablation without changing the rest of the architecture.

## Data and protocol

The included `data/dow30` directory is extracted from the previously audited
Dow30 archive.  The formal defaults are 30 assets, 1,254 common dates,
origins 1113--1248, blocks of 22 origins, five-day paths, 600 steps per
block, one seed (`20260721`), and 500 samples per origin.  The exact protocol
is in `config.yaml`; `manifest.json` records the data audit and code hash.

Runtime filling is forward-only.  The existing `*_filled.csv` files are an
upstream provenance risk and are not treated as a causal audit.

## Local checks

```bash
python -m empirical_runs.world_model_v1.test_smoke
python -m empirical_runs.world_model_v1.evaluate --config empirical_runs/world_model_v1/config_smoke.yaml
python -m empirical_runs.world_model_v1.audit empirical_runs/world_model_v1/outputs/smoke
```

## Formal run

```bash
python -m empirical_runs.world_model_v1.evaluate \
  --config empirical_runs/world_model_v1/config.yaml
python -m empirical_runs.world_model_v1.audit \
  empirical_runs/world_model_v1/outputs/dow30_formal
```

The output contains checkpoints and histories for each block, compressed
Monte Carlo paths, `predictions.csv.gz`, `summary.csv`,
`forecast_comparisons.csv`, `diagnostics.csv`, `manifest.json`, and progress
files.  A failed run retains completed blocks and marks the manifest as
failed.

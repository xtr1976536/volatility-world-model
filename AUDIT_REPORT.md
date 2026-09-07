# World-Model Paper and Code Audit

Date: 2026-09-07

## Scope

The audit compares `paper/main.tex`, the implementation under this package, and the audited Dow30 serial run under `outputs/dow30_serial/seed_20260721`.

## Verified matches

- The target is a five-day path of positive daily RV for a fixed 30-asset Dow30 universe.
- The information set ends at the forecast origin; future observations are not supplied to `imagine()`.
- The default anchor is a direct, block-fitted HAR-IV forecast. Its coefficients are frozen within each rolling block.
- The neural state has one market RSSM and one asset RSSM per ticker, with a shared market deterministic state and learned asset embeddings.
- The observation model is a two-factor low-rank Gaussian residual law plus asset-specific diagonal scales.
- The gate multiplies the residual mean only; it does not multiply the predictive scale or the public factor loadings.
- The likelihood uses the Woodbury identity and matrix determinant lemma, matching the low-rank covariance equation in the paper.
- Standardization is fitted before each block boundary; runtime filling is forward-only.
- The audited run has 7 complete blocks, 136 origins, 30 assets, 5 horizons, 122,400 prediction rows, finite values, unique keys, and target dates after origins.

## Corrections made in this review

- Added author and contact metadata and a repository link placeholder to the manuscript.
- Added five empirical figures generated from saved result files and a figure catalog.
- Removed process-oriented and defensive wording from the introduction and discussion.
- Added explicit benchmark-label qualification: current GHAR/GHAR-IV outputs are cross-sectional-summary models, not a graph-adjacency reimplementation.
- Added the corrected `device: auto` resolution in the evaluator.

## Remaining scientific boundary

The current evidence is a single-seed Dow30 experiment. It supports the reported fixed-seed forecast ranking and exposes path/joint-distribution weaknesses; it does not support a universal superiority claim, causal spillover claim, or complete financial-simulator claim. The upstream files labelled `filled` remain a data-provenance issue.

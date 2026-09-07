# World-Model Manuscript Status

This directory contains the expanded English manuscript and the audited single-seed local results.

## Current manuscript

- `main.tex`: expanded 15-page English manuscript, with approximately 13 pages of text and appendix material before references.
- `build/main.pdf`: locally compiled manuscript PDF.

## Scope frozen in the draft

The proposed method is a generic anchor-conditioned recurrent state-space world model for five-day, multi-asset realized-volatility paths. The current formal instance uses a causal HAR-IV anchor; the RSSM models anchor-relative residual innovations using shared market and asset-specific states, and a two-factor public-shock observation layer.

## What is deliberately not claimed

The numerical tables contain the audited seed-20260721 Dow30 rolling result. The result is a single-seed, single-universe feasibility study. Multi-seed robustness, graph-estimated GHAR, and formal ablations remain future work. The upstream files labelled `filled` remain a data-provenance limitation.

## Approval gate

The implementation and manuscript are currently synchronized. The final local run contains seven complete audited blocks, 136 origins, 30 assets, and 122,400 prediction rows. Further experiments should use the generic anchor interface and preserve the same causal rolling protocol.

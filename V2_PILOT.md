# v2 Observation-Head Pilot

This pilot adds a prior observation decoder for all encoded features:
standardized log-RV, standardized log-IV, and clipped returns. The RV
anchor-relative likelihood remains the primary objective and the full
observation reconstruction is weighted by `observation_loss_weight=0.25`.

The experiment uses the first Dow30 rolling block, seed `20260908`, 20
optimization steps, 8 training samples per step, reduced latent dimensions,
and 32 forecast paths. It passed the smoke and file audits. At this budget,
the hybrid model did not improve the HAR-IV point forecast at horizons 2--5;
the result is a structural diagnostic rather than a model comparison. The
pilot indicates that a full observation head needs a longer training budget
and an explicit auxiliary-loss ablation before it can be interpreted as a
world-model improvement.

The pilot is retained separately and is not merged with the audited
single-seed baseline results.

## 600-step confirmation

The same first-block configuration was trained for 600 steps. The run passed
the audit and produced finite predictions. Hybrid-RSSM QLIKE was 0.000568,
0.000768, 0.001045, 0.001605, and 0.002888 at horizons 1--5. The model did
not establish a point-forecast advantage over the corrected HAR benchmarks.
The mean-path increment SD ratio was 0.424, which is closer to the realized
ratio than the earlier pilot, but the probability and point-loss criteria did
not improve jointly. The full observation head therefore remains a diagnostic
branch rather than the selected v2 architecture.

The companion manuscript `paper/main_v2.tex` uses this confirmation as a
pilot feasibility result and does not import the earlier non-unified result
tables.

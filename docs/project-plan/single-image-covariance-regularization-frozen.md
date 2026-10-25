# N77 — covariance regularization with headroom-aware rates

Run one fresh_00 / seed-5003 trajectory from the identical complete N73
576-update parent, over the unchanged 57,600-image continuation. Preserve the
stopped N76 checkpoint. No new accept-all, fixed-rate or metric-selection sweep.

Keep the measured selector-bank covariance ceiling at 1.30 times original warm.
The upstream bounded, clock-free J gate and its threshold are inherited without
refitting. Its original labels describe unregularized nominal updates: it is an
explicit heuristic for the new update rule, not a newly calibrated forecaster.

For each J-admitted image, combine its unchanged seeded singleton MAE loss with
an exact full-selector-bank penalty. Normalize each reference embedding to unit
length; compute the same float64 squared off-diagonal covariance and variance
as the monitor. Penalize max(0, C/Cwarm - 1.10)^2 with coefficient
10 * (1 + 9 * clip((C/Cwarm - 1.10)/0.20, 0, 1)). Treat this coefficient as a
detached controller value. Also penalize max(0, 1 - V/Vwarm)^2 with weight 0.25.
Use cached current features only to compute the feature-space derivative;
recompute differentiable bank embeddings in batches of 32 and accumulate the
exact full-bank parameter gradient. This explicitly adds reference-bank
representation training, not replay reconstruction or additional labels.

Compute the gradient once per admitted image with the same original view seed.
Clip the combined gradient norm to 1.0. Each image starts at nominal AdamW LR
0.0001. After an actual covariance failure, restore complete pre-image state
and multiply the rate factor by min(0.5, max(0.01, 0.8 * remaining_headroom /
measured_increase)). Exhaust at most 8 trials; stop below factor 1e-6. Keep
the first passing step and its genuine moments; reset parameter-group LR to
nominal for the next image. Every failed trial restores weights, optimizer,
module modes and CPU/CUDA RNG. Share the seeded gradient across independent
trials and retain the post-gradient RNG only on acceptance.

Neither development nor final task outcomes control admission, penalty strength
or LR. Development labels still enter the inherited rich predictor features.
Require GPU preflight: a genuine covariance-decreasing step from stalled N76,
a genuine headroom-based smaller-rate rescue, zero-penalty equivalence to the
original seeded step and exact interrupted resume with optimizer accounting.
These are engineering checks, not new independent efficacy trajectories.

Checkpoint every 384 offered arrivals, with status heartbeats every 32.
Refresh notebook 77 from committed measurements. Show all N76 performance,
health, reconstruction, development and controller metrics, with continuous LR
choices, regularization loss/coefficient and actual parameter displacement.
Include kNN readouts already present in saved assays. Clip saved reference
curves to the current measured horizon; never extend endpoints flat.

Freeze inherited hashes and all new scientific sources before the main run.
Retain two complete checkpoint generations and a two-GiB free-storage reserve.
A stop-request.json stops at the next arrival with a complete committed state.
The live notebook viewer is read-only and reloads the generated notebook and
status; it performs no inference or model evaluations. Exit after this one run.

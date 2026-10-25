# Temporal batch construction at matched exposure

Authorized 14 October 2026, after the health-to-utility bridge failed. This bounded
development experiment tests whether changing complete training batches has useful
leverage before fitting another utility predictor. Freeze this protocol and source
versions before collecting new grouping outcomes. Preserve all previous studies.

## Population and fixed learner

Reuse the six coverage-admission histories (two each short/medium/long dwell),
12,288 ordered arrivals per history, seeds 5003/5009/5011 and existing warm MAE,
AdamW, images, supports and queries. These are inspected development sources and
synthetic clip concatenations with unknown recording sessions. No new-source or
independent-panel qualification is claimed. No downloads or model/pixel caches.

Every arm takes 192 ordinary batch-32 updates. The first eight updates use the
original current-only batches and share an exact calibration state. Then use 23
blocks of 512 arrivals, eight updates per block. Buffer each block until all 512
frames have arrived, for every arm including chronological controls. No model
updates occur during that collection. Selectors see only arrived indices. This
common latency isolates grouping; it is not a zero-latency streaming claim.

## Fixed actions and tunable mechanism

Four arms: current-only, chronological replay, random replay grouping, stratified
replay grouping. Keep replay fraction fixed at 8/32, acceptance/exposure budget
fixed and all SSL settings unchanged. The grouping knob has endpoints chronological
and eight-way temporal stratification; random regrouping is the matched control.
No intermediate knob values, health thresholds or tuning after outcomes.

Current-only reproduces the original uniform-half batches: the first 32 frames
from each chronological 64-arrival slice. Replay arms take the same 24 uniformly
spaced indices in that 32-frame set, using the existing integer linspace rule.
All three replay arms consume precisely the same 192 current identities and 64
historical identities per block. The existing Algorithm-R reservoir-512 and
private 64-identity block draw are unchanged. Historical images must predate the
block, and their eight-per-update identities/order are identical across replay
arms. Admission to memory is independent of training selection.

For current candidates, form an 8-by-24 matrix of consecutive arrival strata.
Chronological uses its rows. Stratified batch j takes columns 3j..3j+2 from each
of the eight rows, concatenated in chronological stratum order. Random uses one
private permutation of all 192 current identities, split into eight groups of 24.
Its seed is stream_seed(stream, seed)+88169+block_index. All replay batches prepend
the same eight historical images. No scene/lighting labels, image embeddings,
health signals, task labels or future outcomes influence grouping.

Matched identities are exact within replay arms over every eight-update block
and the whole trajectory. Their image-to-mask assignments and intermediate AdamW
states can differ: grouping and order are the intervention, not identical per-image
augmentations. Three optimization seeds measure conditional stochastic sensitivity.
The eight index strata are not inferred semantic events. Block/clip alignment in
these prepared streams limits transfer to arbitrary recording boundaries.

## Outcomes and controls

The current and chronological replay controls must reproduce all original update
losses/norms and complete before-return/terminal fingerprints from the audited
adaptive-mixture study. Only then reuse their copied-classifier predictions.
Read out random/stratified at the same before-return and terminal arrival indices.
Hold/warm uses cached unchanged warm predictions. Primary is terminal balanced
50-label, 100-step copied-encoder fine-tuning accuracy averaged over three panels
on 400 queries. Day/night, linear and kNN remain separate; before-return is secondary.
All readers preserve online state, optimizer, modes and CPU/CUDA RNG.

Read all five health metrics on the same fixed 256-image bank at warm,
post-calibration, before-return and terminal. Save compact embeddings for independent
geometry reconstruction. Health is diagnostic, never a utility target or selector.
Compute within-batch current-stratum occupancy, same-stratum pair fraction and
arrival-index separation to verify the intended manipulation without labels.

## Frozen gates

Six history means are the uncertainty units; average seeds/panels within history.
Use paired Student-t intervals, df five, Bonferroni 95% across twelve primary scalars:
stratified balanced minus current/chronological/random (three), stratified day/night
minus those three and warm (eight), and stratified balanced minus warm (one).

A structured-filter lead requires at least 0.25pp balanced mean advantage and a
positive adjusted lower bound versus each of current/chronological/random; useful
SSL requires at least 0.1pp and a positive adjusted lower bound versus warm; every
day/night adjusted lower bound must be at least -1pp. All gates are required.
This still establishes only development evidence. Report random-minus-chronological
and all fixed arms as secondary descriptions; do not promote a secondary winner
or change gates if stratification fails. Report source-wise and seed-wise outcomes.

If the screen fails, preserve the grouping mechanism and findings without fitting
a predictor, launching another parameter sweep or automatically collecting sources.
If it passes, a separately frozen unused-source confirmation and real repeated-control
comparison are necessary. Neither a health shift nor an optimistic hindsight chooser
qualifies downstream filter value.

## Work, storage, cost and recovery

72 trajectories: 36 new grouping trajectories and 36 cached-control reconstructions,
each 192 updates, with every trajectory replayed independently. Thus 6,912 new
grouping updates + 6,912 control-reconstruction updates + 13,824 replay updates +
16 discarded deterministic preflight updates = 27,664 total optimizer updates.
72 new readouts x three primary panels = 216 new fine-tuning fits; 72 cached control
readouts x three panels = 216 reused fits, plus nine unique warm fits reused.
Unit-test optimization and independent CPU verification are separate. No extra GPU
updates are allowed for the independent auditor.

Charge current-frame gathering, memory planning, batch construction, SSL updates
and common buffering. Record health observation, restoration, copied readouts,
forensic replay and artifact writing separately. Include buffer occupancy and update
delay in arrival units; prepared-processing timings are not deployed decoding or
end-to-end efficiency. No efficiency advantage is a primary claim.

Output cap 96MiB; launch reserve 512MiB, per-trajectory reserve 256MiB. Save one
completed/replayed trajectory atomically before advancing; support resume by
verifying frozen hashes and committed files, skipping completed trajectories and
preserving incomplete attempts. Record any recovery work separately. Never rewrite
the original protocol/lock after outcomes. Tests cover exact exposure matching,
private RNG, causal temporal bounds, stratification and corrupted resume evidence.
Independent verification reconstructs plans, grouping, manipulation, health,
raw-prediction scores, source/seed aggregation, work and all gates.

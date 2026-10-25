# Image training impact on health: bounded development pilot

Authorized 14 October 2026. Run alongside the separate capability-response pilot.
Question: can pre-decision measurements predict an individual image's conditional
effect on representation health? This is not a downstream-quality claim.

Population: six existing disjoint-clip budget-reservation histories, checkpoints
before_shift/stable_second/before_return, optimization seed5003 (18 parents).
Take candidate slots0,8,16,24 from the next already-arrived 32 frames. Two private
augmentation/mask draws (seed71011 and71023, offset by parent arrival count) quantify
training-draw sensitivity. All choices are fixed before new outcomes. Parents and
footage are reused development data; no independent capture-session claim.

Primary intervention is full batch32 versus the same forward and random draw with
one candidate image's reconstruction-loss contribution zeroed, keeping denominator
32. The other31 losses are unchanged. Restore all parameters, AdamW moments,
clipping1, module modes and CPU/CUDA RNG before every branch. The omitted branch
still has momentum/decay and advances the optimizer once. This measures marginal
inclusion at weight1/32, conditional on these other31 images; it is not intrinsic
image utility, a replacement image comparison or ordinary batch31 normalization.
Verify all-included per-image loss and state match the ordinary batch32 optimizer
step exactly within declared float tolerance (loss1e-7, parameter absolute2e-7).

Secondary intervention: train a single image alone at weight1 against an exact
hold, using the same restored optimizer. It is reported separately; it changes the
batch recipe and must not be generalized to batch32 marginal effects.

Use the inherited fixed256-image warm bank, excluded from stream/support/query
identities by its original enrollment. Raw encoder embeddings, identical unaugmented
RGB64 images. Primary health is mean per-coordinate sample variance of row-L2
normalized embeddings. Target = (variance_full - variance_omitted)/variance_parent.
Secondary: relative RankMe/covariance changes and differences in CKA/cosine drift
relative to the fixed warm encoder. Compute metrics in float64 to resolve tiny
image effects. No assertion that larger variance or rank means better capability.

Before outcomes: candidate RGB mean/std/edge energy, per-image reconstruction loss
on the paired training view, normalized embedding distance from bank centroid;
parent update count, rank/variance/covariance, warm CKA and image-loss x parent
variance interaction. The candidate loss requires a forward; charge its cost.
Fit fixed ridge10 with train-history-only standardization. Compare zero, training
mean, parent-state-only ridge, image+state ridge (primary cheap candidate).
Mechanistic primary comparator: first-order parent-variance gradient dotted with
the exact candidate-minus-omitted AdamW encoder displacement, divided by parent
variance. This pays for copied updates and a bank backward pass; it avoids post-step
bank observation but is not a cheap image selector. Post-step embeddings never
enter any forecast. Recompute the direct derivative before collecting endpoints.

Hold out each entire history, including every seed/checkpoint/image/draw. Clip IDs
must not overlap histories. Report macro-history MSE, zero-relative skill, correlation,
within-parent/draw rank correlation and repeat-draw agreement. Two primary forecast
candidates each compare against zero and parent-state-only: four paired tests,
Bonferroni family4 across six history errors, df5. Each candidate needs >=10% MSE
reduction and positive adjusted improvement bounds versus both baselines. Additional
metrics and single-image interventions remain secondary regardless of their results.
Near-zero target energy (<1e-16 mean squared relative change) is unresolvable and
cannot qualify any predictor. A small MSE is not evidence without baseline skill.

Scientific updates:36 full,144 omitted,144 single=324; exact replay repeats324.
One ordinary all-included control and one loss-gradient reconstruction control per
parent/draw add72 validation updates; maximum720 total. No discarded scientific
retry, future stream, label access, download or new model/pixel cache. Save compact
embeddings, fingerprints, exact slot IDs, forecasts, timing and source snapshots.
Verify parent/observer invariance and replay every complete branch state and embedding.
Read-only bank gradient must preserve parameters, modes, gradients and RNG. Independent
NumPy reconstruction checks geometry and statistics. Output cap96MiB, reserve512MiB
at launch/256MiB per write. No adaptive policy, metric/seed/horizon sweep or promotion
to broad phase gates. New-source confirmation is a separate subsequent experiment.

## Device repair before restarted collection

The first attempt (`20261014-pilot`) stopped on a CPU/CUDA displacement subtraction
before saving any image-response rows. It completed one full, one omitted and one
single branch, each replayed, plus ordinary and gradient-subtraction controls:
eight discarded updates. Original inputs, source snapshot and embeddings remain.
The repair only converts the inherited CPU parent tensors to the derivative's
device. A CUDA regression test covers it. Restart in
`20261014-pilot-device-repair` with identical scientific settings; account eight
discarded updates separately (728 including the complete720-update restart).
No outcome, threshold, sign, feature or input selection changes.

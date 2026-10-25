# Single-image next-state forecasting

User-directed development experiment, 2026-10-21. Freeze this protocol, the
configuration and sources before collecting scientific observations. The
question is whether information available before learning from the next image
predicts the signed change in health, reconstruction and downstream performance.
This is a new batch-one study, not a rerun or reinterpretation of batch32 results.

Use the existing warm TinyViT MAE, full encoder/decoder AdamW at 1e-4, decay .05,
clip norm one, light augmentation and mask .75. Train on exactly one image per
optimizer step, no accumulation, replay, resets or accept/reject policy. The
generic batch32 loop is not used. Adam starts empty for each trajectory.

Use six existing coverage-admission histories, fresh_00 through fresh_05, and
seeds 5003/5009. From each of the twelve periods take the first clip's first
sixteen timestamp-ordered frames. Thus each history has 192 arrivals, preserving
the day/night dwell pattern but shortening its periods. Chronology holds inside
each clip. Clip joins are synthetic, and this is not an uninterrupted drive or
a real-time deployment. Histories are clip-disjoint. The sources, warm model and
reader panels are previously inspected development material.

Measure state at initialization and after EVERY one-image update on an invariant
256-image health bank and a disjoint reader panel. Health includes float64
RankMe, normalized variance, off-diagonal covariance, warm-relative CKA/cosine,
and feature norm. Reconstruction loss uses unaugmented bank pixels and a fixed
random mask. Also measure the arriving image before/after with a fixed mask.
Keep bank reconstruction mask, supports, queries, reader initialization,
fine-tuning minibatches and evaluation seeds identical across states. Observers
restore modes and CPU/CUDA RNG, never update the live SSL optimizer, and train
fine-tuning copies only. Audit complete learner invariance and exact replay.

Readouts use two scene classes (city street/highway), the first established
50-label support panel, and the first 100 development queries in each
domain/class cell: 400 balanced queries. Linear logistic regression is refit
on support-only standardized embeddings, C=1, max_iter=1000. kNN uses cosine
distance, distance weights and k=20. Record accuracy in percentage points and
cross entropy, balanced/day/night, and fine-tuning-minus-linear accuracy.
Dense fine-tuning uses the established copied-encoder/random-head recipe with
20 steps rather than 100, fixed seed 51036, encoder LR1e-5/head LR1e-3,
AdamW decay .05 and batches32 on labeled support data. The batch-one requirement
applies to STREAMING SSL, not separate labeled diagnostic readers. Also run the
original 100-step assay at states 0,32,...,192; forecast its 32-update changes
as an explicitly secondary, sparse analysis. Twenty-step outcomes must not be
represented as the original hundred-step outcome.

For each transition t -> t+1, predict a signed delta and reconstruct the next
state by adding the prediction to the measured pre-state. Freeze a ridge10
multi-output forecast on the first64 transitions of EACH trajectory; score only
transitions65..192. An otherwise identical online predictor sees an outcome only
AFTER predicting it, then updates by recursive least squares using the fixed
prefix normalization. Earlier observed test outcomes are legitimate inputs to
later forecasts; future outcomes never enter fitting or scaling. No random split,
test-based hyperparameter/model selection, smoothing across the boundary or
pooling of seed replicates as independent sources. This tests same-stream
adaptation, not generalization to unseen histories.

Compare zero-change persistence, last-change, a frozen prefix-mean delta, and
online running-mean delta. Fit frozen/online variants of four feature sets:
current state; current state plus last and trailing-four changes; the latter
plus arriving-image features; label-free health/history plus image features.
Image features are RGB means/stds, edges, difference from previous image,
pre-update fixed-mask reconstruction loss, embedding norm/centroid distance,
and eight fixed random projections of normalized embedding. Inputs never include
labels, domain/clip identities or a copied training step. State/history versions
that use prior downstream scores are diagnostic supervised-observation models;
only the label-free feature set avoids downstream evaluation inputs. Training
targets still require offline labels for downstream measures. No deployed
label-free downstream feedback or useful selection policy is established here.

RMSE and MAE in native units are primary. Errors on levels and deltas are
algebraically identical for these anchored predictions; report persistence
skill to avoid confusing predictable levels with predictable changes. Include
direction accuracy for increase/decrease/unchanged, change-only accuracy,
balanced direction recall, tie frequency, errors by history, boundary proximity,
and source-cluster bootstrap intervals for paired error reductions. Average
seeds within each of six source histories. Intervals are descriptive, without
a multiple-target confirmation claim. Show all frozen models for all targets;
do not promote a different winner per metric. Accuracy resolution is .25pp
balanced/.5pp per domain, so classification ties are expected. Report query
cross entropy as a continuous companion, without equating it to accuracy.

Preflight uses discarded updates only to verify mechanics and runtime. No
outcome-based budget extension, early stopping, model tuning or source replacement.
Save sources/input hashes, selected frame identities, before/after states,
image features, raw readout predictions/probabilities, embeddings, online update
timing, forecast arrays, all-model summaries, numerical audit and plots/report.
Run a training-only replay of every transition and selected observer repeats.
The result answers bounded next-state forecasting under this MAE recipe, not
whether selection improves learning or all possible predictors can work.

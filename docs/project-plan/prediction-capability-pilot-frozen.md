# Capability response prediction: bounded development pilot

Authorized 14 October 2026. Separate from the image-to-health experiment. This
pilot asks whether observations available before accepting training predict its
effect on a declared downstream capability. It does not fit a deployed policy.

Reuse the audited six budget-reservation histories, three checkpoints and three
optimization seeds. Average seeds and the three existing support panels within
each history/checkpoint: 18 points in six groups. These sources and outcomes have
already been inspected. Their original `heldout` names do not make this a fresh
test. Entire histories are excluded from fitting and normalization. Confirm clip
identities are disjoint between histories; unknown capture sessions remain a limit.

Primary target: immediate balanced fine-tuning capability, spend minus hold, in
accuracy percentage points after 16 batch32 updates. This is the existing copied
encoder reader: 50 support images, 100 fine-tuning steps, three panels, 400 balanced
queries. Terminal balanced effects and immediate/terminal day/night effects are
secondary. No change of reader, target, sign or horizon after results. Absolute
classifier accuracy, fixed-head CE and single-image capability are not tested.

Inputs: prior accepted update count; loss on the first already-arrived batch;
parent reference-bank rank, variance, covariance and warm-relative CKA/cosine
movement. Secondary paid inputs: actual first-step virtual geometry changes and
unlabeled reconstruction-anchor change. Incoming loss is measured without updating;
no future frames, downstream scores or query labels enter features. Parent geometry
is expensive instrumentation, not assumed free. Runtime feature-only models use
no labels, but response fitting uses labeled historical targets.

Fixed models: zero effect, training-group mean, count/loss ridge, count/loss plus
parent health (primary), plus one-step preview geometry (secondary), plus anchor
change (secondary). Ridge penalty 10, intercept, training-only centering/scaling;
no parameter search. Report all fold predictions, source errors, MSE/RMSE, Pearson
correlation and sign agreement excluding true ties. Correlation is secondary to
MSE skill. Primary qualification requires >=10% macro-history MSE reduction versus
both mean and count/loss and positive Bonferroni-adjusted 95% paired improvement
lower bounds across six histories (family two, Student-t df5). This is only a
development screen, never independent-population qualification.

Use cached predictions and observations, reconstruct capability effects from raw
saved predictions/labels, verify audited inputs/hashes, and save a source snapshot
before analysis. No new SSL updates, readout fitting, downloads or original-output
edits. Output cap16MiB; launch reserve512MiB. Freeze future independent confirmation
separately. No automatic feature sweep, larger predictor, threshold or controller.

# Long singleton gate: health, admission threshold and allocation

Frozen on 2026-10-22 before collecting these longer trajectories.

## Questions and design

Does the notebook-68 J/rich gate preserve representation health over a longer
stream? Does relaxing its admission threshold improve adaptation? Is its value
in allocating training across clips rather than choosing individual frames?

Reuse its frozen regressors, feature definition, assay banks, support panels,
query images, optimizer and twelve common parent states. This is a targeted
follow-up to an observed result, not a fresh confirmatory test. No model is
refitted, no threshold is selected using these outcomes, and all arms are reported.

Each of six histories retains notebook 68's ten clips, followed by the matching
ten-clip histories from notebooks 69 and 70. The first six clips form the saved
common parent. Continue for 24 clips / 2,304 arrivals, six times the original 384.
Every clip occurs once in the new roster. Verify global clip disjointness and
disjointness from notebook 68's development, support and evaluation membership.
The stream clips were used in earlier studies; this is a longer recombination
of existing held-out material, not newly acquired recording sessions. Retain
the inherited day/night patterns. Six histories and two training seeds are paired.

Every real SSL update uses one image. No replay or gradient accumulation.
Preserve notebook 68's stream augmentation seeds on its original prefix and
continuation; extend the same namespace to later positions.

## Policies

The gate predicts delta-J, not class probability. Admit when prediction < threshold.
Use the original threshold t (approximately -0.001445), t/2 and 0. Moving toward
zero relaxes the required predicted improvement. Zero still requires a predicted
improvement. These are three independent evolving trajectories; their realized
admissions need not be nested because their learner states diverge.

Controls include accept-all, freeze and independent Bernoulli policies with fixed
probabilities 0.10, 0.25 and 0.50. Their complete schedules are fixed before any
gate runs and use neither gate counts nor test metrics.

For each gate also run two explicitly retrospective controls. Global-budget
random selects uniformly from the entire continuation with exactly that gate's
total count, without access to its per-clip allocation. Clip-budget random selects
uniformly within clips with exactly that gate's per-clip counts. Neither is an
independent deployable random policy. Global versus clip matching distinguishes
allocation opportunity from within-clip selection, but differences include state
and timing effects and are not a clean additive causal decomposition.

## Measurements and interpretation

Measure at the common fork and every 96 arrivals on identical fixed banks:
effective rank, feature variance, covariance, CKA, cosine, feature norm,
reconstruction loss and the composite J. Measure both the selector bank and the
separate evaluation query bank. Read fitted linear-probe and kNN accuracy and CE
on the same 400 query images using the three inherited support panels, including
day/night slices. Plot against arrivals and cumulative accepted updates. Retain
per-update augmented training reconstruction loss separately: changing diets make
that loss an unsuitable substitute for the fixed-bank reconstruction assay.

The original 384-arrival horizon is a predeclared intermediate endpoint; the
primary new endpoint is 2,304 arrivals. Report every threshold and random rate.
Report paired differences in accuracy, CE and health, update counts, per-clip
allocations, live feature range violations and charged policy time separately
from measurement time. No supervised fine-tuning or disposable continuation
branches are needed to answer this study's questions.

Whole-history bootstrap intervals retain both seeds and all panels and checkpoints.
They are exploratory, unadjusted, and conditional on one warm initialization and
reused reader panels. Full-update calibration and extended time coordinates may
extrapolate substantially; log that rather than silently clamp features.

## Storage and verification

Reuse existing frame tensors, banks and parent states by reference. Download no
videos. Save compact decisions, fixed-bank assays, reader probabilities and final
states; keep only one resumable in-progress state per active trajectory. Stop
before a checkpoint if free storage falls below one GiB. Cleanup may remove only
redundant source MOV cache files after both their original hash and decoded-frame
hash have been verified, with an itemized cleanup/reacquisition manifest.

Before the full launch, run focused tests and a GPU smoke test covering every arm,
threshold, counterfactual schedule and observation immutability. Verify exact
reproduction of notebook 68's gate and full policy over the first 96 arrivals.
At completion reconstruct decisions, metrics and matched counts independently,
check finite arrays and final fingerprints, and generate notebook 71 with actual
curves. Resume only from matching input/source locks. A failed check stops the run.

# New-source singleton forecasts beyond known clip transitions

Freeze before source acquisition and before any new SSL outcomes. Earlier
expected-effect forecasts are known. The user authorized the proposed new-source,
three-seed transition-controlled test on2026-10-21. Primary target is expected
isolated fixed-bank reconstruction effect; all seven health metrics are retained
to preserve the preceding multioutput ExtraTrees recipe, as secondary outcomes.

Source design: six new highway histories with the same six day/night keyframe
patterns as the preceding held histories. Each contains12 disjoint clips and
96 timestamp-selected frames per clip at16Hz (first6seconds, RGB64). Metadata
classifies the10-second keyframe only and is not asserted to label each training
frame. Select36 day and36 night highway clips by SHA256(namespace:clip_id),
from the original audited warm-disjoint eligible pool. Exclude every clip ID in
known-video inventories, existing output path inventory and preceding signal
input lock (including assay/reader sources). Save the actual exclusion set,
ranked candidates and inventory. Warm/validation membership is inherited from
the audited pool, and source metadata is rechecked. Source independence is at
clip level; recording/session identities are unavailable.

Use the same pinned community archive/index and range-limited fetcher semantics.
Intake budget2GiB; per-member128MiB. Preserve source CRC, SHA256, decoder logs,
timestamps and resized-frame hashes. Up to6 deterministic reserve clips per
lighting stratum may replace only undecodable/too-short/incorrect-dimension
sources BEFORE any SSL. Retain failure evidence. Network errors retry the same
member and never justify scientific replacement. Never select on SSL outcomes.

All trajectories start from the identical established warm MAE well. Train each
of six histories with seeds5003,5009,5011:18 trajectories,1,152 singleton updates
each. These seeds change ongoing augmentation/training randomness, not the warm
initialization. Arrivals receive SHA256(namespace|history|origin|seed) seeds via
the existing deterministic seed helper. No gradient accumulation. At31 origins
128:32:1088, fork the complete live state. Compare six independently seeded
current-image singleton branches with one explicit-zero-gradient AdamW branch.
The mean difference estimates the current image-gradient contribution beyond
momentum/decay, not beyond skipping. All observations restore modes/RNG and never
alter the live stream. Save raw bank features/reconstruction for all branches.

Pre-update base state inputs are health7, ordinary last health difference7,
optimizer6, arrival age1, previous sampled expected effect7 in the same history
and seed (28features). Known transition inputs add the boundary indicator and
frames since the last source boundary (30features). Image inputs add19 established
pixel/reconstruction/embedding summaries (49features). Boundary status is source
metadata available before the update, not an outcome-inferred change point. No
keyframe semantic labels, current counterfactual outcomes or gradient previews
enter inputs. Earlier expected-effect covariates assume costly post-update
counterfactual labels; this study does not solve deployable label acquisition.

Fit one frozen128-tree ExtraTrees model per training seed across all six history
prefixes (72labels per seed, update ends<=512). Preserve depth6,min_leaf5,
allfeatures, random_state84631, singleworker, and training-only input/target
standardization. No tuning or model selection. Compare zero effect,
transition-only prefix means for phases0/32/64, frozen state+transition trees,
and identical trees with image features added. Also save state-only/no-transition
trees and prefix mean/previous-effect controls for diagnostics. A phase mean is
a strong explicit transition-time baseline; all phases occur in the prefix.

Score only origins576:32:1088:102cases per seed,306total, including108 boundary
and198 within-clip cases. The test begins at the start of clip6, so tested clip
IDs are disjoint from fitted clip IDs0..5. Origins512/544 are not fit/test labels
but their matured effects can update covariates. Frozen refers to coefficients;
all models receive current causal observations. All same-time predictions occur
before observing the same-time target.

Primary comparison: image+transition+state versus transition+state trees on
reconstruction RMSE and MAE. Also compare with phase mean and zero. Report pooled,
per-seed and within-clip/boundary splits. Prespecified qualification in a context
requires both point errors lower against all three controls (transition+state,
phase mean,zero), both errors lower against transition+state in EACH seed, and
paired95% whole-history-cluster loss-difference intervals with upper endpoints
below0 against all three controls. Cluster six independent source histories,
keeping their three trajectory seeds together; do not treat seeds or frames as
independent sources. Report contextual gates separately; a join-only pass permits
only a join-specific conclusion. Intervals are unadjusted exploratory95% and six
clusters limit precision. No downstream-accuracy utility claim follows.

Secondary sign check: compare signed predictions with training-prefix majority
sign (global and phase-conditional), report accuracy and balanced sign recall.
Score independently against each disjoint3view target half; do not call their
disagreement an exact6view noise floor. Retain all seven metric scores without
selecting a favorable target after inspection.

Verification: unit tests for causal labels, prefix-only fits, transition timing,
seed isolation and history-cluster bootstrap. Independently reconstruct all raw
geometry, labels, pre-update features, phase means, frozen fits and metrics.
Independently replay all18 full streams and the actual/zero branches at origins
480 and960 in every stream/seed; verify complete checkpoints, losses, fingerprints
and finals. Freeze source/config/protocol hashes before collection. Preserve
all preceding study manifests. Deliver an executed notebook, plots, readable
report, decisions and artifact hashes. This tests new clip sources, conditional
on one warm initialization and the existing health bank.

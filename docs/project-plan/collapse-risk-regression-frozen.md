# Regression forecast of a calibrated collapse failure — bounded development screen

Frozen before new SimSiam outcomes are collected. This is the first stage of the user's
chosen failure-prevention direction (N-E/N-G), not another MAE accuracy predictor.

## Question and scope

Can past, label-free observations predict crossing a calibrated projector-collapse region
eight optimizer updates ahead, without flagging intact adaptation? Phase 0 P0.2.4/5 supplies
the instrument and controls, not transferable numerical thresholds. Recalibrate at this
BDD/32-pixel vehicle. Deliberately weakening the objective is a positive control, not an
incoming-image property. No ordinary image acceptance or general accuracy claim is allowed.

## Fixed vehicle, cohorts and controls

Use existing cached BDD frames only, no downloads or additional labeling. Six reused
`fresh_00..05` development cohorts, seeds 7001/7003, all paired arms share initialization,
warm state, optimizer and update-indexed augmentation RNG. Source/session independence
beyond the manifest's clip identities is unknown. These are development groups, not fresh
population confirmation. Cohorts contain disjoint clips; no windows or seeds are independent
replicates in intervals.

TinyViT: 32-pixel RGB, patch8, width96, depth4, heads3, MLP ratio2. SimSiam projector
hidden256/output128, predictor hidden128, existing augmentation (batch-shared draws as
in Phase 0). AdamW lr0.001/weight decay0.05, batch128, constant learning rate. New
random initialization, not the earlier MAE or pretrained DINO. Hydra instantiation.

Per cohort: periods0..3 (eight clips), frames0,4,..508 produce a 1024-image IID warm
pool. Warm for320 updates (40 pool passes in expectation, sampling with replacement).
Periods4..9 (12 clips), frames0,8,..504 produce768 continuation images. Periods10..11
(four clips), frames0,8,..504 produce a fixed256-image health bank. Banks never train.
All frames are antialiased bilinearly resized64→32. Each nominal continuation batch is
an IID sample from its768-image pool; this is controlled calibration, not a chronological
single-pass claim. Low-diversity draws from the first continuation clip's64 images.

Nine 128-update arms from each warm parent:

| Arm | Objective / data change | Role |
| -- | -- | -- |
| healthy | Intact SimSiam | Reference |
| heavy | Intact, crop minimum0.1/jitter strength1.5 | False-alarm stress, not certified beneficial in advance |
| lowdiv | Intact, one-clip pool | False-alarm stress, may genuinely degrade |
| blend005 | Loss-blend severity0.05 | Fitting |
| blend015 | Loss-blend severity0.15 | Fitting |
| blend100 | Loss-blend severity1 | Forced-collapse calibration pole and fitting |
| soft015 | Soft stop-gradient severity0.15 | Mechanism holdout |
| soft050 | Soft stop-gradient severity0.5 | Mechanism holdout |
| soft100 | Soft stop-gradient severity1 | Mechanism holdout |

Read unaugmented projector and backbone embeddings at0,8,..128. Restore all module
modes and RNG; verify complete-state invariance. Replay every warm trajectory and
every arm exactly, including BatchNorm buffers, AdamW state, CPU/CUDA RNG, losses and
embedding arrays. Save complete CPU warm states for recovery, compact arm artifacts and
hash manifests. Resume only completed records with valid hashes under an unchanged lock;
incomplete arm work is rerun from its warm parent and counted separately if a crash occurs.
No scientific retries with changed parameters.

## Downstream anchor

Reuse the existing900-image development evaluation cache, labels/domains/query400 and
three disjoint-from-query50-example support panels. Resize to32. At random initialization,
warm parent, and each continuation endpoint read backbone embeddings. Fit a fixed ridge10
linear classifier (training-support-only per-feature scaling, unpenalized intercept;
one-hot class targets) for each panel, evaluate balanced/day/night accuracy on query400.
Labels enter only this offline readout. This is not the earlier fine-tuned classifier and
does not establish a universal geometry/accuracy mapping. Store raw embeddings/predictions.
Check clip/keyframe identity overlap using the source manifests before collection.

## Regression and held-out decisions

At origins16,24,..120 predict the next8-update change in log(projector RankMe/warm
RankMe). Fixed ridge10, training-only standardization, unpenalized intercept. Seven
features: current log-rank ratio, its last two8-update increments, log(normalized feature
variance/warm variance), CKA drift from warm, mean past8 training losses, origin/128.
No knob, seed, cohort identity, labels, gradients, preview steps or future values as inputs.
No candidate-image features: this first screen is a continuation warning, not an image filter.

Leave one complete cohort out (both seeds, every arm and origin). Fit only the other five
cohorts' healthy/heavy/lowdiv/blend arms. Never fit any soft-stop-gradient outcomes.
Compare zero-change persistence and last-increment extrapolation. Primary forecast scores
use only held-cohort soft arms; blend and healthy results are separate descriptive reads.

For each held cohort derive healthy and forced endpoint/warm-rank median ratios from the
other five cohorts only. Boundary = geometric mean of those two medians. Never tune a
boundary on held-cohort health or downstream accuracy. A first sustained event is two
successive bank readings at/below that boundary. Eligible first events start at24..120,
allowing an origin with two past increments and one full8-update warning interval. Earlier
events are reported as unforecastable by this observation schedule, not hidden as exclusions.
Primary warning at origin: current rank above boundary and forecast next rank at/below it.
Report trajectory recall and8-update lead, eligible/event counts, missed early failures,
and false warnings among nominal healthy opportunities whose target rank stays above boundary.
Report heavy/lowdiv warnings alongside their actual health and probe outcomes; do not assume
these stressors stayed healthy. Secondary buffers0.1/0.2 in log-rank units illustrate the
tunable tradeoff; no post-hoc operating-point selection or confidence interpretation.

## Gates and stopping rule

All stages are reported even if prerequisites fail, to localize the bottleneck. Promotion
requires **all** of the following, with no secondary rescue:

1. Healthy/forced endpoint RankMe ratio at least2 in all12 matched parent pairs, and
   healthy endpoint median rank/warm rank at least0.8. In other words, the nominal pole
   must not simply collapse along with the positive control.
2. Healthy-minus-forced downstream accuracy at least1pp on average and positive ordinary
   95% t lower bound across six seed/panel-averaged cohorts. Separately report healthy
   versus warm; call a false-alarm control *beneficial* only if this contrast has positive
   lower bound. If none is demonstrated beneficial, the beneficial-adaptation prerequisite
   remains open and promotion fails.
3. On held soft mechanisms, ridge MSE at least10% below both persistence and extrapolation,
   with positive paired cohort improvement lower bounds (Bonferroni family2, df5).
4. At least six eligible soft-mechanism first-event trajectories, at least80% episode
   warning recall, nominal healthy false-warning rate at most5%, and no cohort with
   more than10% nominal healthy false warnings. Early-event coverage must also be at
   least80% of all soft-arm event trajectories (eligible / all events).

No live guard is qualified here: even a pass only authorizes a proposed, separately locked
paired train/hold controller experiment. No physical closed-loop training or intervention
benefit is inferred from shadow warnings. Natural-image failure prevention remains a later
study with an intact objective. These six reused groups cannot establish confidence for
deployment. A failure stops this screen without extra severities, longer horizons or model sweeps.

## Budget and artifacts

12 warm parents +108 continuation arms:17,664 scientific SSL updates and the same17,664
exact-replay updates. Synthetic preflight/test work is separate. No downloads. Cap output
at512MiB and require2GiB free reserve before collection. Expected artifacts: input lock,
source snapshots, role manifest, per-parent states, per-arm health/evaluation arrays,
raw regressions/predictions, summary, independent numerical verification, report figures
and a saved summary notebook. Preserve all earlier experiments.

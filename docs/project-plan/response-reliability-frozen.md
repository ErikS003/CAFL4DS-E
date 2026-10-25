# Repeated local response versus longer-term benefit — study53

Known-cohort retrospective development after study52. Primary question: do
responses averaged over recent batch contents and independently seeded masks
predict ORIGINAL384-update replay-minus-current benefit more reliably than a
single probe? Same72 states, all3 seeds x3 sources x4 contexts x2 pre-diets.
No new histories, downloads, future trajectories or fresh-confirmation claim.
Freeze code/config/tests/synthetic engineering and six figure fixtures plus all
consumed inputs before real reconstruction/probes or response analysis.

Reconstruct each complete fork from its saved historical learner/memory and
original128-step trace. Every update metric/hash/full learner fingerprint and
terminal fork fingerprint must reproduce exactly.9216 physical reconstruction
updates. No silent retry; preserve interrupted work and count all starts.

Cross observed current batches119,123,127 with mask seeds195200+i,195300+i,
195400+i for state index i. Batches are ALREADY ARRIVED before the fork; never
read128+. Each balanced replay batch uses the original schedule at its batch
index:16 current/16 historical images. Thus the batch factor includes both
current content and historical subsampling; it does not isolate these effects.
The same mask seed is used across actions and batch settings within a state.
The seeds are different draws, not evidence of statistical independence across
the whole shared cohort. Temporal batches also need not be independent scenes.

Each of9 settings proposes one current and one balanced inherited-AdamW update
on independent model copies:1296 new proposals. Match each encoder displacement
to0.5 times the state's original pre-action trailing16 actual-step norm, using
the established routine. Keep all other settings fixed; no dose search. Repeat
the batch123/mask1 pair in EVERY state (144 audit updates), require exact full
proposal fingerprints and metrics. Freshly recompute batch127/mask0 and require
exact proposal/feature/response reproduction of study52's primary half-step
anchor. Total physical SSL10656=9216+1296+144. These are reconstruction/probe
costs, not new future384-update action trajectories. No reader optimizer fits.

Zero and every perturbed view infer the support-only200-image tensor twice.
Pad with its first24 support images to full32-image GPU batches and discard
padding outputs. Deterministic CUDAfloat32/TF32off; preserve modes and RNG.
Zero must exactly match cached study44/52 features; repeated features exact.
72*(2 zero passes +18 perturbations*2)=2736 passes,224 images each=612864
forwards (547200 support roles +65664 padding roles), separate from341k SSL
training-image presentations. Save ALL support features, fixed-reader responses,
proposal fingerprints/norms/cosines, audit receipts and journals. Save raw
displacements only for the predeclared repeated batch123/mask1 pair (144 vectors)
for independent encoder/norm reconstruction; others are checked inline and
fingerprinted, with no full vector archive. This bounds output<=512MiB and
keeps1536MiB reserve. Do not delete active scientific artifacts.

Reuse study51's six fixed128-label pools and4-fold96-label OOF heads at both
lambda.01(primary)/.001(sensitivity). Held-out32 rows per fold; validation
support images remain identity-disjoint from the old2048-image benefit query.
No probe refitting, bank choice, support selection or future labels in collection.
Same study52 response definitions: separation/preference changes in BASELINE
within-class-SD units, log SD ratio, fixed-baseline-threshold boundary-mass
change. Eight axes (day/night), all6 pools/both priors. Contrast replay-current.
Primary averages all9 probe contrasts AND6 support-pool descriptors at.01.
Average descriptors, never logits belonging to different support identities.

Before loading old targets, seal every feature/response artifact. Finite3x3
batch x mask ANOVA of support-pool means reports total/batch/mask/interaction
squared variation, checked by exact identity. These are descriptive finite-grid
components, not population noise estimates, ICCs or confidence intervals.
Also retain pool variability and disjoint-batch/mask comparison correlations
across72 state means; repetitions are measurements, never648 independent states.

Use original study43 mean-over12-readouts, fixed-reader night-highway benefit
after384 future updates as primary, joint/all7 components secondary. Reconstruct
the average from saved per-draw benefits and check old targets. Preserve utility
definition; this is not a new confirmation cohort or an extension of horizon.
The paid multi-probe assay also has more pre-action compute than the single
probe, so no equal-compute controller improvement claim is possible here.

Use EXACT study52's seven model designs, ridge10, train-only standardization,
36 seed AND source AND context exclusions (24 train/2 test), all14 targets.
Primary model static_response_added, unchanged23-predictor design; no tuning.
29 panels: all9/.01 (PRIMARY); original anchor; all9/.001; other8 single probes;
3 batch means;3 mask means;3 leave-batch-out;3 leave-mask-out;6 per-pool means.
Every panel retained; no best-setting promotion.7308 forecast ridge fits, each
independently verified with augmented least squares. Anchor forecasts/primary
cheap/static predictions must reproduce study52 exactly (within1e-10).

Retain study52 primary development screen:>=10% MSE reduction versus BOTH cheap
and static, with improvement in EVERY seed and source for each comparison.
Additional averaging-reliability screen requires that screen PLUS lower primary
MSE than the anchor model AND every leave-batch/mask-out panel beating cheap
and static in pooled MSE. Report every clause, whether it passes or fails.
For readout sensitivity, evaluate the FROZEN mean-target forecasts against each
of12 old readout-specific benefits: no per-draw forecast refits or selection.
Report all contexts including4, both diets, classes and shadow policies; all
support/prior/panel variation is conditional on the shared known cohort.

Six predeclared figures: design/exact anchor/audit; finite-grid batch-mask-
interaction and pool variation; response profile by context/diet; MSE and
seed/source stability; single/average/deletion/prior/pool forecast robustness;
per-readout benefit sensitivity and shadow class values. Engineering must
exercise factorial aggregation/ANOVA/forecasts and padded GPU restoration,
paired content/mask proposals and exact repeats on synthetic images/model.
Inspect six synthetic plots before freeze and six real plots before final
delivery; preserve first exports before adding observed findings, execute saved
notebook reconstruction. No causal-health, alarm or controller qualification.

Engineering first completed20 synthetic SSL/38 padded passes, then caught a
leave-factor-out panel shape error from mixed NumPy advanced indexing. Preserve
that attempt/source snapshot. Replace with explicit axis selection and require
all29 panel shapes in unit/engineering checks before freeze. The successful
rerun adds20 SSL/38 passes:40 synthetic SSL/76 passes/17024 image forwards in
total, separate from scientific10656 SSL/2736 passes/612864 forwards.

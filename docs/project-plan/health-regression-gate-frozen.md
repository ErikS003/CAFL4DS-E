# Measured and regression-predicted health to training utility

Authorized 14 October 2026: create the next experiment using regression, not direct
health gradients. Freeze protocol, sources and tests before constructing new targets
or fitting models. This is a bounded retrospective study on already-inspected data.
It is not a new-source confirmation or a physical adaptive-filter rollout.

## Question and scope

First ask whether measured health responses add information about beneficial versus
harmful training beyond the state already observed. Then ask whether regression
forecasts preserve enough of that information for a shadow stop/continue decision.
No health direction is assumed beneficial. Phase 0's positive collapse calibration
was specific to JE projector surfaces; its MAE quality-reader negatives remain
load-bearing. This study neither transfers those thresholds nor retries collapse.

The earlier bridge changed one candidate batch and used optimizer/health derivatives.
This study changes the outcome contrast to eight additional updates versus hold,
uses measured responses as an explicit diagnostic first, and uses only fitted
regressions for prospective prediction. It does not claim individual-image utility.
The prior accumulated-CKA warning used one past movement feature to predict a longer
continuation; here all five forward responses are tested jointly with nested fitting.

## Fixed cohort, actions and labels

Use only the 54 `six_stream` rows in `outputs/health-warning/20261013-gpu`:
six histories, three checkpoints, seeds 5003/5009/5011. Exclude all nine historical
controls from fitting and scoring. Cached state/embedding replays and raw classifier
readouts must have passed their original main and independent audits.

Decision state is continuation update eight. Action train accepts the next eight
chronological batch-32 updates under the unchanged MAE/AdamW recipe. Action hold
leaves the model/optimizer fixed at eight while the same arrivals pass. Utility is
the copied-classifier accuracy at sixteen minus at eight, in percentage points.
This is conditional few-label adaptability, not instantaneous online accuracy.
Primary is balanced 50-label/100-step fine-tuning, averaged over three fixed support
panels and 400 queries. Day/night are separate outputs. Linear/kNN are secondary;
no CE endpoint is declared because probabilities are not retained.

Compute variance, covariance, RankMe, CKA drift and cosine drift in float64 from
cached bank embeddings at 0/4/8/16. CKA/cosine reference the fixed local parent at
zero, not the warm model or a moving reference. Use the same 256-image bank.
Forward response is H16-H8, divided by H8 for variance/covariance/RankMe and absolute
for the two drift channels. Hold response is zero on this fixed reference bank.

## Fixed regression menu and availability

Runtime features contain exactly seventeen scalars: total accepted updates through
eight, last-four actual past SSL-loss mean, five H8 levels, and the two preceding
four-update health changes (H8-H4, H4-H0), normalized like the forward response using
H8. No gradient norm, health gradient, optimizer displacement, virtual update,
future frame, future loss, classifier label, source ID or seed ID is a runtime input.
Candidate future images are not encoded or scored. This is a block-pacing screen.

Fixed ridge penalty ten, training-only mean/std normalization, unpenalized intercept,
multi-output ordinary scalar regressions. No hyperparameter search. Models:

- Training mean and cheap count/loss baselines.
- History: direct utility regression using all seventeen runtime features.
- Measured: history plus the five actual forward health responses. **Post-action
  diagnostic only**; these responses are never available to a prospective decision.
- Predicted: history plus five regression-predicted responses; prospective chain.
- Plug-in: apply the measured-response mapping to forecast responses; secondary.
- Response-only and measured-response × current-health interactions: secondary,
  never replacements for a failed primary measured-response model.

Use outer leave-entire-history-out folds, excluding all checkpoints/seeds of g.
Fit the health forecaster on the other five histories' responses. To generate
utility-training features for history k, fit another health forecaster excluding
both g and k. Fit the utility stage on those inner out-of-history forecasts, and
score g using its outer forecast. Save all masks, generators, scalers and coefficients.
The direct history model uses the same available information; forecast health is
a learned transformation, not an independent new information source. A chain gain
may reflect supervision/regularization and must beat that direct baseline.

## Shadow policy, primary tests and qualification

Primary accept rule: predicted balanced utility >=0.25pp and predicted day/night
utility each >=-1pp; otherwise hold. These are point predictions, not calibrated
confidence bounds or a safety guarantee. Report good/bad/neutral actual outcomes:
good means balanced >=0.5pp with both domains >=-1pp; bad means balanced <=-0.5pp
or either domain <-1pp. Report false accepts and missed beneficial opportunities.

Compare the prospective chain with always train, always hold, the direct-history
regression policy, and an exact expected matched-random selection baseline. The
last baseline takes each held history's realized acceptance fraction times its
mean utility. It is an offline matched-quota expectation, not a causal online policy.
This distinguishes selecting useful decisions from simply accepting fewer blocks.

Six history means set uncertainty; seeds and checkpoints are repeated measurements.
Report ordinary intervals plus Bonferroni 95% paired t intervals, df five, across
sixteen primary scalars: measured and predicted balanced MSE improvements versus
both mean/history (four), chain policy balanced gains versus the four comparators
(four), and day/night gains versus those comparators (eight).

Require all of the following for a development lead:

1. At least two parent/checkpoint means with balanced utility >=0.5pp and both
   domains >=-1pp, and at least two with balanced utility <=-0.5pp. Average seeds
   and panels within parent; these counts alone do not establish repeatability.
2. Measured-response and predicted-response models each reduce MSE at least 10%
   versus mean and direct history, with positive adjusted improvement bounds.
3. Prospective policy gains at least 0.25pp balanced versus every comparator,
   with positive adjusted lower bounds; day/night lower bounds each >=-1pp.

All stages are fitted and reported regardless of gates, to locate the bottleneck.
A failed measured-information gate prevents a pipeline claim, even if another
secondary model appears positive. Report cross-seed and cross-support sign
stability, per-history errors, health forecast skill, coefficient variation and
the predeclared secondary threshold curve 0/0.25/0.5/1pp. Do not select a threshold
or secondary winner from these outcomes. No health-sign or feature sweep follows.

## Integrity, cost and delivery

Zero new SSL updates, zero new downstream classifier fits, no gradients or virtual
updates in this experiment. There are 36 multi-output health fits and 36 multi-output
utility fits: 72 total. Independent sklearn-SVD verification duplicates these CPU
fits separately. Unit-test fits are separate infrastructure work. Output cap 16MiB
before figures; source/result snapshots and saved predictions support an immutable run.
No download, model/pixel cache, old-data deletion or artifact replacement.

Pin all used records, bank embeddings, raw predictions, labels, audit files, source
versions and this protocol before target construction. Reconstruct all raw targets
and source-exclusion rules independently. Test held-history label/response leakage,
decision boundaries, contrast semantics and corrupted evidence handling. Deliver
the full report, saved scientific figures and a summary notebook. A successful
screen would still need unused histories, uncertainty calibration and actual repeated
rollouts. A failed screen closes this regression recipe without automatic new training.

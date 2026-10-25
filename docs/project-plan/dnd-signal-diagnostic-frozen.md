# Cached D/N/D warning-value diagnostic

This is the user-authorized next diagnostic after the completed transfer test.
The collection and its aggregate outcomes have already been inspected. Freeze
this analysis before deriving its new targets or fitting its statistical models;
this is exploratory reuse, not preregistered fresh-source confirmation. Preserve
all earlier scientific files and notebooks. No network, SSL updates, new task
labels, canonical classifier fits, pause/resume branches or deployed controller.

## Counterfactual and horizons

Use the eight complete full-training trajectories from the audited transfer run,
one SSL seed, three shared fixed support draws and the reused query population.
At a checkpoint t, compare holding the current model for H arriving updates with
continuing its exact saved full-training prefix through t+H. Both branches share
the complete state at t and the same subsequent arriving images. Holding leaves
the model unchanged; this analysis cannot reconstruct resumed training after a
pause or a train/skip decision followed by matched further training.

Primary horizon H=32; H=16 and64 are sensitivity analyses. Use only scheduled
monitor clocks t>=140 with an exact observed t+H<=444. Do not interpolate, truncate
an unavailable horizon, or treat a late censored alarm as safe. Include every
eligible full-training checkpoint, even after the earlier guard would stop:
these are hypothetical decisions on the full-training path, not reachable states
of an already stopped policy. All histories share each horizon's checkpoint grid.

Average the three canonical readout scores at each checkpoint first. Targets are:

- Worst observed loss from capability at t: max over day/night and observed
  checkpoints in (t,t+H] of max(0, accuracy(t)-accuracy(s)). This protects capability
  already present at the decision. It differs from decline from a future running
  peak, which can count forfeited future learning as prevented forgetting.
- Final balanced, day and night gains: accuracy(t+H)-accuracy(t). Positive gain
  is learning holding would forgo; a negative gain is damage holding would avoid.

Use all saved offline observation clocks inside each interval for the loss
target. Scores are noisy few-label readouts, not directly observed latent
capability. More observations can mechanically increase a maximum; compare models
within each horizon. Recompute targets separately per support draw to report
reader disagreement; support draws are not additional independent histories.

## Information available at the decision

Compute rank drop, normalized-variance drop and centered linear CKA distance
against each history's fixed update132 bank, using the earlier reading function.
For each instrument use its current level and change since the immediately prior
scheduled observation eight updates earlier. No future embeddings, task scores,
domain/scene metadata, transition clocks, stop times or history identity enter
features. Time is the ordinary update count. References never follow score peaks.

## Held-history prediction and age controls

Leave one entire history out in each of eight folds; fit every horizon and target
separately using only the other seven histories. No random row split and no
hyperparameter search. Constant baseline is the training-fold outcome mean. A
strong age-only baseline is the training-fold mean outcome at the exact same
checkpoint. Every held-out history shares this age grid with training histories.

For each instrument, and all instruments jointly, add a linear health correction
to the age baseline. Demean training features and targets separately at each clock
using only training histories; standardize the demeaned features with training
statistics. Fit ridge on mean squared error with penalty1, no unpenalized residual
intercept. Predict the age mean plus the correction from the held-out features
relative to the training-fold feature mean at that clock. Clip only predicted
worst loss to zero. Gain predictions are signed. This tests incremental health
information beyond a flexible age baseline. There are96 small statistical fits,
not classifier training. Save train history IDs, scaling and coefficients per fold.

Report each history's checkpoint RMSE and MAE, averaged over eight histories;
paired t95% intervals use eight histories, not overlapping windows or support
draws. Also show pooled descriptive risk precision/recall at a fixed1pp threshold.
The exploratory H=32 forecast screen requires >=10% lower mean risk RMSE than age,
a positive paired-history improvement interval, and improvement in >=6 histories.
All four health models are disclosed; passing is only a candidate-screen flag,
not qualification or permission to select favorable horizons. No automatic
pause/resume experiment follows this cached diagnostic.

## Alarms and opportunity cost

Analyze each unchanged10% instrument alarm with two consecutive scheduled
observations, and the original OR-combined persistence rule. Hypothetical warnings
may repeat; they are not a cumulative policy trajectory. Report observed-loss
precision/recall, signed terminal gains, positive balanced gains forgone, and the
fraction warning when balanced gain exceeds0.5pp or either domain gain exceeds1pp.
Call a warning a useful hold descriptively only when observed loss>=1pp and all
three learning-cost tolerances are met. These thresholds are fixed from the prior
capability tolerances, not fitted to these targets.

Compare warnings with the exact same number K of dates in each history selected
by that history's held-out age-only risk forecast (ties choose earlier dates).
Also report the exact expected outcome of uniformly selecting K dates. Neither
control uses held-out outcomes to select dates. Compare protected loss and
forgone positive balanced learning per eligible decision, with source intervals.
Do not sum overlapping-window gains into a realized policy or compute fictitious
training savings. Report each actual first alarm's outcome when its full horizon
is available and explicit censoring otherwise.

## Integrity and delivery

Bind the earlier input/complete/audit chain, raw scores and banks, and freeze this
protocol, config, implementation, focused tests, verifier and renderer before
analysis. Independently rebuild geometry, class-balanced raw probability scores,
targets, train-only regressions, exact-budget warning controls and screen flags.
Tests cover future-feature exclusion, full-horizon censoring, non-anticipating
targets, held-out outcome exclusion, clock-only null behavior and matched counts.
Produce a new numerical report, static scientific plots and an executed display
notebook, detailed study, concise result without result numbers and phase index.
Preserve all previous hashes. Interpret source intervals conditionally on reused
sources/readers; neither statistical forecasts nor warning comparisons establish
that a pause/resume controller will help on a changed training trajectory.

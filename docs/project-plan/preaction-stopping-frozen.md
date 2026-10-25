# Preaction prediction of cumulative stopping effects

The user authorized this separate follow-up after distinguishing retrospective
oracle choices from online decisions. Earlier stopping protocols did not authorize
this automatically; this experiment follows the new instruction, "yes, great!
continue." It preserves all earlier scientific files and numerical gates.

## Question and population

Can inputs available at episode start predict the day/night capability changes of
the five train-then-hold schedules? Reuse every fresh_00–fresh_05 history, parent
128, horizons 8/16/24/32, and the audited 512-draw cumulative-study outcomes.
Zero updates have exactly zero acquired-reference change. The skip comparator
holds throughout; each training prefix also holds after its chosen duration to
the same endpoint. This is duration selection, not marginal image admission or a
skip-and-resume controller. No new SSL training, downstream fitting or labels.

The outcome population, some features and oracle choices have already been
inspected. Freeze this analysis before fitting its new models, but describe it as
exploratory reused-data validation. Outer history exclusion is a fitting boundary,
not fresh confirmation. There are six source units, not 24 independent histories
or 512 independent models. The inherited longer-horizon precision failure remains
visible and cannot qualify protection or deployment.

## Inputs and targets

Only parent-128 features enter forecasts. Use fifteen label-free geometry readings:
warm-relative log rank, log normalized variance, log off-diagonal covariance,
CKA distance and cosine distance, on global/day/night monitor banks. Retain the
inherited coordinate convention and fixed 256-image monitor bank. Day/night bank
metadata are assumed available; city/highway task labels are never inputs.
Use the mean and linear slope of SSL losses at updates 120–127. Do not read future
losses, endpoint geometry, future image content, task-labelled probe scores,
source identity or measured response features into the predictor.

Predict two cumulative gains relative to the acquired parent encoder. Offline
labels supply these training targets and held-out scoring only. The original
pretrained capability is not a forecast target or protection claim in this pilot.

## Models and exclusions

Use fixed ridge penalty 10, train-only scaling and an unpenalized intercept, without
tuning. Let h = duration/32. Cheap controls are [h, h², h*past_loss_mean,
h*past_loss_slope]. The primary health design appends h times the fifteen parent
geometry readings. Baselines are zero change, the training-history mean at each
horizon, and the cheap controls. No feature subset is selected from these results.

Leave one entire history out in six outer folds, excluding all its horizons and
outcomes from fitting and normalization. Save linear outcome operators and fitted
coefficients. As a specificity diagnostic, fit 32 geometry shams using seeds
91601 + 1000*outer_fold + sham_index. Derange whole geometry profiles among the five
training histories, preserving horizons, real losses and real held-out geometry.
No held-out profile can be a training donor. Report the best sham separately per
domain as a stringent diagnostic, not a model selected for deployment.

For horizon-mean, controls and health, calculate a residual-envelope diagnostic:
within the five outer training histories, leave one out again, fit on the other
four, and take the largest absolute prediction residual separately per horizon
and domain. Forecast lower/upper values are the outer forecast plus/minus this
envelope. These are training-only sensitivity bands, not calibrated 95% intervals,
conformal guarantees or simultaneous path coverage. Do not estimate this envelope
from outer held-out errors. Do not fit an envelope for shams.

## Scoring and stopping

Report equal-history day/night RMSE, bias, per-history MSE and horizon-specific
RMSE. Compare the primary model against horizon means, cheap controls and the
best geometry sham. An exploratory prediction gate requires at least 10% MSE
reduction in both domains against all three, improvement in at least five of six
histories per comparison/domain, RMSE <=1pp and absolute bias <=0.5pp in both
domains. Also report paired source t ranges adjusted across the six comparisons;
they are descriptive because the outer fits overlap and sources are reused.
The gate additionally requires positive lower source ranges. Even a numerical
pass does not qualify a controller; no retuning or source removal follows failure.

For each real predictor, choose a duration entirely from its outer-held forecast.
At every visited boundary require both domain forecast gains >=-1pp, then require
terminal balanced forecast gain >=0.25pp. Maximize tanh(balanced_gain/0.5pp), with
fewer updates breaking ties; otherwise hold. Repeat using residual-envelope lower
values, and separately report actual point/conditional acquired-budget outcomes
for each choice. Compare with hold, always train, the acquired-reference oracle,
and exact work-matched random assignments of each rule's six durations (all 720
permutations, duplicates retained). Do not assume random assignments are safe.
Report useful choices, resolved losses and unresolved outcomes without treating
an unresolved result as protected. Never use outer outcomes to adjust a decision.

## Integrity and delivery

Create a separate output with source snapshots and immutable input hashes before
fitting. Bind the inherited completed result and audits, parent inputs, monitor
metadata and new source/tests/protocol. Save extracted causal inputs separately
from targets, all fold membership, sham donor maps, predictions, sensitivity bands,
selected schedules and actual scores. Independently reconstruct parent features,
ridge predictions via augmented least squares, nested envelopes, decisions and
random scores. Test poisoning of held-out labels and future losses, sham exclusions,
training-only normalization and every-boundary budget enforcement. Charge zero
new SSL updates and zero canonical classifier fits. Update the phase routing,
concise result and detailed study; render and visually inspect figures. Do not
launch another experiment or activate admission automatically.

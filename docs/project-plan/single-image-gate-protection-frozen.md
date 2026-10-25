# N74 — one corrected gate and two protection variants

Run exactly three evaluated trajectories, fresh_00 and seed 5003, from N73's
identical complete 576-update parent. Offer the identical 57,600 continuation
images, without downloads or a new accept-all trajectory. Reuse N73's saved full
exposure results as a descriptive reference. This is an exploratory pilot using
previously inspected streams and evaluation panels, with no across-run inference.

The common J predictor excludes optimizer_step_mean and arrival_index. Refit its
unchanged three-view delta-J target from N68's saved development calibration,
using seed 5003, four fit histories and two validation histories. Choose ridge
versus histogram boosting by held-history MSE. Bound retained live coordinates
to fitted development ranges. Freeze the smaller of zero and the 60th percentile
of validation predictions. Refit the chosen family on all six development
histories. This corrects unbounded clock extrapolation; it does not establish
calibration on long filtered states. No online predictor fitting is included.

Policies:

1. corrected_J: accept whenever the corrected predicted delta J is below the
   frozen development threshold. J's objective and weights are unchanged.
2. corrected_covariance: require the same J condition and evaluate a real
   singleton trial update. Accept only if fixed-selector-bank off-diagonal
   covariance remains at most 1.25 times the warm covariance reference. Otherwise
   restore the complete model, optimizer, modes and CPU/CUDA RNG. This is a fixed
   upper-band constraint, equivalent to a hard barrier added to J, not a claim
   that covariance is a validated task-risk surrogate.
3. corrected_development: require the same J condition and evaluate a real
   singleton trial update. Accept only if mean refitted-probe accuracy on the
   200-image DEVELOPMENT query bank remains within 0.02 (two percentage points)
   of each confirmed anchor: overall, day and night. Initialize anchors at the
   common parent. Every 384 arrivals, raise each anchor to the maximum of its
   previous value and the minimum of the last two checkpoint accuracies. This
   avoids anchoring on a single high checkpoint. Restore the entire learning
   state after a failed trial. There is no gradient training on development
   labels; they control selection, making this task-informed filtering.

The two protection variants use measured trial constraints, rather than a new
forecast of the protection metric. Accepted trials are retained as the one real
SSL update; rejected trials are counted and charged, but do not advance the live
optimizer. Each scored arrival has at most one trial. There is no replay,
accumulation or extra accepted-image update. Thus this pilot tests stronger
observed development protection before approximate risk prediction or speed
optimization. The constraints can reduce plasticity or cause freezing.

Thresholds, caps, tolerances and confirmation logic are predeclared design
choices, not optimized on N73 or the final evaluation bank. All three policies
retain the existing 86-coordinate feature computation; only the two clocks are
excluded at predictor input. Log predictions, exclusions/clipping, J admissions,
trial results, protection decisions, actual updates, development scores,
anchors, covariance and costs. Log soft error at temperature 0.05 as a diagnostic;
it is not the probe-protection decision in this pilot.

Measure separate final-query linear/kNN accuracy and health every 384 arrivals.
These readouts cannot affect admissions or anchors. Compare at shared measured
clocks, also plotting each available trajectory. Preserve resumable two-generation
commits, including the protection anchors and all traces. Smoke-test all three
policies and interrupted-resume equivalence before the long run. A rejected GPU
trial must reproduce the original full-state fingerprint. Reserve two GiB free.

Report maximum drawdown, lower-tail and final accuracy, time below the parent and
prior best, recovery behavior, admissions and rejected trials. This single pilot
does not establish generalization, nor guarantee accuracy on the evaluation bank.

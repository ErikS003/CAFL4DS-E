# Singleton filter: objective by feature experiment

User-authorized follow-up to study 66, fixed before new outcomes. Batch 1,
one update per admitted arrival, no accumulation or replay. No prior artifact
or scientific source is modified. This is a development experiment on fresh
clip IDs, not proof of recording-session independence or general deployment.

Cross two prediction targets with two input designs. The old target is actual
after-image minus before J (reconstruction/warm plus .25 warm-relative rank
and .25 variance shortfall). The new target is actual change in balanced
linear-probe CE on a separate 200-image labeled DEVELOPMENT bank, averaged over
three existing 50-example support panels. Refit probes at each measured state.
Predictions target the mean of three training views. All reference objectives
and readout masks are fixed. Linear-probe accuracy remains the primary final
evaluation, with CE, kNN, fine-tuning, health and day/night as secondary outcomes.
No claim that CE improvement necessarily improves accuracy is assumed.

The baseline input is the existing 42-coordinate state/image design. The rich
input adds four current bank-cluster distances; per-cluster reconstruction
means and upper quantiles; similarity to eight recently offered images (one
lagged image is available without additional future information); older health
changes; and candidate-gradient/proposed AdamW update summaries with alignment
to the gradient of a fixed current probe head on a balanced 32-image subset
of the DEVELOPMENT bank. Cluster membership is fitted on warm selector-bank
features only. The probe head is fitted on support labels. Gradient alignment
holds that head fixed and is an approximation to the refitted-probe target.
The proposed AdamW displacement includes clipping, moments, bias correction
and decay. A preflight checks it against an actual step on a disposable copy.
The incoming image has no label, but rich live features DO use development
bank labels. This is task-informed selection, not a fully label-free method.
Do not add an unvalidated health veto or tune warm-based harm boundaries.

Use six old held-source histories and both seeds to collect development pairs
at origins 128..928, stride32. Fit on four histories and select ridge versus
histogram boosting using the other two whole histories. Then refit using all
six DEVELOPMENT histories. Fit J/probe targets separately for state, baseline,
and rich features and separately for both SSL seeds. Choose admission threshold
min(0, validation-prediction 60th percentile), per target/design, then freeze.
Thus only predicted improvements are admitted; validation admission is capped
around60%, but live rates can differ. Report realized exposure explicitly.
No future evaluation label, test feature scaler, live label or outcome refits
a model, threshold, cluster or reference. Offline training uses labeled probe
outcomes even for the baseline-feature probe-target selector.

Fresh intake selects60 highway videos,30 per light, with fixed hashing and
six reserves per light. Exclude recorded prior video paths/inventories,
singleton source histories, original reader membership and health-bank clips.
Decode96 native timestamped frames per clip with pinned mirror CRC/SHA and
decoder trace. Make six ten-clip histories under the fixed patterns. Select
600 fresh local keyframes, balanced across two scenes/two lights, excluding
those videos and prior reader/health membership:200 for development objectives
and400 for final evaluation. They are disjoint by clip ID and exact pixels.
Pool membership inherits the original warm-disjoint partition. No test images
enter gate features, objectives, model fitting or threshold selection. Existing
support panels are reused; final query images are new. Describe this reuse.

For each fresh history/seed, train a shared576-update prefix. Fork identical
complete model/optimizer/RNG states. Process arrivals576..959. Four main gates:
J_base, J_rich, probe_base, probe_rich. Add J_state and probe_state context
controls, accept-all and freeze. For EACH main gate add uniform random image
selection at exactly its realized count in every96-frame clip. These four
offline random controls receive future counts; do not describe them as causal
deployment rules. All accepting arms use the same per-arrival training seed.
Reject means no optimizer.step, rather than a zero-gradient optimizer step.

Record every input, prediction, threshold and decision. Health and development
probe observations follow the current filtered path; lag after rejection is
zero. Rich observations are charged to that policy. At the common fork and
each96 arrivals, measure fresh evaluation-bank linear/kNN accuracy/CE under all
three panels, day/night, and both bank health/objectives. Plot these checkpoints
against offered images AND cumulative accepted singleton updates. Measure
100-step copied fine-tuning at fork and final endpoints, clearly marking that
it has only two measured checkpoints. Do not interpolate it as a dense curve.

In each main gate/clip choose one diagnostic position uniformly from the first
65 positions using a private seed, so32-step continuations stay inside that
clip. This samples a declared restricted phase population, not all arrivals.
Fix the admission decision before diagnostics. Compare a disposable update
against skipping the ENTIRE optimizer step, then continue both copies through
the next31 identical images with identical future full-update decisions/seeds.
Measure DEVELOPMENT and fresh-evaluation probe CE/accuracy immediately and at
H32, plus J. Include accepted and rejected candidates. These labels never feed
back into policy. Research diagnostic cost is separate from charged live cost.

Primary policy endpoints: balanced linear-probe accuracy at960, paired against
each gate's own random-count control and accept-all. Analyze objective and
feature contrasts both in raw endpoints and in gains over their respective
random controls; unequal admission prevents treating raw factorial contrasts
as pure content effects. Forecast RMSE/MAE and balanced nonzero sign accuracy
use the declared sampled same-parent effects. Separate prediction-target
accuracy on the development bank from transfer to fresh query images. Report
H1 and H32 differences; the fitted target is H1, not H32. Holdout accuracy and
CE curves, source/seed/panel differences, counts and wall time remain visible.
Six whole-history cluster bootstrap95% intervals keep seeds, panels and serial
checkpoints together. All comparisons are exploratory and unadjusted; publish
all four gates instead of selecting a winner using final outcomes.

Verification: meaningful unit tests for AdamW displacement versus PyTorch,
feature labels, skip semantics, split exclusion and threshold freezing;
independent raw probability/health and summary arithmetic; all policy decisions,
random budgets and shared-fork identities; physical replay of policy updates,
sampled gradients/diagnostics and final states. Preserve SHA manifests for prior
studies and archive source/config/protocol with executed notebook and inspected
figures. Charge bank/feature/predictor/update time separately; endpoint readers,
calibration, intake, audits and disposable branches are research costs.

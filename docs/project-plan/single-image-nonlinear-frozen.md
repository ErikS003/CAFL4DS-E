# Expected isolated singleton effect: chronological nonlinear forecast protocol

Declared before collecting or inspecting these new averaged labels. Earlier
single-view forecasts and the sparse repeated-view measurement study are known.
This is a follow-up on the same six held highway histories, not a new-source
confirmation. Preserve every earlier study and its source hashes.

Keep the original 1-image SSL stream and optimizer seed5003. At all 31 origins
128:32:1088 in each held history, fork the complete parent. Compare six independent
one-image training views against one explicit-zero-gradient AdamW step at that
parent. The target is their mean difference in each of seven health metrics,
including fixed-bank reconstruction. This isolates image-gradient contribution
relative to momentum/decay, rather than relative to skipping the optimizer.
View seeds are SHA256(namespace|history|origin|draw), first4bytes little-endian
modulo2**31-1; use disjoint draws0:3 and3:6 as measurement checks. No downstream
reader is fitted: earlier reader transfer failed its measurement gate.

At t, predictors may use parent health7, last ordinary health change7 (t-1 to t),
optimizer summaries6, age1, and the most recently observed averaged isolated
effect7 at an earlier sampled origin in the SAME history. The image design adds
the19 already cached pre-update image features. No current branch outcome,
gradient preview or future state enters a predictor. Sparse labels are treated
as available after t+1; obtaining these labels is an expensive measurement oracle,
not a demonstrated deployable online selection system.

Compare standardized multioutput ridge(alpha10), RBF kernel ridge(alpha1,
gamma1/input_dimension), and ExtraTrees(128trees, depth6, min_leaf5, allfeatures,
seed84631, singleCPUworker). Standardize inputs and targets on the fitting data.
Use each family with and without image features. No parameter search.

Calibration labels must end at or before image512 (origins<=480):72rows.
Choose ONE family/design using inner chronological train labels ending<=352
(42rows), validation origins352:32:480(30rows). Selection criterion is mean across
seven metrics of validation RMSE divided by zero-effect RMSE. Ties follow config
order. Refit the selected and all six individual candidates on all72prefix rows.
Evaluate origins544:32:1088(108rows). Frozen models keep their prefix fit; online
models refit on all labels ending strictly before the current origin, pooled
across histories. All six same-origin predictions precede admitting any of their
labels. The unused origin512 can enter online training at544. Both designs may
read their latest matured same-history effect, including for frozen models:
frozen refers to coefficients, not frozen covariates.

Controls: zero effect, fixed prefix mean, expanding causal mean, and previous
same-history effect. Report RMSE and MAE in metric units; direction accuracy is
secondary. Publish every model, and the prefix-selected model, without selecting
on test scores. Report image versus state ablations and prefix-selected gains
over zero and previous-effect controls. Bootstrap paired squared/absolute error
differences using whole-history clusters4000draws, seed84643. Six histories are a
small uncertainty sample. Do not treat correlated frames as independent.

Repeat scoring against each disjoint3view half and report half-to-half RMSE/MAE
to distinguish forecast error from target Monte Carlo variation. The6view target
has no independent6view replicate. It estimates expected measured health under
training augmentation; health is not downstream performance or image utility.

Save raw bank features/reconstruction for every branch; check all original
trajectory losses, complete checkpoints and finals. Independently reconstruct
all raw health values, labels, features, chronology, baselines and scores.
Independently replay all original streams and the branches at480 and960 in every
history. Freeze collector, model implementation, config and protocol in input
hashes before labels; save forecasts, fitted-selection evidence, plots, notebook,
report and an artifact manifest. Tests must exercise future-label exclusion and
same-origin isolation for nonlinear as well as linear models.

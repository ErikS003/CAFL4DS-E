# Functional health profile — study51

Known-cohort retrospective development, motivated by studies48–50. Freeze code,
config, tests, synthetic report/engineering receipts and consumed input hashes
before any new real fitting or profile calculations. Existing outcomes have
already been studied: group exclusion cannot turn this into fresh confirmation.
No new SSL, downloads, encoder inference, interventions or controller deployment.

Use all72 study43 pre-action states:3 seeds x3 sources x4 contexts x2 diets.
Profiles use ONLY the200-image validation support reservoir from the cached2248
evaluation features. The training-partition support is prohibited: it overlaps
the2048-image old benefit query. Verify validation support indices are below200,
metadata is aligned, and identity sets are disjoint in the original manifest.
Capture sessions remain unknown; image separation does not establish session
independence. Reuse all6 fixed128-image study48 validation pools,32 per cell.

For each state/pool/prior use4 stratified folds (8 validation/24 fitting images
per city/highway x day/night cell). Fixed metadata-only seeds in config, identical
across states. Fit96-label binary logistic heads: mean CE + lambda*||beta||²/2,
including bias, raw coordinates, zero init, CPU float64 damped Newton, gradient
infinity norm<=1e-10, max100 steps/60 backtracking. Primary lambda.01; fixed
.001 sensitivity. 3456 collection fits. Repeat primary pool0/fold0 in all72
states; independently optimize these72 with SciPy trust-exact (gtol1e-11,
max100), gradient<=1e-8 and reservoir score discrepancy<=1e-6. Total3600 fits.
Log every fit start/completion, no silent retry. Retain heads, OOF scores, fold
measurements, receipts. Full128 existing study48 heads are DESCRIPTIVE references:
their in-sample profile differs in both training size and evaluation reuse.

Pool the128 OOF logits (four independently trained96-label heads) and compute
per domain: separation=(mu_highway-mu_city)/(2*sigma), preference=(mu_highway+
mu_city)/(2*sigma), log(sigma), and equal-class mass |score|<=.25*sigma.
sigma=sqrt(mean of two within-class population variances), floored1e-8.
These are task/readout-relative, label-requiring diagnostics, not intrinsic or
label-free health. Scores use highway-positive convention. Save both domains
(8 descriptors), all6 pools, both priors, fold descriptors and descriptive raw
boundary mass |score|<=.1. Pooled OOF mixes four fitted readers; within-fold SD
and outer-pool SD document calibration/sampling sensitivity. Six overlapping
pools are measurement repetitions, not six independent state replications.

Seal ALL profiles and fitting receipts BEFORE loading saved intervention
targets or cheap/CKA signals into this pipeline. Reuse original study43 future
replay-minus-current benefits (384 SSL updates after common fork), averaged
over12 old50-label assay support draws. Primary original FIXED-reader night-
highway benefit; joint and all7 class/domain components saved as secondary.
New profiles do not redefine that old utility. Independently reconstruct targets
from saved correctness/cell ordering and reproduce legacy cheap/CKA predictions.

Primary profile uses arithmetic mean of six OOF pool descriptors at lambda.01.
Do NOT average logits across different support identities. Use original strict
holdouts: test both diets of one(seed,source,context); train only rows differing
in ALL THREE axes (24 train/2 test,36 folds). Train-fold-only centering/scaling,
ridge penalty10, unpenalized target mean. Fixed models:
mean; diet; cheap(diet+trailing16 loss/gradient/actual step plus diet interactions);
cka_added(cheap+mean monitor/audit CKA drift and diet interaction);
functional_only(diet+night4 and diet interactions);
functional_added(cheap+night4 and diet interactions, PRIMARY);
cka_functional_added(cka_added+night4/interactions);
day_night_added(cheap+all8/interactions). No feature or hyperparameter selection.
Independently audit every fold/model with augmented least squares. Save both
priors and six single-pool primary-prior sensitivities for every model/target.
These produce2304 forecast ridge fits, separately from3600 logistic reader fits;
independent augmented least-squares checks do not train additional task readers.

Development screen: primary functional_added reduces MSE vs cheap by>=10%,
with positive improvement in each seed and each source. Always report context4
and both diets separately too. This screen does not amend old43 gates or validate
an alarm. Report MSE for all settings, per-axis errors, and shadow policy value
(replay if forecast>0) versus always-current/replay/cheap on all classes. No
actual action selection. Correlations per diet use all72 known states and are
exploratory descriptive associations; no p-values, causal claims or population
confidence intervals. Shared warm start, footage and repeated assay remain.

Six figures fixed before results: design/invariance controls; OOF versus full
profile and sampling sensitivity; night profile by context/diet including4;
strict-holdout MSE and seed/source stability; predicted versus actual benefit
and all-pool/prior sensitivity; per-diet correlations and shadow class values.
Synthetic full-pipeline fixture and all six figures inspected before freeze.
Reserve1536MiB; outputcap256MiB; no cleanup of active scientific artifacts.

# Linear-probe reliability on cached representations

Declared before this new evaluation, 2026-10-08. Outcome queries have already been
inspected under earlier objectives: this is a development diagnostic, not a new
independent confirmation. No encoder training, health prediction or policy fitting.

Primary: balanced linear classification accuracy. Secondary: classification CE
from the **same fitted logistic probe**. Fit a new head on labeled support for each
cached encoder checkpoint; freeze that encoder. Standardization uses support only.
Balance the three scenes within day/night, then weight domains equally. Also report
domains separately. Do not interpret these results as kNN or reconstruction loss.

Use three disjoint 384-image support samples and their overlapping 1152-image pool.
Compare three shared-reader recipes:

1. Accuracy-selected C: [.01, .1, 1, 10], warm-support four-fold CV for seeds
   77017/78011/78017, minimizing mean balanced classification error; ties within
   1e-10 prefer smaller C (stronger regularization). Freeze before new query scoring.
2. Original CE-selected C, copied unchanged from the preceding study.
3. Fixed C=.1 for all support panels and checkpoints.

Use warm features and four real/shuffled supervised-control caches from the larger
study; six exact prefix-12 parent caches and 108 branch feature caches from the
36-branch 12/32/64-update experiment. This gives 119 representation states, with
the same state coverage for every recipe. Reproduce the original CE-selected
accuracy/CE scores. Other primary-objective changes are explicitly retrospective.

Measure real-label gains over warm and matched shuffled controls; alpha 0/.5 gains
against alpha 1; learning changes versus each exact parent; support sign agreement,
unanimous nonzero agreement, ties, repeat agreement and mean absolute repeat
differences. Keep horizon, support, query panel, domain and training draw distinct.
Do not choose a winner by inspecting a favorable query panel. Agreement alone can
come from tied or uninformative outcomes; inspect controls and effect sizes too.

The longer streams repeat 24 original arrival batches. Six parents are nested in
three streams. Both query panels are development evidence; rare nighttime
residential queries total only 45. No new balanced independent labels are available
in the exhausted local cells. This study can diagnose whether the revised linear
objective is more consistent; it cannot qualify generalization or MPC utility.

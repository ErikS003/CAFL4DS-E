# Notebook 90: online admission against saved offline heuristic and full training

Run only B, 57,600 offered images, from Notebook 84's original warm parent,
fresh_00 stream, augmentation seed 5003, action RNG seed 835003. Reuse N84
no-task-rollback as A (offline/frozen heuristic); reuse its saved full-training
reference. Do not rerun A or launch C. Match source hashes, warm parent, offered
order, augmentation seeds, and initial settings. This is one paired stochastic
trajectory, not a multi-seed replication. Full is a reference with a different
training budget, not a matched-update control.

Keep J weights .25/.25, original LR rules, fixed bank and task rollback disabled.
Keep numerical recovery and forced exploration bursts unchanged. Initialize
the task head from 128 accepted updates, excluding first arrival as in the audit.
Predict immediate balanced development soft-error change before each decision.
For ordinary admission in normal/caution mode, multiply the original probability
by 1.2 for negative predicted change and .8 for positive predicted change;
clip to [.05,.95]. Zero/nonfinite predictions leave the probability unchanged.
Use the original per-arrival acceptance draw against the adjusted probability.
Features use the candidate's applicable LR before the acceptance draw.

The head is the N89 discounted ridge model (half-life 256 accepted updates,
ridge 10, intercept .1, initialization-only scaling/clipping). Train only on B's
actual accepted outcomes. Log original and adjusted probabilities, predictions,
readiness, features, actual outcomes and training/health metrics. No evaluation
labels enter the head or action. Verify real GPU exact resume and original A
prefix identity before influence begins.

Plot offline versus online versus full on offered-image axes: balanced/day/night
linear accuracy, kNN accuracy, CE, rank, variance, covariance, reconstruction,
J, acceptance/update counts and training loss where available. Also show common
accepted-update axes with honest coverage. Primary endpoint is mean evaluation
linear accuracy across scheduled 30k–57.6k checkpoints, not a selected peak.
Report final accuracy/CE and total updates. Do not optimize settings during run.

Use transactional checkpoints and the storage reserve. Run only after sufficient
SSD space has been recovered without deleting experiment evidence.

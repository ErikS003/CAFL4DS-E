# Larger measurement and action-gradient development study

Declared before outcome collection, 2026-10-07. The allocation audit found only 12 unused
night/clear residential validation images and 33 unused stream-partition images in that cell.
An entirely new balanced fitting/support/query/three-stream study cannot be allocated.
This experiment is therefore explicitly development research, not independent qualification.

## Data and roles

Use the same warm well, 64-pixel MAE, two domains and three scene classes. Keep the existing
warm-trained coverage anchors and target distribution. Query candidates
are all previously unrecorded validation images capped at 128 per cell, plus previously
unrecorded stream-partition images capped at 64 per cell. Shuffle privately (seed 77011),
split each source/domain/class cell approximately in half into development and holdout.
Keep both query roles disjoint from every fitting, reader-support, monitor and SSL stream role.
Report query origin and counts; the scarce residential/night cell will remain a limitation.
Query holdout is inspected as a development replication, not retained as a future final test.

For other roles, augment the unused stream pool with previously recorded stream-partition
paths from historical input manifests with the same partition seed and fraction. Include
only source manifests explicitly declaring the stream partition. Read per-image attributes
to verify canonical cells. Exclude new queries and every recorded warm image. Historical
SSL exposure is recorded and disqualifies a new-image generalization claim. Never reuse
images between roles in this experiment. Sample 96 fitting, 32 internal validation, three
disjoint 64-image-per-cell support panels, 16 label-free monitors, and three disjoint
128-image-per-cell arrival streams. This is 720 images per cell in nonquery roles. No
silent reduction when capacity is insufficient. No historical response files are read.

## Reader and supervised controls

Fit standardized logistic probes on each support panel and their pooled union. Choose C
from .01/.1/1/10 using only warm-model support features and four balanced folds, private
seed 77017. Freeze all four C values before any control/branch query scoring. The pooled
reader shares data with the individual readers and is not an independent replicate.
Evaluate refitted CE, linear error and cosine kNN error, balanced and separately for A/B.

Two private training seeds 77021/77027 run real and within-domain shuffled labels from
the same warm encoder and matched head initialization. Fixed 20 epochs, batch 32, encoder
LR .0001, head LR .001, AdamW weight decay .01, no augmentation. Save all fitting curves;
score the separate internal-validation set at the final checkpoint. Head-only standardized
logistic fitting at C=.1 is a diagnostic comparator. Do not select checkpoints or extend
training using query results. Score the supervised encoders with the same frozen readers.

An exploratory positive-control screen requires, for both training seeds and both query
roles, pooled-reader CE lower and linear/kNN accuracy at least 5 percentage points higher
than warm, and superiority to the corresponding shuffled control. Both domains must have
nonworsening CE and recognition. Require CE benefit signs to agree across the three separate
support panels. These are declared descriptive gates, not significance tests or population
confidence bounds. Failure keeps the response target unqualified.

## Matched alpha actions and observations

Three arrival streams use A1/B1/A2/B2, six updates per visit, paired private orderings.
Two learning seeds, parent prefix lengths 4 and 12 at alpha=.5, two training repeats,
constant alpha 0/.5/1 and twelve-update branches: 12 parents, 72 branches, 864 branch
updates. Four-update outcomes are horizon sensitivity. All branches use the same parent
model/optimizer/replay/RNG state, 16 current plus 16 replay images and equal work.
Audit action-independent reservoir contents and RNG at every update. Restore RNG after
branches before continuing the shared prefix. Save parent and branch states and features.

Only before branching, collect reconstruction, coverage, embedding/replay signals and
label-free monitor health. Compute last-encoder-block reconstruction gradients without
optimizer steps: per-image norm and cosine to monitor gradient; gradient norm of each
hypothetically selected 32-image training batch; cosine to a fixed replay reference
(first up to 64 reservoir entries). Gradient observations use private masks; they are
local reconstruction diagnostics, not exact AdamW update directions or CE derivatives.
Hypothetical action selection uses private selector copies. Verify method, gradients,
module modes, actuator, optimizer and global CPU RNG are unchanged.

Report gains against .5 and 1, repeat variability, support-panel sign agreement and domain
tradeoffs. Estimate a retrospective best-action upper bound using repeat-averaged results,
clearly labeled optimistically selected and not an achievable online policy. Keep stream
identity for all future prediction splits; rows/repeats are not independent observations.
This small study caches prediction inputs but does not fit a health/gradient regressor.
Reader reliability and independent trajectory expansion remain prerequisites to that fit.

## Verification and artifacts

Unit tests cover allocation capacity/disjointness, private balanced support folds, and exact
invariance of the next optimizer update after backprop-only observations. Run a one-parent,
one-repeat engineering smoke before full collection. Freeze role lists, image/input/source
hashes and the protocol. Fail on parent mutation, unequal work, replay mismatch or source
drift. Save CSV/JSON scores, traces, progress, a report and an executed explanatory notebook.

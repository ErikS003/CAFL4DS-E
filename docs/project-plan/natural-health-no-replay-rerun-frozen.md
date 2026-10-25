# Notebook22 rerun: no replay, complete health and capability readouts

The user requests a physical rerun of the quarter/half/three-quarter/full system
from notebook22, displaying all health signals, kNN, linear probe and fine-tuning.
Preserve earlier scientific files and notebooks. Use new outputs and notebook34.

## Fixed experiment

Run only full training and live adaptive admission. Reuse the exact audited
131,072-frame / 4,096-arrival stream from211 videos in notebook22. Keep frame
timestamps within clips and the pinned archive order between clips. These are
separate videos, without established capture-session chronology. No acquisition,
source expansion, new SSL seed or threshold search is required.

Start each arm from notebook22's exact saved model, empty AdamW and CPU/CUDA RNG
snapshot. This differs from the later reservoir experiment's retained warm Adam.
Architecture: RGB64, patch8, TinyViT MAE encoder width96/depth4/heads3/MLP2;
decoder64/depth2/heads4/MLP2; mask0.75, normalized-pixel loss, crop0.6–1, flip0.5.
Train the full MAE on32 current images per admitted update. AdamW LR1e-4,
weight decay0.05, betas0.9/0.999, epsilon1e-8, clipnorm1. Seed5003 and the
original deterministic GPU settings remain fixed. The inherited model metadata
contains an unused filter specification; this runner never constructs or calls
that filter. There is no memory, replay batch, queue or training on old images.

The original256-image monitor bank supplies the live alarm. Capture its fixed
reference at arrival132 and observe every8 arrivals thereafter. Any relative
rank decline >=0.10, normalized-variance decline >=0.10, or CKA distance >=0.10
on two consecutive scheduled readings confirms an alarm. A confirmed alarm
affects subsequent arrivals, resets integer admission credit, and starts25%
admission. Advance to50/75/100% every32 incoming batches, unless another alarm
resets the ramp. Admit whole batches using quarter-unit integer credit: add
1/2/3/4, train when credit reaches4, subtract4. Rejected batches are discarded.
Full training accepts every arrival; its alarm is observational only. Adaptive
decisions are recomputed online, rather than supplied from an old acceptance mask.

## Measurements

Read capability at0,128,132, every128 arrivals, each arm's shared causal first
alarm and4096, matching notebook22. The first alarm discovered in the full arm
is included in both grids; the two learners are identical until that alarm.
All three readers use the inherited three50-label support draws61001/61007/61013
and3,663 city/highway query images. Balance classes within day/night and then
domains. kNN:20 cosine neighbors, inverse-distance weights. Linear probe:
support-only standardization, logistic C1/max_iter1000. Fine-tuning: independent
encoder/head copy,100 updates, batch32, AdamW encoderLR1e-5/headLR1e-3,
weight decay0.05, clipnorm1; fitting RNG support draw+25. Refit all three kinds
in the new collection; reuse only exact identical states within this rerun.
Labels enter only these offline readouts. Query outcomes never choose actions.

Show RankMe, normalized variance, normalized covariance, CKA distance and cosine
drift on the original bank at every saved monitor observation. Also show the
same five signals on the pre-existing, clip-disjoint256-image audit bank at
capability checkpoints. References remain each arm's arrival132 features;
pre-reference CKA/cosine are missing rather than measured against future data.
Measure held-out SSL loss on both banks at capability checkpoints with private
mask seeds84211/84223. These added diagnostics never enter the controller.
Training loss and pre-clipping gradient norms use accepted updates only;
discarded arrivals have missing measurements, not zero values. All observations
and fits must preserve learner weights, optimizer, gradients, modes and RNG.

## Work, integrity and reporting

Before new SSL or reader outcomes, seal config, runner, report, verifier, tests,
this protocol, original source/input manifests and arrays, initialization,
evaluation data and both banks. Reuse original source audits with unchanged
pixel hashes. New collection is capped at8,192 SSL starts; an exact second
execution is capped at8,192 more. Preserve actual attempts, including failures;
no discretionary retries or scientific expansion. Per collection, at most216
new fits of each reader kind; independent verification at most216 kNN and216
linear refits, plus exactly two fresh fine-tuning checks at full/132 and
adaptive/4096 for support61001. Cache reuse in the exact rerun is explicitly
recorded. Keep at least768MiB free; cap all new scientific outputs at384MiB.

Recompute both policies during the exact rerun. Verify accepted inputs, masks,
alarms, losses, gradients, all original-bank embeddings and saved learner
fingerprints. Compare collection against notebook22's stored physical traces
and fresh fine-tuning probabilities; this is a reproduction, not independent
confirmation. Reconstruct alarm persistence and admission credit independently
from saved features. Refit every new frozen reader, reconstruct class/domain
aggregates, and cross-check geometry. Recompute audit embeddings and both-bank
private losses during the second execution, preserving learner state.

Produce six saved figures and a new notebook: three-reader capabilities,
original-bank geometry, actual alarm inputs, admission/work/training diagnostics,
audit-bank geometry, and both-bank held-out SSL loss. Show balanced/day/night,
all support draws, observed drawdowns, costs and alarm/reset counts. Inspect all
figures; check embeds, links, artifact hashes, work and storage. Keep descriptive
effects distinct from success criteria: no new efficacy gate or controller
qualification is introduced. One reused source stream/SSL draw and reused
evaluation populations do not establish generalization or long-run CF prevention.
No follow-on experiment launches automatically.

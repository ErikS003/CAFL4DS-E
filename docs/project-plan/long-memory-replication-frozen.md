# Long-horizon historical-memory replication

Authorized 18 October 2026 by the user's "great, continue, run this" following
the four-arm design in this chat. This experiment tests whole-trajectory fixed
memory policies. Preserve prior studies. Health is observational; no alarm,
predictor, threshold search, adaptive action, architecture or dose sweep follows.

## Sources and preparation

Reserve three outcome-independent groups of 384 eligible train archive members
after every previous reservation. Exclude all earlier used/reserved clips,
warm-start, monitor, audit and downstream clips. Freeze all three reservations
before acquisition. Decode only each group's ordered prefix required for exactly
131,072 RGB64 frames / 4,096 arrivals. Preserve timestamp order within clips and
pinned archive order between clips. Do not arrange or balance regimes. Unused
reservations stay excluded. No outcome-dependent source substitution or extension.

Use the inherited revision-pinned community mirror and bounded HTTP-range reader.
Cap acquisition at 12 GiB/block, 36 GiB total, one <=128 MiB raw member in flight.
Delete only newly downloaded temporary raw members after CRC/SHA and decoded
manifests are committed, as in the established intake. Retain decoded caches,
timestamps and per-frame hashes. Lossless NTFS compression is allowed with
logical hashes checked. No prior data deletion is authorized or needed by design.
Keep at least 768 MiB free and new scientific outputs <=2,304 MiB excluding source
frames. An incomplete source/resource failure stops before scientific expansion.

Clip-disjoint blocks are not known independent capture sessions. Existing
city/highway support/query populations are reused. This is bounded source and
training-RNG replication, not untouched-population confirmation.

## Learner and fixed policies

Restore M0 SHA256 e372f80d86788e70850bfca407bbd93a2a8a26fcce18f23fffe316e5199d686d,
including its warm AdamW moments/counters. Change only training RNG to
5003/5009/5011. These are draws conditional on one warm model/optimizer.
Keep RGB64 patch8 TinyViT MAE encoder96/depth4/heads3/MLP2, decoder64/depth2/
heads4/MLP2; mask .75; normalized reconstruction; crop .6-1/flip .5; full MAE
training; AdamW LR1e-4, WD.05, betas .9/.999, eps1e-8, clip1; deterministic GPU,
four CPU threads, TF32/cuDNN benchmarking off. No new training loss or schedule.

Each of nine block/seed sets runs current-only32/0, FIFO25616/16 and historical
Algorithm-R25616/16 for all 4,096 arrivals. A zero-update warm reference is shared.
Replay arms start32/0. Sample before ingesting all32 incoming images, including
those not trained immediately. Use private sampling seed73919 and reservoir
replacement seed90101. FIFO/reservoir current subsets must match exactly; every
arm has131,072 optimizer-image exposures. Memory capacity matches; retained
pixels intentionally differ. Log every consumed identity and hash, ages, coverage,
repeat exposure, complete sampler/learner state and actual work. Do not attribute
an effect solely to age: redundancy and source coverage also change.

## Evaluation budget and precision

Capability grid: warm and every128 arrivals through4096 (33 points). Keep the
canonical50-label100-step copied-encoder/head learner, three original curve
draws61001/61007/61013, kNN20 cosine and support-standardized logistic C1.
Primary endpoint uses all256 support/fitting seeds from the completed expanded
measurement study. Archive planning is reconstructed before new SSL; its256-draw
budget is selected for expected ~1pp conditional resolution, not guaranteed
precision on new endpoints. Intermediate curves retain three draws and are
descriptive. Fine-tuning primary is expected adaptability, not frozen recognition.
Read all3,663 class/domain-balanced queries. Labels never affect online learning.

All endpoint predictions (including shared warm) get256 draws per reader. Every
intermediate trained checkpoint gets3. Unique fits/kind are capped at9,679:
28*256 +27*31*3. Encoder-identical states may reuse proven matching fits.
Four isolated reader workers follow the already validated GPU fitting procedure.
Record fitting work and failed attempts separately. Query identities never tune
the reader, source selection, budgets, primary horizon or action choices.

Record all five geometry signals on original/audit256-image banks, plus two-seed
held-out SSL losses. Observe original-bank geometry every8 arrivals from132;
both banks/losses at capability points and132. Each arm's geometry anchor is132;
capability anchor remains M0. Observers preserve learner/optimizer/gradient/mode/
RNG state. Geometry and SSL loss have no automatic interpretation as task quality.

## Analysis and decisions

Primary: final balanced fine-tuning reservoir-minus-current effect. Mandatory
mechanism contrast: reservoir-minus-FIFO. Domain effects and warm-relative gains
are companion requirements. Average reader draws, then training seeds within
source, then equally across three blocks. Keep every block/seed/domain result.
Source blocks, not frames/checkpoints/panels, define source-level replication.

Frozen positive development screen requires all:
1. Mean balanced reservoir-minus-current >=1pp.
2. Positive source means against current in all3 blocks; >=2/3 positive SSL draws
   in every block.
3. Positive mean reservoir-minus-FIFO balanced effect in every block.
4. Each source's final day and night reservoir-minus-current means >=-1pp.
5. Positive mean final reservoir-versus-warm balanced effect.

Use conditional paired support/query resampling for endpoint effects and report
source effects/ranges separately; three sources do not support strong population
inference. Flag unresolved1pp conditional precision separately from the effect
screen. A positive screen alone is not population confirmation. If a required
contrast is imprecise, disposition stays inconclusive even if point gates pass.
Report individual harms; an average budget is not a per-model guarantee.

Separately report per-domain maximum and arrival-averaged shortfall below warm,
peak-relative drawdown, endpoint and warm-relative gain. Compute path metrics on
the fixed grid using trapezoidal integration. Maximize after support averaging
for pooled curves and also show per-draw summaries. Lower peaks can reduce
drawdown. Sparse checkpoints miss transient damage. A protection claim requires
resolved improvement in anchor-relative retention; endpoint benefit alone does
not establish forgetting mitigation. No new success criterion may replace a
failed primary gate.

## Integrity and caps

110,592 collection SSL updates and110,592 exact reproduction updates, total
221,184 physical updates. Bounded GPU preflight is separate, <=64 discarded SSL
updates and <=16 fits. Freeze source identities/pixels, model, config, runner,
statistics, verifier, reader, tests, preparation and input lineage before SSL.
Preserve all attempts. Commit each completed arm and check immutable hashes on
restart. Reproduce every update/input/health/final-state fingerprint exactly.
Independently rebuild memory sampling from primitive RNGs and source pixels,
reconstruct all scores/path metrics/gates, refit frozen readers, and reproduce
preselected fine-tuning fits at warm and final checkpoints. Export a saved-figure
notebook and method/report/audit artifacts; visually inspect actual figures.
Full repository phase gates are unchanged. No follow-on experiment launches.

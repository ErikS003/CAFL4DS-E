# Sustained replay dose and age selection: frozen development protocol

Authorized 14 October 2026 by the user's request to run the proposed experiment.
Freeze this document, collector, selector, tests and independent verifier before
collecting any new accuracy outcome. Preserve earlier completed studies.

## Population, learner and fixed policies

Use the six inspected coverage-admission histories and optimizer seeds
5003/5009/5011. Each history has 12,288 prepared arrivals. Reuse the audited warm
MAE/AdamW recipe, 64-pixel inputs, batch 32 and 192 updates. These are development
sources and synthetic clip concatenations with unknown recording sessions.
Supports, queries and source histories have already been inspected. They cannot
provide independent confirmation, regardless of the number of seeds.

Five arms: current-only, Uniform-25, Uniform-50, Recent-25, Recent-50. The suffix
specifies historical fraction of the batch: 8 or 16 images. Current-only has 32
current images. All arms share eight initial current-only updates. Subsequently
buffer 512 arrivals before eight chronological updates, for all arms. No within-pool
regrouping, new objective, learning-rate change, health selection or context rule.

All-arrival Algorithm-R reservoir capacity is 512, with private generator seed
stream_seed(stream, seed)+74111, identical to the prior admission mechanism.
Snapshot memory before each post-calibration block; historical frames must predate
that block. Uniform permutes the 512 memory entries in their reservoir-slot order
and takes 128. Recent sorts by arrival index, takes the largest 128 and permutes
them. Both use a fresh private generator stream_seed(stream, seed)+94103+block.
stream_seed adds the optimizer seed to the integer first eight SHA256 hexadecimal
digits of the stream name. This never alters learner randomness.

Split historical proposals into eight rows of sixteen. The 50% arm consumes all
sixteen; the 25% arm consumes the first eight of each row. This pairs historical
doses and avoids repeated old identities within a block. Recent refers to relative
age within this common memory, not a fixed real-time window or semantic segment.

Current candidates are the first 32 of each chronological 64-arrival slice.
Select 32-r using integer linspace(0,31,32-r), and prepend r old images. Current
identities match between content policies at the same dose. All arms have 6,144
training exposures. Post-calibration replay totals are 1,472 or 2,944; corresponding
whole-episode current totals are 4,672 or 3,200, versus 6,144 current-only.
Dose changes current coverage intentionally. Reservoir frames may never previously
have been trained on, so this does not isolate forgetting prevention.

## Input-only preflight and controls

Before locking or reading new outcomes, reconstruct all 90 exposure plans. For every
source/seed/dose pair require recent mean old-frame age at most 60% of uniform,
and total variation between their prepared-clip exposure shares at least 0.15.
These are manipulation criteria, not utility tests. Save source hashes, all exposure
summaries, age quantiles, unique/repeated exposures and coverage. Stop before training
if the contrast fails. Prepared clips are arrival blocks of 512; labels never select.

Require current-only to reproduce all 192 original loss/gradient pairs and complete
before-return/terminal states before reusing its readouts. Other arms have new draws
and new readouts. Common calibration fingerprints must match across all five arms.
Repeat the initial eight updates twice in a discarded sixteen-update deterministic
preflight. Every trajectory then has an independent exact complete-state replay
without intervening readouts, including all endpoint health-bank embeddings.

## Outcomes and statistics

Primary outcome: terminal balanced copied-encoder accuracy after fine-tuning on
fifty labels for one hundred steps, averaged equally across three fixed support
panels and 400 queries. Primary contrasts, in order:

1. Uniform-25 minus current-only: including historical data.
2. Uniform-50 minus Uniform-25: increasing historical dose.
3. Recent-25 minus Uniform-25: historical selection at matched dose.

Average seeds and support panels within each source history. Six history means
set paired Student-t uncertainty, df five. Report ordinary intervals and Bonferroni
95% simultaneous intervals across these three primary scalars. The histories share
prepared source material; intervals describe variation across this fixed design,
not independent recording-population uncertainty. Recent-50, before-return capability,
linear/kNN/CE and dose-by-content difference-in-differences remain secondary. No
secondary winner can replace a primary candidate.

Development continuation is stricter than a primary signal. There are three eligible
candidates corresponding to the three contrasts above. For simultaneous qualification
use Bonferroni 95% across eighteen scalars: three primary balanced contrasts, three
candidate-minus-warm balanced contrasts, and twelve candidate-minus-current/warm
day/night contrasts. Each candidate must satisfy every condition:

- Primary mean gain at least 0.25pp and qualification lower bound above zero.
- Nonnegative primary means in all six histories; positive aggregated mean in every
  optimizer seed and every support panel. Use raw integer predictions for panel means.
- Candidate-minus-warm balanced mean at least 0.1pp and qualification lower bound above zero.
- Day and night qualification lower bounds at least -1pp versus both current-only and warm.

The 0.25pp investigation margin and 1pp tolerated domain loss continue the earlier
practical scale. They are declared development requirements, not powered guarantees.
Wide intervals leave benefit or retention unresolved; failed qualification does not
prove zero effect or universal harm. No extra seeds, intermediate doses, new sources,
controller or outcome-driven rescue follows this bounded study automatically.

Measure variance, covariance, RankMe, CKA and cosine on the fixed 256-image bank at
warm, calibration, before-return and terminal. Health is diagnostic with no assigned
beneficial direction. Readers and observers must preserve weights, optimizer,
module modes and CPU/CUDA RNG. Save raw predictions and compact bank embeddings.

## Work, provenance and completion

Ninety trajectories: 17,280 scientific updates, comprising 13,824 new diet updates
and 3,456 current-control reconstruction updates. Exact replay adds 17,280;
discarded preflight adds sixteen. Total planned optimizer work is 34,576 updates.
There are 432 new primary fine-tuning fits (43,200 copied optimization steps),
108 identity-checked cached current fits, and nine unique cached warm fits.
Two endpoints are read per trajectory, three panels each. Interrupted attempts
are preserved and separately charged; never silently replace committed evidence.

Output cap is 128MiB, launch free-space reserve 512MiB and per-trajectory reserve
256MiB. No downloads, persistent pixel/model caches, or old-artifact deletion.
Collector wall-clock cap is two hours; check between trajectories and preserve
completed units if the cap is reached. No repeated failed preflight without a
documented infrastructure diagnosis. One bounded study, no implicit confirmation.

Charge selection, gathering, construction and SSL separately from health, restoration,
offline readouts, exact replay and artifact work. Equal training exposure is not a
deployment-efficiency claim; prepared caches do not price deployed decoding/storage.
Record interruption work separately. Resume unchanged committed units only after
hash validation; preserve incomplete units before restarting from warm.

Pin all inherited inputs and helper versions plus new scientific sources, this
protocol, tests and manipulation preflight. Save independent causal selection,
memory/dose/current matching, geometry, raw-prediction and statistical verification.
Complete focused tests, report, saved summary notebook and the documentation routing.
Any qualified development lead still requires a separately frozen unused-source and
untouched-query confirmation. State-dependent prediction requires a later paired
common-parent action study; whole-trajectory rankings alone do not justify it.

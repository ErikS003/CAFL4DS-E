# Transfer of the frozen health stop across new D/N/D histories

The user authorized the proposed transfer test: new histories, varied transition
times, unchanged guard thresholds, fixed-time and shuffled-time controls, and
offline-only labels. Preserve the original D/N/D and all earlier experiments.
Freeze this protocol and intake code before selection/decoding; freeze all learner,
analysis and verifier sources before new learner outcomes.

## New population and fixed guard

Use eight clip-disjoint histories, each448 batch32 SSL updates (14,336 frames).
First-day lengths are [160,192,224,160,192,224,160,192]; night lengths are
[64,96,64,128,64,96,96,128]. Returning day fills the remaining448 updates. Each
visit uses balanced counts of city/highway clips,512 distinct chronological frames
per clip at nominal16Hz. There are28 clips per history and224 total. Whole clips
remain within one visit; real timestamp gaps are retained. Streams concatenate
drives, with capture sessions unknown. Use one SSL seed5003 in all histories to
isolate source/timing variation; optimization-seed generalization is not tested.

Select clear-weather train clips from the existing audited eligible stream pool,
which excludes the original warm partition and validation. Additionally exclude
every clip in all existing intake/selection manifests, all local video IDs, and
every monitor/evaluation image ID. Sort each cell by SHA256 of
`dnd-health-transfer:20261016:` plus clip ID, then allocate in history/visit/pair
order. No learner outcome selects footage. Maximum eight corrupt/insufficient
decode replacements use the next ranked same-cell reserve before any training;
preserve failure logs and reasons. No outcome-driven source removal is allowed.

Retain the exact prior `HealthStop` implementation, threshold0.10 and patience2.
Activate and fix the acquired representation reference at update132 in every
history. Observe only132,140,148,...,444. The controller receives geometry and the
ordinary update counter, never domain metadata or transition clocks. Do not reset
or re-arm at a transition. Stop permanently after two consecutive rank/variance/
CKA alarms, with no rollback or resume. Endpoint448 means the stream ended if no
alarm fired; it is not retrospectively counted as a successful alarm. Do not add
observations at phase boundaries to the controller. No new threshold tuning.

## Evaluation and controls

Use the existing expanded canonical learner and evaluation pool: fifty support
labels per fit,100 encoder-plus-head steps, the first three fixed support draws
61001/61007/61013, and3,663 clear-weather day/night city/highway queries. Offline
task labels supply readout fitting and evaluation only. All histories share the
same three support draws; those draws are not independent source histories. This
is new training-clip transfer with reused evaluation data, conditional on one SSL
seed and three support draws; it is not broad independent-session confirmation.

The universal offline grid is [0,132,140,148,...,444,448] plus148/264 and all possible
phase boundaries160/192/224/256/288/320. Every history has the same49 observation
clocks. Auxiliary offline checkpoints never increment alarm persistence. Retain
continuous training, warm frozen, stop148, stop264 and the frozen health guard.
All permanent-stop arms share exact chronological prefixes. Histories are the
eight inference units; group all support panels within history.

Enumerate all8! permutations of the health rule's eight stop counts, including
duplicate assignments. Every assignment preserves total SSL training work and
is evaluated using each recipient history's own full prefix. This conditional
random-timing benchmark breaks state/timing association; it is not a deployed
random policy or a source-generalization guarantee. Report per-history stop times
relative to both transitions, and counts of stops during initial day/night/return.

Primary risk is each history's maximum of the day and night checkpoint drawdowns
from their own observed running maxima. Also report separate-domain drawdowns,
final balanced/day/night accuracy, gains versus warm/full/fixed controls, exposure
saved, and post132 duration more than1pp below the observed running maximum.
For duration and mean deficit use left-observation values over subsequent update
intervals, weighted by actual interval length. There is no claim to observe losses
between checkpoints or during the sparse initial0–132 interval. Guard references
never follow the evaluation's running maxima.

Report a predeclared exploratory added-timing-information flag if the health
rule reduces mean worst-domain drawdown versus exact shuffled timing by >=20%
and >=0.5pp, with a positive paired-history t95% lower endpoint, while mean final
balanced accuracy is no more than0.5pp worse and neither domain is more than1pp
worse than shuffled timing. Separately show the same final-quality tolerances
against continuous training and whether either fixed stop dominates health on
risk, final balanced capability and update count. Do not treat these two flags as
controller qualification. All source intervals are exploratory and conditional on
shared readout pools; no simultaneous safety claim. Execute the full fixed budget
regardless of interim scientific findings; engineering failures still halt.

## Resource and integrity plan

Use the already pinned archive index/revision and bounded HTTP ranges. Limit total
intake network traffic to6GiB, including retries and ZIP index reads. Download one
temporary video at a time, verify CRC/length/SHA, decode512 frames, retain frame
hashes/PTS/source identity, and remove only that newly created temporary raw video
after verifying the cached bytes. No existing scientific file is removed. Strong
lossless NTFS compression may reduce the new frame caches' disk use; retain logical
hashes. Never create whole-stream float pixel caches: gather one batch from memmaps.

Freeze intake exclusions, ranking, exact clip roster and independent intake audit
before training. Freeze model/config/monitor/evaluation/code/test sources. Keep
512MiB free reserve during scientific writes. A selection or decode error preserves
evidence and uses only the declared infrastructure reserve rule, never new outcomes.

Scientific work:8*448=3,584 SSL updates.49 clocks give48 learned states per history,
each with three fits:1,152 new learned-state fits plus one shared three-fit warm
readout,1,155 canonical fits total. Primary probabilities cover the full canonical
4,063 layout; scoring uses only the3,663 primary queries. Reuse support/query roles
and the unchanged fine-tune/scoring-batch law. No additional labels are acquired.

Independent exact replay adds3,584 SSL updates and two extra classifier fits:
history00/update148/draw61001 and history07/update264/draw61001. Reproduce every
saved bank embedding, state fingerprint, loss/gradient trace and guard event.
Observation/readout/skip must preserve model, Adam, modes and CPU/CUDA RNG. Save
one stopped complete state per history for actual hold verification, compact
embeddings and probabilities elsewhere. Save all attempts and per-fit bindings;
never silently overwrite completed fits or completed scientific results.

Independent arithmetic reconstructs metadata allocation, bank geometry, alarms,
raw class-balanced counts, source aggregation, every control path, time-weighted
deficits, all40,320 shuffle assignments and scientific flags. Measure monitoring
forward/geometry separately from SSL; offline readouts/audits are additional work.
Tests cover metadata exclusion from the decision clock, future-prefix causality,
constant-clock equivalence, exact work preservation and duration weighting. Render
and inspect a new companion notebook and figures, update the detailed/concise study
and phase router, and preserve earlier notebooks. No automatic threshold search,
new predictor, pause/resume experiment or deployed controller follows.

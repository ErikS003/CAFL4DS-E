# Provisional temporal state development: execution protocol

Frozen before new accuracy outcomes, 2026-10-11. User instruction: run the
paired-state experiment. This is **provisional local-annotation development**:
the local scene labels have an undocumented provider. Local file hashes and
metadata consistency are checked; annotation authenticity and capture-session
independence are not claimed. The sealed final query role is never scored.

## Inputs and preparation

Use the repaired 2,256-row allocation and pinned community video archive
revision `2e48a0340eb591fa1918f3375d55601a7f0bbab3`. Reuse six qualified smoke
clips and acquire the remaining 138. Initial intake is capped at 3 GiB;
the complete new intake including replacements is capped at 4 GiB. Original
allocation, completed historical studies, and failed intake logs are preserved.

Decode 256 strictly advancing real timestamps per clip at 8 Hz, RGB64 PIL
bilinear resizing, with the qualified decoder and frozen timestamp-slot rule.
Insufficient duration or a video screen flag triggers an outcome-free same-cell,
same-role replacement. Reserve order is SHA256 of `duration-reserve:{clip_id}`,
excluding used IDs and identifier-prefix families owned by another role.
Screen keyframes and footage before any query predictions. Preserve every
failure; at most nine footage attempts per slot. Stop if reserves or transfer
budget are exhausted rather than weakening the screen.

Screen every decoded frame for exact RGB64 matches against historical frame
caches, historical/readout keyframes and accepted new clips. For near matching,
sample arrivals 0,64,128,192,255; compare dHash Hamming <=4 and RGB8 RMSE <=.03
jointly against all historical train/val keyframes, support/query keyframes,
sampled historical video frames and sampled accepted new clips. This sampled
screen cannot exclude all video overlaps or establish independent drives.
Reader/final pixels are used for exposure screening only, not scoring or tuning.

## Fixed cohorts and decisions

Use six disjoint 24-clip cohorts, four clips in each of the six clear-weather
day/night scene cells. Preserve chronology inside each clip. The concatenation
between clips is an explicitly artificial stream, with no natural recurrence
claim. This replaces the draft's proposed stable/transition/recurrence cohort
labels: opportunity is assessed against measured past content instead.

Canonical cells are day-city, day-highway, day-residential, night-city,
night-highway, night-residential. Sort each cell by SHA256 of
`order:{clip_id}`. Place four rounds of six clips; round shifts are the cohort
index modulo six, except the fourth round shifts three additional cells. This
gives matched scene classes and opposite day/night sources at the two anchors.
No ordering uses learner outcomes.

Use seeds 5003,5009,5011, full and uniform-half histories, and parents after
windows **26 and 74**. Both anchors occur halfway through a 32-second clip.
Each full/half decision covers the **same next two FIFO windows**, 128 arrivals
or approximately 16 seconds, ending at that clip's boundary. This supersedes
the draft's boundary anchors and four-window horizon so past content has a
continuous within-clip future to predict. History updates never cross clip
boundaries inside a window.

Full performs two chronological 32-frame updates per closed 64-arrival window;
half performs one update from positions 1,3,...,63. No compensation, replay,
encoder change, reader search or query-driven branch/state selection. Use the
hash-locked image-MAE warm checkpoint, mask .75, AdamW lr 1e-4, weight decay .05,
gradient clipping 1, deterministic CPU and four Torch threads.

There are 72 paired parents and 144 trained branches. Reconstruct histories
through window 74: 3,996 history updates plus 432 branch updates = **4,428 main
scientific SSL updates**. Each full branch has an independent repeated
continuation control: 288 separately counted discarded updates. Compare with
the unchanged parent as a hold/no-update readout reference.

## Small qualification and readers

Two six-clip, one-per-cell bundles are the first round of cohorts 0 and 1.
Use seed 5003, frozen/full/half, and unchanged readout settings. This costs
144 scientific SSL updates. Qualification is descriptive, with no half-rate
global noninferiority prerequisite, no outcome-based exclusion and no tuning
to force useful SSL. A negative result remains visible in the report.

Support draws are 51011,51017,51023, each from its own reserved support-panel
role. Select 25 images per scene class using SHA256 of
`support:{draw}:{clip_id}` within cells. Day counts are
`12 + (panel_index + class_index) % 2`; night supplies the remaining labels.
Draws are mutually disjoint and disjoint from streams and query roles.
Development queries are 128 per cell, 768 total. Load 225 selected support
images plus 768 query images; all image/label hashes must match allocation.

Primary accuracy is class-balanced scene accuracy after 100-step copied-model
fine-tuning on each 75-image support draw. Retain the historical fine-tuning
recipe, linear probe and cosine distance-weighted kNN-20. Average draws inside
each training seed; save raw query predictions. Readouts must preserve all
SSL model/optimizer/RNG/mode state. The main 216 branch/parent readouts cost
64,800 copied fine-tuning steps, plus 1,500 steps for the five qualification
readouts (one shared frozen and four trained). No complete-run ETA is frozen.

## Features, measurements and analysis

Seal mean adjacent pooled-RGB change, within-window pooled-RGB spread,
90th-percentile change and preceding-window mean shift from already observed
frames. Features may not use future arrivals, query labels/CE, branch training
loss or outcome. Record history identity and arrival/update clocks.

Primary response is full-minus-half fine-tuning accuracy in pp, balanced and
separately day/night. Retain full/half gains over hold and processing cost.
Measure selection plus training on resident-cache arrivals; decode, checkpoint
and readout overheads are separate and resident-cache savings do not establish
deployment savings. Alternate branch order deterministically across cohort,
seed, history and parent index.

Average nuisance seeds inside cohort/history/anchor states. Plot action response
against causal change and spread. The prespecified exploratory linear forecast
uses change, spread, full/half history and the late-anchor indicator. Compare
with intercept-only forecasting using leave-one-cohort-out predictions, with
feature scaling fitted only on training cohorts. No feature/model/threshold
search on held-out outcomes. Six artificial clip cohorts and shared queries
support development discovery, not strong generalization or independent
population claims. Do not substitute seed/panel/parent counts for source groups.

Report the proposed 1 pp quality-loss and 25% processing-saving bands only as
practical reference bands; do not certify half-preferred states from this run.
Record unresolved outcomes as unresolved. Sealed confirmation, an independently
powered interval family, a frozen filter policy and repeated policy rollouts
remain later experiments. Health, MPC, federation and GPU changes remain later.

## Execution, recovery and audit

Preparation and collection run under a hidden supervisor with scoped automatic
idle-sleep prevention. Manual sleep pauses CPU work. Save complete history
checkpoints after every window, parents at both anchors, and branch terminal
states. Resume only with unchanged source/configuration/data hashes. Preserve
failed stages and original evidence. Source snapshots are saved before scoring.
Verify exact restoration, parent immutability, every repeated full continuation,
finite SSL updates, and unchanged inputs at completion. Report scientific and
discarded controls separately. Do not mark source authenticity or session
independence verified merely because this development run finishes.

# Long BDD100K archive-order quarter-admission experiment

## User request and ordering

The user requests a longer, more natural stream and explicitly chooses video
frames in order, clips in dataset order. Use the existing pinned video archive's
member order. Preserve that order after excluding official non-train clips,
warm-training clips and fixed monitor/evaluation clips. Do not group, balance,
shuffle or select clips by illumination, weather, scene or measured outcomes.
Unknown attributes and rare scenes remain eligible. The metadata annotations
are for offline description and warm/excluded membership checks only.

This is an archive-order sequence of separate videos, not established capture
chronology or a continuous drive. The available metadata lacks ride/session
identities. The mirror revision and archive-member integrity remain pinned;
CRC and SHA256 establish retrieval consistency, not independent authentication
against an original BAIR archive. Primary dataset description:
https://bair.berkeley.edu/blog/2018/05/30/bdd/.

## Outcome-free intake

Reproduce the original warm partition from complete local train metadata:
valid time/weather/scene, scene floor 50, per-cell filename hash seed 20260921,
warm fraction .2 and Python round as in the existing source. Exclude that warm
set, the fixed monitor bank and all evaluation clip identities. Take the first
256 eligible archive members. Record every exclusion without changing order.
Previously inspected stream clips need not be excluded; report any overlap.

Decode actual source frames at the first advancing PTS at or after k/16 seconds,
using the established native RGB then PIL bilinear 64x64 resize. Retain source
frame indices, PTS, pixel hashes and real gaps. Decode complete selected videos
in order until 131,072 frames are obtained, truncating only the terminal prefix.
If the 256-video cohort ends sooner, use its complete prefix, dropping at most
31 terminal frames to form full batches. Require at least 2,048 batches before
training; otherwise record an infrastructure failure. Never interpolate,
repeat frames or substitute a failed clip. Decode failures stop preparation.
The final arrival count and all pixel hashes freeze before learner outcomes.

Network cap 8 GiB including retries and archive indexing. Keep a 768 MiB intake
and 512 MiB science disk reserve. Fetch only bounded member ranges. Remove only
new verified temporary videos after saving their cache and provenance; earlier
science is untouched. Intake attempts and failures remain recorded.

## Physical scientific comparison

Keep the previous warm TinyViT-MAE, full optimization scope, masking, optimizer,
batch size 32, seed 5003, fixed unlabeled monitor bank and canonical downstream
reader. One uninterrupted learner state per arm; no reset at clips or regimes.
Batches can cross clip boundaries. The desired horizon is 4,096 arrivals, over
nine times the earlier 448-arrival study; actual horizon follows the intake rule.

Extend only the monitor clock beyond the old 444 endpoint: reference at arrival
132, observations at 140,148,... through the stream. Two consecutive readings
with rank drop >=10%, normalized variance drop >=10% or CKA distance >=.10 fire.
The reference never changes. A confirmed alarm sets subsequent admission to
25%, then 50/75/100% in 32-arrival stages. A new alarm resets the stages, streak
and integer admission credit. Admission rejects whole batches; rejected inputs
are never replayed. Persistent alarms can maintain quarter-rate training.

Physically run full training, adaptive recovery, fixed quarter after the first
causal alarm, and evenly paced admission with the adaptive arm's exact final
suffix count. All share the same initial state and first-alarm convention.
Pacing uses a future total known offline: it is a conditional benchmark, not
an online policy. Derive permanent-stop capability from the exact first-alarm
state and reuse warm frozen. Entire identical masks may alias an executed
trajectory with identical initial state; record this explicitly. A no-alarm
stream legitimately gives identical arms and does not justify retuning.

Read capability at warm, 132, every 128 arrivals, the first alarm, and the final
arrival. Full training captures its causal first-alarm readout too. Canonical
fine-tuning uses the same three 50-label/100-update support/fitting draws and
3,663 primary validation queries. Labels and regime annotations never enter
admission, monitoring, checkpoint selection or optimization. State-identical
readouts may be reused only with complete learner fingerprints and matching
reader metadata. Readouts preserve learner modes, Adam and CPU/CUDA RNG.

Budget: at most 16,384 collection SSL updates, 432 new classifier fits, and
16,384 replay SSL updates plus two exact classifier replays. Intake horizon
and checkpoint count determine a tighter pre-outcome bound in the input lock.
No seed or horizon expansion after outcomes. Preserve all attempt accounting.

## Outcomes, verification and limits

Report complete day/night/balanced trajectories; gain over warm; largest
observed domain drawdown; exposure-weighted deficits and arriving batches below
earlier observed maxima; alarm/reset/rate traces; accepted updates/images and
timing. Deficits use sampled checkpoints and left-observation weights after
132. Report counts in arrival units; unrelated clips have no known physical
inter-clip time. Do not claim continuous observation between checkpoints.

Compare recovery with permanent stop, quarter, full and exact-work pacing.
Report all three readers, source attributes in archive order, temporal sections
and observed disadvantages. One natural stream and one SSL seed do not supply
independent-source uncertainty; do not calculate source-confidence intervals
from checkpoints, clips or the three shared readers. All efficacy remains
descriptive development evidence; no controller qualification flag can pass.

Before collection freeze config, code/tests, this protocol, intake/frames,
prior scientific dependencies, warm state, evaluation inputs and monitor bank.
Replay every unique physical trajectory and compare accepted masks, alarms,
loss/gradient traces, monitor embeddings and full learner fingerprints. Replay
two classifier fits. Independently reconstruct admission and raw probability
metrics, confirm dataset order, exclusions and all input/work hashes. Generate
four saved figures and a new notebook, update study/navigation docs, visually
review every figure and check final notebook embeddings and links. Earlier
scientific artifacts and notebooks remain byte-identical.

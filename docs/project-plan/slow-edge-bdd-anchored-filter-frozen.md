# Frozen temporal BDD anchored-change comparison

Frozen 2026-10-11 before collection. The user authorized implementation and launch
of the five-policy experiment in [the revision plan](slow-edge-bdd-filter-revision.md).
Prior protocols, evidence and sources are preserved. This protocol fixes the new
collection; the revision memo's later controller studies remain proposed.

## Question and causal rule

Does regular temporal sampling retain useful SSL adaptation at half the updates,
and does adding local content change improve on uniform and random sampling?
These are separate development claims. Do not require a content contribution to
recognize a qualified simpler uniform filter, or infer control from static success.

Use the locked 48 chronological development clips and all 6,144 RGB64 frames from
`outputs/temporal-bdd/20261011-frames-v3/intake.json`. Every policy acts after each
64-frame FIFO closes, with no window crossing a clip boundary. There are 96 windows
and two per clip. No later frame/window, task label, time-of-day metadata, query
outcome, learner feature, health metric or replay image affects decisions.
Delivery delay is up to roughly 8.5 seconds at the sampled cadence; this is a
windowed causal filter, not instantaneous per-frame acceptance.

Anchored change partitions each window into sixteen four-frame blocks. Retain each
block's last frame (3, 7, ..., 63). Average-pool RGB64 to RGB16; a frame's change
score is mean absolute difference from its immediate predecessor's pooled RGB.
Set the first score to zero, resetting every window. Do not center, L2-normalize
or fit a scale. Within each block's first three frames, retain the highest-scoring
frame; ties prefer earliest position. Exactly two frames per block train, in
chronological order. The score for a block's first frame uses the preceding frame
within this window. No score uses a predecessor from another window or clip.
The selector guarantees inclusion of the latest frame and no more than three
consecutive skipped frames; smoothing does not establish semantic utility.

## Policies and budget

| Policy identifier | Retained positions per 64 arrivals | Updates per trajectory |
|---|---|---|
| full | All 64 | 192 |
| fixed_half | Last 32, preserving the historical latest-half control | 96 |
| uniform_half | 1, 3, ..., 63 | 96 |
| random_half | Uniform 32 without replacement, sorted | 96 |
| anchored_half | Sixteen anchors and sixteen change-ranked candidates | 96 |

Five paired seeds: 5003, 5009, 5011, 5021, 5023. Random uses a private Torch
generator seeded learner-seed + 62003. No selector consumes learner RNG.
Budget: 25 newly collected trajectories and 2,880 SSL updates; discarded first-window
infrastructure preflight costs 2+1+1+1+1 = 6 updates separately. No new downloads.
Rotate policy execution order by seed index. Retained frames train in batches of
32; no skipped frame is accumulated, replaced or replayed.

Keep the full-method warm TinyViT-MAE, mask .75, AdamW lr 1e-4/weight decay .05,
gradient clipping 1, deterministic CPU Torch with four threads. No rate/objective/
learning-rate changes, extra seed collection or post-outcome descriptor alternatives.

Newly measure FIFO gathering, pooling/change selection, frame assembly and SSL in
the online processing boundary. Report selection and SSL components and all paired
costs. Acquisition, decoding, initialization, readout and audit are excluded.
Resident-cache timings do not establish deployment or energy savings.

## Correctness and readout

Validate both prior feasibility and content audits, every audited artifact hash and
their source/input locks before collection. Snapshot and lock new collector,
selector, report, supervisor and this protocol. Verify the restricted validation
pixel tensor against the earlier paired-study hash; reconstruct the same 900-image
readout allocation and hash the actual evaluation images/arrays.

Reproduce warm frozen exactly. Save deep-copied initial and final complete model,
Adam, RNG and learning-clock snapshots; initial states must match within seed.
The new full/latest-half seed-5003 controls must exactly reproduce the historical
chronological learning states, optimizer/RNG, loss/gradient sequences and raw
terminal predictions. Stop on mismatch, preserving the failed attempt.

Primary: copied-model fine-tuning using the same three support panels, 25 labels
per class, 100 steps and 600 query images. Linear and kNN are secondary. Offline
evaluation must preserve the live model, Adam, RNG and training mode. No query
labels enter selection. Report all policies and all domain contrasts.

## Fixed decisions

Paired two-sided Student-t 95% intervals over the five training seeds remain
conditional on reused development clips, the shared warm model and fixed panels.
They are not independent driving-session intervals or simultaneous family
confirmation. Uniform and anchored are two prespecified efficiency candidates.

Each candidate qualifies for static efficiency only if all hold:

1. Candidate-minus-full lower bounds exceed -1 pp for balanced, daytime and nighttime
   primary fine-tuning accuracy separately.
2. Candidate-minus-warm-frozen balanced lower bound exceeds zero.
3. Mean paired resident processing savings versus freshly timed full are at least 25%.
4. Every exact acceptance/update and audit requirement passes.

Report per-candidate gates separately. Joint decision is `static_efficiency_candidate`
if either qualifies, otherwise `no_qualified_static_efficiency`. Do not pool domains
to hide a failed guardrail or select a winning seed/reader.

Added content value is a separate intersection: anchored-minus-uniform and
anchored-minus-random balanced lower bounds both exceed zero. Report this even if
efficiency fails. A content advantage alone cannot qualify efficiency or control.
Latest-half and random contrasts remain mandatory descriptive comparators.

The independent report reconstructs all 2,400 window decisions, chronological
indices, counts and timings; verifies initial equality, final optimizer counters,
control reproductions and locks; and recomputes fine-tuning/linear/kNN results
from raw predictions. It publishes audit hashes and the fixed decisions.
`controller_qualified` and `independent_confirmation` always remain false.

## Following gate

No automatic rate sweep, health fitting, encoder sweep or new-data acquisition is
authorized by completing this collection. Review qualified static candidates and
unresolved outcomes. The separately frozen next study would measure same-state
temporal full/half action responses before any content/history controller, with
full-rate as fallback. A later fresh-data confirmation must freeze one policy
before new outcomes; revisiting these development queries is not confirmation.

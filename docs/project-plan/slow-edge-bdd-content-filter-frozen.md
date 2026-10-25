# Frozen temporal BDD content-dependent filtering

Frozen 2026-10-11 before new content-filter outcomes. The user authorized this
direction after the audited temporal feasibility study. Existing sources/protocols
and all completed evidence remain unchanged.

## Evidence and bounded question

The chronological study established a full-rate fine-tuning gain over warm frozen
of +0.867 pp [0.651, 1.083]. Fixed half-rate saved 47.5% resident processing time,
but its nighttime effect relative to full-rate was -1.333 pp [-1.919, -.748], so
the joint quality screen failed. Across clips, mean adjacent RGB difference was
.0281 versus .0901 for within-clip shuffling. This motivates a content test; pixel
redundancy does not establish downstream action utility.

Question: at exactly half the accepted frames and optimizer updates, does a cheap
causal content-coverage selector improve on both fixed and random selection, retain
full-rate quality in both domains and save processing time including its overhead?
This is a single static frame-filter test. It does not fit health or action-value
regressors. It is not independent confirmation or evidence for MPC/federation.

## Observation and causal contract

Keep the 48 locked development clips and all 6,144 ordered RGB64 frames from
`outputs/temporal-bdd/20261011-frames-v3/intake.json`. Preserve clip/frame/PTS trace
and warm/validation clip exclusion. Use chronological order only; do not add a
shuffled sweep. This is the same constructed daytime-to-night stream over different
drives, not one continuous ride. Frame labels remain absent at runtime.

Every policy receives a bounded FIFO of **64 already-arrived frames**. It selects
after the window closes, using no later frame. No window crosses a clip boundary:
each 128-frame clip contributes two windows. Retained frames stay in original
order, train in batches of 32, and leave the FIFO after the decision. No replay,
cross-window image memory or replacement of discarded images is permitted.

This is a windowed streaming filter with latency up to one 64-frame window
(about 8.5 seconds at the sampled cadence), not instantaneous per-frame selection.
All policies share that arrival/decision clock. Waiting wall time is a delivery
latency, not CPU processing time. Full-rate executes both 32-frame batches after
window closure; fixed-half executes its second batch. Their ordered optimizer
inputs and RNG sequence remain identical to the audited chronological controls.

Content descriptor: average-pool RGB64 to RGB16, subtract each frame's per-channel
spatial mean, flatten and L2 normalize (epsilon 1e-6). This removes overall
brightness/color and global contrast scale; constant images yield zeros. No learned
encoder, scene/time label, SSL loss, query outcome, future image, per-domain threshold
or online normalization fit enters this descriptor.

Content coverage selects 32 of the current 64 descriptors. Seed with the window's
first frame; repeatedly select the remaining descriptor maximizing its Euclidean
distance to the nearest selected descriptor. Ties prefer earliest remaining index.
Return the selected frames chronologically. This is a simple geometric coverage
baseline, motivated by [core-set selection](https://arxiv.org/abs/1708.00489), not
a reproduction of that paper's supervised active-learning model or a new method.
Only this descriptor and seed rule are collected: no post-query alternatives.

## Matched policies, learner and budget

| Policy | Frames retained per 64 arrivals | Updates per window | Total updates |
|---|---:|---:|---:|
| Full current-only | 64 | 2 | 192 |
| Fixed half | Last 32 | 1 | 96 |
| Random half | Uniform 32 without replacement | 1 | 96 |
| Content coverage half | Diverse 32 | 1 | 96 |

Random half uses its own Torch generator, seed = learner seed + 62003; draws at
each completed window and restores chronological selected order. It never consumes
the learner RNG. Content also consumes no learner RNG. Exactly 3,072 images train
each filtered arm, compared with 6,144 full-rate exposures. All arms see 6,144
arrivals and 96 closed windows. Random and content therefore cannot win by more
updates, higher acceptance in one domain, accumulating extra frames, or query tuning.

Keep the same full-method warm TinyViT-MAE well, architecture, objective, mask .75,
AdamW lr 1e-4/weight decay .05, gradient clipping 1, CPU/four threads and deterministic
Torch. No replay, learning-rate compensation or separate SSL family search.
Five paired seeds: 5003, 5009, 5011, 5021, 5023. All four arms are newly collected
for matched timing: **20 trajectories and 2,400 SSL updates**. A discarded first-window
preflight executes 2+1+1+1 = 5 updates, counted separately. There are no new downloads.

Rotate execution order by seed index through [full, fixed_half, random_half,
coverage_half] to reduce consistent order bias. Measure resident FIFO gathering,
descriptor computation, pairwise/greedy selection, selected-frame assembly and SSL
training inside the same online boundary. Save selection and SSL timing components.
Exclude acquisition, decoding, initialization, offline readouts and audit time.
Report individual paired timings and resident-cache scope; deployment cost remains
unmeasured. No cached content decision replaces runtime selection in timed arms.

## Initialization, readout and correctness

Validate the prior audit and all 108 prior artifact hashes and prior input hashes.
Lock new sources/protocol and shared caches. Load the existing validation source
and reproduce its complete restricted pixel tensor hash against the earlier paired
study, not only label allocation. Reconstruct the same three 25-label/class panels,
600 query positions and 100-update copied-model fine-tuning reader. Reproduce warm
frozen readout exactly; linear and kNN remain mandatory secondary outcomes.
Cache and hash the 900 actual readout images for the new independent audit.

Save initial full learning snapshots using deep copies and pre-update fingerprints;
require equal model/optimizer/RNG per seed. Save final snapshots, 96-window selection
trace, chronological source indices, losses/gradients and costs. Offline evaluation
must preserve model, optimizer, training mode and Torch RNG. For seed 5003, newly
collected full and fixed-half final model/optimizer/RNG, loss sequences and terminal
predictions must reproduce the corresponding prior chronological arms exactly.
Different delivery latency cannot justify altered learning states. Stop on mismatch.

## Frozen primary practical screen

Use the same copied fine-tuning balanced scene accuracy as primary. Compute paired
two-sided Student-t 95% intervals over the five fixed training seeds; these remain
conditional development estimates, not independent driving-session uncertainty.
The content filter qualifies only if every condition holds:

1. Content-minus-random and content-minus-fixed-half balanced lower bounds are > 0 pp.
2. Content-minus-full lower bounds are > -1 pp for balanced, daytime and nighttime
   separately. All guardrails must pass; averaged accuracy cannot hide nighttime loss.
3. Content-minus-warm-frozen balanced lower bound is > 0 pp.
4. Exact half acceptance/update counts hold, and mean paired processing-time savings
   relative to freshly measured full-rate are >=25%, including selection overhead.

All pass: `content_half_qualified_for_separate_policy_test`. Otherwise report
`content_half_not_qualified` and which lower-bound/cost screens failed or remained
unresolved. This intersection screen does not cherry-pick passing domains/comparators.
Do not expand seeds, tune descriptors or weaken margins after outcomes. Report
random-minus-fixed and all secondary reader results, even when unfavorable.

## Audit and following gate

The separate audit reconstructs every content/random/fixed decision from observed
windows and the private generator, checks budgets and source/frame provenance,
shared initial states, complete optimizer counters, control reproduction and source
hashes, and recomputes terminal accuracy from raw predictions. It derives the joint
decision and publishes all effects and costs with artifact hashes. Static filter
success never sets `controller_qualified` or `independent_confirmation` to true.

If qualified, freeze a separate local action-response study and then a simple
causal content/history policy with held-out clip groups and repeated rollouts.
Expose retention count as the future actuator only after its action response is
measured. Add learner/health observations in a separate ablation after the content
baseline works. If not qualified, close this content recipe and review the evidence;
no automatic descriptor sweep or health fitting is part of this collection.

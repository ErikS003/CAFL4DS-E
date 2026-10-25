# Paired full-rate versus half-rate update responses

Date: 2026-10-10. Frozen prospective development protocol. The user authorized a study of states where half-rate may be preferable. The [forward plan](slow-edge-ssl-filter-next-plan.md) proposes this bounded opportunity diagnostic. No prior experiment verdict is changed.

## Parents, actions and work

Use the five recurrence/fixed-half trajectories from the audited `outputs/update-filtering/20261010-half-rate` study, seeds5003/5009/5011/5021/5023. Decision offsets are36/108/180/252/324/396/468/540 arriving batches. These40 parents were saved prospectively with full model, AdamW, Torch RNG, mode, arrival-order hash and clocks. Parents within a trajectory are dependent. Responses are conditional on half-rate histories and reused BDD images/evaluation; they do not establish full-rate-history or independent-source transfer.

For each parent restore two independent complete copies, then process the identical next36 batches (1,152 images). Full-rate executes36 updates; half-rate executes18, accepting even absolute one-based arrival indices. Skips discard current batches without model/optimizer/RNG changes. No replay, label-based admission, altered learning rate, delayed accumulation, optimizer reset or new SSL objective. Use the original clean-warm TinyViT-MAE configuration, exact data allocation/runtime, four CPU threads and deterministic algorithms. Numerical learning settings remain unchanged.

Both copies restore the same saved training RNG before their first arrival. Unequal accepted-update counts intentionally consume different later RNG draws. Alternate branch order by seed position plus offset position. Runtime decisions depend only on arrival index and declared action; future images and query outcomes are not observations. No controller is fitted.

New collection:80 terminal branches,2,160 optimizer updates,69,120 optimizer-image exposures and92,160 image arrivals counted across branches. Compatibility: continue the first parent for72 arrivals under half-rate (36 updates), reproducing its original arrival108 model/optimizer/RNG/mode and all accepted losses exactly. Reproduce its original linear/kNN panels and health at108. Four discarded full-rate updates on a separately restored first parent validate finite full-action work. These40 preflight/compatibility updates are separate from collection.

## Terminal evaluation and cost

Primary: final equal-domain balanced accuracy after copied-model fine-tuning,25 labels/class, three support draws51011/51017/51023 and100 support-only updates per draw. Both domains' accuracy are guardrails. Always perform fine-tuning at the terminal arrival, including terminals that were not legacy visit boundaries. Linear/kNN and losses/gradient summaries are secondary. Reuse exact support/query identities; no online task labels, final-query fitted predictors or support-selected actions.

Evaluate only at the prespecified36-arrival terminal. Do not select a more favorable horizon. Warm-frozen evaluation is reproduced once for compatibility/reference; it does not turn a local branch comparison into full-system qualification. Copied-model work:24,000 collection steps,300 compatibility and300 warm-frozen =24,600 additional labeled support-only steps. No parent fine-tuning is needed to compare the two terminal branches.

Measure per-arrival resident-data CPU processing time, including image lookup, acceptance decision, accepted optimization, numerical loss conversion and trace construction. Exclude shared load/restoration, downstream evaluation and research serialization. Report restoration and total elapsed times separately. Costs are unequal by design. Paired phase timing reductions use the mean of five per-seed half/full ratios; this is a descriptive wall-time screen with all seed timings retained, not a replicated deployment-latency claim.

## What counts as a preferred action

Use eight prespecified offset strata, each with five paired seeds. Average support draws within seed; do not count40 parents or three supports as independent replicates. For each offset compute half-minus-full differences for balanced, A and B primary accuracy.

The quality tolerance is1pp, inherited as an explicit development preference from the preceding plan. Half-rate is an **efficiency-preferred phase** only when balanced and both domain lower bounds exceed-1pp and the observed mean half/full online-time ratio is<=0.75. This means acceptable quality for less work, not necessarily better accuracy. Half-rate has an accuracy advantage only if the corresponding balanced lower bound also exceeds0.

To avoid choosing whichever of eight phases looks favorable, primary preference labels use simultaneous Bonferroni Student-t bands across the24 prespecified quality contrasts: critical value t_(1-0.05/(2*24),df=4). Report ordinary paired95% intervals beside these conservative bands. These intervals quantify conditional seed variation, not source independence or all downstream sources of uncertainty. No offset, seed, margin or confidence rule changes after outcomes.

Full-rate is quality-required when the adjusted balanced or either domain upper bound for half-minus-full is below-1pp. Other states/phases remain unresolved. A negative mean or a nonsignificant difference alone does not settle preference. Individual seed-parent preferences are descriptive only; primary evidence concerns the eight phase strata, not a deployable per-image decision.

Frozen diagnostic verdicts: `candidate_state_dependent_opportunity` if at least one efficiency-preferred phase and one quality-required-full phase exist; `local_half_rate_candidate` if any efficiency-preferred phase exists without an established full-required phase; otherwise `no_qualified_local_half_rate_opportunity`. Always preserve all nominal/adjusted intervals, seeds, domain guardrails and timings. An outcome-selected best-action envelope is optimistic and is not a policy evaluation. No success claim follows from extra geometric diversity or a fitted forecast.

If a candidate exists, the next study must check other/full-rate histories, causal observations and repeated rollouts against static and content/history baselines. If opportunity is unresolved or absent, stop this local update-filter mechanism in the current environment; do not expand offsets/seeds until a pass or launch a cadence sweep automatically. Temporal-video qualification remains the separate alternative.

## Provenance, execution and independent audit

Before collecting branch outcomes, validate the complete parent audit, protected input/artifact hashes, numerical runtime, exact source/config/data reconstruction and restored-state compatibility. Snapshot new sources and this protocol; seal all input hashes. Each branch records parent digest, mask, every arrival identity, arrival/optimizer clocks, finite losses/gradients, timing, terminal raw predictions, final model/optimizer/RNG state and terminal scores. Parents are read-only.

The independent audit reconstructs absolute-index masks and next-arrival windows without the runtime selector, validates all parent bindings and restore fingerprints, budgets and cumulative Adam clocks, final-state finiteness, unchanged inputs, saved-output scoring, paired inference and the frozen preference rules. Automatic reporting writes the complete report, plot and phase1/latest-results documents only after successful collection/audit. Source/protocol edits are prohibited while active. Preserve failures; never overwrite a partial collection silently.

# Half-rate SSL update filtering: frozen development protocol

Date: 2026-10-10. The user authorized this experiment after reviewing the [forward plan](slow-edge-ssl-filter-next-plan.md). Freeze this document and scientific sources before outcome collection. Completed recurrence and MinRed decisions remain intact.

## Objective and information budget

Test whether useful few-label downstream adaptation is retained at half the optimizer-update rate, and whether deterministic timing improves over random timing. This is SSL-preserving efficiency development on reused BDD images, allocation, five seeds and clean-warm initialization. It is not independent confirmation, health control, forgetting mitigation or a formal adaptation-floor promotion.

Keep the exact completed recurrence study's TinyViT-MAE64px, width96/depth4, mask .75, full-backbone AdamW lr1e-4, weight decay .05, clip1, CPU four threads and deterministic runtime. The four A1-B1-A2-B2 visits contain distinct images; shuffled controls contain identical identities. Seeds5003/5009/5011/5021/5023. Each arm sees18,432 images in576 batches of32. No replay, frozen geometry, labels or health observations are used for runtime selection. The fixed-rate choice is motivated by explicit compute savings, not post hoc MinRed subgroup outcomes.

## Policies and clocks

Always-update current-only and warm-frozen accuracy controls are reused only after audited input/artifact/runtime validation, complete data/config/frozen readout reconstruction and an exact576-update current-only seed5003 reproduction. The new runner itself performs the reproduction, with identical predictions, trajectories, losses and encoder delta required. A failed check stops collection.

New arms: fixed-half and random-half, both schedules, all five seeds:20 trajectories. Fixed-half trains on even one-based arrival steps. Random-half selects72 out of each144-step index block by private CPU Torch randperm, generator seeded with stream seed+62003. Generate masks before training; use the same seed-specific random mask on both schedules. This is image/model/label-independent and does not consume global learner RNG. Fixed phase, mask rule and seed offset never change after outcomes.

The exact-budget random baseline assumes a known benchmark horizon; it is not an unknown-horizon deployment policy. Block boundaries depend on arrival indices, not domain labels. Each batch is retrieved on arrival. Rejected batches are discarded; there is no delayed accumulation or compensation. Skips do not run SSL forward/backward, consume augmentation/masking RNG, call optimizer steps or alter optimizer state. Optimizer-step and arrival clocks are logged separately. Accepted updates use all32 current images. Keep constant lr; no extra epochs, tuning or model resets. Training RNG is paired by initialization; unequal numbers of updates intentionally imply unequal random draws.

## Outcomes, timing and decision

Evaluate at arrival0 and every36 arrivals through576. Copied-model fine-tuning at0/144/288/432/576,25 labels/class, three support draws51011/51017/51023,100 support-only steps, remains primary. Average support draws within seed. Linear/kNN, CE where existing readers supply it, domain trajectories, interference/recovery and shuffled results are secondary. Use exact old support/query identities; labels never enter online SSL or selection. All intervals are paired five-seed Student-t95% intervals, conditional on reused data; guardrails are nominal, not simultaneous coverage.

Primary policy/schedule: recurrence/fixed-half. Declare `half_rate_efficiency_lead` only if every gate passes:

1. Final balanced fine-tuning fixed-half minus always-update lower bound >-1pp.
2. Current-domain endpoint differences at144A/288B/432A/576B each have lower bound >-1pp.
3. Final fixed-half minus warm-frozen lower bound >0.
4. Each filtered arm has288 updates and9,216 optimizer-image exposures, exactly half of always-update.
5. Recurrence/fixed-half seed5003 measured online seconds <=75% of the matched seed5003 always-update reproduction.

The1pp margin caps a practically meaningful development quality loss in the existing primary readout; it is not inferred from observed half-rate outcomes or a powered equivalence guarantee. The25% wall-time screen demands substantial savings after overhead despite50% fewer updates. Timing is a descriptive single-machine/single-run gate, not a confidence claim; show all seeds, paired fixed/random timing and block variability, and distinguish quality qualification from timing qualification. Do not estimate comparable time from legacy elapsed logs.

Both arms are CPU/resident-data, fixed-thread runs. Warm up with discarded finite updates before timing; exclude shared load/initialization, evaluation, research artifact serialization and checkpointing. Each per-arrival timer includes batch lookup, mask lookup and execute/skip work, loss/gradient reporting conversion and ledger-entry construction. Sum per-arrival seconds; report four block totals. This covers online processing conditional on resident data, not disk-decoding/end-to-end deployment latency. Fixed-half seed5003 runs immediately after the control reproduction; later arms alternate policy and schedule order by seed to expose order variability. No per-run sleeping/system-tuning intervention is used.

Mandatory comparisons: fixed/random versus always and frozen, fixed-minus-random, both schedules, all readers. A positive recurrence fixed-minus-random final primary lower bound establishes a conditional timing-policy lead. Passing efficiency without this difference establishes simple reduced-rate learning only. Random-half qualification is secondary and cannot replace the primary fixed-half gate. A failed or unresolved gate records `no_primary_efficiency_lead`; report the failed dimensions and preserve uncertainty rather than declare harm or extend collection to obtain a pass.

## Work, state and independent audit

New collection5,760 updates/184,320 optimizer-image exposures and368,640 arriving images (counted across20 arms). Compatibility adds576 updates/18,432 exposures. Disposable preflight adds eight accepted updates (four always, two fixed, two random); all discarded. Copied downstream optimization adds25,500 steps:300 warm-frozen,1,200 compatibility and24,000 collection. Frozen models have zero online updates. All costs remain visible.

Persist mask manifests, every arrival identity/acceptance/cumulative-update count and timing, accepted-step finite losses/gradients,17 readouts, four encoder boundary checkpoints and eight complete model/optimizer/Torch-RNG decision snapshots at arrival36/108/180/252/324/396/468/540. Snapshots are for potential future paired branching; no branches or health fitting run in this experiment. Save and hash exact source/protocol/config/runtime/data inputs. Protect the prior recurrence and MinRed studies and their original locks.

Audit independently reconstructs masks using separate code, checks exact per-block budget, causality/exposure identities, all finite accepted updates, arrival/order hashes, saved-state counters, unchanged inputs and compatibility. Rescore all new and reused control predictions independently, recompute intervals/guardrails and timing gates, and retain every seed/schedule/readout. Only a successful audit writes the final report and experiment summaries. No source/protocol edits during active collection or audit.

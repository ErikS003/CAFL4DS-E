# Matched-exposure history order: frozen development experiment

Primary question: after identical exposure and update counts, does day→night versus night→day history change the benefit of further updates on new day footage? This is a new focused experiment, not a relaxed reanalysis of earlier simultaneous screens.

## Histories and actions

Start both histories from the same per-seed warm learner, fresh Adam and CPU/CUDA RNG. Seeds5003/5009/5011. Use the original day1 and night1 frames,4224 each, preserving within-regime order. DN presents day then night; ND presents night then day. Both train on exactly the same8448 distinct historical images once,264 updates of32, with unchanged optimizer/SSL recipe. Save the immutable warm state before either history. Reconstruct both orders for all seeds:1584 scientific history updates. DN complete learner and predictions must exactly reproduce the audited original night-end parent.

From each order/seed parent, branch hold versus full on identical1024-frame continuations. Primary:12 new day blocks, each512 city plus512 highway frames. Secondary:five previously decoded new-to-parent night blocks from the crossed-state study; these are reused diagnostics, not fresh confirmation. Alternate city/highway order by block parity. Full32 updates, hold zero. Total204 branches/3264 continuation updates plus1584 parent updates =4848 scientific updates. Primary fine-tuning fits324:18 shared-parent fits and306 full-branch fits. Preflight and exact replay separately charged.

## New-source allocation and media eligibility

Select12 clear-day train clips per city/highway cell using SHA256("history-order:20261013:"+filename). Preserve the original warm/stream partition seed20260921/fraction0.2/minimum scene count50. Exclude all previously downloaded video IDs and original48 clips. Freeze the full ranked eligible pool and selection before decoding. Source clips remain disjoint from parents/warm/evaluation. Decode512 real increasing-PTS frames using the unchanged16Hz/PIL RGB64 algorithm with its requested count set to512. This count was fixed before any new clip decoding or scientific collection; the prior draft using528 decoded frames is preserved in the intake directory. All512 decoded frames are exposed.

Before any new model outcomes, a clip that cannot provide512 real frames may be replaced by the next unused member of its frozen same-cell eligible pool. Preserve original allocation, failed log, replacement identity and hashes; no outcome-dependent source changes. Fetch only missing pinned-archive members, with an aggregate768MiB transfer cap across retries/replacements. Validate member CRC/length, source SHA256, timestamps and pixel hashes. Capture sessions unknown; source-block inference is conditional on clip-level separation and reused parent/evaluation populations.

## Fixed primary test and gatekeeping

For each day block, average the three support panels and three optimization seeds. Compute each order's full-minus-hold primary balanced accuracy, then ND response minus DN response on that identical block. The single primary two-sided95% paired Student-t interval across12 block means (df11) must exclude zero and the absolute mean effect must be>=0.5pp to qualify a history-dependent response. No training-age difference exists by construction; a positive effect means ND responds more favorably.

Budget planning uses the earlier day interaction's source-block SD≈1.55pp. With12 blocks the expected ordinary half-width is≈0.98pp. This is a planning assumption, not a guaranteed realized precision. Fix12 blocks without optional stopping or automatic expansion.

Only after the primary history-effect gate passes, test a joint action-preference flip on day footage. Use Bonferroni six scalar bounds (DN and ND full-minus-hold × balanced/day/night; alpha0.05/6) for this gated family. Prefer full when balanced lower bound>0 and mean>=0.1pp, with both domain lower bounds>−1pp. Prefer hold using the reversed bounds and identical tolerances. A flip requires opposite preferred actions at the same horizon on the same primary block population. Report ordinary/guard-adjusted intervals regardless; primary effect without a flip establishes changed response, not a qualified controller. Night comparisons and interactions, other readers and observer associations remain secondary. Earlier broad-family gates remain unchanged.

Primary readout remains the400-query two-class day/night few-label encoder fine-tuning task with three fixed50-image support panels. Linear/kNN are secondary; no reader selection from outcomes. Save all predictions and reconstruct statistics independently. The primary test concerns admission utility, not absolute parent accuracy: both orders may have different hold baselines.

## Observer and audits

Before an action, after first32 continuation frames arrive, record the same current reconstruction loss, difference from the last four actual history-training losses, and normalized feature-centroid distance from the preceding128 frames in that order. Isolate observer RNG and preserve learner/Adam/modes/CPU-CUDA RNG. No labels/future frames choose actions; no new signals, thresholds or controller fitting. Record6MiB raw past-buffer footprint and observer time; original maintenance overhead is not charged.

Freeze scientific sources/config/data before parent construction or fits. Audit identical initial states, equal image multisets/update/Adam clocks, order-specific index traces, original DN reproduction, both history and branch exact replay, hold invariants, all210 complete scientific terminal states (six parents plus204 branches),972 new prediction sets,102 observer reproductions and independent primary/guard arithmetic. No automatic follow-on training or policy fitting. This isolates data-order/history effects as a bundle including resulting optimizer/RNG state; it does not isolate a single health mechanism.

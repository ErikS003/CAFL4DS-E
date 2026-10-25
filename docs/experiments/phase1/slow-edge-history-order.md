# Matched exposure, reversed history order

The [crossed-parent study](slow-edge-cross-state.md) found weaker update responses in later parents, but those parents had more training exposure. This experiment holds the historical images and update count fixed and changes their order. It tests whether training age alone can explain the response difference.

## Method

The [frozen protocol](../../project-plan/slow-edge-history-order-frozen.md) starts both histories from identical immutable warm learner/Adam/RNG states within each seed. DN presents the original4224 day1 frames followed by4224 night frames. ND presents the same frames in reversed regime order, preserving within-regime order. Both finish at8448 historical arrivals and264 updates. The historical frame multiset and optimizer clocks match. Both orders are reconstructed for seeds5003/5009/5011; DN learner states and predictions must reproduce the original audited night-end parents exactly.

Each parent branches into hold or32 full updates on identical1024-frame blocks. The primary population is12 new day city/highway clip pairs,512 real frames per clip. Five previously decoded night pairs are secondary diagnostics. The24 primary clips were selected under a fixed hash from the warm-disjoint train partition, excluding all previously downloaded videos. A512-frame decoder count was fixed before decoding; its RGB64 resize and16Hz real-timestamp sampling algorithm are unchanged. No source replacements were needed. Night outcomes for DN were already known and are not fresh confirmation.

The unchanged primary reader measures few-label encoder fine-tuning capability on400 balanced two-class day/night queries using three fixed50-image support panels. Linear and kNN are secondary. Average support panels and three optimization seeds within each source block. The single primary contrast is ND-minus-DN full-versus-hold response on identical new day footage. Estimate its paired95% t interval across12 block means (df11). Qualification requires a nonzero interval and absolute mean>=0.5pp. After that gate, a six-scalar Bonferroni family checks each order's balanced/day/night action effect for a joint preference flip. Day/night guardrails remain−1pp; no gate is relaxed after outcomes.

## Findings

| History | Continuation | Hold balanced % | Full balanced % | Full minus hold pp [ordinary95% CI] |
|---|---|---:|---:|---|
| Day→night | New day |59.806|56.745|−3.060 [−3.381,−2.740]|
| Night→day | New day |57.306|57.176|−0.130 [−0.337,+0.078]|
| Day→night | Reused night |59.806|59.850|+0.044 [−0.972,+1.061]|
| Night→day | Reused night |57.306|57.272|−0.033 [−0.959,+0.893]|

The primary matched-age history interaction is **+2.931pp [+2.485,+3.376]**. Every one of the12 new day blocks favors ND over DN in update response. This passes the frozen history-effect rule: training order changes admission utility even at identical exposure and update age. The realized primary half-width is0.445pp, below the≈0.98pp planning estimate; no source-block expansion or optional stopping occurred.

The effect is a change from harmful updating to approximately neutral updating. It is not a demonstrated beneficial-update alternative. DN day training reduces primary day capability by2.162pp and night capability by3.958pp. Hold is preferred there under the gated six-scalar bounds. ND day updating is near-neutral in balanced capability, and neither full nor hold clears balanced superiority. Thus no same-age hold/full preference flip qualifies. Decision: `history_changes_action_response`.

Absolute parent quality differs. DN's hold baseline59.806% is higher than ND's57.306%; after full day continuation ND ends57.176% versus DN56.745%. The interaction compares changes relative to each parent's own baseline. It does not establish that ND is the better parent overall. Baseline capability/headroom is part of the saved-state difference and remains conditional on the fixed evaluation procedure.

Secondary readers corroborate the day interaction: linear+1.502pp [+1.064,+1.941], kNN+2.234pp [+1.736,+2.732]. DN day updating hurts both secondary balanced readers, while ND day changes remain unresolved. The Night diagnostic primary interaction is−0.078pp [−1.680,+1.524]. Its reader responses disagree: ND night updating lowers linear accuracy but improves kNN. No nighttime promotion is claimed.

## Observer measurements

The observer only receives the first32 arrived images and preceding128 images in that parent's actual historical order. It records reconstruction loss, difference from the last four training-batch losses and normalized feature-centroid distance. All102 observations reproduce with learner/Adam/modes/CPU-CUDA RNG unchanged.

| Cell | Incoming loss | Difference from recent training | Feature distance |
|---|---:|---:|---:|
| DN/day |1.246555|+0.462390|0.621553|
| ND/day |0.910230|+0.031237|0.266243|
| DN/night |0.843668|+0.059502|0.174132|
| ND/night |0.921783|+0.042790|0.924029|

Large relative reconstruction loss accompanies DN day harm, while ND day is near-neutral with a smaller loss difference. These observations are candidate inputs, not validated thresholds. Large feature distance also occurs in near-neutral ND night adaptation. No utility predictor, health threshold or controller is trained. Observer timing and the6MiB historical pixel-buffer footprint are retained; original buffer maintenance is not charged and no end-to-end efficiency claim is made.

## Validation and trace repair

Scientific work: six parent histories plus204 branches,4848 SSL updates and324 primary fits. Discarded preflight:4 updates and3 fits. Exact planned replay:4848 verification updates. All210 complete terminal states,972 new prediction sets,108 reader/metric intervals, both image multisets/Adam clocks and102 observer reproductions pass. Independent verification checks2404 locked inputs,12672 historical source frames and17568 continuation source frames. All hold states remain invariant. Every DN parent and prediction reproduces the prior audited result. Twelve targeted tests and scoped lint/format checks pass.

The first invocation saved the first DN parent state but failed while serializing its trace: its callback returned tensor-bearing state instead of metadata. No parent readout or continuation branch had run. The repair changes callback return metadata and adds preservation/checking of existing warm and parent states. The original source, input lock, status, cached-parent fingerprint and repair are archived in `trace-serialization-repair/`. Replaying the cached264-update parent reconstructed its missing trace and matched the complete state exactly; these264 extra updates are verification work, not another independent scientific trajectory. No learner settings, source allocation, primary test or guard changed. Total updates including all checks are9964. The earlier attempt took28.900 seconds; the successful runner took692.412 seconds. The successful invocation reused the original preflight. Video transfer was below the768MiB cap, without replacements.

## Interpretation for adaptive filtering

Update age alone does not explain the measured response magnitude. Training history changes the effect of updating on common footage, with secondary-reader corroboration. This supports considering history/state in admission research.

However, both matched-age parents lack a qualified positive-update option here. A fixed stop at this age remains a serious baseline: changing harm to neutrality does not demonstrate that a controller should choose different actions. The experiment does not qualify a resume decision, transferable utility predictor or adaptive policy. Further research must earn additional value over fixed cadence/stopping before controller promotion; no follow-on collection was launched.

The effect is conditional on the same warm initialization, three optimization seeds, historical clips and reused downstream evaluation panels. Twelve new source blocks do not represent twelve independent training histories. Capture-session identifiers remain unknown. The intervention changes the resulting weights, Adam and RNG/history bundle; it does not isolate a single health or forgetting mechanism. The intervals quantify source-block variation conditional on the fixed parents/readout, not full parent/query-population uncertainty. Formal phase and sealed-test gates remain unchanged.

## Artifacts and reproduction

Output: `outputs/history-order/20261013-gpu`. Frozen inputs/source snapshots, immutable `warm-parents`, `history-indices.json`, six `parents` state/trace/readout directories, per-branch states/traces/predictions, observer JSON, summary/report, `history-order-contrasts.png`, replay and independent audits, runtime freeze, repair record and closure are retained. New-source intake/pool are under `outputs/history-order/20261013-intake`; decoded primary frames and contact sheet are under `outputs/history-order/20261013-frames`.

Run `.venv-gpu\Scripts\python.exe -m scripts.run_history_order_experiment --output outputs/history-order/20261013-gpu`, then `.venv-gpu\Scripts\python.exe -m scripts.verify_history_order_experiment --output outputs/history-order/20261013-gpu`. Use `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic verification. The primary test and preference guards are explicitly distinct in `summary.json`; the plot labels both matched-age histories and marks reused-night diagnostics separately.

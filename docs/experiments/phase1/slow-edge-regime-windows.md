# Sustained temporal day/night/day: windows and filters

## Question and fixed method

User-directed development pilot under the [frozen protocol](../../project-plan/slow-edge-regime-windows-frozen.md). Test sustained adaptation, absence-period retention and return response with admission widths32/128/512, full SSL references and uniform/random/live-loss half filters. The earlier recurrence study used shuffled still images; this study preserves real order within video clips and allows windows to cross regime switches.

The available temporal allocation has no night residential clips. Use city street/highway as shared scene classes, matched four clips/class in each visit, with disjoint day1/night/day2 clips. Outcome-independent hash ordering chooses24 original train videos. Each contributes528 distinct timestamped frames at a nominal16Hz over33 seconds; each visit4224 arrivals, roughly4.4 minutes of concatenated footage,12672 total. These are constructed visits across unrelated drives, not a continuous day-to-night ride. Keyframe attributes construct the benchmark and do not guarantee constant per-frame scene labels. Fixed first-clip/midpoint/last-frame contact sheets confirm the broad illumination distinction and illustrate changing scene content within a clip.

Retain the warm TinyViT MAE, normalized reconstruction objective, mask.75, light augmentation, batch32, AdamW1e-4/decay.05 and clip1. Model/optimizer never reset at regime or clip boundaries. No historical replay or health steering. Three paired seeds5003/5009/5011;36 trajectories. Full admits all frames/396 updates; every half filter admits6336 frames/198 updates, at every width. Full references share each width's waiting boundaries.

Uniform selects odd window positions; random samples half without replacement with private RNG. Live-loss selects highest pre-update MAE reconstruction loss, earliest ties, with isolated scoring RNG/modes and fresh training views. Width32 half admission queues16 accepted frames until a32-frame learner batch can form. Large buffers close naturally, with no flush at4224 or8448. Width512 windows mix regimes; the terminal384-frame partial window closes only at12672.

Five checkpoint clocks4224/4480/8448/8704/12672 represent first-day endpoint,256 arrivals into night, night endpoint,256 into returning day and final day endpoint. Readouts never train pending or accepted-but-untrained frames. The complete buffers, private selector RNG and model/Adam/modes/CPU-CUDA RNG are saved. Width512 can show no early night response because the next mixed window is still pending; compare with its matched full reference to understand delay.

## Readout and interpretation

Original validation images/labels are verified against the earlier cache. Restrict original support/query membership to the two shared classes:50 support labels per panel (25/class), three fixed panels,400 queries in four100-image cells. Primary readout is a fresh encoder copy with a two-class head and100 support-only fine-tuning steps; linear/kNN secondary. Labels never enter online SSL or admission. This measures later few-label adaptability, rather than an online supervised classifier, and is a new two-class task rather than exact replication of preceding three-class accuracy numbers.

Day/night/balanced curves show first-visit learning, early and endpoint night adaptation, day change while day is absent, early and endpoint day recovery, and night change during day2. Observations are sparse checkpoint samples, not exact recovery time or continuous learning curves. Without a no-training continuation branch, decline during a regime block cannot be attributed uniquely to regime-induced forgetting rather than continued updates more generally.

All paired ranges use exact integer counts, panels averaged within optimization seed, and descriptive Student-t95% intervals with three seeds/df2. These are reused-data conditional diagnostics, not independent-drive confidence, multiplicity-adjusted confirmation, or proof of useful control. No forgetting is assumed. A provisional absence-decline flag requires positive first-day learning and negative day-retention upper bounds. Provisional efficiency flags require useful final SSL, matched-full final day/night/balanced and retention-change -1pp guardrails, and25% charged savings; every candidate is reported.

## Preparation and integrity

An inherited100ms timestamp-slot screen rejected a genuine395ms source gap. Before any model outcome, the original protocol/decoder and both failed logs were preserved, and preparation was amended to retain/report natural gaps while requiring528 distinct source frames with strictly increasing PTS. Successful frames were not duplicated or interpolated. A separate first GPU input-verification attempt failed on a missing audit subdirectory; zero SSL updates/fits preceded its repair. The current scientific protocol is the amended source lock.

Preflight136 discarded SSL updates/3 primary fits tests every width/policy, scoring purity, exact nonempty Adam/RNG continuation and copied-readout isolation. Seventeen targeted tests cover window budgets, transient queues, prefix causality, partial tail flushing, scoring RNG/modes, exact-count ties, retained source gaps and JSON replay metadata. Critical lint/format checks pass. All8910 scientific SSL updates/543 primary fits completed, including one shared frozen readout and180 learned checkpoints. Exact audit replayed8910 updates without downstream refitting;1629 saved prediction sets and source/selection/transient-buffer statistics were independently reconstructed.

The initial replay reproduced all five complete states of the first full trajectory, then stopped because JSON list records were compared directly with native loss/gradient tuples. Only that metadata comparison was normalized. Original collection lock/source and failed audit are retained; an AST comparison verifies that every non-audit function is unchanged. The audit-only closure reused all fits/predictions. Its successful8910-update replay is separate from the396 updates in the failed audit. The repaired audit source/lock and hash chain are recorded in audit-metadata-repair.json.

## Results

**Completed decision: no_efficiency_pilot_lead_qualified.** Useful final SSL is positive in every arm, but none of the nine half-policy/window combinations clears all frozen efficiency guardrails. No absence-decline flag qualifies for day performance during the night block. This does not certify universal retention or a working health controller.

The primary full-SSL means atwidth32 (width128 shares these endpoints) are:

| Observation | Day accuracy % | Night accuracy % | Balanced accuracy % |
|---|---:|---:|---:|
| Warm frozen |53.167|58.833|56.000|
| After day1 |54.000|59.667|56.833|
| After night |55.833|63.778|59.806|
| After returning day2 |54.389|62.056|58.222|

**The primary trajectory does not show the expected day loss during night.** Full day change during night is +1.833pp [+0.029,+3.638], and night adaptation is +4.111pp [+2.244,+5.978]. Initial full day learning is +0.833pp [-0.823,+2.489], so its first-visit learning component alone remains unqualified despite positive means in all three seeds.

**Returning day2 is the problematic block on the primary readout.** Full day change from night-end to final is -1.444pp [-2.306,-0.583]; night change is -1.722pp [-3.053,-0.391]. Both are negative in all three seeds. Full final balanced gain versus warm frozen remains +2.222pp [+0.875,+3.569]. This supports useful SSL while identifying a later capability decline. It does not identify whether illumination recurrence, the particular return clips, continued optimization or another source/task factor caused the decline.

Reader dependence matters: full linear day accuracy changes62.167% ->62.000% during day2, and kNN54.278% ->54.167%, while linear night falls64.778% ->60.611% and kNN59.056% ->57.111%. The night decline is visible on both secondary readers; the larger simultaneous day decline is a property of the primary fine-tuning capability, not equally strong frozen-feature classification degradation. These secondary values are descriptive means; their uncertainty is not substituted for the primary gates.

**Window size changes arrival-time response.** Atwidth512 the first switch leaves128 day arrivals pending at4224 and384 total pending at4480. No update occurs between those observations, so early-night response is exactly zero for every policy. Width32/128 have already updated. At the return checkpoint8704 the mixed512-frame window closes. This is genuine algorithmic delay, not a failed readout or a boundary flush. All full and uniform-half terminal learner states and predictions match exactly across widths: the admitted/training batch sequence is unchanged by width for those policies, although intermediate update timing changes. Random/live-loss can change their selected populations and therefore terminal outcomes.

| Window | Full final balanced % | Uniform half % | Random half % | Live-loss half % | Uniform/random saving % | Live-loss saving % |
|---|---:|---:|---:|---:|---|---:|
|32|58.222|57.361|57.361|57.389|49.7 /51.2|21.9|
|128|58.222|57.361|57.111|57.139|47.2 /48.8|20.2|
|512|58.222|57.361|57.417|56.833|51.0 /50.9|24.6|

Uniform/random save about half the charged processing work, but full-reference domain/retention guards remain unqualified. For example uniform-width32 final-minus-full is -0.861pp [-1.903,+0.181] balanced and -1.556pp [-2.597,-0.514] daytime. Live-loss never reaches25% mean savings and does not establish added content value. Atwidth512 its final-minus-full daytime contrast is -2.000pp [-2.717,-1.283], a particularly poor outcome on this conditional pilot, and balanced contrast is -1.389pp [-2.584,-0.194]. Report all candidates; do not select a deployed winner from these reused results.

Crossing windows affect allocation: live-loss-width512 averages1976.3 night admissions (46.8% of the night block) versus2232.3 returning-day admissions (52.8% of day2), despite exactly50% admission per completed window. Uniform admits2112 per visit. This diagnoses a selection bias in mixed windows, not a causal explanation of the accuracy differences or proof that reconstruction loss measures semantic value.

![Primary checkpoint trajectories](../../../outputs/regime-windows/20261013-gpu/trajectories-minutes.png)

Plot lines connect sparse sampled seed means; shaded time is the constructed nominal16Hz schedule, not an actual continuous drive. Axes share scales within each evaluation domain; the source's395ms gap is retained. Intervals are three-seed/df2 conditional diagnostics, with no simultaneous multiple-comparison or source-independence claim.

## Verification and implication

Independent verification passed2199 locked hashes, all12672 source timestamps/identities,36 trajectories,180 complete checkpoint/budget/Adam-clock checks, all transient-buffer pixels,1629 prediction sets and480 primary paired intervals. It independently regenerated admission/exposure queues and all24 crossing windows, checked shared-class/disjoint-visit allocation, and verified identical full/uniform terminal states and predictions across widths. Scientific readouts were not repeated after the audit repair. The original collection attempt including failed audit took953.246s; successful audit-only closure245.420s. Decoder preparation and the failed57.173s input-verification attempt are separate. GPU scientific work8910 updates, successful verification8910 updates, failed audit396 updates, discarded preflight136 updates;543 scientific primary fits plus3 discarded preflight fits.

Keep SSL and full as the quality reference. This pilot's useful next causal question concerns the return block: compare continued-day training with holding the prior state and a matched continued-night control, before fitting health steering or tuning a window/score. The observed primary decline should not be renamed automatic regime-induced forgetting; the controlled source groups and reader distinction remain essential. No additional experiment is launched by this interpretation, and formal phase gates remain unchanged.

## Artifacts

[Full all-arm report](../../../outputs/regime-windows/20261013-gpu/report.md), [summary and contrasts](../../../outputs/regime-windows/20261013-gpu/summary.json), [independent verification](../../../outputs/regime-windows/20261013-gpu/independent-verification.json), [training replay](../../../outputs/regime-windows/20261013-gpu/training-replay-audit.json), [secondary readers](../../../outputs/regime-windows/20261013-gpu/secondary-readouts.json), [regime admissions](../../../outputs/regime-windows/20261013-gpu/regime-admission-counts.json), [closure](../../../outputs/regime-windows/20261013-gpu/closure.md), [frozen protocol](../../project-plan/slow-edge-regime-windows-frozen.md), [decoded intake](../../../outputs/regime-windows/20261013-frames/intake.json), [input lock](../../../outputs/regime-windows/20261013-gpu/input_lock.json), [source contact sheet](../../../outputs/regime-windows/20261013-gpu/source-contact-sheet.png). Raw frames/timestamps, original collection/audit source snapshots, state/prediction artifacts and actual dependency versions are retained beside the run.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:MPLCONFIGDIR = "$PWD/.cache/matplotlib"
$env:PYTHONUTF8 = '1'
.\.venv-gpu\Scripts\python.exe -m scripts.run_regime_windows_experiment --output outputs/regime-windows/20261013-gpu --frames outputs/regime-windows/20261013-frames
.\.venv-gpu\Scripts\python.exe -m scripts.close_regime_windows_experiment --output outputs/regime-windows/20261013-gpu
.\.venv-gpu\Scripts\python.exe -m scripts.verify_regime_windows_experiment --output outputs/regime-windows/20261013-gpu
```

Host-preloaded research data do not imply whole-process memory bounded by the admission window. Processing timing includes all live-loss scoring, gathering, transfers, admission/assembly and synchronized SSL; it excludes preparation/decoding, snapshots and evaluation. Report these separately. Source/capture independence and fresh confirmation remain outstanding; no sealed final queries or provisional annotation archive are used.

# Accepted-batch coverage admission

The [frozen protocol](../../project-plan/slow-edge-coverage-admission-frozen.md) tests whether accepted-content coverage improves allocation of a192-update SSL budget beyond cheap paced admission. The four-arm design was authorized by the user. It is a content-coverage test, not a learned utility/health controller or a leave-budget-unused policy.

## Method

Four reused scheduling development streams provide a fixed feasibility screen; six new warm-disjoint clip-disjoint streams provide the primary comparison. Each fresh stream contains24 clear day/night city/highway clips,512 timestamp-selected16Hz RGB64 frames per clip,12288 arrivals total. Short/medium/long pair-domain patterns and both starting regimes match the stopping-calibration design. Real chronology is within clips; concatenation is synthetic. Keyframe metadata defines offline allocation, not per-frame SSL labels. Capture sessions are unknown. Warm model and downstream evaluation panels are reused; no sealed final panel was opened.

Early stop trains on the first6144 frames then holds. Uniform, random and coverage observe four32-image batches per128-arrival window, then accept two; all96 windows use the same capacity/delay and exactly192 updates/6144 exposures. Uniform accepts positions0,2; random uses a private seeded draw; coverage greedily maximizes minimum cosine distance from32 FIFO accepted-batch centroids. All selected batches train chronologically. Skipped batches never enter memory. Coverage features come from a fixed warm encoder: normalized image features averaged per batch, renormalized in float64 CPU geometry. Fixed geometry avoids backbone drift masquerading as novelty; current-model utility remains untested. No memory-size/score sweep, replay, objective/LR changes or predictor fitting occurs.

The pre-action MAE-loss diagnostics use private masks/RNG and preserve the learner. They are logged equally for all four arms but never influence admission. Their cost is recorded separately as offline research overhead, not claimed free for a future policy. Early-stop learner states reproduce the previous immediate baseline exactly despite grouped diagnostic instrumentation. Full controls are measured on fresh sources; audited development full controls and warm predictions are reused.

Deterministic float32 RTX2080, unchanged MAE/AdamW1e-4/batch32 and seeds5003/5009/5011. Readouts at6144/12288 arrivals use copied support-only100-step encoder fine-tuning, three50-image panels and400 balanced city/highway x day/night queries. Early terminal predictions alias its unchanged midpoint. Primary terminal balanced capability, day/night guards; linear/kNN secondary. Seeds/panels are averaged within source, with paired t intervals over six fresh source means (df5).

## Results

|Arm|Updates|Balanced %|Day %|Night %|Policy processing seconds|
|---|---:|---:|---:|---:|---:|
|early_stop|192|57.060|53.620|60.500|3.884|
|uniform|192|56.597|53.815|59.380|3.839|
|random|192|56.569|53.648|59.491|3.801|
|coverage|192|56.648|53.741|59.556|4.619|
|full|384|56.940|55.065|58.815|7.818|

Frozen warm primary balanced accuracy is56.000%. Coverage improves it by+0.648pp [0.182,1.114], an ordinary prespecified supporting-context interval. Useful SSL remains present, but **coverage content value does not qualify**.

Coverage-minus-uniform is+0.051pp [−0.224,+0.326]; coverage-minus-random+0.079pp [−0.129,+0.286], ordinary secondary95% intervals. Both are below the frozen0.5pp materiality floor, and neither establishes a positive lower bound. The12-scalar primary family (coverage versus uniform/random/early/full, balanced/day/night) uses Bonferroni-adjusted intervals; balanced ranges against uniform/random are[−0.482,+0.584] and[−0.323,+0.480]. No outcome-driven recipe change occurs.

Coverage-minus-early-stop is−0.412pp [−1.630,+0.806]; minus-full−0.292pp [−2.101,+1.518], ordinary intervals. These do not establish inferiority, but the adjusted quality guards also fail. Day lower bounds are−1.399pp against uniform and−6.850pp against full; the coverage-minus-full day mean is−1.324pp. Domain gaps can reflect learning forgone as well as interference; no direct catastrophic-forgetting claim follows. Decision: `no_coverage_content_lead_qualified`. Gates: beats paced controls=false; quality guards=false; positive frozen context=true; processing saving=true.

## Costs and scope

Coverage policy processing averages4.619s versus3.839s uniform and7.818s continuous. Including all reference forwards, geometry/memory operations, gathering/transfers and synchronized training, it saves40.92% against continuous, while doing the same optimizer work as uniform/random. Its approximately0.779s extra cost over uniform buys no established quality lead.

Common offline diagnostics add2.056s to the coverage run. If a future policy uses those loss measurements, that cost must be charged; naively adding it here would reduce coverage's saving against continuous to about14.62%, before any predictor overhead. Source IO/decode, checkpoint instrumentation, offline readers and replay are excluded. This is prepared-arrival processing, not end-to-end deployment efficiency.

This recipe forces two admissions per local window. It cannot shift unused quota between windows, reject all harmful candidates, or reserve more budget for an unknown later regime. A negative result therefore closes this fixed-reference local coverage recipe, not every possible budget-aware admission gate. Do not fit a utility predictor merely to rescue this outcome; a different signal requires its own causal benefit/interference evidence.

## Integrity, data intake and storage

138 new physical trajectories (120 four-arm and18 fresh full) use29952 scientific SSL updates and738 new primary fits. Reuse12 development full trajectories supplies72 fits, plus9 warm fits:81 reused fits. Exact replay repeats29952 updates and reproduces276 new complete endpoint states, selection/memory/RNG, diagnostic values and exposure/loss-gradient traces. Discarded preflight4 updates/3 fits gives59908 total SSL updates including verification/preflight.

Independent verification checks4361 locked inputs,122880 pixel/timestamp source frames,300 endpoint states including24 reused full states,2430 prediction sets,2880 independently reconstructed coverage windows and72 reader/metric interval comparisons. Eleven targeted tests and scoped lint/format checks pass. Successful runner elapsed2872.677s. No scientific retry, numerical repair, sample expansion or gate relaxation occurred.

Final144-source intake transferred2831.707MiB within4096MiB, with two infrastructure-only short-video replacements before scientific outcomes. Original selections, allocations, failed logs/videos and amendments remain archived and locked. All657 previously downloaded movie paths were excluded in fresh selection.

SSD space was checked before intake and collection. Available space subsequently fell more than the output projection; an inventory identified older movies outside input-lock references, but that alone was insufficient to establish that they were safe to discard. Optional cleanup was reviewed and skipped; no old video/run/data was removed. All prior locked evidence remains intact. The completed run leaves approximately6.41GiB free. Any future large intake needs a new storage check and explicit artifact-retention assessment.

## Interpretation

A192-update cap is compatible with useful SSL, but this novelty/coverage rule does not improve budget use over uniform or random pacing. It adds scoring overhead without an established content benefit. Retain the cheap paced controls and early-stop baseline. No adaptive controller is promoted, no loss/novelty threshold is validated and no automatic next collection is launched.

The result is conditional on frozen warm features, batch-centroid memory32, two-of-four local quotas, the narrow scene/time target and reused fine-tuning panels. Only six fresh source streams support inference; two per dwell scale are descriptive, with unknown capture sessions. Existing phase and sealed-test gates remain unchanged.

## Artifacts

- `outputs/coverage-admission/20261013-gpu`: source/input lock, collection/status, warm references, decisions/diagnostics, complete states, copied readouts/predictions, summary, exact replay and independent verification.
- `outputs/coverage-admission/20261013-intake`: selection/ranked eligible pool, pinned downloads, two amendments and logs.
- `outputs/coverage-admission/20261013-frames`: final allocation/intake, frame caches and timestamp/pixel manifests.
- [Figure](../../../outputs/coverage-admission/20261013-gpu/figures/coverage-admission.png), SVG/PDF exports.
- Runner `python -m scripts.run_coverage_admission --output outputs/coverage-admission/20261013-gpu`; audit `python -m scripts.verify_coverage_admission --output outputs/coverage-admission/20261013-gpu`; render `python -m scripts.render_coverage_admission --output outputs/coverage-admission/20261013-gpu`. Use established GPU/CUBLAS deterministic settings. Existing evidence is immutable, not a second independent run.

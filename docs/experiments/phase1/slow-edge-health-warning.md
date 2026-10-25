# Accumulated CKA early-warning trajectories

## Question and method

Does accumulated representation movement warn about capability loss that additional SSL training would cause? The prior [health measurement](slow-edge-health-measurement.md) separated selected beneficial and harmful endpoints after 32 updates, but did not establish warning lead time. This test distinguishes future damage from damage already incurred. The [frozen protocol](../../project-plan/slow-edge-health-warning-frozen.md) fixes the primary landmark, population and qualification criteria before collection.

Reuse six budget-reservation stream histories, three checkpoint positions and three optimization seeds per history: 54 continuations grouped into six source histories. Nine separate controls reuse the historical day-end/night-end/early-return states and never enter forecast fitting. The sources and evaluation population have previously been inspected; these are development results. Seeds repeat optimization, not independently sampled histories. Capture-session identity is unknown.

Restore the complete learner, Adam state, module modes and CPU/CUDA RNG. Train on the next 1,024 chronological images with the unchanged MAE recipe: 32 updates of batch 32. Observe after 0, 4, 8, 16 and 32 updates. A fixed bank of 256 warm-partition images supplies unaugmented encoder embeddings; CKA drift is measured against the local starting state. Its enrollment, exclusions and metadata caveats are inherited from the prior measurement. Runtime health readings use no labels.

At each landmark, the stop alternative sees the remaining arrivals without updating. Because rejected arrivals leave the complete learner untouched, its capability is the landmark capability. The continuation alternative is the update-32 capability. Both use the unchanged fine-tuning reader, three 50-label support draws and 400 queries, with balanced/day/night accuracy; linear/kNN are secondary. Query labels are offline targets only. Readouts operate on copied encoders and must preserve the online state.

For example, at update eight, the target is the effect of the remaining 24 updates relative to stopping at eight. It is not the total effect relative to the initial parent. Average panels and optimization seeds within each stream/checkpoint, then define signed worst-domain remaining damage as minus the smaller day/night continuation effect. Positive denotes a domain loss still ahead; negative means both domains improve. Balanced damage and already-incurred damage are separate diagnostics.

The primary forecast at update eight compares a training mean, update count plus the last four actual SSL losses' mean, and those inputs plus accumulated CKA drift. Counts include prior accepted updates. Fixed ridge-one regression holds out an entire history, with training-fold normalization/intercepts. Eighteen checkpoint means yield six paired source error means. Qualification requires at least 10% MSE reduction versus both baselines with positive family-two adjusted lower bounds (df five), positive CKA coefficients in at least five folds, and at least two observed examples each of large future domain damage (at least 1pp), no future domain damage, and beneficial balanced continuation (at least 0.5pp). The population condition does not make any individual observed effect statistically qualified.

Update-four and update-sixteen timing forecasts, balanced-damage forecasts and already-incurred-damage forecasts are descriptive secondary diagnostics. They cannot replace a failed primary. No metric, sign, threshold or landmark is selected after outcomes. Selected historical controls only provide conditional trajectory examples; seed ranges are not population confidence intervals. Even a successful development screen would require unused-history confirmation and false-alarm calibration before controlling admission.

## Results and interpretation

| Warning landmark | Mean forecast MSE | Count/loss MSE | Count/loss/CKA MSE | CKA error change versus count/loss |
|---|---:|---:|---:|---:|
| Four, secondary | 1.656941 | 1.868964 | 3.635244 | +94.51% |
| Eight, primary | 1.693141 | 1.982261 | 2.569509 | +29.63% |
| Sixteen, secondary | 1.407572 | 1.806751 | 2.033793 | +12.57% |

MSE units are squared accuracy percentage points. At the primary landmark, CKA increases error 29.63% versus count/loss and 51.76% versus the training mean. The paired MSE-improvement intervals are −0.587248 [−1.884300, +0.709804] versus count/loss and −0.876368 [−2.156098, +0.403362] versus mean, adjusted for the two primary comparisons. These intervals cross zero: improvement is unqualified, rather than inferential proof that CKA always worsens prediction. Only two of six conditional CKA coefficients are positive; the sign gate also fails.

The population screen passes: among 18 update-eight checkpoint means there are nine observed future domain losses of at least 1pp, five continuations without a future domain loss and four balanced gains of at least 0.5pp. Failure is therefore not simply absence of both-sided observed outcomes. Individual effects have not been qualified statistically by this count. The update-four and update-sixteen secondary forecasts also fail incremental prediction; five positive coefficients at sixteen do not rescue its higher error.

| Historical control at update eight | Accumulated CKA drift | Balanced effect already incurred, pp | Balanced effect still ahead, pp | Night effect still ahead, pp |
|---|---:|---:|---:|---:|
| Day-end | 0.00536 | +0.278 | +3.056 | +2.944 |
| Night-end | 0.02437 | −1.222 | −1.222 | −3.278 |
| Early return | 0.03875 | −0.861 | −0.472 | −2.667 |

These selected controls retain a conditional early separation: the two harmful histories have more mean movement at eight than the beneficial history. It does not transfer to the six-history remaining-damage predictor. At sixteen, early return has drift 0.11757 and already-incurred balanced loss −1.750pp, but continuing to 32 recovers +0.417pp balanced; its remaining night effect is −0.167pp. Large accumulated movement therefore cannot be interpreted directly as a reason that further adaptation will harm balanced capability. This conditional example also does not prove a beneficial stop/continue preference under uncertainty.

Already-incurred-damage diagnostics are mixed: at four, adding CKA lowers error descriptively (0.100457 versus count/loss 0.132710 and mean 0.120668); at eight and sixteen it does not beat the mean. These are secondary diagnostics without qualified improvement bounds. Balanced future-damage forecasts likewise increase error when CKA is added at every landmark. No secondary metric or horizon is promoted.

The primary early-warning recipe is unqualified. Keep CKA as instrumentation for accumulated representation change; do not build a pause or admission threshold from the selected control separation. The next filter direction should remain grounded in demonstrated action utility and cheap stopping/pacing baselines, rather than interpreting global movement as future harm. This closes the bounded CKA screen without a controller or automatic follow-on collection. It does not disprove all health signals.

## Integrity and storage design

Every complete landmark state and embedding array is exactly replayed without intervening readouts. Six-stream update-sixteen states and traces match the audited spend branch; historical update-32 states and traces match the audited full branch. Alias cached readouts only after full-state equality, resolving the historical hold-to-parent aliases explicitly. Hold readouts at zero provide the stop baseline. Preserve all original evidence.

Save compact embeddings, predictions, scores, traces and fingerprints. No new online model weights, pixel caches, downloads or video decoding are needed. There are 2,016 scientific and 2,016 replay updates, plus eight discarded preflight updates. Of 315 readout landmarks, 126 reuse verified readouts (378 primary panel fits) and 189 require new readouts (567 primary panel fits). Offline fine-tuning optimization is separate from online SSL work.

The main audit checks exact updates, complete states, observers, aliases and inherited input hashes. Independent float64 numerical reconstruction checks CKA from embeddings, every prediction accuracy, aggregation order, future targets, all grouped forecasts, adjusted contrasts and qualification gates. It audits saved readout predictions rather than independently retraining every copied fine-tuning encoder. Warm-pool enrollment is inherited rather than rescanned. Four focused tests cover remaining-effect semantics, day/night tradeoffs, held-history target isolation and grouping requirements.

All audits pass: 315 complete-state/embedding landmark replays, 315 independent CKA scalars, 2,835 prediction sets, 54 remaining-effect targets, 27 grouped forecast models and 5,529 inherited/new input hashes. Primary and secondary gates are reconstructed independently. The four focused tests, scoped E9/F lint and formatting pass. Scientific sources and frozen criteria remain unchanged after locking.

Storage caps are 128 MiB output, 768 MiB launch reserve and 256 MiB per-write reserve. Monitoring forward/geometry, state restoration, SSL updates and offline readouts are timed separately; forensic fingerprint checks and artifact writes are additional job costs. No production efficiency claim follows.

Mean measured reference forward is 25.12ms, geometry 3.47ms and SSL update 18.95ms. A monitor reading plus geometry costs about 28.59ms, equivalent to 1.51 measured updates; checking once per eight updates would add roughly 18.85% to those isolated update times, excluding other costs. Mean parent restoration, including its disk load, is 84.03ms per trajectory. New offline readouts total 685.12 seconds; collection, replay and final hashes total 939.59 seconds (15.66 minutes), with the independent audit additional. The run preserves every prior source and uses approximately 31 MiB including reporting. No old data was deleted. Final free bytes are recorded in `storage.json`.

The output directory retains its preparation-date identifier `20261013-gpu`; closure is recorded on 2026-10-14. Cached footage is synthetic clip concatenation, and scene/lighting exposure changes within some continuation windows. These results do not isolate illumination, forgetting or a single state mechanism. Evaluation queries/supports are reused, the predictive fits have only six history groups, and no deployed stream efficiency or independent-session generalization is established.

## Artifacts and reproduction

Local run root: `outputs/health-warning/20261013-gpu`. Preserve `input_lock.json`, `source_snapshot/`, `preflight.json`, `collection.json`, compact per-trajectory `embeddings.npz`, new readout `predictions.npz`/`readout.json`, `summary.json`, `verification.json`, `independent-verification.json`, role/storage manifests and `health-warning.{png,svg,pdf}`. Existing readouts are referenced at their locked original paths. No automatic rerun into an existing locked directory is allowed.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.run_health_warning --output outputs/health-warning/20261013-gpu
.venv-gpu\Scripts\python.exe -m scripts.verify_health_warning --output outputs/health-warning/20261013-gpu
.venv-gpu\Scripts\python.exe -m scripts.render_health_warning --output outputs/health-warning/20261013-gpu
```

Core: `cafl4ds/health_warning.py`; focused tests: `tests/unit/test_health_warning.py`. RTX 2080, deterministic float32 CUDA, four CPU threads, unchanged established MAE/Adam recipe. Renderer uses headless Agg. Local pre-commit is unavailable; scoped Ruff and pytest are the executed software checks. Broad P1.1 and controller qualification remain unchanged.

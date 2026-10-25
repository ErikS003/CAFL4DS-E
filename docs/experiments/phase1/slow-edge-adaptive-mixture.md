# Context-adaptive batch mixtures at matched replay dose

## Question and method

Can adjusting the past/current batch fraction to observed context changes make SSL training more useful than current-only training, a fixed mixture or random mixture timing? The prior [smoothing experiment](slow-edge-diet-smoothing.md) tested preset historical tapers and found no qualified protection. This [frozen protocol](../../project-plan/slow-edge-adaptive-mixture-frozen.md) tests a causal context rule with matched historical identities and dose, not a damage predictor.

Use six existing coverage-admission stream histories, three optimization seeds and four arms. Each episode has 12,288 arrivals and 192 updates of batch 32. Short histories alternate lighting every period; medium and long histories have longer dwells. They are synthetic clip concatenations with repeated city/highway scenes, inspected development data and unknown capture sessions. The initial warm MAE/Adam recipe and evaluation task remain unchanged.

Each update follows 64 arrivals. The current candidate set is the first 32 frames, reproducing the earlier uniform-half training inputs. The first eight updates are identical current-only calibration across all arms. After calibration, the mixture arms use 23 eight-update blocks, each with exactly 64 historical exposures. A 512-index Algorithm-R reservoir admits every past arrival, including images not trained on immediately. At each block start, draw and freeze 64 unique past identities. All three mixture arms consume exactly that same sequence. A fixed mixture replays eight per update; random timing proposes zero or sixteen with private fair coins; adaptive timing proposes zero or sixteen according to the context score. Remaining-quota constraints enforce the common historical dose.

The frozen original warm encoder reads 64 newly arrived images per adaptive update. Row-normalize embeddings and compare the normalized incoming centroid with the centroid of the last 256 actual training exposures, including replay. During the first eight updates collect seven past-only scores, then freeze their 90th percentile plus a numerical floor as the threshold. Above it, propose sixteen replay images; below it, propose zero. This reads context change, not calibrated harm or illumination. The training diet feeds back into the recent-content centroid, while the reference encoder remains frozen.

For replay count r, admit 32-r uniformly spaced current candidates and prepend r historical images. Mixture trajectories contain exactly 1,472 old and 4,672 current exposures, versus 6,144 current exposures in current-only training. Total optimizer work and image exposures are matched. Past identities/order match within every mixture quota; current identities and exposure timing deliberately differ. The random control tests whether context conditioning adds value beyond variable fractions. Quota enforcement can override proposals and is explicitly logged.

Readouts occur when the second lighting regime ends, before the first return, and at stream end. Primary capability is encoder fine-tuning on three fixed 50-label support draws and 400 balanced scene-class queries; report day/night separately. Earlier endpoint and linear/kNN are secondary. All readers train copied models and preserve the complete online learner, Adam buffers, modes and CPU/CUDA RNG.

Primary terminal contrasts are adaptive minus current/fixed/random/warm, each balanced/day/night. Average support panels and three optimizer seeds within each of six source histories. Twelve primary scalars receive Bonferroni paired intervals, df five. Qualification requires at least 0.5pp balanced advantage with positive adjusted lower bounds against all training controls, useful SSL versus warm (at least 0.1pp and a positive lower bound), and both domain lower bounds at least -1pp against every comparator. No outcome-driven threshold, buffer, fraction or objective sweep is allowed. A passing development screen would still require unused histories and quality/cost confirmation.

## Results and interpretation

| Arm | Terminal balanced % | Day % | Night % |
|---|---:|---:|---:|
| Initial warm frozen | 56.000 | 53.167 | 58.833 |
| Current-only | 56.597 | 53.815 | 59.380 |
| Fixed mixture | 56.926 | 53.509 | 60.343 |
| Quota-random mixture | 56.829 | 53.454 | 60.204 |
| Adaptive mixture | 57.046 | 53.741 | 60.352 |

The adaptive arm has the highest primary mean among these arms, but its incremental lead does not qualify. Useful SSL versus initial warm passes the frozen adjusted balanced criterion. Adaptive content value and joint domain retention fail. These are development findings, not evidence of a deployable filter or independent confirmation.

| Adaptive minus comparator | Balanced gain, pp | Ordinary interval, pp | Family-twelve interval, pp |
|---|---:|---|---|
| Current-only | +0.449 | [+0.034, +0.864] | [−0.355, +1.253] |
| Fixed mixture | +0.120 | [−0.078, +0.319] | [−0.264, +0.505] |
| Quota-random | +0.218 | [−0.023, +0.458] | [−0.249, +0.684] |
| Initial warm | +1.046 | [+0.533, +1.559] | [+0.052, +2.040] |

All three training-control gains fall below the 0.5pp materiality floor; their adjusted balanced bounds cross zero. The useful-SSL gain is positive in every source history. Thus the test preserves evidence for SSL while failing to establish that context-dependent mixture timing is worth adding.

Most observed nighttime improvement is already present with fixed mixing. Adaptive minus current night gain is +0.972pp, whereas adaptive minus fixed night gain is +0.009pp. Adaptive improves day relative to fixed by +0.231pp under ordinary bounds, but its adjusted interval crosses zero. Its mean day effect relative to current is −0.074pp. Joint retention fails because adjusted day lower bounds versus current and warm are −1.889 and −3.586pp. Relative to warm, two source means show day losses exceeding 1pp (−1.722 and −1.500). Balanced improvement therefore does not establish preservation of both domains.

The reservoir includes past arrivals that were never trained on immediately. Mixing changes raw-frame coverage as well as repetition and temporal diversity. Its observed night advantage does not isolate a forgetting-prevention mechanism. Fixed mixing is a useful comparator here, not a newly qualified deployment rule.

Before the first return, adaptive balanced accuracy is 56.241%, fixed 56.245%, random 56.153% and current 55.755%. Adaptive minus fixed is −0.005pp, with ordinary interval [−0.262, +0.253]. A nighttime advantage over random at this secondary endpoint does not qualify the terminal balanced or retention gates. No early endpoint or secondary reader replaces the primary result.

Secondary terminal balanced means differ: linear favors current (61.454%) over adaptive (61.130%), while kNN favors adaptive (54.458%) over current/fixed/random (54.231/54.245/54.264%). This reader dependence limits broader capability claims.

The signal is not constant, but quota enforcement materially constrains it. Adaptive proposes the high replay fraction in 59.1% of post-calibration updates, ranging from 7.8% to 94.4% across histories. The quota overrides 40.9% of its proposals, versus 14.2% for random timing. The test therefore measures context-conditioned timing under a strict matched-dose constraint; it does not show what an unconstrained adaptive replay allowance would do. There is no outcome-driven recalibration.

Decision: close this frozen context-fraction recipe as unqualified. Replay mixing remains a candidate batch-construction mechanism, but its apparent benefit here does not establish the value of this adaptive signal. Further work must distinguish useful replay content or timing from simply including past images, on unused histories with charged costs and explicit domain constraints. No automatic new experiment or controller follows.

## Integrity and cost boundaries

Current-only terminal states and 192-update traces must reproduce the previously audited uniform-half trajectories. Those 18 terminal readouts and three unique warm readouts can be aliased after complete-state/encoder identity checks. Common eight-update calibration states must agree across all arms. Every trajectory is replayed without intervening readers; exposures, loss/gradient tuples and complete endpoint fingerprints must match exactly. Adaptive reference features must reproduce across all seeds and remain finite. Selectors access only already-arrived feature rows, despite full arrays being retained for audit.

The numerical audit independently rebuilds Algorithm-R histories, block samples, random decisions, context centroids/thresholds, quota constraints, actual image selections and grouped prediction-derived comparisons. It does not independently retrain every copied fine-tuning model. No task labels select online training images. Lighting metadata locates offline readout checkpoints only.

The harness keeps reservoir indices into existing prepared frame caches. It does not implement or price a deployed decoding/pixel-retention pipeline. Full per-frame feature arrays are compact audit artifacts; a deployed bounded-memory implementation would retain only active reservoir/quota/recent features. Input gathering, reservoir planning, frozen feature extraction, batch decisions, SSL updates, restoration and offline readouts are timed separately. Fingerprint checks, replay and artifact writes are additional job costs. No end-to-end or deployment-efficiency qualification follows.

Scientific budget: 13,824 SSL updates, 13,824 replay updates and sixteen discarded preflight updates, totaling 27,664. There are 126 new readouts/378 primary panel fits, 18 terminal aliases/54 reused panel fits and three unique warm readouts/nine reused fits. No new videos, pixel caches or online model weights are saved. Output cap128MiB, launch reserve768MiB, write reserve256MiB; preserve prior evidence.

All main and independent audits pass: 144 complete endpoint replays, 3,456 context-observer invariance checks, 1,242 matched replay sequences, 13,824 causal update-exposure records, 1,323 prediction sets and 144 reader/metric/statistical intervals. Current traces reproduce prior uniform trajectories, and all 18 four-arm calibration comparisons match. Five focused tests, scoped E9/F lint and formatting pass. Local pre-commit is unavailable; these are the executed software checks.

The main lock checks 4,448 inputs. A supplementary provenance record covers five imported helper files absent from the coverage-derived lock. During collection, they were checked against their already-frozen stopping-headroom ancestor hashes and separately snapshotted. A final check confirms those versions remain unchanged. This adds provenance without changing the running scientific sources, settings, exposures or gates; `dependency-verification.json` preserves the exact lineage.

| Arm | Measured prepared-processing seconds per trajectory |
|---|---:|
| Current-only | 4.118 |
| Fixed mixture | 4.356 |
| Quota-random | 4.285 |
| Adaptive mixture | 7.546 |

This sum includes input gathering, reservoir planning, batch decisions, frozen feature extraction and SSL optimization. Adaptive is 73.24% slower than fixed under these measured boundaries. Its reference forwards cost 1.991 seconds per trajectory, compared with 4.581 seconds of adaptive SSL updates. The timings exclude restoration, copied readouts, forensic fingerprints, replay, artifact writes and the unimplemented deployed decoding/pixel-retention pipeline. Interleaved verification and hardware/cache conditions can affect measured update times, so these are research-job processing measurements, not isolated deployment benchmarks.

Main collection, preflight, replay and final source hashes take 1,600.65 seconds (26.68 minutes); independent numerical verification and reporting are additional. The output is approximately 42MiB including reports/plots, below the 128MiB cap. Final footprint/free bytes are recorded in `storage.json`. No old data was removed and all prior evidence is preserved. Source reuse, unknown capture sessions, repeated queries, synthetic recurrence and the fixed scene-class fine-tuning target remain inference limits. Broad phase/controller gates are unchanged.

## Artifacts and reproduction

Root: `outputs/adaptive-mixture/20261014-gpu`. Preserve input/source locks, preflight, compact `trajectory.json` records, six frozen-reference feature arrays, new readout predictions/audits, collection/summary, main/independent verification, role/storage manifests and `adaptive-mixture.{png,svg,pdf}`. Reused warm/current readouts remain at their locked original paths. Outputs are gitignored local artifacts; this document is the tracked result record.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.run_adaptive_mixture --output outputs/adaptive-mixture/20261014-gpu
.venv-gpu\Scripts\python.exe -m scripts.verify_adaptive_mixture --output outputs/adaptive-mixture/20261014-gpu
.venv-gpu\Scripts\python.exe -m scripts.render_adaptive_mixture --output outputs/adaptive-mixture/20261014-gpu
```

Scientific collection refuses an existing lock. Verification/rendering reuse completed outputs. Core: `cafl4ds/adaptive_mixture.py`; tests: `tests/unit/test_adaptive_mixture.py`. Deterministic float32 CUDA on RTX2080, four CPU threads, unchanged existing MAE recipe. Five focused tests cover causal snapshots, exact old/current dose and old-identity order, label-free context response, temporal eligibility and immunity to future feature changes. Scope and formal phase/controller gates remain unchanged.

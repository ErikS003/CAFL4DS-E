# Cached stopping headroom and domain retention

## Question and method

Is there enough benefit from history-specific stopping to justify another adaptive signal search? The [frozen retrospective protocol](../../project-plan/slow-edge-stopping-headroom-frozen.md) uses the completed [health-warning trajectories](slow-edge-health-warning.md), without new training. It assesses five possible local stopping points rather than an arbitrary filter or a whole-stream stopping budget.

The 54 six-history trajectories supply 18 history/checkpoint cases with three optimizer seeds and three few-label support panels per case. The nine selected historical controls are excluded. Reconstruct fine-tuning, linear and kNN accuracy directly from 270 existing landmark readouts. Average seeds and support panels within each case. The primary target is balanced scene-class accuracy with separate day/night retention. Source groups are the six stream histories; three checkpoints are averaged within each for intervals. These sources, query images and outcomes were previously inspected.

Compare fixed additional stopping doses 0/4/8/16/32. Each saved parent already contains SSL learning. Stopping at zero means preserving that parent through the local continuation. The unconstrained hindsight choice selects the highest observed balanced accuracy for each case. A retention-aware version requires neither domain to lose more than 1pp relative to that case's zero-update state. Ties favor fewer updates. The fixed comparator chooses one pooled best dose. Choices and evaluations on the same full queries form an optimistic diagnostic; their intervals do not correct outcome-selection bias.

For selection-noise sensitivity, hash existing query identities into balanced 200-query halves. Each contains 50 examples in every class/domain cell. Choose using A and score B, then reverse. The same prefit support models supply predictions; no model is retrained. These retrospective halves are not fresh test data, because their queries and labels were already used in earlier studies.

Also compare with a common, state-independent distribution over fixed doses. A frozen linear program maximizes selection-half balanced accuracy with expected mean updates no greater than the selected oracle's mean, and pooled expected domain effects at least -1pp relative to zero. This is a common resource ceiling, not equal consumed work. Distribution-level retention is an expectation, whereas the oracle's selection guard is per case. Scores and violation counts on the other half are reported for both.

An investigation screen requires both query directions to achieve at least 0.5pp balanced advantage over that mean-budget baseline, positive adjusted advantage lower bounds, day/night lower bounds at least -1pp versus both baseline and zero, and positive balanced differences in at least four source groups. The ten contrasts across the two directions use family-ten adjusted paired intervals, df five. This conditional screen could only prioritize independent opportunity confirmation, never qualify a deployed controller. Other readers remain secondary.

## Results

| Fixed additional updates | Balanced accuracy, % | Balanced effect versus stop now, pp | Day effect, pp | Night effect, pp | Cases with a domain loss >1pp |
|---|---:|---:|---:|---:|---:|
| 0 | 56.398 | 0.000 | 0.000 | 0.000 | 0 |
| 4 | 56.168 | −0.230 | −0.241 | −0.219 | 1 |
| 8 | 56.034 | −0.364 | −0.367 | −0.361 | 5 |
| 16 | 56.049 | −0.349 | −0.306 | −0.392 | 4 |
| 32 | 55.948 | −0.451 | −0.404 | −0.497 | 11 |

The highest pooled primary mean belongs to immediate pause. This is a descriptive mean ranking, not qualified superiority of every hold case or a claim that initial SSL is unnecessary. All parents have already adapted, and the trajectory collection samples particular local checkpoint windows.

Both full-query hindsight rules choose identical doses and reach balanced accuracy 56.776%, an advantage of 0.378pp over pause. Day/night effects are +0.142/+0.614pp. The conditional ordinary interval for balanced gain is [0.072, 0.684]pp; every source mean is positive. This interval is selection-optimistic and does not establish a transferable action advantage. The point estimate falls below the frozen 0.5pp materiality floor.

Hindsight consumes 11.778 additional updates on average: eight cases stop immediately, one at four, two at eight, two at sixteen and five at thirty-two. Its largest observed domain loss is −0.944pp, with no >1pp violations on the full queries used to select it. The guard changes no full-query choice, so its success there is not independent retention evidence.

| Retrospective query direction | Balanced advantage, pp | Ordinary interval, pp | Family-ten interval, pp | Mean additional updates | Evaluation cases with domain loss >1pp |
|---|---:|---|---|---:|---:|
| Choose A, score B | +0.145 | [−0.111, +0.401] | [−0.330, +0.620] | 9.556 | 0 |
| Choose B, score A | +0.142 | [−0.480, +0.764] | [−1.013, +1.297] | 12.444 | 4 |

Both halves select zero as their best common fixed dose. Independently optimized mean-budget mixtures also choose zero with probability one. Thus the reported advantages are against an alternative that consumes no additional SSL work, despite being allowed the hindsight rule's mean update allowance. Forcing the comparator to spend that allowance would weaken it without a scientific justification.

Neither direction reaches the 0.5pp floor or positive adjusted gain bound. The B-to-A night-retention lower bound is −1.899pp, failing the −1pp screen. The four violating cases in that direction include a worst domain loss of −3.000pp. A selection-half retention guard therefore does not establish retention on even the complementary reused queries. Source-positive counts use the frozen exact sign rule; one A-to-B positive is numerically near zero and should not be interpreted as substantive utility.

Secondary readers do not rescue the opportunity. Linear's full-query retained hindsight gain is 0.497pp, with transferred gains 0.343/0.099pp. kNN favors fixed thirty-two updates, unlike the primary reader, and retained hindsight adds 0.207pp; transferred gains versus the mixture are −0.185/+0.039pp. Both secondary investigation screens fail.

## Interpretation

There are local cases that benefit from more updates, but their aggregate exploitable stopping margin is small in this cached action set. The full-query selection overstates the margin relative to the query-half check. Domain protection also fails to transfer consistently. The investigation screen does not support another predictor search on these cached 32-update windows.

Retain useful SSL and make simple stopping/pacing with explicit domain constraints the practical benchmark. Before reopening adaptive timing, obtain a materially larger, replicated beneficial-versus-harmful action opportunity on unused histories. This result does not bound policies with other objectives, batch choices, resume schedules or longer horizons. No new experiment or controller is launched automatically. Broad phase gates remain unchanged.

## Verification, costs and storage

Main and independent audits pass. Independent reconstruction checks 2,430 saved prediction sets and 2,430 score scalars across all readers/partitions, query identities/strata, chosen doses, mean update counts, violation counts, worst-domain effects and intervals. All six mixture problems are checked by enumerating feasible LP vertices independently of the main HiGHS solver. Both query directions' investigation gates are reconstructed. The audit locks 5,861 inherited/new input hashes.

Five focused tests verify cheaper ties, hidden domain losses, state-independent work constraints, disjoint balanced query halves and genuine heterogeneous stopping utility. Scoped E9/F lint and formatting pass; local pre-commit remains unavailable. Scientific sources and criteria are unchanged after input locking.

There are zero new SSL updates, zero new downstream fits, zero downloads and no new model/pixel caches. Initial source hashing and analysis take 38.72 seconds; final hashes and independent verification are additional. Compact plots/reporting remain below the 32MiB cap. Final footprint/free bytes are in `storage.json`. No old data is removed; all original evidence is retained. This audit measures update counts rather than production processing time. The oracle and mixtures have access to offline outcomes and are not deployable admission rules.

## Artifacts and reproduction

Root: `outputs/stopping-headroom/20261014-cached`. Preserve `input_lock.json`, `source_snapshot/`, `query_split.npz`, `score_tables.npz`, `summary.json`, main/independent verification, role/storage manifests and `stopping-headroom.{png,svg,pdf}`. Case identities and selected doses are in the summary; original readouts remain at their locked paths. Outputs are gitignored local artifacts; this document is the tracked result record.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.run_stopping_headroom --output outputs/stopping-headroom/20261014-cached
.venv-gpu\Scripts\python.exe -m scripts.verify_stopping_headroom --output outputs/stopping-headroom/20261014-cached
.venv-gpu\Scripts\python.exe -m scripts.render_stopping_headroom --output outputs/stopping-headroom/20261014-cached
```

Collection refuses an existing lock. Verification/rendering can reuse the completed output. Core: `cafl4ds/stopping_headroom.py`; tests: `tests/unit/test_stopping_headroom.py`. Rendering uses headless Agg. Existing deterministic model fits are inherited; this analysis runs on CPU.

# A heuristic adaptive J controller

2026-10-24. Proposed next pilot following user feedback. This replaces the larger W/C/A sequence as the immediate recommendation, without deleting that design. Specification only: no runner or training launched. Settings below are deliberate starting guesses informed by observed mechanisms, not fitted optima or safety guarantees.

## Intended behavior

Keep learning when J becomes pessimistic. Slow down when health worsens. Back away from demonstrated task damage. Give a stalled trajectory a short sequence of small updates instead of repeatedly asking whether one isolated update improves J.

Use one live learner and one recoverable checkpoint. Do not require candidate promotion, a new predictive model, a new labeled audit panel per decision, or another feasibility study before this exploratory pilot. The inherited development labels already used by the rich observer supply slow task feedback, so this controller is task-informed, not label-free.

## 1. Predictions bias opportunities; they do not veto them

Reuse the bounded clock-free corrected-J predictor. No raw arrival/optimizer clocks are restored. Compute the same prediction and feature-clipping diagnostic as before. Interpret a negative score as a suggestion about nominal-step J, not a probability of useful task learning or a calibrated prediction at the smaller executed rate.

Outside a burst or numerical-failure cooldown, use one seeded random draw per arrival:

| Prediction | Probability of taking an update |
|---|---:|
| Below the existing negative corrected-J threshold | 0.50 |
| Between that threshold and zero | 0.25 |
| Zero or above | 0.10 |

Threshold equality belongs to the middle bin; zero belongs to the last bin. Nonfinite predictions or more than 25% clipped feature coordinates use the neutral 0.25 probability. This fallback handles unavailable/out-of-range predictions without pretending to know their utility. A separate RNG supplies action draws and is saved with controller state.

Even an entirely pessimistic gate therefore retains opportunities to move. Positive J predictions do not cause nominal-rate updates. The floor means exploration, not assurance that the permitted updates help.

## 2. Health controls step size

Start in NORMAL at nominal LR / 4. This is motivated by N79's frequent quarter-rate rescues, not by an established quarter-rate optimum. Keep each accepted step as a real AdamW update with consistent moments, decay and seeded augmentation.

Check fixed-bank health at 128-arrival boundaries, using the same bank, preprocessing and masks. Maintain a fixed-at-attempt health anchor; initialize it from the original complete parent and replace it only when the slow task check accepts a new backup checkpoint. Do not move it merely because covariance is rising.

Enter CAUTION at LR / 16 after either condition holds at two consecutive health checks:

* Rank or variance is below 85% of the health anchor.
* Covariance is above 125% of the health anchor AND fixed-bank reconstruction is above 105% of the anchor.

Remain in CAUTION for at least two checks; return to NORMAL only after two consecutive checks with neither condition. Reset the clear counter on renewed warning. These conjunctions are heuristic and may miss harm; covariance alone cannot stop training. Warm-relative health ratios remain logged but do not veto updates. Do not add a covariance-gradient regularizer in this first version.

No step is rejected solely for increasing J or crossing the old 1.30-times-warm covariance ceiling. Health checks request a lower rate, not endless rollback of the same action. Tiny denominators and nonfinite assays produce explicit invalid-measurement events; do not silently interpret them as a healthy state.

## 3. Stalls trigger a short learning burst

Count actual committed updates in the preceding 256 arrivals, including their displacement, without resetting the count merely because one tiny step succeeded. If fewer than eight were committed, schedule a burst of 16 consecutive singleton updates at LR / 16, overriding J probabilities. A burst takes precedence over ordinary admission, as the N81 scheduling failure requires. At least 256 arrivals must separate burst starts.

There is no covariance or measured-J veto during the burst. Numerical failure and the slow task brake still apply. The sixteen-step sequence, rather than a single pulse, tests a route out of the gate's current state. Both 16 and eight are trial settings. Log whether the following 256 arrivals regain ordinary updates and whether task performance improves; successful parameter movement is not success by itself.

The ordinary 10% probability floor should make bursts infrequent. Frequent bursts expose scheduling, numerical or recovery failures and must not be hidden by aggregate acceptance counts.

## 4. Task feedback provides a slow brake, not a per-image hurdle

Every 384 offered arrivals, read balanced/day/night development linear accuracy with the inherited fixed support/query panels. Use final evaluation panels only for reporting. Keep a recoverable complete checkpoint whose task scores and health anchor are recorded together.

Initial backup is the original complete parent. Replace it when balanced development accuracy is at least 1 pp above the existing backup for two consecutive readouts and neither subgroup is more than 1 pp below its backup score or 2 pp below its original-parent score. Save the current complete state, not a reconstructed mixture of two checkpoints. Confirmation on the same panels reduces reacting to a single excursion but is not independent statistical confirmation. Not replacing a backup never blocks continued learning.

Rollback to that backup if balanced accuracy or either subgroup is more than 2 pp below the backup for two consecutive readouts, or more than 5 pp below it at one readout. Clear consecutive-check counters after rollback. This permits small or temporary task deterioration and attempts to interrupt sustained/severe damage. It cannot guarantee retention between readouts or generalization beyond development queries.

After rollback, enter RECOVERY: LR / 64, neutral update probability 0.25, for the next 256 offered arrivals. Consume new arrivals normally; never rewind to repeat the rejected segment. At recovery end, enter CAUTION and collect two fresh health checks before allowing NORMAL. Recovery overrides the burst trigger; a due burst can start after recovery, using its usual spacing rule. This bounded recovery interval cannot become a permanent freeze just because J stays unfavorable.

Numerical failure immediately restores the latest complete finite pre-step state, consumes that arrival and starts a 64-arrival numerical cooldown. After cooldown enter RECOVERY. Save the numerical failure reason. Repeated failures are an engineering failure, not a scientific protection result.

All restore operations include model, decoder, optimizer moments, modes and CPU/CUDA RNG. Stream position, action-draw RNG, resource counters, cooldown and recovery timers advance as controller state, rather than being rewound with learner state. This separation prevents replaying the same random decision or restarting a failed interval indefinitely. Report discarded updates and compute explicitly.

## Decision precedence

1. Process a due task readout and any rollback; a rollback cancels the active burst.
2. Honor numerical cooldown; no training during its bounded interval.
3. RECOVERY uses probability 0.25 and LR / 64; suppress burst triggers.
4. An active or newly due burst uses probability 1 and LR / 16.
5. Otherwise, J chooses the probability and NORMAL/CAUTION chooses LR.
6. Attempt at most one real update on this arrival. No rate search, trial cascade or J/covariance rejection.
7. Record committed update and parameter movement; process health checks and persist both controller and learner state at scheduled boundaries.

## What comes from the runs, and what remains a hypothesis

| Observation | Design response | Unproven assumption |
|---|---|---|
| Old clocks eventually drove almost-full admission and unstable performance | Keep corrected bounded clock-free predictor | Its remaining rankings help at smaller LR |
| N79 rescued 910 predictor rejects | Give all prediction bins nonzero opportunity | Exploration can improve task outcomes, not only J |
| N78's zero cutoff still stalled | Use a probability floor, not another cutoff relaxation | Random opportunities can escape the pessimistic state |
| N82 froze near the fixed covariance ceiling | Remove that ceiling; health only slows steps | Trend combinations provide useful rate guidance |
| N79 measured-J guard eventually blocked all tested rates | Permit temporary measured-J worsening | Useful sequences exist beyond local J descent |
| Original J gains could later disappear | Keep a task-checked backup and slow rollback | Development protection transfers to final queries |
| Repeated rate-search trials were expensive | One attempted update per selected arrival | Lower trial cost outweighs imperfect action choice |

This rule set intentionally prioritizes a plausible coherent trajectory over isolating every mechanism in advance. It remains vulnerable to a misleading health proxy, developmental panel overfitting, missing task damage between checks, and rates that are too small or large. Preserve those uncertainties in the results.

## Small practical comparison

First run the full heuristic from the original N73 parent on the same 57,600 arrivals, alongside a new simple control with probability 0.25 and LR / 4 throughout, no health adaptation and no task rollback. Reuse corrected J, full exposure and N82 as historical descriptive references; label all horizons honestly. Do not compare only to the worst full-exposure endpoint.

If the heuristic shows useful sustained gain, run one subtraction next: keep everything, replace J probabilities with constant 0.25. This asks whether predictions earn their cost. Then test constant LR / 4 with the same predictor, burst and task-brake logic to assess health-driven rate changes. These are complete-policy contrasts, not identical-accepted-update comparisons. A later action-count-matched analysis must be labeled retrospective. Do not launch a broad sweep of thresholds.

For the pilot, judge task gain over the original parent, retained improvements, drawdown, day/night tradeoffs, time below parent, rollback count, ordinary learning after bursts, rejected/discarded work and total wall time. Plot accuracy with actual update counts, effective LR, modes, covariance, J and backup changes on the same offered-arrival axis. A flat protected trajectory below the parent is not the desired outcome.

The existing rich predictor still requires expensive observation features. Removing trial searches does not make this a demonstrated cheap controller. Preserve current feature refresh semantics after committed updates, account separately for selector/health/task-reader cost, and make no streaming-efficiency claim until measured. Only the slow controller's additional decisions use the 128/384-arrival cadence.

If promising, freeze the heuristic and replicate on fresh histories/seeds and fresh final-query panels. The first reused stream is development evidence only. Before a scientific run, implement and verify exact rollback/resume, precedence, disjoint controller/final-query roles and accurate computation accounting. No implementation or GPU validation is claimed by this proposal.

Evidence: N73 drawdown analysis, N74 matched-prefix comparison, N79 controller-state review, N80 stopped retention-budget report and completed N82 summary in the existing `outputs/` directories. The earlier W/C/A proposal remains available as a more controlled follow-up if a heuristic result warrants it.

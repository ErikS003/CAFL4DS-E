# Next controller experiments after N82

Update, 2026-10-24: the user prefers a heuristic next controller. The immediate recommendation is now [the adaptive J heuristic](single-image-heuristic-controller.md). This W/C/A design is preserved as a possible controlled follow-up; it is not a prerequisite for the heuristic pilot.

Date: 2026-10-24. Status: concrete experimental design; not a frozen execution protocol or implemented runner. No training is launched by this document. Companion specification: `cafl4ds/configs/single_image_next_controller_experiments.json`.

The user accepts the existing evidence that restrictive controllers suppress useful learning opportunities. Do not repeat the proposed stalled-state feasibility diagnostic as a prerequisite. The next question is how to preserve useful adaptation over a trajectory.

## Evidence and design commitments

N73 demonstrates useful excursions and severe subsequent losses. N74 demonstrates strong restriction after the bundled calibration correction. N79 finds 910 measured-J-improving rescues of predictor rejects, then stalls despite covariance headroom. N80's 1% J budget also stalls. Completed N82 retains 308 updates in 57,600 arrivals, none after 17,156, and finishes at 57.917% versus corrected J's 60.917% and the unchanged parent's 61.083%. Its final covariance is 1.29827 times warm against a 1.30 ceiling.

These results justify changing the decision horizon and acceptance target. They do not establish that any new controller will work. Do not require predicted or measured J improvement for the principal new controller. Do not introduce a replacement hard health threshold into the no-covariance arm.

Three studies are staged, with separate causal questions:

1. W: can bounded candidate learning and task-based promotion retain useful adaptation?
2. C: does the covariance veto improve that same controller, or prevent useful sequences?
3. A: does action-aware feedback improve on fixed pacing and step size?

W and C share runs. A follows once window outcomes exist; its predictor is not a prerequisite for W.

## Shared experimental contract

Use the same SSL vehicle, preprocessing, singleton loss, optimizer, parent construction, reader recipe and seeded augmentation as N73/N82. Preserve complete model, decoder, optimizer moments, module modes and RNG in every retained and candidate checkpoint. A rejected candidate must not change the retained optimizer. Do not add replay, regularization gradients, parameter noise or optimizer resets to these studies.

Development pilot: original N73 complete parent, history fresh_00, SSL seed 5003, the same 57,600 continuation arrivals. This is an exploratory development comparison, since this history has already informed the design. Do not select an endpoint from the best-looking partial prefix. Existing full exposure, corrected J and N82 are descriptive references; the new paced control below is the relevant mechanism control. The unchanged parent's evaluation readout supplies the frozen baseline.

Confirmation: lock all settings after development, then use three new source-disjoint histories and three independent SSL seeds per history (nine paired history/seed cells), each with 57,600 arrivals. Construct matched parents by the same recipe; do not claim that different histories share identical parent weights. The exact IDs, seeds, split manifest and hashes must be resolved and saved before execution. Never silently substitute reused clips when fresh material is unavailable. Nine cells provide a bounded first confirmation, not a power guarantee; report uncertainty primarily across histories and disclose the three-history limitation.

Match offered arrivals and the predeclared opportunity schedule. Do not call LR-factor sums equivalent optimizer steps. Report actual attempts, updates, parameter displacement, full reader/selector cost, wall time, storage and discarded work. Extra rejected work is part of the controller cost.

## W: bounded candidate learning with task-based promotion

### Arms

| ID | Training | Adoption | Covariance veto |
|---|---|---|---|
| W0 paced | Every fourth arrival, nominal LR / 16 | Always adopt | None |
| W1 window | Identical opportunities and LR | Window promotion described below | None |
| C1 capped window | Identical opportunities and LR | Identical window promotion | 1.30 times warm, on each proposed update |

The small LR and pacing are declared starting settings, not established optima. Hold them fixed across these three arms. W0 versus W1 tests the complete retain/candidate mechanism. W1 versus C1 isolates adding the covariance veto; subsequent path and update-count differences are consequences of that intervention.

All three start from the original complete parent, not N79/N82 endpoints. No J predictor is needed for these arms. Keep J and health as logged diagnostics.

### Exact candidate lifecycle

Maintain a retained state R and a separate candidate state C. At the beginning of an attempt, clone the complete R into C. The retained model supplies deployment-facing outputs; C receives real AdamW updates on every fourth arrival. Each image is used once at its arrival, with the existing view-seed mapping. Both states are evaluated on identical support/query identities when compared.

One window is 256 offered arrivals, hence 64 scheduled updates. Evaluate C and unchanged R at the end of each window. If the promotion rule passes, replace the complete R with C and begin a new attempt on subsequent arrivals. Otherwise continue the same candidate for at most three windows (768 arrivals, 192 scheduled updates), allowing temporary J and task deterioration. Do not restart a promising or uncertain candidate after every failed window.

After three unsuccessful windows, discard C and begin a new attempt from R. Arrivals already consumed remain consumed: no rewind or replay. Nonfinite model state or loss terminates that attempt immediately and is recorded as a numerical failure. No task-gain decision is made from final evaluation queries. Persist both the retained state and candidate state even when there has been no promotion.

A candidate can temporarily deteriorate substantially; this design isolates it from the retained model. Report candidate downside rather than implying that learning itself was monotone. No guarantee is made for an architecture that serves candidate predictions directly.

### Promotion rule and data roles

This is a task-informed controller, not a label-free one. Separate source identities for (i) control/calibration, (ii) promotion audit and (iii) final evaluation. Existing final-query panels have already informed research design; fresh confirmation needs new held-out final queries. Assign all roles before the first candidate update and verify source/clip disjointness.

At each window boundary, refit the established linear reader for both C and R with the same three support draws. Use balanced accuracy and day/night subgroup accuracy. Development starting rule, with all values in percentage points:

* Mean balanced gain C minus R must be at least 1.0 pp on the control panel.
* At least two of the three support-panel contrasts must be positive.
* Neither day nor night mean may fall more than 1.0 pp below R, and neither may fall more than 2.0 pp below the original parent's score on that panel. The original-parent constraint prevents repeated small permitted subgroup losses from accumulating without bound.
* A control-passing candidate receives one independent promotion-audit check. Require positive balanced gain and the same subgroup limits there. Audit outcomes may only accept/reject promotion; they cannot tune LR, window size or tolerances within the locked run.

These tolerances are explicit pilot choices, not statistical confidence bounds. Same-bank support draws are not independent replications. Allocate a declared budget of audit panels/checks before training, use each audit query block at most once, and reserve no more than one audit attempt per three-window candidate attempt. If an audit fails, the candidate may finish its remaining windows but cannot be submitted again; then discard it. If the audit budget is exhausted, finish in retained-only mode and report exhaustion, not adaptive success. Preflight must establish a feasible disjoint audit supply; any smaller budget is declared in the lock, never silently reused.

Use the final evaluation readout only for reporting. Evaluate retained and candidate trajectories at fixed 768-arrival intervals and at the endpoint. Record development checks at every window and promotion event. Sampled maximum drawdown is explicitly a lower bound on unsampled excursions.

### Success and failure

Primary outcome is paired retained-model time-averaged balanced accuracy over offered arrivals relative to both frozen and W0. Use a fixed integration rule over measured checkpoints. Endpoint gain, worst-group retention and maximum drawdown are key secondary outcomes. Require gains to persist over later scheduled readouts; disclose immediate reversals.

Promotion count, candidate updates and low retained drawdown are not standalone success criteria. Zero promotions means adaptation was not demonstrated. Positive control gains without final-query gains mean development selection did not transfer. W1 retaining less loss but never beating frozen is a protection result, not beneficial adaptation.

Inspect whether rejected windows later recover within the three-window horizon, how often promotion requires temporary J deterioration, and whether gains survive the next two reporting intervals. These are descriptive mechanism analyses; do not use them to change the locked run.

## C: covariance ablation

C1 adds only the original-definition 1.30-times-warm per-step covariance veto to W1. On failure, restore C's complete pre-trial state, consume the arrival and continue to the next scheduled opportunity. There is no 64-arrival cooldown or rate search: adding either would confound this ablation. A rejected update does not extend the 256-arrival window or three-window attempt budget.

Log trial covariance and measured post-trial health before rollback. Compare promotion frequency, accepted candidate steps, retained task utility, candidate downside and total cost. Report the fraction of C1 attempts with no feasible updates and the duration of stalls.

After W/C collection, inspect W1 promotions whose candidate path crossed 1.30 times warm. These show whether useful promoted paths can coexist with cap violations; they do not by themselves identify the counterfactual effect of each rejected C1 image. C1 versus W1 complete trajectories supplies the policy contrast.

Do not sweep cap multipliers on the evaluation stream. If C1 loses and W1's promotions transfer, remove the cap from this architecture. If C1 improves task retention enough to offset lost adaptation and cost, retain it as an empirically supported component. If neither learns, do not attribute failure solely to covariance.

## A: action-aware window controller

Run this only after W/C have produced usable window outcomes. It asks whether choosing an action adds value beyond the fixed action; it does not ask again whether the stalled model can learn.

Keep the winning declared W/C promotion architecture and identical opportunities. Action set: hold, LR/16, LR/4. Choose an action at the start of each 256-arrival window; no per-image veto and no multi-rate search. The holder consumes the same stream interval. Reset the three-window candidate budget by lifecycle rules only, not by changing LR.

Use a small regularized linear model initially. Predict window-end development task gain relative to a paired hold branch and worst measured within-window subgroup loss; record later persistence separately. Inputs are available before the window: current task accessibility, recent changes, covariance/variance/rank, reconstruction, optimizer summaries, prior action and candidate age. Action identity is explicit. Exclude unbounded absolute arrival/optimizer clocks. Avoid rich per-image features until a state-only model beats fixed pacing.

Collect action labels on development histories with randomized action assignment independent of J; log assignment probabilities. Allocate 10% of windows, selected with a locked RNG, to a paired alternate-action shadow from the identical complete parent and stream. Restore state and RNG exactly; charge shadow training/readout work to the policy budget. Untried actions have missing outcomes, never zero-gain labels. The existing nominal-step forecaster cannot supply labels for smaller or regularized actions.

Fit and select regularization using history-held-out development only. During online refinement: predict and log first, execute and measure second, update last. A separate randomly sampled audit subset is withheld from predictor updates. Compare:

| ID | Choice policy |
|---|---|
| A0 | Fixed LR/16 (same W architecture) |
| A1 | Independent randomized action schedule with declared probabilities |
| A2 | Action-aware selection, with 10% randomized exploration |

All arms receive the same declared shadow/readout allocation where used for a compute-matched comparison. Also report operating cost without research-only shadow measurements. A2 chooses the highest predicted task gain among actions passing a development-calibrated downside criterion; if none passes, choose hold. Freeze that criterion before confirmation. Exploration runs in C, never bypasses promotion into R.

Judge complete retained trajectories, not just prediction error. Report action calibration, gain over A0 and A1, subgroup costs, stalls, promotion rates and computation. A2 must demonstrate extra value beyond changing its total training dose; include a separately labeled retrospective total-action-budget schedule analysis, not a purported prospective controller. If A2 does not outperform simple policies, keep the simple controller.

## Execution order and implementation acceptance

1. Implement the reusable R/C lifecycle, paired reader API, data-role manifest and persistent event log. Prepare the three W/C arms together. This is the first execution batch.
2. Run CPU logic checks and bounded GPU engineering checks: exact resume mid-window and after failed promotion; complete rejected-state restoration; correct stream advancement; identical opportunity/view schedules; candidate-to-retained optimizer transfer; cap-only difference; disjoint data roles; audit exhaustion; numerical-failure recovery. These checks validate machinery, not scientific success.
3. Run the full development W0/W1/C1 trajectories. Lock any revised settings before fresh confirmation; report revisions as development tuning.
4. Confirm W/C across fresh paired cells. Build A's randomized labels from development runs/additional declared development histories, then freeze and confirm A separately.

Required durable records include parent and source hashes, offered arrival, attempted and committed updates, LR, displacement, candidate age, per-step veto reason, promotion/control/audit scores, audit identity, promotion/reset reason, both checkpoint hashes and full resource accounting. Keep promotion data inaccessible to predictor fitting and final evaluation data inaccessible to controller decisions.

Resource preflight must estimate all model generations, assay banks, checkpoints and output volumes. The recent runs had limited free disk. Set a hard 2-GiB reserve, stop cleanly before violating it, and preserve at least the last two complete resumable generations for both R and C. Do not remove earlier experiment artifacts to make space automatically.

Outstanding implementation work is explicit: no new runner or notebook exists yet; exact fresh-history and promotion-audit IDs are not bound; GPU preflight and scientific runs are not performed. The companion JSON is a design manifest, not a Hydra training configuration. Future execution must resolve those inputs and create a hashed frozen protocol.

## Local evidence

* `outputs/single-image-gate-pair-drawdown-analysis/report.md`
* `outputs/single-image-gate-protection-comparison/20261023-33792/report.md`
* `outputs/single-image-controller-state-review/20261024/review.md`
* `outputs/single-image-retention-budget/20261024-uniform/report.md`
* `outputs/single-image-stall-priority/20261024-corrected-priority/summary.json`
* `docs/project-plan/single-image-plastic-controller-proposal.md`

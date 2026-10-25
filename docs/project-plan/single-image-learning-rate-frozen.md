# N75 — adjustable learning rate under the unchanged covariance rule

One evaluated trajectory, fresh_00, seed 5003, restarts from N73's identical
576-update complete parent and its 57,600-image continuation. Reuse N74's
exact corrected J forecaster, threshold and covariance ceiling, without refit.
Existing N74 fixed-rate covariance and corrected-J records supply descriptive
references. No new fixed-rate or accept-all trajectory is run.

The sole scientific policy change is per-image learning-rate backtracking. An
image must first pass the unchanged corrected J admission condition. Try one
singleton AdamW update at nominal rate. If the selector-bank covariance exceeds
the unchanged 1.25-warm ceiling, restore the complete pre-image model, optimizer,
modes and CPU/CUDA RNG and retry the SAME image and view at factors 1/4, 1/16,
then 1/64 of nominal. Retain the first passing candidate. Exhausting the four
rates skips the live update. A rejection by J is still skipped with no search.

Every trial starts from the same parent state. Failed trials must not accumulate
momentum, optimizer steps, weight decay or RNG changes. Accepted trials become
one live update. Reset optimizer group LR values to nominal after retaining the
candidate so the next image begins the same ordered search; keep its genuine
accepted-step moment buffers. Scale all parameter-group rates consistently.
Use actual AdamW steps, not a scaled approximation to an old gradient.

No final-query or development-probe outcome selects a learning rate. The fixed
covariance check selects it. Development and final probes are measurements only.
This is an exploratory test of LR flexibility, not a complete plasticity
controller. It does not address J-induced stalls or prove covariance is an
appropriate task proxy. A fixed cap may remain binding even at the smallest rate.

Record every attempted rate, actual trial covariance, training loss and gradient
norm, rejected-trial restoration, selected rate and predictions. Count live
updates separately from trial steps. Report sum of selected LR factors as nominal
rate-equivalent exposure, while explicitly noting that differing momentum makes
this an accounting quantity rather than equivalent model training. Report task
accuracy and drawdown alongside update counts to avoid equating tiny updates with
learning. Measure checkpoints every 384 arrivals, with a two-GiB storage reserve.

Smoke checks must include a real nominal-rate covariance failure followed by an
accepted smaller-rate update from the same source state, exact interrupted resume,
and reconstruction of every ordered decision. Synthetic optimizer tests check
that multiple rejected trials do not advance the live step count. The source
N74 worker is stopped at user request and its completed/partial artifacts remain.

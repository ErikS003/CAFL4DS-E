# Expanded precision budget for the same downstream capability

Frozen design date: 2026-10-15. User authorization: continue with a larger support
and query measurement budget. This is a separate follow-up to the failed
[measurement gate](../experiments/phase1/target-alignment-measurement.md).
Previous results, scientific files and criteria remain unchanged.

## Fixed capability and states

Keep city/highway classification from fifty labels, the same 100-step encoder
and head fine-tuning procedure, and the same day/night accuracy definitions.
Keep the 200-image support candidate pool and joint support/fitting seed law.
Keep all eighteen 32-update contrasts across six inspected histories and SSL seed
5003. Load the thirty-six saved encoders directly. No new SSL training or state
reconstruction is needed. This is conditional measurement, not fresh-source
confirmation or health/filter efficacy.

## Budget planning and query allocation

Use the original documented-ID exclusions and the original fifty-per-cell support
pool. Expand queries to all other eligible local clear-weather day/night,
city/highway validation images in that population. The expected cell counts are
733 day/city, 314 day/highway, 1,933 night/city and 683 night/highway: 3,663 queries.
The 1,024 pilot queries are reused; 2,639 additional queries are scored. Class and
domain weights stay equal despite different cell sizes. No support/query overlap,
pixel duplicates, new downloads or annotations. Session independence remains
unknown. These are additional documented identities under the original inventory,
not proof that every earlier image read was logged.

Before new outcomes, project support and crossed precision using the twelve-panel
pilot response arrays. For candidate counts 48/96/144/192/256, resample empirical
pilot panels at the proposed count and empirical pilot queries at each expanded
cell count. Use 1,000 planning resamples and seed 62501. Choose the smallest count
projecting both original precision fractions at least 80%; the anticipated
selection is 256. Stop before new outcomes if selection differs. Planning assumes
pilot response variation is representative. Its resampled duplicates create no
new evidence and do not guarantee achieved precision.

Preserve the original twelve seed IDs as a prefix, then add 244 integer seeds
62001 through 62244. Each fit still uses fifty labels and fitting seed `draw + 25`.
More draws reduce uncertainty about the expected learner under the fixed support
pool; they do not improve every individual classifier or broaden the support
population. Save per-panel results as well as means and disagreement rates.

## Fixed stages and stop rule

First fit 64 draws on all thirty-six states. Compute primary support/query/crossed
95% sensitivity ranges using 2,000 paired bootstrap resamples, as before. The
query-only range must have half-width <= 1pp in at least 80% of the 36 domain
contrasts to fund the remaining draws. This resource gate uses precision only;
neither help/harm direction nor a health metric selects continuation. Failure
ends the study without final qualification. It is not a proof that unlimited
draws could never improve the query influence estimate.

If that resource gate passes, continue to exactly 256 draws per state. Retain
the original final gates: support-only half-width <= .5pp and crossed half-width
<= 1pp in at least 80% of domain contrasts; helpful balanced gains under the
declared rule in at least two sources; domain loss exceeding 1pp under the
declared range rule in at least two sources. Helpful requires mean >= .25pp and
crossed lower endpoint > 0; harmful requires a domain's crossed upper gain
endpoint < -1pp. No promotion at 64 draws, threshold change, optional final
sample-size extension or outcome-selected subset. No health model is fitted.

Use bootstrap seed `62551 + panel_count`, resampling panels jointly and queries
separately within class/domain. These are conditional finite-pool sensitivity
ranges, not population coverage or per-head safety guarantees. Pilot-only and
old-query effects remain secondary diagnostics. Expanded-query effects can
change because the evaluation sample changes; do not call that a learner change.

## Execution, equivalence and cost

An engineering preflight repeats eight stored fits serially and thirty-two in
four independent processes. All forty must reproduce probabilities exactly.
Choose four processes only if warm fitting throughput improves at least 20%;
otherwise use one. It produced 1.804x throughput and selected four processes.
This preflight uses previously known outcomes only and adds no new study cases.
Any helper smoke replays are separately logged and charged.

Use the unchanged canonical fitting routine. Score the original 1,424-image
pilot/old-query layout separately from additional queries to preserve exact
readout numerics. The first twelve panels on all thirty-six states must reproduce
all stored probabilities exactly. Each process has private RNG and model copies.
Parallelism does not batch together examples from different learners.

Intended unique fits: 2,304 at the interim stage; at most 9,216 in total at 256
draws (921,600 supervised steps). Forty preflight replays and two final audit fits
are additional work. No SSL updates, health fits, downloads or optimizer previews.
Require 3GiB free disk at launch; output cap 2GiB excluding source snapshots.

Freeze source/config/protocol and raw input hashes before new outcomes. Each fit
is atomically saved and state-bound. Completed fits may be reused only after hash
validation; incomplete fits must be replayed and charged. Per-worker attempt
journals record started, completed and reused work. A worker failure halts launch
of later stages; preserve partial outputs for recovery without changing settings.
Completed collection is immutable. Do not automatically expand past 256 draws.

Independent verification reconstructs roles, support samples, exact bootstrap
multiplicities, stage/final effect ranges, resource/final gates and work accounting.
Reuse the independently written statistic reconstruction from the previous audit.
Check all 432 inherited panel replays and repeat two new canonical fits: the first
new seed on fresh_00/128, and the final included seed on fresh_05/296. Inspect
presentation figures and bind notebook outputs to the audited artifacts.

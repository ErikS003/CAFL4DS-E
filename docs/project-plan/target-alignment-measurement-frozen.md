# Target-aligned few-label measurement gate

Frozen design date: 2026-10-15. This implements the first gate of the
[target contract](health-target-alignment.md). User authorization: continue the
experiments. Scientific code/configuration and this protocol are hashed before
new downstream outcomes. Existing six-history outcomes are already inspected;
this is a conditional measurement diagnostic, not fresh-source confirmation.

## Objective and fixed population

Protect expected city/highway classification after the canonical 50-label,
100-step encoder-plus-head fine-tuning procedure. Preserve its learning rates,
optimizer, preprocessing, minibatch size, seed rule and clipping unchanged.
Use all six histories from the completed fresh functional guard. Fix SSL seed
5003, with parent ages 128/256/264, and compare each with its 32-update endpoint.
The 18 contrasts require 36 unique SSL encoder states. Neither contexts nor
support panels become independent source replications.

The previous run did not save these encoder states to disk. Reconstruct each
296-update path from the same warm model and footage. Require exact equality
of all 296 loss/gradient pairs and all thirteen prior complete-state fingerprints.
Readout must preserve the complete SSL learner, optimizer, module modes and RNG.
Save encoders and all fitted probabilities for direct inspection and replay.
This work is charged as reconstruction, never reported as zero SSL computation.

## Evaluation identities and randomness

Inventory prior `input_lock.json`, `data_lock.json`, data-role manifests,
evaluation manifests and corpus manifests outside source snapshots. Exclude all
documented BDD identities in them. Eligible target images are local validation
JPEGs with clear weather, daytime/night and city/highway labels. This conservative
inventory does not prove every historical image read was logged. Capture-session
identity is unavailable; independence is at documented clip/image identity only.

Within each domain/class cell, rank remaining identities by SHA256 of
`target-alignment-measurement-20261015:<clip_id>`. Reserve the first 50 as the
support pool and the next 256 as queries: 200 support candidates, 1,024 queries.
There is no outcome-based replacement, additional download or new annotation.
Require disjoint identities and no resized-pixel duplicate within the new set
or against the inherited 900-image evaluation cache.

Use the twelve support seeds in the config. Sample uniformly without replacement
within each cell using the inherited Torch generator and class/domain order.
Each panel contains 25 images per class and a counterbalanced 12/13 allocation
across domains. The resulting domain totals are 24/26 or 26/24. Fit on those
50 labels, with downstream seed `draw + 25`. Use identical panels and seeds
on the parent and post-update state. Variation is over this specified joint
support/fitting law; this does not separately identify initialization variability.

The same fitted heads also score the old 400 city/highway queries as a secondary
query-population sensitivity diagnostic. They do not affect selection or gates.
Those images have been repeatedly inspected. New support and query labels remain
offline; no health monitor or online filter consumes them.

## Estimation and declared gates

Primary outcome is the paired accuracy gain, averaged equally over classes within
each domain and then equally across domains. Save all twelve panel gains and
their positive/negative fractions. Retain probabilities for secondary CE analysis;
CE cannot replace failed primary accuracy. No optimal-reader or threshold search.

Use 2,000 paired bootstrap draws: sample twelve panels with replacement and sample
queries with replacement separately within each class/domain cell. Save exact
multiplicities. Compute support-only, query-only and crossed percentile 95% ranges.
The same sampled support or query applies to both endpoints. These are conditional
finite-pool sensitivity estimates. They are not guaranteed population coverage,
independent event confidence intervals or protection for every downstream head.

Declare measurement sufficiently resolved for a subsequent health study only if:

1. At least 80% of the 36 domain contrasts have support-only half-width <= .5pp.
2. At least 80% have crossed support/query half-width <= 1pp.
3. At least two distinct histories contain a balanced gain >= .25pp whose crossed
   lower endpoint exceeds zero.
4. At least two distinct histories contain a domain loss > 1pp whose crossed
   range is wholly below -1pp in gain units.

The .5pp precision reference aims to keep support uncertainty below half the
1pp domain-loss band. The crossed precision band and population requirements are
bounded research go/stop rules. They are not an operational safety contract or
multiplicity-adjusted claims of source-level efficacy. Report all contrast ranges,
not only the qualifying ones. No automatic panel/query expansion follows a failure.

If any gate fails, stop before health fitting and explain which measurement or
opportunity limitation remains. If all pass, prepare a separate frozen health
bridge protocol; this run does not automatically launch it. No health regressions,
incoming-image predictors, filter policies, new thresholds or gradient previews
are fitted in this study.

## Work limits, audit and recovery

Budget: 1,776 SSL reconstruction updates, 432 new 100-step downstream fits, and
36 saved encoders. Independent verification adds one 296-update state replay and
two complete downstream fit replays. Total intended SSL work is 2,072 updates;
supervised work is 43,400 steps in 434 fits. No acquisition/network work. Require
2GiB free disk before collection; new output cap 512MiB excluding source snapshots.

The verifier reconstructs data roles, support draws, resampling multiplicities,
all paired effect/range tables and gate decisions independently. It compares one
entire reconstructed SSL path with prior traces and saved encoders, and repeats
two predetermined fits (fresh_00/128/first draw; fresh_05/296/last draw) exactly.
Unit tests include zero-change, opposing domains and literal paired-bootstrap
controls. Existing locked studies and input files are not modified.

Resume only with identical hashes. Completed per-panel fits may be reused after
their hashes and state binding are checked. Partial path reconstruction must be
charged again and disclosed; do not silently rewrite completed outcomes. A
completed collection is immutable. Source/state mismatch halts the run without
scientific retries or recipe changes.

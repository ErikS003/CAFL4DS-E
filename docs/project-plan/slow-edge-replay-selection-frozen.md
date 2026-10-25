# Fixed-allocation replay selection: frozen protocol

## Question and scope

Can choosing *which* old images train the encoder yield a repeatable improvement over
standard online learning plus uniform replay? If so, do task gradients identify that
improvement, and can unlabeled features support a cheaper selection/gating policy?

This is a local decision study using existing BDD images, support/query panels and
12 audited transition parents from three original pool groups. It is exploratory
evidence on reused data, not independent task confirmation or a closed-loop MPC trial.
`--smoke` uses one parent and two draws to validate execution, not to qualify efficacy.

## Interventions

All arms use the same16 current images and16 replay images,32 total. The256-image
all-arrival reservoir, its random generator, stream arrivals and optimizer settings
are identical. A shared random permutation supplies a64-image shortlist; uniform
uses its first16, which is also an ordinary uniform draw from the full reservoir.

1. **Uniform:** standard16/32 replay.
2. **Coverage:** greedy farthest-first cosine coverage on parent encoder features,
   initialized with the16 selected current images; choose16 from the shortlist.
3. **Task gradient:** partition the shortlist into eight disjoint blocks of eight.
   Score each using two copied AdamW steps on16 current +8 candidate images, dotted
   with the labeled task CE gradient; choose the best two blocks. This24-image
   surrogate is a ranking approximation, not an additive prediction of the final
   32-image update. Validate it separately against copied steps on each final batch.
4. **Mixed:** keep the first eight uniform images and choose the best gradient block
   excluding the overlapping first block. No replay image is duplicated.

All selected sets are frozen before outcome collection. Final batches use
the same shortlist order; equal selected sets therefore yield
identical ordered batches, rather than changing stochastic-mask assignment by rank.
Task head fitting uses
support1, gradients use disjoint support2, and support3 tests transfer. Two secondary
rotations record choice disagreement without selecting the primary rotation using
outcomes. Head C=.1, solver tolerance1e-10. Task-gradient and mixed action generation
require labels; neither is an unlabeled health controller.

Only the **first** update differs. The next three use ordinary paired uniform16/32
replay. Score after1 update (diagnostic) and4 updates (primary delayed benefit). This
isolates one decision and avoids reading future arrivals to select later actions.

## Randomness, targets and readout

Three new outcome draws, seeds193001/194009/195023 plus the recorded parent seed,
are paired across all rules. Observation seeds181003/182009 are separate. Every
branch starts from identical encoder, decoder, AdamW moments and reservoir state.
Selection must preserve ingestion and future selected image identities across arms.

Primary outcome: held-out query balanced linear-probe accuracy with the pooled1152
support images and inherited support-selected C. Secondary readouts: three separate
384-image support heads, development query, CE, day/night metrics, and frozen task-head
CE on support2/support3. Refit probes at tolerances1e-8 and1e-10 for every endpoint
and parent. Normalization and fitting use support only. Reuse of past-selected C is
declared; new queries never choose C or the task-gradient rotation.

Predeclared numerical gate: maximum paired accuracy-contrast change between solver
tolerances <=.01 percentage points. Larger changes block tiny-gain qualification.
Minimum interesting mean gain: .1 percentage points; report smaller effects without
qualifying them. Also show each pool, draw, reader and domain; avoid treating12
related states or image-level scores as independent datasets. No confidence claim
based on three original pool groups.

## Analysis and health bridge

Report paired rule-minus-uniform gains and selection overlap. Diagnose opportunity
by choosing a rule from the *other* outcome draws, then scoring on the untouched draw,
with a .1pp threshold and uniform fallback. This diagnostic uses future query labels:
it is not an operational policy or a proven upper bound. A same-draw hindsight
maximum is displayed separately and must not be interpreted as achievable benefit.

Predecision unlabeled features include reconstruction, rank/variance/drift on the
shortlist, selected-set coverage, SSL gradient norms/cosines and virtual SSL loss.
Task-gradient forecasts are stored separately and excluded from health models.

After collecting draw0, fit prespecified Ridge(alpha100) coverage-versus-uniform
accuracy-response models on four-update development outcomes from the other two original pools.
Compare constant, reconstruction, and health+SSL-gradient features. Freeze all
choices before training draws1/2. Predicted gain must exceed .1pp to choose coverage;
otherwise choose uniform. This health bridge chooses only uniform/coverage, whose
generation is unlabeled. It cannot choose the label-dependent task/mixed arms.
With only three reused pools and few states, this is an exploratory gate, not a
qualified health controller; retain exact fits, inputs and choices for auditing.

## Budget and artifacts

Full study:12 parents ×4 arms ×3 draws ×4 updates =576 real optimizer updates;
24 copied observation steps and3 task backward observations per parent, plus two
current SSL backward observations. Observation compute is additional to training.
Endpoint probe fitting cost and numerical audits are recorded separately.

Save protocol/source/input hashes, frozen selections, block scores, predecision
features, model coefficients, cross-pool choices, branch decisions, checkpoints,
features, heads, solver iterations, paired contrasts and a report with graphs.
Rehash inputs and frozen decisions after collection. Preserve every historical run.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_replay_selection outputs/image-utility/20261009-replay-selection-smoke --smoke
.\.venv\Scripts\python.exe -m scripts.run_replay_selection outputs/image-utility/20261009-replay-selection
.\.venv\Scripts\python.exe -m scripts.report_replay_selection outputs/image-utility/20261009-replay-selection
.\.venv\Scripts\python.exe -m scripts.verify_replay_selection outputs/image-utility/20261009-replay-selection
```

Report figures live in the run directory. This study does not add another notebook
to the curated main folder; inspect its standalone report before promoting results.

## Completed run: 9 October 2026

The completed four-update-target run is `outputs/image-utility/20261009-replay-selection-delayed`.
The earlier `20261009-replay-selection` folder was stopped before outcome training
to correct the gate target from one update to four; its record is preserved.
The completed run contains 144 branches, 576 optimizer updates and 6,600 independently
recomputed readouts. See its `delayed_report.md` for conclusions and reuse limits.
The existing overview notebook includes this result; no new numbered notebook was added.

To regenerate the delayed analysis from saved results, without training:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_delayed_replay_selection outputs/image-utility/20261009-replay-selection-delayed
```

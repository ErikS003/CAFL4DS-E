# Local functional response profile — study52

Retrospective known-cohort development after study51's failed static-profile
incremental forecast. No alarm, controller qualification or fresh confirmation.
Freeze sources/config/tests/synthetic receipts and every consumed input before
real reconstruction, SSL probe updates, encoder inference or response analysis.

All72 study43 states (3 seeds x3 sources x4 contexts x2 pre-diets), original
utility contract, all14 procedure/component targets. Faithfully reconstruct each
fork from its saved complete512-update historical learner/memory with the same
128 pre-action updates. Verify every loss/gradient/displacement and complete
learner fingerprint against the original trace, and final full fork fingerprint.
Original observers preserve state/RNG, so omit them. Budget9216 reconstruction
SSL updates, explicitly counted physical work despite reproducing known states.

Probe only on ALREADY OBSERVED current batch127 (never future128+), verified
against the original trace's current-batch pixel hash. Replay batch uses the
same fixed balanced schedule at127,16 current/16 saved historical images.
No future batches or benefit query inputs enter collection. From independent
copies of the exact fork, propose one inherited AdamW SSL update per action,
using the same case-specific seed195200+case_index for both actions. Repeat
each proposal from the original fork/seed and require exact response, encoder
and full learner fingerprints.144 proposals +144 repeats. Total physical SSL
budget9504, no silent retries. These steps occur on copies and are not new
future continuation trajectories. Save both raw encoder displacements and
their parameter names, batch hashes, raw norms and replay/current angle.

Use the SAME proposed encoder direction for two fixed magnitudes:0.25 and0.5
times the original positive pre-action trailing16 actual-step norm. Normalize
encoder parameters only, using the existing exact-step procedure; retain the
proposal's decoder/Adam only inside its disposable copy. Assert realized norm
within max(1e-9,2e-5*target), independently verify vector direction and dose ratio.
Do not recompute gradients or optimizer proposals for the second magnitude.

For zero step and every action/dose, infer all200 pre-action validation-support
images. Before freeze, prepare a support-only tensor by copying the first200
images from the pinned original image file; collection never loads its query
images. This input preparation performs no inference or training. Infer these
images twice in deterministic CUDAfloat32/TF32off/batch32 mode with observer
modes and RNG restored. Require exact repeated features; zero step must exactly
reproduce study44's cached support features.720 passes/144000 encoder image
forwards:72 states x(2 zero passes +2 actions x2 magnitudes x2 repeats)x200.
SSL internal training forwards are separately included in9504 updates/304128
training-image presentations. No new probe/readout optimizer fits. Disk reserve
1536MiB/outputcap512MiB. Cache only support features, proposal deltas, journals
and receipts; preserve completed states and any interrupted work.

Readout measurements use study51's FIXED96-label OOF heads,6 pools x2 priors
x4 folds, always evaluated on each head's held-out32 support images. Do not
refit readers after perturbation or choose priors from utility. All support
IDs lie within200 validation images, identity-disjoint from old2048 query.
For each domain and pool/prior, compute changes in mean-score class separation
and preference divided by BASELINE within-class pooled SD; log(after SD/before
SD); and boundary-mass change with threshold FIXED at0.25*baseline SD. Equal
class weighting and SD floor1e-8 as study51. Frozen reader bias cancels from
score differences; no reference-bank Procrustes alignment is needed.

Define response vector as replay response minus current response at matched
actual norm. Primary: half-step/lambda0.01/arithmetic mean of6 pool vectors,
night4 axes. Save all8 axes, both actions, both magnitudes/priors, all pools.
Static profile remains the exact study51 OOF profile. Preserve finite-dose
nonlinearity (half response minus2*quarter response) and pool variability; no
post-hoc dose or feature selection. Seal ALL response arrays before target
loading. Known outcomes already exist, so separation is procedural only.

Use original fixed-reader night-highway replay-minus-current384-update benefit
as primary. Original joint/all7 component outcomes remain secondary. Strict
seed AND source AND context exclusion (36 folds,24 train/2 test), train-only
standardization, ridge10. Fixed models: mean; cheap(diet +trailing loss/gradient/
actualstep +diet interactions); static(cheap+night static4/interactions);
response_only(diet+night response4/interactions); response_added(cheap+response);
static_response_added(cheap+static+response, PRIMARY); cka_response_added(cheap+
CKA+response). No tuning.10 panels: mean pools at both doses/priors (4), then
each single pool at primary half/.01 (6).2520 forecast ridge fits, independently
verified by augmented least squares, separate from SSL/readout budgets.

Primary development screen requires>=10% MSE reduction relative to BOTH cheap
and static, with improvement in EVERY seed and source for each comparison.
No promotion based on secondary models/panels. Report all contexts including4,
both diets, all classes, shadow value vs always-current/replay/cheap/static,
and per-diet exploratory correlations. No causal-health interpretation or
population confidence intervals: same known cohort, shared warm weights/footage,
unknown capture sessions, overlapping support pools. A one-batch response can
be noisy, reflects recently seen frames and need not predict384-step behavior.

Six figures frozen before collection: design and response algebra controls;
reconstruction/repeat/zero audit and dose nonlinearity; response components by
context/diet; forecast MSE and seed/source stability; dose/prior/pool sensitivity
and predicted vs observed benefit; per-diet correlations and shadow class values.
Validate collection mechanics on synthetic images/model state and six synthetic
figures before freeze. Inspect all six real figures, reexecute notebook saved
reconstruction, preserve original exports before adding observed interpretation.
Engineering includes a logged first synthetic attempt (4 SSL/10 passes) whose
final guard mistakenly compared deliberately changed global RNG; restore the
fork RNG before checking the untouched original learner. The successful second
attempt adds4 SSL/10 passes, totaling8 synthetic SSL/20 passes/4000 forwards,
separate from the9504 real reconstruction/probe updates and720 real passes.

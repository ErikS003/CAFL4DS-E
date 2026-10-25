# Single-image signal, reader reliability and longer forecasts

User-authorized follow-up, 2026-10-21. Freeze this protocol and scientific sources
before collection. Keep the warm TinyViT MAE, mask .75, light augmentation,
AdamW learning rate 1e-4, decay .05 and clipping at one. Every SSL update uses
exactly ONE image. No accumulation, learner resets or deployed selection policy.
This study separates training randomness, reader variation and optimizer motion
before claiming image-specific forecast value.

## Sources and timing

Use the existing 144 verified coverage-admission clip caches. The first clip of
each of twelve periods makes a development history; the second makes a held-out
history. There are six of each, with no clip overlap. Use each clip's first 96
ordered frames, giving 1,152 arrivals per history. Preserve short/medium/long
day/night dwell patterns and synthetic joins. These sources were inspected in
older project studies; held-out means excluded from this forecast fitting, not
newly collected population confirmation. The first-clip sources also occurred
in the prior batch-one pilot. The second-clip sources did not occur in that pilot.

Use one optimizer seed, 5003; repeated training draws in the diagnostic do not
stand in for independent histories. Derive a private training seed from history
and arrival clock. All counterfactual branches share future arrival seeds, so
skipping an update cannot shift subsequent masks/augmentations. Record this RNG
schedule change from the earlier pilot. Initialization uses a fresh AdamW state.

## Longer time series

Forecast origins are 128,160,...,1088. Observe states at t-1,t,t+1,t+8,t+32,
and initialization. At every required state measure the same 256-image health
bank and 400 balanced development queries. Main readers use the original first
50-label support panel, refitted linear/kNN and a fresh 100-step fine-tuned
encoder copy with the original recipe. Report balanced/day/night accuracy and
cross entropy. Average probabilities/losses in float64; retain all raw query
probabilities and bank features. Diagnostic support fits retain batch32 and
never touch the live batch-one SSL learner.

For each horizon H, target is m(t+H)-m(t); next-level and anchored-delta errors
are identical. H=1 stays primary. H=8/32 means sequential ONE-image updates,
never a larger training batch. Required baseline is H times the last observed
single-update change m(t)-m(t-1), plus no change and a calibration mean.

Use ridge10 to predict the residual beyond the last-change baseline. Fixed
feature recipes: state/history; history plus optimizer summaries; history plus
image observations; history plus optimizer plus image; the latter plus paid
pre-update gradient summaries; and label-free health/optimizer/image/gradient
inputs. Image features match the prior pilot. Optimizer features summarize
parameter/first-moment/second-moment norms and optimizer age. Paid gradient
features condition on the actual upcoming training view and include loss,
gradient norm, gradient/momentum alignment, preconditioned gradient norm and
encoder-gradient fraction. They require an extra backward pass; they are not a
cheap selector and cannot be presented as a free image observation. No copied
post-update metric enters any predictor.

Compare three modes: frozen transfer trained only on development histories'
prefixes; frozen prefix trained on development plus held histories' prefixes;
online prefix initialized identically to frozen prefix and subsequently updated.
Only labels whose target clock is <=512 may fit either frozen model or prefix
normalization. At a later origin t, online updates may use held-history outcomes
only after target clock <=t. Predict every held history at t before admitting
any same-origin future target. Never fit development outcomes after clock512.
Score held-history origins 544,...,1088: 108 cases per horizon. This explicitly
distinguishes source transfer from calibration on a new history's first N images.

RMSE and MAE in native units are primary; report direction/ties and all recipes.
The main image ablation is online history+optimizer+image versus online
history+optimizer. The paid gradient extension is separate. Do not select a
different winning model after inspecting each target. Whole-history bootstrap
keeps all times from each of six held histories together; intervals remain
descriptive and unadjusted across the 25 dependent metrics and three horizons.
Call an image forecast useful only if both errors beat both matched history
and last-change controls on held histories. No controller is qualified here.

## Matched counterfactual diagnostic

At state768 in each of six held histories, restore the complete learner state.
Compare hold (no optimizer step), a real zero-gradient AdamW step, and training
one image. A zero gradient is an explicit tensor for parameters used by this
MAE; None must not be substituted, because it skips an optimizer step. Separate
the exact contrasts actual-minus-hold, zero-minus-hold and actual-minus-zero.
The last contrast is a defined gradient-increment counterfactual, not a claim
that AdamW momentum and current gradients contribute linearly.

Candidate indices are t, t-16 and t-64: the currently arriving image and two
already available historical images. This is a controlled image-substitution
diagnostic, not a policy comparison or three simultaneously novel arrivals.
Use three matched augmentation/mask draws for every candidate. All scientific
branches use one image and the same starting optimizer. Report image contrasts
after centering within parent, to avoid mistaking different states for image
signal. Compare between-image variation with training-draw variation, leave-one-
draw-out agreement and cross-reader agreement. Do not count 54 branches as 54
independent sources.

Observe immediate outcomes with three original disjoint support panels. For
each panel run both 20-step and 100-step copied fine-tuning, with matched panel
initialization/minibatches before and after. Measure state767 with the same
panels to provide the last-change baseline. Compare 20/100 effect signs and
readout agreement; the shorter assay cannot replace the longer assay by default.

For the currently arriving candidate only, continue its three draw branches,
the zero-gradient branch and the hold branch through identical future images
and matched private training seeds. Read all three 100-step panels at horizons
8 and32. This measures whether a single acceptance/skip difference persists;
it differs from observational next-state forecasts of 8/32 ordinary updates.

## Verification and boundaries

Preflight uses discarded updates only, checks singleton updates, probe purity,
actual-view loss agreement, explicit-zero versus None behavior and observer
invariance. Retain prior artifacts unchanged. Lock source/config/input hashes,
selected clip identities, parents, states, raw probabilities/features, gradient
and optimizer observations, all forecasts, counterfactual effects and work counts.
Replay all longitudinal updates without observers, matching every loss/gradient,
saved checkpoint and final model/optimizer/RNG state. Replay selected diagnostic
branches, reconstruct geometry/readers from raw arrays and regressions by a
separate direct least-squares implementation. Do not adjust sources, budgets,
models, learning rates or targets after seeing outcomes. A measurement failure
limits a forecasting claim but does not suppress the requested descriptive fits.

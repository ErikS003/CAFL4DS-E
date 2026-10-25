# Historical training-diet smoothing

The question was whether gradually mixing old night frames with returning-day frames prevents harmful adaptation. This changes the filter's training diet while keeping the SSL learner and optimization budget fixed. Continued-night footage provides a control for deterioration without a day/night switch.

## Method

The [frozen protocol](../../project-plan/slow-edge-diet-smoothing-frozen.md) restores the same three audited night-end parents used in the paired return-action study. Each parent branches into hold, full current-only training, gradual mixing and abrupt replay, for both day2 and night2. No labels select training images. All training arms use132 updates and4224 image exposures. Hold performs no updates.

Memory contains512 historical images: the last64 frames of each of eight already observed night1 clips. Gradual mixing uses16,16,15,15,...,1,1 historical images per batch for32 updates, then current-only batches. Abrupt replay uses eight entirely historical batches and one half-historical batch, then current-only batches. Both schedules expose exactly the same272 historical images in the same sequence and3952 new images. New-image identities and exposure timing differ by design. Full exposes4224 new images. Thus gradual versus abrupt tests scheduling at matched historical/current dose; gradual versus full combines replay, new-image admission and scheduling.

Every arm observes4224 continuation arrivals. Readouts at256 and4224 arrivals use the existing two-class city/highway task,400 balanced queries and three fixed50-image support panels. Primary accuracy measures few-label encoder fine-tuning capability. Linear and kNN are secondary. Readouts preserve learner/Adam/RNG state. Paired95% intervals average support panels within each of three optimization seeds; these are conditional development intervals, not independent-drive uncertainty.

## Findings

| Continuation | Action | Final day % | Final night % | Final balanced % |
|---|---|---:|---:|---:|
| Returning day | Hold |55.833|63.778|59.806|
| Returning day | Current only |54.389|62.056|58.222|
| Returning day | Gradual mixing |51.833|62.722|57.278|
| Returning day | Abrupt replay |53.444|62.778|58.111|
| Continued night | Hold |55.833|63.778|59.806|
| Continued night | Current only |56.222|56.444|56.333|
| Continued night | Gradual mixing |56.167|56.167|56.167|
| Continued night | Abrupt replay |56.111|56.500|56.306|

Returning-day gradual minus current-only balanced accuracy is−0.944pp [−1.064,−0.825]. Gradual mixing retains slightly more nighttime primary capability (+0.667pp [+0.253,+1.081]), but loses2.556pp [−3.034,−2.077] daytime capability. It therefore trades a small retention gain for a larger current-domain loss.

Gradual minus abrupt replay is−0.833pp [−2.396,+0.730] on returning day and−0.139pp [−1.387,+1.109] on continued night. Neither scheduling contrast establishes an advantage. Gradual minus hold balanced accuracy is−2.528pp [−3.549,−1.507] on returning day and−3.639pp [−3.878,−3.400] on continued night.

At256 returning-day arrivals, gradual mixing's day accuracy54.111% exceeds current-only53.333%, but its final day accuracy51.833% is lower. Abrupt replay has early day accuracy56.222%, while having trained exclusively on historical frames at that point. These early differences do not demonstrate successful new-domain learning or sustained protection. Early historical/current exposure is deliberately unmatched.

Hold meets the frozen joint quality-preference screen on returning day. No action meets that screen on continued night because hold's daytime guardrails against some training arms remain uncertain. Hold nevertheless has significantly higher balanced primary capability than gradual mixing in both contexts. Decision: no gradual-smoothing lead; retain the previously established local pause result.

Secondary readers limit the interpretation. On returning day, gradual versus full linear day accuracy is60.944% versus62.000%, whereas kNN day accuracy is55.167% versus54.167%. Thus the large primary daytime deficit is not uniform across readers. Continued-night declines relative to hold persist in both secondary readers. The results do not establish a universal catastrophic-forgetting mechanism or failure of every smoothing design.

## Cost and validation

Scientific collection:24 branches,2376 SSL updates and117 primary fits. Discarded preflight:32 updates and3 fits. Exact replay:2376 additional verification updates,48 complete checkpoints. Runner elapsed281.911 seconds including preflight and replay. Independent verification checks2245 locked hashes,8448 source timestamps/pixels,351 new prediction sets and164 paired primary intervals. Hold learner/Adam/RNG invariance and prior full-day states/predictions reproduce exactly. Nine targeted tests and scoped lint/format checks pass.

Historical float32 pixels occupy25,165,824 bytes (24MiB). Retrospective memory construction took0.032 seconds; this excludes maintaining the memory during the original trajectory. Selection and training timings are archived, but no end-to-end efficiency claim is made. All training arms have the same update work, and mixing additionally retains memory. Full versus mixing differs in unique current-frame coverage.

## Research consequence

Close this fixed taper recipe's promotion claim. The experiment does not justify automatically expanding taper lengths or replay ratios. Keep SSL and the hold/full references. The next proposed qualification concerns where additional updates are beneficial versus harmful along the trajectory, using observations available before the action. Health-driven tuning remains conditional on that opportunity. No subsequent training has been launched.

## Artifacts and reproduction

Output: `outputs/diet-smoothing/20261013-gpu`. Scientific inputs and source snapshots are in `input_lock.json`, `data-role-manifest.json` and `source_snapshot/`. Per-branch `training.json`, `state.pt`, `readout.json` and `predictions.npz` preserve exposure traces, complete state and predictions. `summary.json`, `training-replay-audit.json`, `independent-verification.json`, `cost-breakdown.json`, `memory-construction.json` and `smoothing-contrasts.png` record the conclusion and checks. `collection-report-original.md` preserves the runner report before presentation-only table formatting.

Run `.venv-gpu\Scripts\python.exe -m scripts.run_diet_smoothing_experiment --output outputs/diet-smoothing/20261013-gpu --night-frames outputs/return-actions/20261013-night2-frames`, then `.venv-gpu\Scripts\python.exe -m scripts.verify_diet_smoothing_experiment --output outputs/diet-smoothing/20261013-gpu`. Existing output resumes only under the identical lock. Deterministic CUDA settings are set in the runner. Sources and parents are reused development data; three seeds and unknown capture groups prevent source-independent confirmation.

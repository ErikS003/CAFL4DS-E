# Frozen persistent-novelty admission experiment

13 October 2026, frozen before downstream outcomes. The user authorized this filter-first test while retaining SSL. No new population or parameter sweep is introduced.

## Scientific question and environment

Can persistent changes plus guaranteed temporal exploration produce a better SSL training diet than uniform or random half admission at the same accepted-image/update budget?

Return to the earlier positive chronological temporal setting: all 48 selected train clips, the original order and the 128 timestamped RGB64 frames per clip in `20261011-frames-v3/intake.json`. Exactly 6144 arrivals. Preserve clip boundaries and reset content references there; concatenate clips only as the existing constructed day-to-night schedule, never call it a continuous drive. No global shuffle, selected subepisodes, replaced clips, new seeds or horizon tuning.

Use the same clean-warm TinyViT MAE well, decoder, masking .75, normalized-pixel objective, light spatial augmentation, batch32, AdamW LR .0001/weight decay .05, gradient clipping1 and no replay. Train seeds5003/5009/5011/5021/5023. CUDA RTX2080 float32, deterministic algorithms, AMP/TF32 off; do not pool CPU/GPU timings or require cross-runtime bitwise CPU reproduction.

## Fixed filter rule

Process each completed FIFO window of64 arrived frames. Frozen warm-encoder features are normalized to unit length. Within a clip, the reference is the normalized mean of the last16 previously accepted feature vectors; the first window bootstraps with its first frame. No validation or runtime labels enter selection.

For candidate position t, compare the residual feature vectors at t,t+1,t+2 with that reference. Persistence score = minimum residual norm across the three × the nonnegative minimum pairwise cosine of their directions. This favors departures that remain present and directionally consistent. It does not certify semantic utility or identify sensor noise. Zero residuals safely yield zero. Final two positions have no complete confirmation and receive zero content priority. No later window is inspected.

Reserve16 exploration anchors at positions3,7,...63, including the latest frame. Choose16 additional nonanchor positions from0..61 by descending persistence score, earliest ties; train all32 selected images chronologically once. All-zero scores use the same deterministic tie rule. Keep the last16 selected embeddings as the next window's reference population. Clear frame storage after each block. Never carry the reference across unrelated clips. Fixed confirmation3, history16, exploration16/32; no alternate settings after inspection or outcomes.

| Arm | Selection from64 arrivals | Updates per full stream |
|---|---|---:|
| Warm frozen | No SSL |0|
| Full-rate | All64, two32-frame updates |192|
| Uniform half | Positions1,3,...63, one update |96|
| Random half |32 without replacement, sorted chronologically, private CPU RNG seed+62003 |96|
| Persistent novelty half |16 uniform anchors plus16 persistent changes, one update |96|

All trained arms use the same64-arrival waiting boundary for this comparison. Maximum FIFO wait63 intervening arrivals; content confirmation waits at most two subsequent arrivals within that window. Full retains the earlier update sequence; timing measures this shared-wait implementation. Half arms have identical work and latency; comparisons with full intentionally include the training-budget difference.

## Before outcomes

Verify original frame/cache hashes, advancing timestamps, clip order and zero warm/validation overlap from the preserved selection reconstruction. Reconstruct the original validation pool and verify exact pixels, labels, domains, support/query memberships against the cached earlier content study; use local validation labels, not the provisional test annotation archive from the latest short-buffer screen. No sealed query panel is accessed.

Inspect candidate selections on these development clips without downstream fitting. Require exact32-per-window admission, all16 anchors, reference causality and clip resets, and at least10% selected-slot replacement relative to uniform averaged across windows. Record per-stratum acceptance and score/selection descriptions using keyframe metadata solely for diagnosis; each clip must have exactly64 accepted arrivals regardless of stratum. Archive representative overlays from fixed hash-selected clips; no outcome-based clip choice or threshold tuning. An ineffective/invalid selector stops before scientific collection rather than being retuned.

Test persistent-versus-isolated synthetic signals, directional disagreement, ties, budgets, private RNG, causal prefix invariance and reference reset. Discarded real-MAE preflight requires exact model/Adam/CPU-CUDA-RNG continuation and downstream-copy isolation. Count preflight work separately. Hash executable sources, protocol and data before fitting.

## Readout and bounded budget

Use the original 900-image validation allocation and three75-label support draws51011/51017/51023, 600 queries. Primary balanced scene accuracy after copied-encoder100-step supervised fine-tuning, encoder LR1e-5/head1e-3, AdamW decay.05, clipping1, draw+25. Day/night separate; linear and kNN mandatory secondary diagnostics. Labels are only used in evaluation. Preserve learner parameters, Adam, all module modes and CPU/CUDA RNG through readouts.

Four trained arms × five seeds =20 trajectories and2400 scientific SSL updates. One shared frozen endpoint plus20 trained endpoints =21 evaluations and63 primary fine-tuning fits. The identical warm-frozen readout is reused across seeds instead of pretending duplicate fits are independent. Terminal-only scoring. Complete state, selector history/RNG, traces and raw predictions are saved. Resume only exact locked inputs; completed fits are reused. Replay every trained trajectory and independently regenerate selection scores/indices and reconstruct all189 prediction sets. Audits/preflights do not enlarge the scientific sample.

## Decision and costs

Average support draws within each optimization seed, then use paired two-sided Student-t95% ranges across the five seeds. These are conditional training-seed diagnostics on a reused single stream, not confidence intervals over independent drives.

Content value requires persistent-minus-uniform and persistent-minus-random primary lower bounds both positive, mean gains each at least .1pp, and at least4/5 positive paired seeds for each. Useful filtered SSL requires persistent-minus-frozen primary lower bound positive. Quality guardrails require persistent-minus-full lower bounds above-1pp for balanced/day/night. Efficiency additionally requires at least25% mean paired measured processing savings versus full. Report these separately: a quality lead can merit qualification even if monitoring costs fail the efficiency target. A cheap uniform filter can be an efficiency lead without content value.

The .1pp floor is an exploratory materiality threshold; the1pp margin and25% savings target are prespecified practical development screens, not powered guarantees. Report every arm/readout and all domain/seed/panel contrasts. A failed screen is unqualified/inconclusive, not proof that SSL universally fails. Full-rate need not qualify before evaluating the candidate; the filtered system must demonstrate useful learning itself. No health model, acceptance-rate sweep or new architecture is authorized by this protocol.

Charge host-cache gathering, transfers, frozen reference encoding of every arrival for the candidate, score/history updates, assembly and synchronized SSL. Rotate arm order by seed before outcomes. Report component costs and excluded acquisition/decoding/resize/checkpoint/readout costs. Selection features are never free. Untouched source-group confirmation and slow health steering remain later, separately frozen work.

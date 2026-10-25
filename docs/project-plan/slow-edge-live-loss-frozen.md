# Frozen live SSL-loss admission experiment

13 October 2026. User-authorized five-arm filter-first GPU comparison; frozen before scientific downstream outcomes.

## Fixed environment and work

Reuse exactly the persistent-novelty study's positive 48-clip chronological BDD train allocation, 128 RGB64 timestamped frames per clip, 6144 arrivals, original clean-warm TinyViT MAE and original validation readout. Preserve the mask ratio .75, normalized-pixel objective, light spatial augmentation, batch32, AdamW LR1e-4/decay.05 and clipping1. No replay. Seeds5003/5009/5011/5021/5023. Deterministic RTX2080 float32, AMP/TF32 off. Clips are a constructed schedule; capture-session independence is unknown.

## Frozen admission rule

Each completed 64-arrival window: full admits64/two updates; uniform half admits positions1,3,...63; random half draws32 without replacement using private CPU RNG seed+62003; live-loss half admits the32 highest current MAE per-image reconstruction losses, earliest exact ties, then trains chronologically in one update. One shared frozen endpoint has no online training.

Score before the window update, eval mode/no gradients, two chronological batches of32, with the baseline light augmentation (one crop/flip draw shared within each batch) and one independently sampled patch mask per image. Scoring CPU/CUDA generators are temporarily seeded seed+72003+window_index and restored afterward. Training draws a fresh augmentation/mask on selected images. Restore every module mode; scoring must preserve parameters, buffers, Adam and training RNG. No anchors, variance score, label use, threshold fitting or mask-count sweep. Confirm scoring purity at warm and nonempty Adam states, exact continuation and readout-copy isolation in discarded preflight. Require positive measured preflight net saving before full collection; no retuning if it fails.

## Inputs, readout and audits

Verify original clip/frame/source hashes and overlap checks; reconstruct original900-image validation cache against raw pixels and original BDD labels. Three75-label support panels51011/51017/51023, 600 queries in six100-image cells. Primary copied-encoder100-step fine-tune, encoderLR1e-5/headLR1e-3, AdamW.05, clip1, draw+25. Day/night reports, linear/kNN secondary. No sealed queries. Labels only enter evaluation.

20 trained trajectories, 2400 scientific SSL updates, 21 endpoints and63 primary fine-tuning fits. Terminal-only evaluation. Discarded preflight14 updates/3 fits; preparation scoring is separately recorded. Complete model/Adam/mode/CPU-CUDA-RNG/private selector state and per-window scores/indices retained. Replay all2400 updates, independently rank saved losses with Python and reconstruct189 prediction sets. Preserve source/runtime/data locks and raw artifacts. Independent score replay uses the same MAE implementation, whereas ranking and integer readout arithmetic are separately reconstructed; do not claim independent MAE implementation verification.

## Decision and costs

Use exact correct-count contrasts, support panels averaged within each seed, paired two-sided Student-t95% intervals across five seeds. Conditional on this reused single constructed stream, not independent-drive uncertainty.

Content-value qualification requires candidate-minus-uniform and candidate-minus-random balanced fine-tune lower bounds positive, mean each at least .1pp, and at least4/5 positive seeds for each. Candidate useful SSL requires candidate-minus-frozen lower bound positive. Quality requires candidate-minus-full lower bounds above -1pp for balanced/day/night. Efficiency requires useful SSL, quality and at least25% mean paired charged processing saving. Keep component screens distinct. Unqualified intervals do not establish a true loss larger than the margin.

Charge host-cache gathering, transfers, all-arrival live MAE scoring, RNG/mode management, selection/assembly and synchronized SSL. Rotate arm order by seed. Exclude acquisition, decoding/initial resize, checkpoint IO and readout; report these boundaries. The research runner preloads the full stream into host memory: FIFO decision causality does not imply whole-process64-frame storage. No model changes, health controller, extra seeds or outcome-driven tuning. Confirm any lead on separate source groups before steering it.

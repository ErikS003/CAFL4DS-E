# Frozen five-arm short-buffer development experiment

Frozen on 13 October 2026 before scientific outcomes. User authorized execution on the available GPU. This implements the bounded experiment in the SSL/tunable-filter forward plan; it does not alter completed verdicts.

## Question and allocation

Does short-term batch construction retain useful SSL adaptation at half the work, and does fixed-reference content selection improve on random buffered selection?

Use the first six clips from the outcome-independent `order_cohort` ordering of development cohorts 0 and 1 in the archived 20261011-development-v1 prepared allocation. Each bundle contains one clip per daytime/nighttime and scene-class cell. Keep 256 frames per clip, 1536 per bundle, artificial chronological concatenation, seeds 5003 and 5009, and support draws 51011/51017/51023. No outcome-based replacement or extra seeds. IDs, hashes and roles are locked before scoring.

Use the existing clean-warm TinyViT MAE checkpoint `mae_bdd_train20_s101.pt` with encoder width96/depth4, image64, patch8, normalized pixel reconstruction, masking .75, decoder64/depth2, AdamW LR .0001/weight decay .05, clipping1, batch32 and no schedule compensation. Preserve the light spatial augmentation. No persistent replay, health predictor or objective change.

| Arm | Arrival/block rule | Updates per episode |
|---|---|---:|
| Frozen | No online optimization | 0 |
| Full | Immediate 32-arrival batches, every frame once | 48 |
| Uniform half | Accumulate 64 arrivals, accept positions1,3,...63, one32-frame batch | 24 |
| Buffered random half | Accumulate128, select64 without replacement, sort chronologically, two32-frame batches | 24 |
| Buffered diverse half | Accumulate128, embed all128 using the frozen warm encoder; normalize features; greedy farthest-first Euclidean coverage, oldest seed and earliest ties; sort64 selected chronologically; two32-frame batches | 24 |

Clear every block after training. Random selection uses a private CPU generator seeded optimization-seed+62003, never learner RNG. No future frame beyond a completed block is inspected. Maximum waiting is127 intervening arrivals; clip timestamps determine the corresponding seconds. Every selected frame trains once. Full and uniform update timing intentionally differs from buffered timing; random versus uniform tests this mechanism, diverse versus random tests content at matched timing/work.

## GPU and evidence prerequisites

RTX2080, CUDA float32, AMP/TF32 off, deterministic algorithms on, synchronized GPU timers. A separate pinned project-local environment is recorded; preserve uv.lock. Do not pool CPU and GPU timing or claim historical bitwise training reproduction across runtime versions.

Before collection: inspect all72 historical paired parents,144 terminals and36 resumable histories; check saved fingerprints, finite model/Adam tensors, update/arrival clocks, exact load/snapshot round trips and no parent aliases. Record that this is tensor/restoration auditing, not reexecution of historical CPU trajectories. For the new GPU collection, require exact stochastic continuation after restoring nonempty Adam plus CPU/CUDA RNG; causal/exact-budget tests; and downstream evaluation isolation. A discarded preflight runs22 SSL updates and three fine-tuning fits, warms CUDA kernels, and records representative duration/memory. This work is separate from scientific allocation.

The original local test-label JSON files may be missing in this transferred workspace. Use archived selected reader attributes and saved arrays only after checking allocation identity, class/domain assignments, source-image hashes and exact resized-pixel equality. Record missing originals explicitly. Do not invent provenance, copy keyframe labels across video frames, or score sealed final queries. Unknown provider and capture sessions permit this reused-data development screen, not independent confirmation.

## Outcomes, cost and decision

One terminal evaluation per bundle/seed/arm:20 evaluations, three fixed75-label support panels each, exactly60 primary fine-tuning fits. Primary balanced scene accuracy after100 supervised support-only fine-tuning steps on an encoder copy: AdamW encoder1e-5/head1e-3, weight decay.05, batch32, clipping1, seed draw+25. Day/night reported separately. Linear/kNN are secondary. All9 prediction sets per evaluation are saved and independently reconstructed. Evaluate the complete episode target, not a next-window action forecast. Preserve model/optimizer/module modes and both CPU/CUDA RNG across readouts.

Charge host-resident RGB gathering, host-device transfers, selection and every reference-encoder pass on all arrivals, plus synchronized SSL time. Decoding/acquisition, checkpoint IO and downstream fitting are outside the online metric and disclosed separately. Never make cached features free. Rotate arm execution order by bundle/seed before outcomes. Report four paired timings, summed-time and mean-paired savings; this is a development timing estimate, not a production latency guarantee.

Freeze descriptive **pilot** screens: useful SSL mean gain versus frozen at least .1pp, positive means in both bundles, both optimization seeds and all three support panels; quality guardrail every bundle/seed/panel half-minus-full balanced/day/night greater than -1pp; charged aggregate saving at least25%. A budget-quality lead requires all three. Content-value pilot screen uses the same .1pp and sign-consistency rule versus buffered random. The .1pp is an exploratory materiality floor carried from earlier work;1pp is a practical pilot quality tolerance and25% an engineering savings target. These are not powered noninferiority/confidence gates. With only two reused bundles, do not construct independent-drive confidence intervals from seeds, panels or images.

480 scientific SSL updates. No per-window fits, added arms, tuned readers or post-outcome sample expansion. Report all outcomes, including negative/inconclusive ones. A lead advances only to a separately frozen source-group qualification. No useful SSL lead sends work back to the task/data/learning premise; buffering-only value does not justify a health model. Sealed queries remain untouched.

# Static memory admission under useful SSL

Frozen development protocol, 2026-10-10, before collecting filtered-model outcomes. The user authorized this study. It follows the completed recurrence qualification; it does not change that study's decisions.

## Question and comparisons

Does minimum-redundancy memory admission improve downstream learning versus Algorithm-R reservoir at a fixed image-memory capacity and optimizer-exposure budget? No health controller is fitted. The completed recurrence study showed useful fine-tuning gains versus frozen but no established primary forgetting impairment; this study therefore targets filter utility, not confirmed forgetting mitigation.

Use the exact clean-warm MAE, configuration, five seeds5003/5009/5011/5021/5023, data allocation, A1-B1-A2-B2 orders and shuffled controls of `outputs/adaptation-bdd/recurrence-retention/20261010-ssl-control-restart`. New outcomes: minimum-redundancy replay on both schedules, ten learned arms. Primary schedule: recurrence. Shuffled is secondary, with all results retained. Reservoir, current-only and frozen controls are reused only after input/artifact validation, exact frozen evaluation reproduction and a complete576-update recurrence/reservoir reproduction on seed5003. Failure stops collection; controls are not silently replaced or settings retuned. Reproduction adds576 actual SSL updates, recorded separately.

This is prospective filter development on reused images, evaluation pools and stream seeds, conditional on a common warm initialization. It is not independent population confirmation. Existing annotation ambiguity remains. Online training and selection use no scene/time task labels; attributes define benchmark allocation only.

## Fixed learner and causal memory policy

TinyViT-MAE64px, width96/depth4, full-backbone AdamW lr1e-4, weight decay.05, mask ratio.75, clip1;576 updates per arm. Memory capacity256 images. Replay16 historical images plus16 randomly selected current images after startup; first update uses32 current. All32 arrivals are offered to memory after replay sampling. Uniform retrieval is unchanged. No optimizer/memory reset at domain boundaries.

Reference representation: an independent, frozen eval-mode copy of the same warm encoder, raw96-dimensional `embed` output. Encode only the currently arrived batch before the SSL update. Normalize descriptors in float64, rejecting nonfinite or near-zero vectors. The reference is never updated and no future-arrival features are consulted. This isolates memory membership; it does not implement evolving-encoder MinRed or the fast model-to-filter feedback yet.

Offer arrivals sequentially. Append the candidate to the memory and, if capacity is exceeded, identify the maximum off-diagonal cosine similarity (the closest pair). Evict the newest member among all members whose nearest-neighbor similarity equals that maximum. Buffer order is chronological among retained arrivals. Similarities are rounded to12 decimal places before comparisons for reproducible numerical ties. The candidate may itself be rejected. This deterministic tie rule, geometry and capacity remain fixed.

Use the original private Torch generator sequence: replay randperm, current randperm, then one Algorithm-R randint for each arrival after capacity fills. Minimum redundancy consumes the same admission RNG draws but ignores their proposed slots. This preserves retrieval/current random-number scheduling versus the existing reservoir arm; memory membership causes the differences. Training RNG is independent and evaluation preserves it.

Trace all arrivals, selected-current/replay identities, each admission/eviction and final memory membership. Save raw reference descriptors in arrival order for independent membership reconstruction. Record reference encoding and admission time and persistent image/descriptor/similarity/reference-model bytes. Capacity and SSL exposures match, but end-to-end compute and total auxiliary storage are not assumed equal.

## Readouts and frozen development rule

Use the unchanged support/query allocation and readers. Primary: final equal-domain class-balanced accuracy after copied-model fine-tuning (25 labels/class, three support draws,100 support-only steps). Average draws within each seed. Evaluate quarter visits for linear/kNN and boundaries for fine-tuning. Report full domain trajectories, interference, recovery, CE where already available, geometry and costs. Geometry is not the success outcome.

Primary comparison: recurrence/minimum-redundancy minus recurrence/reservoir. A development memory-policy lead requires mean final gain >=1pp and positive lower bound of the paired95% Student-t interval across five seeds. The1pp reference is a declared practical development screen, not a powered guarantee or deployment threshold. Also require current-domain fine-tuning noninferiority: lower bound >-1pp at each of four visit endpoints (A,B,A,B), compared with reservoir, and positive final advantage versus warm frozen. Bounds are nominal and conditional on reused data; the conjunction is not advertised as simultaneous95% coverage.

Compare final minimum redundancy against current-only as well. Report separately whether its paired lower bound is positive, inconclusive or negative. A reservoir-relative lead without an advantage over current-only is a candidate memory mechanism, not justification that memory overhead is worthwhile. Current-only, reservoir and frozen all remain visible. No tuning of geometry, ties, C, label budget, seed subset or horizon after inspecting outcomes.

If static minimum redundancy consistently wins, report a useful static filter. State-dependent advantages are descriptive leads; they require a separate frozen action-response and controller study. If memory membership changes without downstream utility, stop this recipe and consider admission/update gating or a temporal-video environment. Do not infer health-controller utility from increased diversity alone.

## Work, audit and execution

New collection:5,760 SSL optimizer updates and184,320 optimizer image exposures. Compatibility reproduction:576 updates and18,432 exposures. Disposable finite-update preflight adds8 updates, discarded; a separate320-arrival selection-only prefix checks independent reconstruction past capacity without SSL optimization. Frozen/reference models perform no optimization. Copied-model fine-tuning and reference encoding are additional measured work. Persistent-byte counts describe tensor payloads, excluding Python/container overhead and research trace archives.

Before training, verify prior audit/input hashes, exact configuration/data/runtime compatibility, unchanged warm/reference state and causal replay. Snapshot sources and record input hashes. On completion, independently rescore saved predictions and paired inference, reconstruct membership from descriptors/arrival identities/private RNG, verify finite updates, exposures and unchanged inputs. Preserve all null/negative outcomes. Automatic reporting writes a study report after successful collection and audit.

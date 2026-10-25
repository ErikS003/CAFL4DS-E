# N78 — easier J admission under unchanged covariance protection

Queue one fresh_00 / seed-5003 trajectory after N77 completes successfully.
The user explicitly selected predicted delta J < 0, allowing more images than
N77's cutoff of -0.001152126437698479. This is an easier acceptance bar and a
higher numerical cutoff. Equality at zero is rejected.

Start from N77's identical original N73 complete 576-update parent, not its
endpoint. Preserve the same 57,600-image stream, seeds, predictor weights,
covariance/variance regularization, measured LR backtracking and 1.30-warm
hard cap. Change only the J admission threshold. No control sweep or new
accept-all trajectory. Neither development nor final outcomes set the cutoff.

The existing frozen J predictor is an upstream heuristic for the changed
regularized update. Easing its threshold need not prevent J-driven inactivity.
Covariance/variance regularization acts on the proposed gradient; actual LR
reduction occurs only after a measured covariance-cap violation. Every image
begins its trial at nominal LR. Accepted smaller steps retain real moments.

Freeze N78 configuration, runner, reporter and inherited inputs while queued.
The worker uses no model setup or GPU work before N77 finishes. Require N77's
complete trajectory, passing branch verification and unchanged input hashes.
If N77 fails or is stopped, do not start N78 automatically. A cancel-queue.json
file cancels a waiting job. Record the immutable completed N77 reference hashes
before the new run. Queue acquisition is exclusive; stale leases fail closed.

Reuse N77's implementation with a fresh threshold-specific 16-arrival exact
resume preflight. Show notebook 78 with the same seven plot groups and N77 as
the direct matched reference, clipped to shared measured arrival horizons.
Publish metrics every 384 arrivals and status every 32. Keep the two-GiB free
storage reserve, complete rejected-trial rollback and resumable checkpoints.

## Authorized queue amendment, 2026-10-24

The user explicitly requested cancelling N77 and running N78 plus the new
smaller-rate J rescue experiment. This overrides the original wait for full
N77 completion. Retire the old waiting worker, preserve its protocol/input
lock, and release N78 only after N77 publishes a complete stopped checkpoint
and exits. Freeze the stopped checkpoint hashes in authorized-release.json.
The image threshold, model update rule, parent, stream and measurements do not
change. This exception does not automatically apply to any other failed run.

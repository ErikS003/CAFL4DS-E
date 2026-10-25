# N76 — modestly relaxed covariance ceiling

Run exactly one fresh_00 / seed-5003 trajectory from the same complete
576-update N73 parent, over the same 57,600 continuation images. Stop N75
at the user's request and preserve its last committed state. No new accept-all
or fixed-rate reference trajectory is run.

The only scientific change from N75 is an absolute covariance ceiling of
1.30 times the original warm selector-bank covariance, replacing 1.25 times.
Use the identical corrected J predictor and threshold without refitting.
Reuse N75's exact ordered AdamW LR trials at nominal, quarter, sixteenth and
sixty-fourth rate, the same image/view seeds, complete rejected-trial rollback,
nominal-rate reset after acceptance and checkpoint cadence of 384 arrivals.
Neither final-query nor development-probe outcomes control admission or LR.
This remains a hard ceiling; it may merely postpone the stall. The value
1.30 was chosen exploratorily after inspecting N75 and is not independently
calibrated. Fixed-path trial feasibility does not predict a new trajectory.

Reuse N75's GPU preflight: original 64-image prefix reproduction, exact
interrupted resume and a real smaller-rate rescue from the earlier stalled
N74 state. Validate ordered decisions and live optimizer-step accounting.
Handle a transient Windows publication failure only after verifying the
fully saved pending state's hashes and fingerprint; resume complete state.
This I/O recovery does not change scientific decisions.

Plot final-query balanced/day/night linear accuracy, running drawdown,
development accuracy and soft error, covariance with both ceilings, variance,
RankMe, J, warm-relative drift, fixed selector/held-out reconstruction and
actual accepted-image training reconstruction. Also plot admissions,
retained/rejected trials, chosen LR and exposure. Empty training intervals
are missing observations, never zero loss. Reference policies are clipped
to the new run's current arrival horizon. Reference final-query values are
descriptive and never control the new trajectory.

Use existing saved assays for reporting without extra model evaluations.
Record the new config, protocol, runner and reporter in the source lock;
retain all inherited frozen hashes. Reserve two GiB of free storage and
keep the preceding complete checkpoint generation. The stopped N75
checkpoint is an immutable reference, not the new trajectory's parent.

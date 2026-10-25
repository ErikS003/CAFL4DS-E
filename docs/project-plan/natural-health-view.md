# Focused full-training versus quarter-recovery readout

The user requests only the two trajectories, all five established fixed-bank
health signals, kNN and a frozen-encoder linear probe. This is a descriptive
readout extension of the completed natural-order experiment, not a new policy
comparison or threshold search. Preserve the original scientific artifacts,
notebook and delivery hashes. Write a new focused notebook and output directory.

Replay only full training and quarter recovery from their recorded initial
state and exact accepted masks. Verify each accepted raw-input hash, SSL loss
and gradient norm; monitor embeddings; complete learner fingerprints at saved
observations; and both final states. New evaluation must preserve all module
modes, optimizer state and CPU/CUDA RNG. Never optimize on labels during replay.
Do not refit the previously recorded fine-tuning reader.

At the original thirty-five capability checkpoints, extract frozen unaugmented
features for the existing evaluation image pool. Reuse only identical encoder
fingerprints. Keep the three original support draws, each fifty scene labels
(twenty-five per class), and the same disjoint 3,663 primary queries. Use the
existing checkpoint-robustness recipes: cosine-distance, distance-weighted kNN
with k=20; and logistic linear classification, support-only coordinate
standardization, C=1, max_iter=1000. No tuning, query-based normalization, new
labels, additional support draws or expanded query pool. Treat convergence
warnings as errors rather than silently changing the recipe.

Plot balanced/day/night class-balanced accuracies for kNN, linear and the
previous fine-tuning readout. Show all three support draws as thin lines and
their mean as a thick line. Their range is reader sensitivity, not a source
confidence interval. These remain one ordered stream and one SSL seed.

Compute raw RankMe, mean row-normalized feature variance, normalized
off-diagonal covariance, CKA distance and paired cosine drift from the original
256-image monitor embeddings. CKA and cosine use the unchanged step-132
reference. Before that point show the other three geometry levels only; do not
imply a reference was available online earlier. Display the exact alarm-driving
relative rank/variance declines and CKA distance too, distinguishing the two
supporting metrics (covariance and cosine) from the three actual trigger inputs.
Show admission, cumulative accepted batches, SSL loss and pre-clipping gradient
norm on the same arriving-batch clock. Loss/gradient smoothing uses the previous
thirty-two accepted updates, retains gaps for rejected arrivals, and is labelled
as training-diet-dependent rather than a matched held-out quality measure.

Before new readouts, freeze this plan, configuration, implementation, tests and
all consumed input hashes. Record every replay update and head/neighbour fit,
including verification refits. The cap is 6,025 replay SSL updates, at most 210
linear and 210 kNN fits for collection, and the same head-fit cap for verification.
Independently reconstruct scores from saved predictions, recompute geometry
with the existing float32 instruments, and refit both readouts on every unique
saved feature state. Render, inspect and embed all figures; verify embedded
bytes and local links. Scientific failures preserve their artifacts and journal.

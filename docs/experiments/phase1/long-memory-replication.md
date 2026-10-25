# Long-horizon historical memory: source replication

The fixed reservoir passes all six effect requirements. The full frozen development screen remains inconclusive because day-domain uncertainty exceeds the one-percentage-point precision target.

| Reader | Reservoir − current: balanced/day/night (pp) | Reservoir − FIFO: balanced/day/night (pp) |
|---|---|---|
| finetune | +5.473 / +6.627 / +4.318 | +5.132 / +6.002 / +4.261 |
| knn | +2.794 / +1.954 / +3.634 | +2.616 / +1.826 / +3.407 |
| linear | +1.372 / +1.584 / +1.159 | +1.184 / +1.561 / +0.807 |

| Source | Reservoir − current: balanced/day/night (pp) | Reservoir − FIFO: balanced/day/night (pp) |
|---|---|---|
| 1 | +5.550 / +7.649 / +3.452 | +4.838 / +6.849 / +2.827 |
| 2 | +5.643 / +6.188 / +5.098 | +5.249 / +4.186 / +6.313 |
| 3 | +5.225 / +6.046 / +4.404 | +5.308 / +6.973 / +3.643 |

| Comparator | Conditional balanced / day / night 95% ranges (pp) |
|---|---|
| current | [+4.589, +6.378] / [+5.091, +8.255] / [+3.367, +5.250] |
| fifo | [+4.291, +6.012] / [+4.555, +7.559] / [+3.330, +5.150] |
| warm | [+4.652, +6.461] / [+3.255, +5.987] / [+5.313, +7.642] |

These ranges resample the paired reader draws and query images. They describe conditional evaluation sensitivity, not confidence over new capture sessions.

| Development requirement | Decision |
|---|---|
| balanced gain at least one pp | Pass |
| positive each source | Pass |
| two positive training draws per source | Pass |
| positive vs fifo each source | Pass |
| each source domain budget | Pass |
| positive mean gain over warm | Pass |

Conditional one-point precision: False.

Three new clip-disjoint 131,072-frame blocks, three training draws each, and three fixed policies.
Primary terminal scores use 256 paired reader draws; intermediate curves use three. The warm reference is shared.
Source blocks and the evaluation population do not establish independent capture-session/population confirmation.
Fine-tuning measures adaptability after a fixed labeled budget. Frozen readers answer a separate question.
Replay can include observed frames not previously selected for training. Gains may reflect added coverage as well as retention; age, source coverage and redundancy change together.
Health never chooses an action. No health controller is qualified. Path metrics are descriptive on a sparse fixed grid.

Scientific SSL: 110,592 collection plus 110,592 exact reproduction updates. Independent source/diet/reader audits pass.
All 9,679 kNN and linear fits are independently refitted; 28 fixed fine-tuning fits reproduce exactly.
Engineering checks add 64 discarded SSL updates and six fine-tuning fits, recorded separately from the scientific comparison.

[Frozen methods](../../project-plan/long-memory-replication-frozen.md) · [Full report](../../../outputs/long-memory-replication/20261018/report.md) · [Notebook](../../../notebooks/sums/35_long_memory_replication.ipynb)


The model weights are warm, but the prescribed saved M0 checkpoint has an empty AdamW state. The earlier wording about retained warm optimizer moments was inaccurate; all arms restored the exact same checkpoint. This clarification changes no scientific input or result.

Disk exhaustion interrupted the original four-worker evaluation after 649 panels were committed. All committed panels were preserved and missing panels resumed with the unchanged recipes in sequential processes. There are 9,679 unique panels and 9,683 physical collection fits per reader kind, including four fits whose results could not be saved. Failed writes and original counters are preserved and independently reconciled. SSL training was not repeated. The original frozen-reader audit used eight BLAS threads instead of the four used in collection. A one-query kNN difference reproduced with eight threads and matched exactly with four. The full exact audit was rerun with the matching thread context; original failures and diagnostic refits are accounted separately.

[Initialization clarification](../../project-plan/long-memory-initialization-clarification.md) · [Reader recovery accounting](../../project-plan/long-memory-reader-storage-recovery.md) · [Audit thread correction](../../project-plan/long-memory-audit-thread-clarification.md)


Interpretation: the reservoir improves final balanced fine-tuning accuracy by
5.473 pp versus current-only and 5.132 pp versus FIFO. Its source means versus
current-only are +5.550, +5.643 and +5.225 pp; all nine training-draw balanced
means are positive. kNN and linear probing also favor reservoir over both trained
controls, although balanced linear-probe accuracy remains below the warm reference.

The conditional interval half-widths versus current-only are 0.894 / 1.582 / 0.942
pp for balanced/day/night; versus FIFO they are 0.860 / 1.502 / 0.910 pp. All
reported endpoint intervals are positive, but the day half-widths exceed the
prespecified precision requirement. This is a precision failure, not a failed
point-effect requirement. The original decision and draw budget remain unchanged.

Positive means do not guarantee benefit for every support/fitting draw. Of 2,304
paired source/training/support comparisons per domain, reservoir is lower than
current-only in 306 day comparisons and 354 night comparisons; the worst individual
domain difference is -13.331 pp. These paired fitting outcomes are dependent and
are not independent source replications.

Fixed-anchor retention is descriptive: mean worst-domain shortfall below warm
is 0.136 pp for reservoir, 0.439 for current-only and 0.483 for FIFO on the
three-reader-draw grid. Source 3 has a larger worst-domain peak-relative drawdown
with reservoir than with either control. No uniform protection claim is qualified.

The next research priority is to separate historical age, replay repetition and
access to observed frames that were not previously trained, while holding work,
current subsets and replay dose fixed. The saved diets show that reservoir revisits
a much smaller set of identities than FIFO, despite reaching further into the
past. This experiment cannot identify which of those changes causes the gain.
Resolve domain precision in a fresh design and confirm on fresh evaluation data
and additional warm initializations before attributing value to health alarms.
No follow-on experiment was launched.

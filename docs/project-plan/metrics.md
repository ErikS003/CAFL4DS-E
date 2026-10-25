‹ [Project Plan index](index.md)

# Metrics — signal analytics, health, control, evaluation

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution (see
[index](index.md#novelty-at-a-glance)). The `N-x` novelty claims are stated in full in [novelty.md](novelty.md);
baselines (B5, B-open, …) in [experiments.md](experiments.md#baselines).*

## System / project context

**Variables:** the selection policy + steerable filter F-c are the *independent* side (what we turn); the **health
metrics** are the *dependent* readout (the thermometer). The **controller** (monitor→filter) is **not** a readout — once
the loop is closed it is part of the *intervention* (independent) side.

**We probe freely from the start.** The goal is to study the dynamics, not to ship a real-time edge pipeline — so labels
are used wherever they sharpen a metric, from day one. The "label-free vs. needs-probe" tag below is a *recorded
property* (it bears on eventual deployment), not a constraint on the study. Read signals **jointly** (multivariate
change-point / covariance tracking), not per-channel — single-channel thresholds miss correlation flips and subspace
rotations.

## Diagnostic vs. validating axes

Keep the two axes separate, and **establish the link between them** — that link is part of the contribution.

- **Diagnostic (explanatory).** The health signals, read primarily as the **divergence between the SSL loss and the
    health metrics** (loss flat or improving while effective rank / uniformity / drift degrade — the silent-degradation
    signature). Explains *why* a config degrades. *Metrics are `[EXT]`* (RankMe, VICReg variance/covariance, CKA drift);
    *using them as a live explanatory axis coupled to selection is `[NEW]`.* Necessary but not sufficient: a model can
    forget or overspecialize **without** collapse, so health is **multivariate** (rank + drift + forgetting), never
    collapse alone. Trustworthy only after the positive control (PC) has validated the instruments.

- **Validating (payoff) `[STD]`.** Downstream quality (frozen linear probe + kNN, vs. **B5**) and forgetting (per-era
    probe → **Backward Transfer** / **Forgetting Measure**). A config "wins" only if a diagnostic advantage **translates
    here**. For N-E/N-G: closed-loop must beat **B-open** on this axis.

> **The link `[NEW]` (leading-indicator test).** Does the diagnostic axis predict the validating axis with usable lead
> time — a rank drop or rising drift *before* forgetting shows in the probe? This is what licenses a label-free signal
> as a control input at all. Cheap to run early, since both axes are already logged — *except* the **offline audits**
> below (kNN cross-probe, recoverability, relearning-savings), the most expensive rung of the validating axis: they need
> extra data + retraining, so a cheap online signal (recon retention gap, drift) earns its keep precisely by
> *predicting* one of those offline verdicts without paying for it.

### Signal catalogue

*Each tagged with `[architecture]` it applies to and whether it needs a probe (labels)*

**Geometry / collapse (label-free):**

- **Effective rank / RankMe** — # embedding dims actually in use; the core collapse readout. Necessary-but-not-
    sufficient for downstream quality (Garrido et al. 2023). `[general]` (Roy & Vetterli 2007; Garrido et al. 2023)
- **Intrinsic dimension** — a geometry-based label-free proxy that, in a 260-model comparison, tracks downstream
    performance **more reliably** than spectral rank metrics (still moderated by architecture/objective); a candidate
    corroboration / monitor signal alongside RankMe, not a replacement. `[general]` (IdEst — Mordacq et al., ICML 2026;
    260-model comparison — Arputharaj et al., TMLR 2026)
- **Per-dimension variance** — flags dimensions collapsing to a constant. `[general]` (VICReg, Bardes et al. 2022)
- **Off-diagonal covariance / decorrelation** — flags informational (redundancy) collapse. `[general]` (VICReg, Bardes
    et al. 2022)
- **Alignment & uniformity** — positive-pair closeness + spread on the hypersphere; uniformity drops as embeddings
    clump. `[joint-embedding]` (uniformity general; alignment needs pairs) (Wang & Isola 2020)
- **Prototype-assignment entropy / usage** — how many prototypes are actually used (collapse = a few dominate); the
    specified version of "entropy." `[prototype: DINO/SwAV]` (Caron et al. 2020, 2021)

**Dynamics / stability (label-free):**

- **Representation drift** — CKA / cosine churn of a *fixed* probe set's embeddings across checkpoints; how fast the
    coordinate frame moves (ties to forgetting + the moving-reference problem). Read with two caveats: **benign vs.
    harmful** drift is distinguished by *non-mean-reverting change without performance loss* (benign) vs. change *with*
    info loss (forgetting) — test via mean-reversion + probe-recovery; and **CKA is not fully reliable** (sensitive to
    outliers / function-preserving shifts, biased in high-dim/low-sample), so use the **debiased** estimator + a
    complementary metric (Procrustes/CCA). `[general]` (CKA, Kornblith et al. 2019; reliability — Davari et al. 2023;
    drift formalism — representational-drift line)
- **MAE reconstruction retention gap** — held-out *past-era* reconstruction loss, read as a positive-control-vs-replay
    *retention* gap; the MAE-native, label-free forgetting corroborator (recon on the era being overwritten degrades
    relative to a replay control). **Caveat (P0.3.8):** over-fires on a low-level *output*-distribution shift the
    representation shrugs off (e.g. colour→grayscale — the decoder's output moves but the encoder does not), so it is
    trustworthy only when the new era is not itself an output shift versus the scored era. `[MAE]` (reconstruction
    objective, He et al. 2022)
- **SSL loss trajectory** — necessary but weak alone; for joint-embedding a *low* loss can be a collapsed trivial
    solution, so read it *against* the collapse signals. `[general]` (collapse caveat: Jing et al. 2022)
- **Gradient norm / gradient diversity** — instability/divergence (norm) and training-diet diversity (diversity); not a
    collapse signal. `[general]` (gradient diversity: GSS, Aljundi et al. 2019)

**Downstream / forgetting (needs a labels / probe):**

- **Linear probe + kNN accuracy, current data** — is the representation useful now (vs. B5). `[general]` (kNN protocol:
    Wu et al. 2018; Caron et al. 2021)
- **Per-era / past-class accuracy → forgetting** — the direct forgetting measure: per-era probe accuracy, **Backward
    Transfer** (Lopez-Paz & Ranzato 2017), and the **Forgetting Measure** (Chaudhry et al. 2018). The ground truth for
    MAE's degradation mode. `[general]`

**Offline audits — genuineness / permanence (offline, need held-out data + a retraining budget; run *once* after a fire,
not live readouts):**

Unlike every signal above — cheap, continuous, mostly label-free readouts sampled each era — these are one-shot audits
that *qualify a fired forgetting event*, trading extra data and a retraining budget for a direct read of the ground
truth. They are the most expensive rung of the validating axis, and (per the leading-indicator link above) the verdicts
the cheap online signals aim to *predict*:

- **kNN cross-probe** — genuine representational degradation vs. a linear-probe *accessibility* artifact (a rotation
    into a linearly-inaccessible basis): if kNN craters too, the loss is real, not a probe artifact. `[general]`, needs
    labels. (P0.3.5)
- **Recoverability re-learn** — permanence vs. transient interference: re-learn the forgotten era and measure how far
    accuracy returns toward its pre-crater level (R00). `[general]`, needs labels + a relearn budget. (P0.3.5)
- **Relearning savings (savings fraction)** — genuine loss vs. apparent/recoverable interference: relearn the era from
    the *forgotten* init and from a *from-scratch* init; genuine forgetting ⇒ relearns as slowly as scratch, apparent ⇒
    as fast as a retained model. `[MAE / general]`, needs data + retraining. (P0.3.7/P0.3.8)

**Architecture coverage:** MAE exposes loss, the geometry suite, drift, and its own reconstruction retention gap
(collapse signals sparse → lean on the forgetting metrics + the recon gap); joint-embedding adds alignment/uniformity;
prototype methods add assignment entropy. This is why the plan demonstrates collapse on joint-embedding and forgetting
on MAE — and why each mode gets its **own** positive control in Phase 0 to calibrate its metrics (collapse-PC → the
geometry suite; a deliberately-forgetting run → drift + the forgetting metrics), a metric being trustworthy only once it
fires on its mode and stays quiet when healthy. See [Phase 0](../experiments/phase0/index.md).

### Two questions the labels-allowed freedom lets us separate

1. **Does closing the loop help, given a good signal?** Drive the controller with the best available signal — *including
    a probe-based oracle* (held-out labeled forgetting) — to test the control idea unhandicapped.

1. **Can a label-free signal detect degradation in time to drive the loop?** The leading-indicator / deployability
    question: does a Tier-free signal (rank drop, drift) *predict* the probe-based forgetting before the labels would
    reveal it, with usable lead time?

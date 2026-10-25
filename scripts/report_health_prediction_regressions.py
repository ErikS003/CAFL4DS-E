"""Rebuild a visual notebook of previous health regressions without fitting models."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from cafl4ds.artifacts import atomic_json
from scripts.report_prediction_pilots import figure_cell, markdown
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research-summary/20261014-health-prediction-regressions"
NOTEBOOK = ROOT / "notebooks/sums/09_health_prediction_regressions.ipynb"
TIME = ROOT / "outputs/prediction-health-timeseries/20261014-pilot"
IMAGE = ROOT / "outputs/prediction-image-health/20261014-pilot-device-repair"
DERIVATIVE = ROOT / "outputs/prediction-health-derivatives/20261014-extension"
METRICS = ("variance", "covariance", "rank", "cka", "cosine")
NAMES = ("Feature variance", "Feature covariance", "RankMe", "CKA drift", "Cosine drift")
MODELS = ("persistence", "local_trend", "ar1", "ar2", "delta_ar2", "varx")
MODEL_LABELS = ("Persist.\nzero change", "Recent\ntrend", "AR1", "AR2\nprimary", "Change AR2\nsecondary", "VARX")
BLUE, ORANGE, DARK, GRAY = "#1677a5", "#d47b32", "#183746", "#8a969e"
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(a, b):
    if not np.allclose(a, b, atol=1e-12, rtol=1e-10):
        raise ValueError("Saved health regression evidence does not reconstruct")


def features(record, metric, origin, model):
    """Reconstruct only already-observed inputs; no scientific source import."""
    h = np.array(record["health"])
    if model == "ar1":
        return h[origin, [metric]]
    if model == "ar2":
        return h[[origin, origin - 1], metric]
    if model == "delta_ar2":
        return np.array([h[origin, metric] - h[origin - 1, metric], h[origin - 1, metric] - h[origin - 2, metric]])
    if model == "varx":
        return np.concatenate(
            (
                h[origin],
                h[origin] - h[origin - 1],
                [
                    np.mean(record["loss"][origin - 4 : origin]),
                    np.mean(record["gradient_norm"][origin - 4 : origin]),
                    record["parent_updates"] + origin,
                ],
            )
        )
    raise ValueError(model)


def evidence():
    """Bind saved sources to prior report hashes, then verify targets and forecasts."""
    hashes = {}
    for directory, old_report in (
        (TIME, "20261014-health-timeseries"),
        (IMAGE, "20261014-prediction-pilots"),
        (DERIVATIVE, "20261014-health-derivatives"),
    ):
        previous = ROOT / "outputs/research-summary" / old_report / "provenance.json"
        hashes[str(previous)] = digest(previous)
        provenance = read(previous)["sources"]
        for name in ("summary.json", "verification.json", "independent-verification.json"):
            path = directory / name
            if str(path) not in provenance or digest(path) != provenance[str(path)]:
                raise ValueError("Previously published evidence changed")
            hashes[str(path)] = digest(path)
        if not all(read(directory / name)["passed"] for name in ("verification.json", "independent-verification.json")):
            raise ValueError("Original audits required")
    series_path = TIME / "series.json"
    if (
        digest(series_path)
        != read(ROOT / "outputs/research-summary/20261014-health-timeseries/provenance.json")["sources"][
            str(series_path)
        ]
    ):
        raise ValueError("Dense trajectories changed")
    hashes[str(series_path)] = digest(series_path)
    summary, series = read(TIME / "summary.json"), read(series_path)["records"]
    targets, predictions, scores = 0, 0, 0
    for horizon in (1, 4):
        for k, metric in enumerate(METRICS):
            result = summary["forecasts"][str(horizon)][metric]
            rows = result["test_rows"]
            y = np.array([r["target"] for r in rows])
            groups = np.array([r["stream"] for r in rows])
            for row in rows:
                h = np.array(series[row["series"]]["health"])
                if not 20 <= row["origin"] <= 28 or row["target_update"] != row["origin"] + horizon:
                    raise ValueError("Forecast origin mismatch")
                check(h[row["target_update"], k] - h[row["origin"], k], row["target"])
                targets += 1
            for model in MODELS:
                saved = result["models"][model]
                rebuilt = []
                for row in rows:
                    record, t = series[row["series"]], row["origin"]
                    if model == "persistence":
                        estimate = 0.0
                    elif model == "local_trend":
                        estimate = horizon * (record["health"][t][k] - record["health"][t - 4][k]) / 4
                    else:
                        fold = result["folds"][model][row["stream"]]
                        x = features(record, k, t, model)
                        estimate = fold["intercept"] + ((x - fold["center"]) / fold["scale"]) @ np.array(fold["beta"])
                    rebuilt.append(estimate)
                check(rebuilt, saved["predictions"])
                predictions += len(rebuilt)
                residual = y - np.array(rebuilt)
                source_mse = [float(np.mean(residual[groups == g] ** 2)) for g in sorted(set(groups))]
                check(source_mse, [saved["source_mse"][g] for g in sorted(set(groups))])
                check(np.mean(source_mse), saved["mse"])
                check(
                    100 * (1 - saved["mse"] / result["models"]["persistence"]["mse"]),
                    saved["skill_vs_persistence_percent"],
                )
                scores += 1
            for model in ("ar1", "ar2", "delta_ar2", "varx"):
                for excluded, fold in result["folds"][model].items():
                    train = [r for r in result["training_rows"] if r["stream"] != excluded]
                    if len(train) != fold["fit_rows"] or any(
                        not 4 <= r["origin"] <= 16 or r["target_update"] > 20 for r in train
                    ):
                        raise ValueError("Source/time training isolation mismatch")
                    design = np.array([features(series[r["series"]], k, r["origin"], model) for r in train])
                    scale = design.std(0)
                    scale[scale < 1e-12] = 1
                    check(design.mean(0), fold["center"])
                    check(scale, fold["scale"])
    derivative = read(DERIVATIVE / "summary.json")
    for metric in METRICS:
        data = derivative["interventions"]["marginal"]["metrics"][metric]
        group = np.array(derivative["interventions"]["marginal"]["groups"])
        for model in data["models"].values():
            residual = np.array(data["targets"]) - model["predictions"]
            mse = np.mean([np.mean(residual[group == g] ** 2) for g in sorted(set(group))])
            check(mse, model["mse"])
            check(100 * (1 - mse / data["models"]["zero"]["mse"]), model["skill_vs_zero_percent"])
    return (
        summary,
        series,
        read(IMAGE / "summary.json"),
        derivative,
        hashes,
        {
            "passed": True,
            "time_series_targets": targets,
            "saved_forecast_predictions": predictions,
            "model_error_tables": scores,
            "scalers_and_source_time_isolation": True,
            "image_health_error_tables": 25,
            "new_ssl_updates": 0,
            "new_model_fits": 0,
        },
    )


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=120, bbox_inches="tight", facecolor="white", pil_kwargs={"optimize": True})
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heat(ax, values, xlabels, ylabels, title, limits=(-100, 100), percent=True):
    artist = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=limits[0], vmax=limits[1])
    ax.set(
        xticks=range(len(xlabels)), xticklabels=xlabels, yticks=range(len(ylabels)), yticklabels=ylabels, title=title
    )
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            label = f"{value:.1f}%" if percent else f"{value:.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.65 * limits[1] else DARK,
            )
    return artist


def panel_grid():
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), layout="constrained")
    axes.flat[-1].axis("off")
    return fig, axes.flat


def diagrams():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), layout="constrained", height_ratios=(1.3, 1))
    ax = axes[0]
    ax.axis("off")
    cards = [
        (0.01, "OBSERVE THROUGH t", "$H_t, H_{t-1}, H_{t-2}$\nLast two health changes", "#edf4f7"),
        (0.355, "SAVED LINEAR REGRESSION", "Fixed ridge penalty = 10\nScalers fitted on other histories", "#e8f2f7"),
        (0.70, "PREDICT", "$\\widehat{H_{t+4}-H_t}$\nFour more accepted updates", "#fff0e4"),
    ]
    for x, title, body, color in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.24), 0.285, 0.6, boxstyle="round,pad=0.014", fc=color, ec="#cbd7df", transform=ax.transAxes
            )
        )
        ax.text(x + 0.143, 0.69, title, ha="center", transform=ax.transAxes, fontsize=10, color=DARK, weight="bold")
        ax.text(x + 0.143, 0.45, body, ha="center", va="center", transform=ax.transAxes, fontsize=12, linespacing=1.5)
    for x in (0.308, 0.653):
        ax.annotate(
            "",
            xy=(x + 0.038, 0.54),
            xytext=(x, 0.54),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2},
        )
    ax.text(
        0.5,
        0.04,
        "Target: a health change under continued training. No image utility or accuracy label is implied.",
        ha="center",
        transform=ax.transAxes,
        fontsize=11,
        color=DARK,
    )
    ax = axes[1]
    ax.broken_barh([(4, 12)], (0.65, 0.42), facecolors=BLUE)
    ax.broken_barh([(4, 16)], (0.18, 0.15), facecolors="#abcbdc")
    ax.broken_barh([(20, 8)], (-0.5, 0.42), facecolors=ORANGE)
    ax.broken_barh([(24, 8)], (-0.94, 0.15), facecolors="#f3cbab")
    ax.set(
        xlim=(0, 33),
        ylim=(-1.1, 1.25),
        yticks=(0.86, -0.29),
        yticklabels=("Other 5 histories: fit", "Held history: score"),
        xticks=(0, 4, 16, 20, 24, 28, 32),
        xlabel="Additional accepted optimizer updates",
    )
    ax.text(10, 0.86, "Origins 4–16", ha="center", va="center", color="white", weight="bold")
    ax.text(24, -0.29, "Origins 20–28", ha="center", va="center", color="white", weight="bold")
    ax.text(12, 0.255, "Training outcomes finish by 20", ha="center", va="center", fontsize=9)
    ax.text(28, -0.865, "Targets 24–32", ha="center", va="center", fontsize=9)
    fig.suptitle("What was predicted, and how was leakage avoided?", fontsize=17, color=DARK)
    save(fig, "01-forecast-contract")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, series, image, derivative, hashes, verification = evidence()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "figure.titlesize": 16,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "grid.alpha": 0.16,
        }
    )
    groups = sorted({r["stream"] for r in series})
    diagrams()
    fig, axes = panel_grid()
    for k, (metric, name, ax) in enumerate(zip(METRICS, NAMES, axes, strict=False)):
        for g, group in enumerate(groups):
            curves = []
            for row in series:
                if row["stream"] != group:
                    continue
                h = np.array(row["health"])[:, k]
                curve = 100 * (h / h[0] - 1) if k < 3 else 1000 * (h - h[0])
                curves.append(curve)
                ax.plot(range(33), curve, color=PALETTE[g], alpha=0.16, lw=0.8)
            ax.plot(range(33), np.mean(curves, axis=0), color=PALETTE[g], lw=1.8, label=group)
        ax.axhline(0, color=GRAY, lw=0.7)
        ax.set(
            title=name, xlabel="Accepted updates", ylabel="Change from parent (%)" if k < 3 else "Drift change × 1,000"
        )
    axes[5].legend(*axes[0].get_legend_handles_labels(), loc="upper left", title="Held-history groups", frameon=False)
    axes[5].text(
        0.02,
        0.35,
        "18 short continuations\n3 parent checkpoints per history\n33 readings on a fixed 256-image bank\n\nThin lines: individual trajectories\nThick lines: within-history mean",
        transform=axes[5].transAxes,
        va="top",
        linespacing=1.5,
    )
    fig.suptitle("The measured health trajectories", color=DARK)
    save(fig, "02-health-trajectories")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for ax, horizon, title in zip(
        axes, (4, 1), ("Four updates ahead — primary horizon", "One update ahead — secondary horizon"), strict=True
    ):
        values = [
            [s["forecasts"][str(horizon)][m]["models"][model]["skill_vs_persistence_percent"] for model in MODELS]
            for m in METRICS
        ]
        artist = heat(ax, values, MODEL_LABELS, NAMES, title)
        ax.set_xlabel("Frozen model menu")
        ax.tick_params(axis="x", labelsize=9)
    fig.colorbar(artist, ax=axes, shrink=0.8, label="MSE reduction vs persistence (%)")
    save(fig, "03-regression-scorecard")

    fig, axes = panel_grid()
    for metric, name, ax in zip(METRICS, NAMES, axes, strict=False):
        f = s["forecasts"]["4"][metric]
        target = np.array([r["target"] for r in f["test_rows"]])
        predicted = np.array(f["models"]["delta_ar2"]["predictions"])
        source = np.array([r["stream"] for r in f["test_rows"]])
        scale = f["models"]["persistence"]["rmse"]
        for group, color in zip(groups, PALETTE, strict=True):
            mask = source == group
            ax.scatter(target[mask] / scale, predicted[mask] / scale, color=color, s=18, alpha=0.65, edgecolors="none")
        lo = min(target.min(), predicted.min()) / scale
        hi = max(target.max(), predicted.max()) / scale
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=GRAY, lw=1)
        ax.axhline(0, color=GRAY, lw=0.5)
        ax.axvline(0, color=GRAY, lw=0.5)
        ax.set(
            title=f"{name} · skill {f['models']['delta_ar2']['skill_vs_persistence_percent']:.1f}%",
            xlabel="Measured change / persistence RMSE",
            ylabel="Predicted change / persistence RMSE",
            xlim=(lo - pad, hi + pad),
            ylim=(lo - pad, hi + pad),
            aspect="equal",
        )
    axes[5].text(
        0.02,
        0.92,
        "Every point is a saved out-of-history\nforecast from the secondary change model.\n\n162 overlapping forecasts per metric\n6 histories, not 162 independent sources\n\nDashed diagonal: perfect forecast\nColours match trajectory histories\n\nPlot scale uses each metric's persistence\nRMSE; original-unit errors are tabulated\nin the notebook.",
        transform=axes[5].transAxes,
        va="top",
        linespacing=1.5,
    )
    fig.suptitle("Predicted versus measured health changes", color=DARK)
    save(fig, "04-predicted-versus-measured")

    fig, axes = panel_grid()
    # Display the first saved series, chosen by index rather than forecast accuracy.
    row = series[0]
    for k, (metric, name, ax) in enumerate(zip(METRICS, NAMES, axes, strict=False)):
        f = s["forecasts"]["4"][metric]
        indices = [i for i, r in enumerate(f["test_rows"]) if r["series"] == 0]
        origins = np.array([f["test_rows"][i]["origin"] for i in indices])
        times = origins + 4
        h = np.array(row["health"])[:, k]
        ax.plot(range(16, 33), h[16:], color=DARK, lw=2.2, label="Measured health")
        for model, color, style, label in (
            ("persistence", GRAY, ":", "Persistence"),
            ("ar2", ORANGE, "--", "Level AR2 (primary)"),
            ("delta_ar2", BLUE, "-", "Change AR2 (secondary)"),
        ):
            forecasts = h[origins] + np.array(f["models"][model]["predictions"])[indices]
            ax.plot(times, forecasts, color=color, ls=style, marker="o", ms=3, lw=1.6, label=label)
        ax.set(title=name, xlabel="Target update", ylabel="Health level", xticks=(16, 20, 24, 28, 32))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 3), useMathText=True)
    axes[5].legend(*axes[0].get_legend_handles_labels(), loc="upper left", frameon=False)
    axes[5].text(
        0.02,
        0.46,
        f"Fixed illustration: {row['stream']}\nParent: {row['checkpoint']}\n\nEach marker is a new four-update forecast\nfrom a later observed origin.\nThis is not a single open-loop rollout.\n\nAll-history results appear separately.",
        transform=axes[5].transAxes,
        va="top",
        linespacing=1.5,
    )
    fig.suptitle("What does the regression look like on one continuation?", color=DARK)
    save(fig, "05-one-continuation")

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    values = [
        [
            100
            * (
                1
                - s["forecasts"]["4"][m]["models"]["delta_ar2"]["source_mse"][g]
                / s["forecasts"]["4"][m]["models"]["persistence"]["source_mse"][g]
            )
            for g in groups
        ]
        for m in METRICS
    ]
    artist = heat(ax, values, groups, NAMES, "Does the secondary change model transfer across histories?")
    fig.colorbar(artist, ax=ax, label="MSE reduction vs persistence (%)")
    save(fig, "06-held-history-skill")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    for ax, field, title in zip(
        axes,
        ("level_acf", "delta_acf", "detrended_level_acf"),
        ("Health levels", "One-update changes", "Detrended health levels"),
        strict=True,
    ):
        values = [s["diagnostics"]["metrics"][m][field]["mean"][1:] for m in METRICS]
        artist = heat(ax, values, range(1, 9), NAMES, title, (-1, 1), False)
        ax.set_xlabel("Lag in accepted updates")
    fig.colorbar(artist, ax=axes, shrink=0.8, label="Autocorrelation")
    fig.suptitle("Temporal dependence: high persistence also appears in the changes", color=DARK)
    save(fig, "07-autocorrelation")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained", width_ratios=(1, 1.4, 1.4))
    cross = s["diagnostics"]["cross_health_delta_correlation"]
    values = [[cross[a + ":" + b]["mean"][0] for b in METRICS] for a in METRICS]
    artist = heat(axes[0], values, ("Var", "Cov", "Rank", "CKA", "Cos"), NAMES, "Cross-health changes", (-1, 1), False)
    for ax, field, title in zip(
        axes[1:],
        ("loss_ccf", "gradient_norm_ccf"),
        ("SSL loss → health change", "Gradient norm → health change"),
        strict=True,
    ):
        values = [s["diagnostics"]["metrics"][m][field]["mean"] for m in METRICS]
        heat(ax, values, range(-4, 5), NAMES, title, (-1, 1), False)
        ax.set_xlabel("Lag; positive = loss/norm precedes change")
    fig.colorbar(artist, ax=axes, shrink=0.8, label="Descriptive correlation")
    save(fig, "08-cross-correlations")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for index, (ax, title) in enumerate(
        zip(axes, ("Weight on latest change", "Weight on previous change"), strict=True)
    ):
        for k, metric in enumerate(METRICS):
            folds = s["forecasts"]["4"][metric]["folds"]["delta_ar2"]
            weights = [folds[g]["beta"][index] / folds[g]["scale"][index] for g in groups]
            ax.scatter(weights, np.full(6, k) + np.linspace(-0.13, 0.13, 6), c=PALETTE, s=40, zorder=3)
            ax.plot([min(weights), max(weights)], [k, k], color="#c4d1d8", lw=3, zorder=1)
        ax.axvline(0, color=GRAY, lw=1)
        ax.set(yticks=range(5), yticklabels=NAMES, title=title, xlabel="Coefficient in original health units")
        ax.invert_yaxis()
    fig.suptitle("The saved regression coefficients: six held-history folds", color=DARK)
    save(fig, "09-regression-coefficients")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    data = derivative["interventions"]["marginal"]["metrics"]
    y = np.arange(5)
    for ax, model, color, title in zip(
        axes,
        ("image_state", "adamw_derivative"),
        (ORANGE, BLUE),
        ("Cheap image + state ridge", "Paid AdamW-aware derivative"),
        strict=True,
    ):
        skill = [data[m]["models"][model]["skill_vs_zero_percent"] for m in METRICS]
        ax.barh(y, skill, color=color, height=0.6)
        ax.axvline(0, color=DARK, lw=1)
        ax.set(
            yticks=y,
            yticklabels=NAMES,
            title=title,
            xlim=(-110, 120),
            xlabel="MSE reduction vs zero marginal effect (%)",
        )
        ax.invert_yaxis()
        for i, value in enumerate(skill):
            if value < -60:
                position, align = value / 2, "center"
            else:
                position = value + (3 if value >= 0 else -3)
                align = "left" if value >= 0 else "right"
            ax.text(position, i, f"{value:.2f}%", ha=align, va="center", fontsize=10, color=DARK)
    fig.suptitle("A separate question: predicting one image's marginal health effect", color=DARK)
    save(fig, "10-image-health-comparison")

    publish(s, series, image, derivative, hashes, verification)


def executed(source, count, namespace):
    """Execute only this report's own displayed CPU code; save its actual output."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, "<health-regression-notebook>", "exec"), namespace)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": count,
        "source": source.splitlines(keepends=True),
        "outputs": [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(keepends=True)}],
    }


def publish(s, series, image, derivative, hashes, verification):
    table = "| Health signal | Level AR2, four updates | Change AR2, four updates | Change AR2, one update | Histories improved at h=4 |\n|---|---:|---:|---:|---:|\n"
    raw = "| Health signal | Persistence RMSE | Change-regression RMSE | Predicted/measured r |\n|---|---:|---:|---:|\n"
    residuals = "| Signal | Persistence residual lag-1 ACF | Change-model residual lag-1 ACF |\n|---|---:|---:|\n"
    for metric, name in zip(METRICS, NAMES, strict=True):
        result = s["forecasts"]["4"][metric]
        zero, delta = result["models"]["persistence"], result["models"]["delta_ar2"]
        improved = sum(delta["source_mse"][g] < zero["source_mse"][g] for g in zero["source_mse"])
        table += f"| {name} | {result['models']['ar2']['skill_vs_persistence_percent']:.2f}% | {delta['skill_vs_persistence_percent']:.2f}% | {s['forecasts']['1'][metric]['models']['delta_ar2']['skill_vs_persistence_percent']:.2f}% | {improved}/6 |\n"
        raw += f"| {name} | {zero['rmse']:.6g} | {delta['rmse']:.6g} | {delta['correlation']:.3f} |\n"
        acf = result["residual_acf_per_trajectory"]
        residuals += f"| {name} | {np.mean(np.array(acf['persistence'])[:, 1]):.3f} | {np.mean(np.array(acf['delta_ar2'])[:, 1]):.3f} |\n"
    image_table = "| Marginal image effect | Cheap image/state ridge skill | Paid derivative skill |\n|---|---:|---:|\n"
    for metric, name in zip(METRICS, NAMES, strict=True):
        models = derivative["interventions"]["marginal"]["metrics"][metric]["models"]
        image_table += f"| {name} | {models['image_state']['skill_vs_zero_percent']:.2f}% | {models['adamw_derivative']['skill_vs_zero_percent']:.2f}% |\n"
    # The original variance screen belongs to the original pilot, not the extension family.
    check(
        image["interventions"]["marginal"]["metrics"]["variance"]["models"]["adamw_derivative"][
            "skill_vs_zero_percent"
        ],
        derivative["interventions"]["marginal"]["metrics"]["variance"]["models"]["adamw_derivative"][
            "skill_vs_zero_percent"
        ],
    )

    intro = r"""# Predicting SSL health: the regression results

**A visual return to the completed prediction studies — no new training or model fitting.**

Recent changes in a health signal can forecast its next few changes on the saved continuations.
The clearest secondary leads are **RankMe, CKA drift and cosine drift**. The original primary,
which regressed on health *levels*, did not pass its adjusted screen. That distinction stays visible here.

This notebook asks: **given health observed so far, what happens to health if the same SSL recipe continues?**
It then contrasts that forecast with the separate experiment predicting the marginal effect of one image.
Neither result gives health changes a beneficial direction or predicts downstream accuracy.

The ten figures are saved inline. Blue indicates forecast skill; orange is used for contrasting methods.
History colours stay consistent across the trajectory, scatter and coefficient figures.
"""
    scope = r"""## 1. What was measured?

The time-series study replayed **18 continuations: six histories × three parent checkpoints**.
There is one optimization seed, 32 accepted batch-32 updates per continuation, and 33 readings
on the same 256-image health bank. Parent checkpoints within a history share sources.

| Signal | Instrument |
|---|---|
| Feature variance | Mean feature variance after row-normalizing embeddings |
| Feature covariance | Off-diagonal covariance penalty on row-normalized embeddings |
| RankMe | Effective rank from the raw embedding singular-value spectrum |
| CKA drift | One minus centered linear CKA against the warm representation |
| Cosine drift | Mean cosine distance from the warm representation |

Four additional accepted updates are the primary forecast horizon; one update is secondary.
Every forecast predicts a *change*: $Delta_h H_t = H_{t+h}-H_t$.
Warm-relative drift and the additional forecast change are different quantities.
Reading the bank costs encoder forwards, even when no optimizer preview is used.
"""
    regression = r"""## 2. The actual linear regressions

Define the most recent one-update change as $d_t = H_t-H_{t-1}$.
The secondary change model uses just two inputs for each health signal:

$$
\widehat{\Delta_4 H_t} = b + w_1 d_t + w_2 d_{t-1}.
$$

The saved fit standardized its inputs using training histories only, then used ridge penalty 10
and an unpenalized intercept. The expression above is the same fitted model in original units.
Its coefficients are $w_j=\beta_j/\sigma_j$ and $b=\bar y-\sum_j w_j\mu_j$.

| Method | Inputs available at the origin | Role |
|---|---|---|
| Persistence | Predict zero future change | Main baseline |
| Recent trend | Last four-update slope, extrapolated over the horizon | Baseline |
| AR1 | Current health level | Fixed comparison |
| AR2 | Current and previous levels | **Frozen primary regression** |
| Change AR2 | Last two one-update changes | **Predeclared secondary regression** |
| VARX | All five levels and changes, past loss/norm, training age | Fixed comparison |

**Evaluation holds out an entire history**, including all its parents. A fold fits early origins
4–16 in the other five histories. Training targets finish by update 20. It scores origins 20–28
in the held history, with all inputs ending at that origin. Each fold has 195 fit rows and 27
scored windows. Overlapping windows are dependent; there are six history groups, not 162 independent sources.
"""
    score_text = (
        r"""## 3. Which regressions actually forecast health?

We report improvement against persistence:

$$\text{skill} = 100\left(1-\frac{\text{macro-history MSE(model)}}{\text{macro-history MSE(persistence)}}\right).$$

Positive means less squared error; negative means worse than predicting no change.
**Skill is not accuracy, probability of being right, or $R^2$ against a fitted mean.**
The calculation averages the per-history MSEs before taking their ratio.

The level-based primary AR2 passes **none** of its adjusted tests. The change model is a
strong secondary lead, particularly for RankMe and drift. It remains a secondary result;
the primary model and its ten-comparison screen were not replaced after outcomes.

"""
        + table
    )
    parity = (
        """## 4. Predicted versus measured changes

These are all the saved four-update forecasts from the change model, not in-sample fits.
Each panel rescales both axes by that metric's persistence RMSE so the point clouds are comparable.
The diagonal represents a perfect prediction. Correlation alone does not establish calibration.
The original-unit errors remain visible below.

"""
        + raw
    )
    example = """## 5. A concrete forecast sequence

The example is the first saved series (`fresh_00 / before_shift`), selected by index rather than fit quality.
Each prediction at target update 24–32 uses a newly observed origin four updates earlier.
The reconstructed level is $H_t+\\widehat{\\Delta_4H_t}$.
This is nine overlapping rolling forecasts, **not** a nine-step open-loop forecast from one state.
The all-history scatter and table carry the performance claim.
"""
    transfer = """## 6. Does the lead hold across histories?

RankMe, CKA and cosine improve on persistence in all six histories at the four-update horizon.
Variance and covariance each have a history with negative skill. All five metrics improve in every
history at the secondary one-update horizon. These are descriptive cross-history results on reused
data, not an unused-source confirmation. No confidence interval is implied by the cell colours.
"""
    dependence = """## 7. Why look at autocorrelation?

The health levels are smooth. Their one-update changes also have strong positive lag-1 dependence
(roughly 0.86–0.89), which makes a regression on recent changes worth testing. High autocorrelation
alone is not a forecast evaluation: the poor level-AR2 result is a direct example.

ACF measures correlation with earlier observations at the displayed lag. Each trajectory is centered
separately; boundaries are never concatenated. Three parent trajectories are averaged within each
history before histories are averaged. Full-series detrending is a descriptive diagnostic only;
it never enters a forecast. Short series cannot establish general stationarity or seasonality.
"""
    cross_text = (
        """### Cross-signal relationships and remaining errors

Variance/covariance increments correlate **0.987**; CKA/cosine increments correlate **0.787**
on these continuations. That suggests redundancy worth measuring, not interchangeable sensors.
The loss/norm panels use the full prespecified lag grid. Positive lag means an earlier loss or norm
precedes a later health change; zero lag is a same-update association. None establishes causation.

The forecast errors also retain temporal structure. Each residual series has only nine overlapping
test windows. The lag-1 values below are descriptive averages, with no significance or white-noise claim.

"""
        + residuals
    )
    coefficients = r"""## 8. What did the regression learn?

These are the saved four-update change-model coefficients converted back to original units,
with one dot per held-history fold. The latest change receives a positive weight and the previous
change a negative one. The model is sensitive to how the local rate of change is evolving.
This is a description of the fitted relation, not a causal mechanism or a universal health law.
The model also has an intercept; the figure isolates the two slope coefficients.
"""
    image_text = (
        r"""## 9. A different target: what does one image do to health?

The image experiment compared a full batch-32 update with a matched update omitting one image's
loss contribution, keeping the denominator at 32. It used **144 marginal contrasts**:
18 parents × four candidate slots × two mask/augmentation draws. This is different from forecasting
a continuation from its last health readings. The observations share parents and draws.

The cheap image/state ridge has **negative skill for all five metrics**. The paid derivative
closely approximates the saved local effects:

$$\widehat{\Delta H} = \nabla_\theta H(\theta)^\top
\left(\Delta\theta_{\mathrm{full}}-\Delta\theta_{\mathrm{omit}}\right).$$

That method requires copied AdamW updates and health gradients. It is **not a fitted regression**,
and it is not a cheap image-only prediction. Bars in this section use a zero *marginal-effect*
baseline; they must not be pooled with the time-series persistence scores.

"""
        + image_table
        + """
The original marginal-variance derivative passed its bounded adjusted development screen.
The four new metrics in the retrospective extension did not pass their full adjusted screen,
despite very small descriptive errors. This does not erase the signal or establish independent
generalization. Variance image effects correlate only 0.314 across the two training-view draws;
image impact depends on the state, background batch and sampled training views.
"""
    )
    takeaway = """## 10. What this evidence supports

- **Short continuation forecasts:** recent-change regression is a promising secondary model for health under the same training recipe.
- **Cheap incoming-image forecasts:** the tested image/state regression did not establish calibrated marginal-effect prediction.
- **Paid local health forecasts:** optimizer-aware derivatives approximate saved one-update image effects very closely.
- **Training decisions:** none of these results calibrates which health change is good, predicts downstream accuracy reliably, or qualifies an adaptive image filter.

The histories, parent states and reference images are reused development data. Capture-session independence
is unknown, the dense time-series study uses one optimization seed, and horizons are short.
This notebook reorganizes existing evidence. It adds **zero SSL updates and zero model fits**.

Sources: `outputs/prediction-health-timeseries/20261014-pilot`,
`outputs/prediction-image-health/20261014-pilot-device-repair`, and
`outputs/prediction-health-derivatives/20261014-extension`.
The original numerical audits passed. This report also reconstructs saved targets, predictions,
training-only scalers and macro-history scores before plotting. Original result hashes are
checked against the earlier published provenance records and again after rendering.

For earlier full studies, see [time-series forecasts](05_health_timeseries.ipynb),
[metric derivatives](04_health_metric_derivatives.ipynb), and
[the original prediction pilots](03_prediction_pilots.ipynb).
Rebuild only this presentation from the repository root:
`python -m scripts.report_health_prediction_regressions`.
"""
    # All example code below is executed now; outputs are actual CPU results.
    namespace = {}
    load_code = """from pathlib import Path
import json
import numpy as np

root = Path.cwd()
while not (root / "cafl4ds").is_dir() and root != root.parent:
    root = root.parent
run = root / "outputs/prediction-health-timeseries/20261014-pilot"
summary = json.loads((run / "summary.json").read_text())
series = json.loads((run / "series.json").read_text())["records"]
assert all(json.loads((run / f).read_text())["passed"]
           for f in ("verification.json", "independent-verification.json"))
print(f"Loaded {len(series)} saved continuations; no fitting or SSL updates.")
"""
    reconstruct_code = """# Reconstruct every held-history CKA forecast from saved coefficients.
result = summary["forecasts"]["4"]["cka"]
reconstructed = []
for row in result["test_rows"]:
    t = row["origin"]
    health = np.asarray(series[row["series"]]["health"])[:, 3]
    x = np.array([health[t] - health[t-1], health[t-1] - health[t-2]])
    fold = result["folds"]["delta_ar2"][row["stream"]]
    predicted_change = (fold["intercept"] +
                        ((x - fold["center"]) / fold["scale"]) @ fold["beta"])
    reconstructed.append(predicted_change)
saved = result["models"]["delta_ar2"]["predictions"]
assert np.allclose(reconstructed, saved, atol=1e-12, rtol=1e-10)
print(f"Reconstructed {len(saved)} CKA forecasts from the saved folds.")
print("Four-update MSE reduction vs persistence: "
      f"{result['models']['delta_ar2']['skill_vs_persistence_percent']:.2f}%")
"""
    cells = [
        markdown(intro),
        markdown(scope),
        executed(load_code, 1, namespace),
        markdown(regression),
        figure_cell(OUT / "01-forecast-contract.png", 2),
        executed(reconstruct_code, 3, namespace),
        figure_cell(OUT / "02-health-trajectories.png", 4),
        markdown(score_text),
        figure_cell(OUT / "03-regression-scorecard.png", 5),
        markdown(parity),
        figure_cell(OUT / "04-predicted-versus-measured.png", 6),
        markdown(example),
        figure_cell(OUT / "05-one-continuation.png", 7),
        markdown(transfer),
        figure_cell(OUT / "06-held-history-skill.png", 8),
        markdown(dependence),
        figure_cell(OUT / "07-autocorrelation.png", 9),
        markdown(cross_text),
        figure_cell(OUT / "08-cross-correlations.png", 10),
        markdown(coefficients),
        figure_cell(OUT / "09-regression-coefficients.png", 11),
        markdown(image_text),
        figure_cell(OUT / "10-image-health-comparison.png", 12),
        markdown(takeaway),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    report_text = "\n\n".join(
        (
            intro,
            scope,
            regression,
            score_text,
            parity,
            example,
            transfer,
            dependence,
            cross_text,
            coefficients,
            image_text,
            takeaway,
        )
    )
    (OUT / "reading-guide.md").write_text(report_text, encoding="utf-8")
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Existing result changed during presentation build")
    atomic_json(OUT / "reconstruction-verification.json", verification)
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    hashes[str(ROOT / "scripts/report_prediction_pilots.py")] = digest(ROOT / "scripts/report_prediction_pilots.py")
    atomic_json(
        OUT / "provenance.json",
        {
            "sources": hashes,
            "figures": {str(p): digest(p) for p in sorted(OUT.glob("*.png"))},
            "svg_exports": {str(p): digest(p) for p in sorted(OUT.glob("*.svg"))},
            "notebook": str(NOTEBOOK),
            "notebook_sha256": digest(NOTEBOOK),
            "figure_count": 10,
            "new_training": False,
        },
    )
    print({"notebook": str(NOTEBOOK), "figures": 10, "reconstruction": verification})


if __name__ == "__main__":
    main()

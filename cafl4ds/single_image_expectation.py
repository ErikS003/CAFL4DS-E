"""Independent-repeat agreement for expected singleton image-effect measurements."""

from __future__ import annotations

import numpy as np


def center_candidates(values):
    """Remove each parent's candidate mean while retaining its repeat dimensions."""
    return values - values.mean(axis=1, keepdims=True)


def replication_pairs(effects):
    """Construct fixed, nonoverlapping training-view and held-reader comparisons."""
    if effects.ndim != 4 or effects.shape[1:] != (3, 6, 3):
        raise ValueError("Require parent x3 candidates x6 views x3 readers")
    result = {}
    for n in (1, 2, 3):
        a = effects[:, :, :n].mean((2, 3))
        b = effects[:, :, 3 : 3 + n].mean((2, 3))
        result[f"views_{n}_raw"] = (a, b)
        result[f"views_{n}_centered"] = (center_candidates(a), center_candidates(b))
        if n == 3:
            result["current_image_views_3"] = (a[:, :1], b[:, :1])
    readers = effects.mean(2)
    source, held = [], []
    for r in range(3):
        for other in range(3):
            if r != other:
                source.append(readers[:, :, other])
                held.append(readers[:, :, r])
    result["readers_1_centered"] = (
        center_candidates(np.stack(source, axis=-1)),
        center_candidates(np.stack(held, axis=-1)),
    )
    source, held = [], []
    for r in range(3):
        source.append(np.delete(readers, r, axis=2).mean(2))
        held.append(readers[:, :, r])
    result["readers_2_centered"] = (
        center_candidates(np.stack(source, axis=-1)),
        center_candidates(np.stack(held, axis=-1)),
    )
    a, b = effects[:, :, :3].mean(2), effects[:, :, 3:].mean(2)
    source, held = [], []
    for r in range(3):
        source += [a[:, :, r], b[:, :, r]]
        held += [np.delete(b, r, axis=2).mean(2), np.delete(a, r, axis=2).mean(2)]
    result["joint_centered"] = (
        center_candidates(np.stack(source, axis=-1)),
        center_candidates(np.stack(held, axis=-1)),
    )
    return result


def agreement(a, b, groups, draws, tolerance=1e-10):
    """Score replication, with bootstrap clusters equal to complete source histories."""
    if a.shape != b.shape or len(groups) != len(a):
        raise ValueError("Require matched effects and one history per parent")
    error = a - b
    aa, bb = a.flatten(), b.flatten()
    ac, bc = aa - aa.mean(), bb - bb.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
    corr = float(ac @ bc / denominator) if denominator > 1e-28 else None
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    if len({len(v) for v in members}) != 1:
        raise ValueError("Declared strata require equal parent counts per history")
    indices = np.array(members)[draws].reshape(len(draws), -1)
    boot_a, boot_b = a[indices].reshape(len(draws), -1), b[indices].reshape(len(draws), -1)
    boot_a = boot_a - boot_a.mean(1, keepdims=True)
    boot_b = boot_b - boot_b.mean(1, keepdims=True)
    denom = np.sqrt((boot_a**2).sum(1) * (boot_b**2).sum(1))
    valid = denom > 1e-28
    bootstrap_corr = (boot_a * boot_b).sum(1)[valid] / denom[valid]
    ci = np.quantile(bootstrap_corr, [0.025, 0.975]).tolist() if len(bootstrap_corr) else None
    zero_rmse = float(np.sqrt(0.5 * np.mean(a**2 + b**2)))
    zero_mae = float(0.5 * np.mean(abs(a) + abs(b)))
    rmse, mae = float(np.sqrt(np.mean(error**2))), float(np.mean(abs(error)))
    nonzero_a, nonzero_b = abs(a) > tolerance, abs(b) > tolerance
    both = nonzero_a & nonzero_b
    return {
        "correlation": corr,
        "correlation_ci95": ci,
        "valid_bootstrap_draws": int(valid.sum()),
        "rmse": rmse,
        "mae": mae,
        "symmetric_zero_rmse": zero_rmse,
        "symmetric_zero_mae": zero_mae,
        "rmse_ratio_to_zero": rmse / zero_rmse if zero_rmse > 0 else None,
        "mae_ratio_to_zero": mae / zero_mae if zero_mae > 0 else None,
        "cross_repeat_signal_power": float(np.mean(a * b)),
        "half_difference_noise_power": float(0.5 * np.mean(error**2)),
        "observed_power": float(0.5 * np.mean(a**2 + b**2)),
        "both_nonzero_cases": int(both.sum()),
        "nonzero_direction_agreement": float(np.mean(np.sign(a[both]) == np.sign(b[both]))) if both.any() else None,
        "zero_fraction_a": float(np.mean(~nonzero_a)),
        "zero_fraction_b": float(np.mean(~nonzero_b)),
        "parent_count": len(a),
        "history_count": len(members),
        "values_in_comparison": a.size,
    }


def measurement_gate(view, joint):
    """A positive repeat correlation alone cannot establish usable precision."""

    def passed(result):
        ci = result["correlation_ci95"]
        return bool(
            ci is not None
            and ci[0] > 0
            and result["rmse_ratio_to_zero"] is not None
            and result["rmse_ratio_to_zero"] < 1
            and result["mae_ratio_to_zero"] < 1
        )

    return passed(view) and passed(joint)

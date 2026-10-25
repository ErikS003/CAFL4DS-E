"""Predict-then-learn audit of saved accepted updates; never trains the SSL learner."""

import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/online-controller-audit/20261025-chronological"
FEATURE_NAMES = [
    "current_rank", "current_variance", "current_covariance", "current_bank_reconstruction",
    "preceding_arrival_change_rank", "preceding_arrival_change_variance",
    "preceding_arrival_change_covariance", "preceding_arrival_change_bank_reconstruction",
    "training_view_ssl_loss", "candidate_gradient_norm", "gradient_momentum_cosine",
    "proposed_adamw_displacement_norm", "development_CE_gradient_displacement_cosine",
    "warm_centroid_distance", "nearest_recent_offered_cosine_distance",
]
J_METHODS = ["zero", "frozen", "rate_scaled_frozen", "frozen_plus_recent_bias", "prefix_ridge_correction", "online_ridge_correction"]
TASK_METHODS = ["zero", "recent_mean", "prefix_ridge", "online_ridge"]


def replay(features, outcomes, frozen, rates, protocol):
    prefix = protocol["prefix_accepted_updates"]
    assert len(features) > prefix and np.isfinite(features).all() and np.isfinite(outcomes).all()
    center, scale = features[:prefix].mean(0), features[:prefix].std(0)
    scale[scale < 1e-10] = 1
    x = np.c_[np.ones(len(features)), np.clip((features - center) / scale, -5, 5)]
    labels = np.c_[outcomes[:, 0] - frozen, outcomes[:, 1]]
    target_scale = labels[:prefix].std(0)
    target_scale[target_scale < 1e-12] = 1
    regularizer = np.diag([protocol["intercept_ridge"]] + [protocol["ridge"]] * features.shape[1])
    a, b = regularizer.copy(), np.zeros((x.shape[1], 2))
    discount = 2 ** (-1 / protocol["half_life_accepted_updates"])
    mean = np.zeros(2)
    # Prefix outcomes train only the initial models; none are scored as predictions.
    for i in range(prefix):
        a = discount * a + (1 - discount) * regularizer + np.outer(x[i], x[i])
        b = discount * b + np.outer(x[i], labels[i] / target_scale)
        mean = labels[i] if i == 0 else discount * mean + (1 - discount) * labels[i]
    fixed = np.linalg.solve(a, b)
    pred_j = np.full((len(features), len(J_METHODS)), np.nan)
    pred_t = np.full((len(features), len(TASK_METHODS)), np.nan)
    for i in range(prefix, len(features)):
        online = x[i] @ np.linalg.solve(a, b) * target_scale
        static = x[i] @ fixed * target_scale
        # All predictions above and below use only outcomes preceding i.
        pred_j[i] = [0, frozen[i], frozen[i] * rates[i], frozen[i] + mean[0],
                     frozen[i] + static[0], frozen[i] + online[0]]
        pred_t[i] = [0, mean[1], static[1], online[1]]
        # Reveal this outcome only after its predictions have been stored.
        a = discount * a + (1 - discount) * regularizer + np.outer(x[i], x[i])
        b = discount * b + np.outer(x[i], labels[i] / target_scale)
        mean = discount * mean + (1 - discount) * labels[i]
    return pred_j, pred_t


def metrics(y, p):
    nonzero = np.abs(y) > 1e-12
    both = nonzero & (np.abs(p) > 1e-12)
    rho = float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else None
    return dict(n=len(y), mae=float(np.mean(np.abs(y-p))), rmse=float(np.sqrt(np.mean((y-p)**2))),
                spearman=rho, sign_accuracy=float(np.mean(np.sign(y[both]) == np.sign(p[both]))) if both.any() else None,
                direction_coverage=float(both.mean()), improvement_prevalence=float(np.mean(y < 0)),
                actual_mean=float(y.mean()), prediction_mean=float(p.mean()))


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    schema_path = ROOT / "outputs/single-image-filter-factorial/20261021-probe-feature/feature-schema.json"
    schema = json.loads(schema_path.read_text())["rich"]
    columns = [schema.index(k) for k in FEATURE_NAMES]
    hashes = {}
    def remember(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    remember(schema_path); remember(OUT / "protocol.json"); remember(Path(__file__))
    results, plotted = {}, {}
    for name in ["20261024-no-rollback", "20261025-bounded-lr"]:
        branch = ROOT / "outputs/single-image-heuristic" / name / "trajectories/fresh_00/seed_5003/heuristic"
        trace = sorted((branch / "resume_states").glob("step_*/trace.npz"))[-1]
        remember(trace)
        record_path = trace.with_name("commit.json"); remember(record_path)
        record = json.loads(record_path.read_text())
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == record["file_hashes"]["trace.npz"]
        with np.load(trace) as z:
            accepted = np.flatnonzero(z["accepted"][:record["done"]])
            accepted = accepted[accepted > 0]  # first arrival lacks a preceding trace row
            assert not z["task_rollback"].any() and not z["numerical_failure"].any()
            idx = accepted
            # Confirm selected health features are PRE-update, not outcome measurements.
            np.testing.assert_allclose(z["features"][idx, :7], z["health"][idx-1], atol=1e-12)
            features = np.c_[z["features"][idx][:, columns], z["prediction"][idx], z["rate"][idx],
                             (z["mode"][idx] == 1).astype(float), z["development"][idx-1, 3]]
            outcomes = np.c_[z["J"][idx] - z["J"][idx-1], z["development"][idx, 3] - z["development"][idx-1, 3]]
            frozen, rates = z["prediction"][idx], z["rate"][idx]
            clipped = np.mean(z["clipped"][idx] > .25)
        pj, pt = replay(features, outcomes, frozen, rates, protocol)
        # Causality checks: future outcome/feature changes cannot alter earlier predictions.
        cut = 400
        altered = outcomes[:600].copy(); altered[cut:] += np.array([.1, -.1])
        aj, at = replay(features[:600], altered, frozen[:600], rates[:600], protocol)
        np.testing.assert_array_equal(aj[:cut+1], pj[:cut+1])
        np.testing.assert_array_equal(at[:cut+1], pt[:cut+1])
        changed_features = features[:600].copy(); changed_features[cut:] += 100
        fj, ft = replay(changed_features, outcomes[:600], frozen[:600], rates[:600], protocol)
        np.testing.assert_allclose(fj[:cut], pj[:cut], atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_allclose(ft[:cut], pt[:cut], atol=1e-12, rtol=0, equal_nan=True)
        scored = np.arange(len(idx)) >= protocol["prefix_accepted_updates"]
        windows = {"all_scored": scored}
        for lo, hi in [(0,10000),(10000,20000),(20000,30000),(30000,40000),(40000,57600),(30000,57600)]:
            windows[f"{lo}-{hi}"] = scored & (idx >= lo) & (idx < hi)
        result = dict(accepted_cases=len(idx), prefix=protocol["prefix_accepted_updates"],
                      first_scored_arrival=int(idx[protocol["prefix_accepted_updates"]]+1),
                      frozen_fallback_fraction=float(clipped), windows={})
        for label, mask in windows.items():
            result["windows"][label] = dict(J={k:metrics(outcomes[mask,0],pj[mask,i]) for i,k in enumerate(J_METHODS)},
                                           task_soft_error={k:metrics(outcomes[mask,1],pt[mask,i]) for i,k in enumerate(TASK_METHODS)})
        result["late_blocks"] = []
        for lo in range(30000,57600,1920):
            mask=scored & (idx >= lo) & (idx < min(lo+1920,57600))
            if not mask.any(): continue
            result["late_blocks"].append(dict(start=lo, n=int(mask.sum()),
                task_online_mae=float(np.mean(np.abs(pt[mask,3]-outcomes[mask,1]))),
                task_recent_mae=float(np.mean(np.abs(pt[mask,1]-outcomes[mask,1]))),
                task_zero_mae=float(np.mean(np.abs(outcomes[mask,1]))),
                J_online_mae=float(np.mean(np.abs(pj[mask,5]-outcomes[mask,0]))),
                J_recent_bias_mae=float(np.mean(np.abs(pj[mask,3]-outcomes[mask,0])))))
        results[name] = result
        np.savez_compressed(OUT / f"{name}-predictions.npz", offered=idx+1, actual=outcomes,
                            J_predictions=pj, task_predictions=pt, scored=scored,
                            J_methods=J_METHODS, task_methods=TASK_METHODS)
        plotted[name]=(idx+1,outcomes,pj,pt,scored)
    (OUT / "results.json").write_text(json.dumps(results,indent=2))
    (OUT / "input-hashes.json").write_text(json.dumps(hashes,indent=2))
    (OUT / "verification.json").write_text(json.dumps(dict(passed=True,trace_hashes=True,pre_update_features=True,
        future_outcome_perturbation_invariant=True,future_feature_perturbation_invariant=True,
        prefix_only_scaling=True,no_final_evaluation_labels=True,no_SSL_training=True,no_controller_changes=True,
        feature_names=FEATURE_NAMES+["frozen_J_prediction","applied_LR_factor","caution_mode","prior_development_soft_error"]),indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for row,(name,(idx,y,pj,pt,scored)) in enumerate(plotted.items()):
        x=idx[scored]; yy=y[scored]; j=pj[scored]; t=pt[scored]; w=256
        for k in [0,1,2,3,5]:
            axes[row,0].plot(x[w-1:],np.convolve(np.abs(j[:,k]-yy[:,0]),np.ones(w)/w,mode="valid"),label=J_METHODS[k])
        for k in [0,1,2,3]:
            axes[row,1].plot(x[w-1:],np.convolve(np.abs(t[:,k]-yy[:,1]),np.ones(w)/w,mode="valid"),label=TASK_METHODS[k])
        for ax in axes[row]:
            ax.set_yscale("log");ax.set_xlabel("Offered arrivals");ax.set_ylabel("Trailing MAE, 256 scored updates (log scale)");ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[row,0].set_title(name+": immediate J change")
        axes[row,1].set_title(name+": development soft-error change")
    fig.savefig(OUT/"chronological_prediction_errors.png",dpi=150);plt.close(fig)
    print(json.dumps({name:r["windows"]["30000-57600"] for name,r in results.items()},indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

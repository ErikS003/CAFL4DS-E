"""Synthetic factorial GPU probes and all29 forecasting/report settings."""
from __future__ import annotations

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.functional_response_profile import MODELS, forecast, screen
from cafl4ds.response_reliability import MASK_OFFSETS, grid_variance, panels
from scripts.report_response_reliability import make_report
from scripts.run_response_reliability import OUT, base, digest, padded_embeddings, read, write


def main():
    out = OUT/'engineering'; out.mkdir(parents=True, exist_ok=True); base.setup(); torch.manual_seed(195353)
    workpath = OUT/'engineering-work.json'; assert not workpath.exists(), 'Preserve attempted synthetic work'
    work = {'SSL_started': 0, 'SSL_completed': 0, 'passes_started': 0, 'passes_completed': 0}
    def update(model, opt, batch):
        work['SSL_started'] += 1; write(workpath, work)
        result = base.optimize_current(model, opt, batch, True)
        work['SSL_completed'] += 1; write(workpath, work); return result
    def infer(model, images):
        result = None
        for repeat in (False, True):
            work['passes_started'] += 1; write(workpath, work); value = padded_embeddings(model, images)
            if repeat:
                np.testing.assert_array_equal(value, result)
            else:
                result = value
            work['passes_completed'] += 1; write(workpath, work)
        return result
    encoder = instantiate(OmegaConf.create({'_target_': 'cafl4ds.models.vit.TinyViTEncoder', 'img_size': 64,
        'patch_size': 8, 'in_chans': 3, 'embed_dim': 96, 'depth': 4, 'num_heads': 3, 'mlp_ratio': 2.}))
    model = instantiate(OmegaConf.create({'_target_': 'cafl4ds.ssl.factory.build_mae', 'mask_ratio': .75,
        'decoder_dim': 64, 'decoder_depth': 2, 'decoder_heads': 4, 'decoder_mlp_ratio': 2., 'norm_pix_loss': True}), encoder=encoder).cuda()
    config = OmegaConf.create({'_target_': 'torch.optim.AdamW', 'lr': 1e-4, 'weight_decay': .05}); opt = instantiate(config, params=model.parameters())
    images = torch.rand(200, 3, 64, 64); current = torch.rand(3, 32, 3, 64, 64).cuda(); historical = torch.rand(3, 16, 3, 64, 64).cuda()
    fork = base.capture(model, opt); original = base.fingerprint(fork); infer(model, images)
    for b, m, action in np.ndindex(3, 3, 2):
        batch = current[b] if action == 0 else torch.cat([current[b, :16], historical[b]])
        local, lo = base.restore(model, config, fork); torch.manual_seed(MASK_OFFSETS[m]); torch.cuda.manual_seed_all(MASK_OFFSETS[m])
        parameters = [p for p in local.encoder.parameters() if p.requires_grad]; before = [p.detach().clone() for p in parameters]
        metric = update(local, lo, batch); fp = base.fingerprint(base.capture(local, lo))
        if b == m == 1:
            again, ao = base.restore(model, config, fork); torch.manual_seed(MASK_OFFSETS[m]); torch.cuda.manual_seed_all(MASK_OFFSETS[m])
            np.testing.assert_array_equal(metric, update(again, ao, batch)); assert base.fingerprint(base.capture(again, ao)) == fp
        base.set_encoder_step(parameters, before, .005); infer(local, images)
    torch.set_rng_state(fork['cpu_rng']); torch.cuda.set_rng_state_all(fork['cuda_rng']); assert base.fingerprint(base.capture(model, opt)) == original
    assert work['SSL_completed'] == 20 and work['passes_completed'] == 38
    rng = np.random.default_rng(195353)
    groups = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3, 4) for _ in range(2)])
    diets = np.tile([0, 1], 36); h = rng.normal(size=(72, 6, 2, 8)); signals = rng.normal(size=(72, 16))
    state = rng.normal(0, .01, (72, 1, 1, 1, 2, 8))
    delta = state+rng.normal(0, .006, (72, 3, 1, 1, 2, 8))+rng.normal(0, .003, (72, 1, 3, 1, 2, 8))+rng.normal(0, .002, (72, 3, 3, 6, 2, 8))
    target = rng.normal(3, 9, (72, 2, 7)); by_draw = target[:, :, None]+rng.normal(0, 2, (72, 2, 12, 7)); target = by_draw.mean(2)
    settings = panels(delta, h); all_pred = []; screens = {}
    for name, rr, static in settings:
        predicted, _, _ = forecast(signals, diets, groups, static, rr, target.reshape(72, 14))
        pp = np.stack([predicted[m].reshape(72, 2, 7) for m in MODELS], 1); all_pred.append(pp)
        screens[name] = screen(target[:, 1, 6], {m: pp[:, j, 1, 6] for j, m in enumerate(MODELS)}, groups)
    pp = all_pred[0]; draw = np.array([np.mean((pp[:, :, 1, 6]-by_draw[:, 1, d, 6, None])**2, axis=0) for d in range(12)])
    np.savez_compressed(out/'analysis-arrays.npz', contrast=delta, groups=groups, diets=diets, target=target,
        predictions=all_pred, grid_variance=grid_variance(delta[..., 0, :].mean(3)), draw_MSE=draw)
    write(out/'summary.json', {'states': 72, 'physical_SSL': 20, 'encoder_passes': 38, 'image_forwards': 8512,
        'panels': [n for n, _, _ in settings], 'screens': screens,
        'primary_MSE': {m: float(np.mean((pp[:, j, 1, 6]-target[:, 1, 6])**2)) for j, m in enumerate(MODELS)}})
    delivery = make_report(out, out/'synthetic.ipynb', engineering=True)
    archived = list((OUT/'engineering-attempts').rglob('*'))
    prior_ssl = sum(read(p)['physical_SSL'] for p in archived if p.name == 'failure.json')
    prior_passes = sum(read(p)['encoder_passes'] for p in archived if p.name == 'failure.json')
    write(OUT/'engineering-verification.json', {'passed': True, 'visual_QA_pending': True,
        'synthetic_SSL': 20, 'synthetic_encoder_passes': 38, 'synthetic_image_forwards': 8512,
        'cumulative_synthetic_SSL_including_failed_fixture': 20+prior_ssl,
        'cumulative_synthetic_passes_including_failed_fixture': 38+prior_passes,
        'cumulative_synthetic_forwards_including_failed_fixture': (38+prior_passes)*224,
        'paired_mask_content_proposals_and_audit_repeats_exact': True, 'padded_feature_repeats_and_original_state_exact': True,
        'all29_forecast_settings_and_six_figures_exercised': True,
        'hashes': delivery['hashes'] | {str(p): digest(p) for p in [workpath, out/'analysis-arrays.npz', out/'summary.json']+archived if p.is_file()}})
    print('Synthetic crossed probes,20 SSL/38 passes, all29 forecast settings and six figures passed; visual QA pending.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

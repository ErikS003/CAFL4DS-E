"""Synthetic GPU copies/proposals/inference plus complete forecast/report fixture."""
from __future__ import annotations

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.content_step_intervention import set_encoder_step
from cafl4ds.functional_response_profile import MODELS, forecast, screen
from cafl4ds.health_calibration import embeddings
from cafl4ds.reservoir_action_replication import capture, restore
from cafl4ds.update_filtering import optimize_current
from scripts.report_functional_response_profile import make_report
from scripts.run_functional_response_profile import OUT, csv_write, digest, fingerprint, read, setup, write


def main():
    out = OUT/'engineering'; out.mkdir(parents=True, exist_ok=True); setup(); torch.manual_seed(195252)
    encoder = instantiate(OmegaConf.create({'_target_': 'cafl4ds.models.vit.TinyViTEncoder', 'img_size': 64,
        'patch_size': 8, 'in_chans': 3, 'embed_dim': 96, 'depth': 4, 'num_heads': 3, 'mlp_ratio': 2.}))
    model = instantiate(OmegaConf.create({'_target_': 'cafl4ds.ssl.factory.build_mae', 'mask_ratio': .75,
        'decoder_dim': 64, 'decoder_depth': 2, 'decoder_heads': 4, 'decoder_mlp_ratio': 2., 'norm_pix_loss': True}), encoder=encoder).cuda()
    config = OmegaConf.create({'_target_': 'torch.optim.AdamW', 'lr': 1e-4, 'weight_decay': .05})
    optimizer = instantiate(config, params=model.parameters())
    images = torch.rand(200, 3, 64, 64); batches = [images[:32].cuda(), images[32:64].cuda()]
    fork = capture(model, optimizer); original = fingerprint(fork); actual_SSL = 0; passes = 0
    zero = embeddings(model, images); passes += 1; np.testing.assert_array_equal(zero, embeddings(model, images)); passes += 1
    for action in range(2):
        saved = None
        for repeat in (False, True):
            local, opt = restore(model, config, fork); torch.manual_seed(195200); torch.cuda.manual_seed_all(195200)
            parameters = [p for p in local.encoder.parameters() if p.requires_grad]; before = [p.detach().clone() for p in parameters]
            metrics = optimize_current(local, opt, batches[action], True); actual_SSL += 1
            if repeat:
                assert saved[0] == fingerprint(capture(local, opt)); np.testing.assert_array_equal(metrics, saved[1])
            else:
                saved = fingerprint(capture(local, opt)), metrics
        proposed = [p.detach().clone() for p in parameters]
        for scale in (.25, .5):
            with torch.no_grad():
                for p, v in zip(parameters, proposed, strict=True):
                    p.copy_(v)
            _, actual, _ = set_encoder_step(parameters, before, scale*.01)
            assert abs(actual-scale*.01) < 1e-7
            z = embeddings(local, images); passes += 1
            np.testing.assert_array_equal(z, embeddings(local, images)); passes += 1
    # Probe RNG is deliberately reseeded; restore it before comparing the
    # original learner's complete state. Parameters/optimizer were never edited.
    torch.set_rng_state(fork['cpu_rng']); torch.cuda.set_rng_state_all(fork['cuda_rng'])
    assert fingerprint(capture(model, optimizer)) == original
    rng = np.random.default_rng(195252)
    groups = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3, 4) for _ in range(2)])
    diets = np.tile([0, 1], 36); h = rng.normal(size=(72, 6, 2, 8)); signals = rng.normal(size=(72, 16))
    action = rng.normal(0, .01, (72, 2, 2, 6, 2, 8)); action[:, :, 1] = 2*action[:, :, 0]+rng.normal(0, .0005, action[:, :, 1].shape)
    delta = action[:, 1]-action[:, 0]; target = rng.normal(2, 6, (72, 2, 7)); all_pred = []; screens = {}
    panels = [(f'mean_dose{m}_prior{k}', delta[:, m, :, k].mean(1), h[:, :, k].mean(1)) for m, k in ((1, 0), (0, 0), (1, 1), (0, 1))]
    panels += [(f'pool{o}_dose1_prior0', delta[:, 1, o, 0], h[:, o, 0]) for o in range(6)]
    for name, rr, static in panels:
        pred, _, _ = forecast(signals, diets, groups, static, rr, target.reshape(72, 14))
        pp = np.stack([pred[m].reshape(72, 2, 7) for m in MODELS], 1); all_pred.append(pp)
        screens[name] = screen(target[:, 1, 6], {m: pp[:, j, 1, 6] for j, m in enumerate(MODELS)}, groups)
    np.savez_compressed(out/'analysis-arrays.npz', contrast=delta, target=target, groups=groups, diets=diets, predictions=all_pred)
    write(out/'summary.json', {'states': 72, 'physical_SSL': actual_SSL, 'encoder_passes': passes, 'image_forwards': passes*200,
        'primary_MSE': {m: float(np.mean((all_pred[0][:, j, 1, 6]-target[:, 1, 6])**2)) for j, m in enumerate(MODELS)}, 'screens': screens})
    write(out/'audit-summary.json', {'states': [{'cosine': float(v)} for v in rng.uniform(-.4, 1., 72)]})
    rr = delta[:, 1, :, 0].mean(1)
    csv_write(out/'correlations.csv', [{'diet': diet, 'axis': k, 'r': float(np.corrcoef(rr[diets == diet, k], target[diets == diet, 1, 6])[0, 1])} for diet in (0, 1) for k in range(8)])
    shadows = []
    for model in ('always_current', 'always_replay', 'cheap', 'static', 'static_response_added'):
        take = np.zeros(72, bool) if model == 'always_current' else np.ones(72, bool) if model == 'always_replay' else all_pred[0][:, MODELS.index(model), 1, 6] > 0
        for k in range(7):
            shadows.append({'model': model, 'component': k, 'value_pp': float(np.mean(target[:, 1, k]*take))})
    csv_write(out/'shadow-values.csv', shadows)
    delivery = make_report(out, out/'synthetic.ipynb', engineering=True)
    paths = [out/n for n in ('analysis-arrays.npz', 'summary.json', 'audit-summary.json', 'correlations.csv', 'shadow-values.csv')]
    attempt = out/'attempts/001.json'
    write(attempt, {'passed': True, 'synthetic_only': True, 'physical_SSL': actual_SSL,
                    'encoder_passes': passes, 'image_forwards': passes*200})
    paths += list((out/'attempts').glob('*.json'))
    write(OUT/'engineering-verification.json', {'passed': True, 'visual_QA_pending': True, 'synthetic_SSL_updates': actual_SSL,
        'synthetic_encoder_passes': passes, 'synthetic_image_forwards': passes*200,
        'cumulative_engineering_SSL_updates_including_failed_attempt': sum(read(p)['physical_SSL'] for p in (out/'attempts').glob('*.json')),
        'cumulative_engineering_passes_including_failed_attempt': sum(read(p)['encoder_passes'] for p in (out/'attempts').glob('*.json')),
        'original_model_unmodified': True, 'exact_proposal_and_inference_repeats': True,
        'hashes': delivery['hashes'] | {str(p): digest(p) for p in paths}})
    print('Synthetic GPU proposal/dose/observer mechanics and all10 forecast panels/six figures complete; visual QA pending.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

"""Frozen padding correction; preserve and count the unpadded aborted attempt."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

import scripts.run_functional_response_profile as core

ORIGINAL_EMBEDDINGS = core.embeddings
OUT = core.OUT; ROOT = core.ROOT


def padded_embeddings(model, images):
    n = len(images); pad = (-n) % 32
    tensor = torch.cat([images, images[:pad]]) if pad else images
    return ORIGINAL_EMBEDDINGS(model, tensor)[:n]


def engineering():
    core.setup(); torch.manual_seed(195253)
    encoder = instantiate(OmegaConf.create({'_target_': 'cafl4ds.models.vit.TinyViTEncoder', 'img_size': 64,
        'patch_size': 8, 'in_chans': 3, 'embed_dim': 96, 'depth': 4, 'num_heads': 3, 'mlp_ratio': 2.})).cuda()
    model = torch.nn.Module(); model.encoder = encoder
    images = torch.rand(224, 3, 64, 64)
    full = ORIGINAL_EMBEDDINGS(model, images)
    short = ORIGINAL_EMBEDDINGS(model, images[:200])
    padded = padded_embeddings(model, images[:200])
    np.testing.assert_array_equal(full[:192], short[:192])
    np.testing.assert_array_equal(full[:200], padded)
    path = OUT/'padding-engineering.json'
    core.write(path, {'passed': True, 'synthetic_only': True, 'SSL_updates': 0, 'encoder_passes': 3,
        'image_forwards': 648, 'short_batch_maximum_error': float(np.max(abs(full[:200]-short))),
        'padding_matches_full_batch_exactly': True, 'padding_uses_only_first24_support_images': True})
    print(core.read(path))


def freeze():
    assert not (OUT/'padding-amendment-lock.json').exists()
    core.check(core.read(OUT/'input_lock.json')['hashes'])
    assert core.read(OUT/'padding-engineering.json')['passed']
    work = core.read(OUT/'work.json')
    assert work == {'SSL_started': 128, 'SSL_completed': 128, 'passes_started': 2, 'passes_completed': 2, 'in_progress': 0}
    assert not (OUT/'response-ledger.json').exists() and not (OUT/'target-start.json').exists()
    archive = OUT/'attempts/000-unpadded'; archive.mkdir(parents=True)
    (OUT/'work.json').replace(archive/'work.json'); shutil.copyfile(OUT/'run.log', archive/'run.log')
    core.write(archive/'audit.json', {'physical_SSL': 128, 'encoder_passes': 2, 'image_forwards': 400,
        'all128_recorded_updates_and_full_fork_exact': True, 'probes_or_targets_reached': False,
        'zero_feature_cached_equality_failed_at_short_final_batch': True, 'maximum_difference': 2.3841858e-7})
    paths = [Path(__file__), OUT/'padding-engineering.json', OUT/'input_lock.json',
        ROOT/'docs/project-plan/functional-response-profile-padding-amendment.md', *archive.iterdir()]
    core.write(OUT/'padding-amendment-lock.json', {'hashes': {str(p): core.digest(p) for p in paths},
        'frozen_before_resumed_reconstruction_or_probes': True, 'frozen_unix': time.time(),
        'complete_collection_SSL': 9504, 'aborted_reconstruction_SSL': 128, 'total_physical_SSL': 9632,
        'complete_collection_passes': 720, 'aborted_passes': 2, 'total_passes': 722,
        'complete_collection_forwards': 161280, 'aborted_forwards': 400, 'total_forwards': 161680,
        'profile_image_roles': 144000, 'padding_forwards': 17280, 'synthetic_extra_passes': 3, 'synthetic_extra_forwards': 648})
    core.status('padding-amendment-frozen', total_SSL=9632, total_image_forwards=161680)


def metadata_correction(value):
    return {**value, 'physical_SSL': 9632, 'collection_SSL': 9504, 'aborted_reconstruction_SSL': 128,
        'encoder_passes': 722, 'collection_encoder_passes': 720, 'aborted_encoder_passes': 2,
        'image_forwards': 161680, 'collection_image_forwards': 161280, 'aborted_image_forwards': 400,
        'profile_image_role_forwards': 144000, 'padding_image_forwards': 17280,
        'padding_only_existing_support_images': True, 'padding_amendment_lock_sha256': core.digest(OUT/'padding-amendment-lock.json')}


def run():
    core.check(core.read(OUT/'padding-amendment-lock.json')['hashes'])
    core.embeddings = padded_embeddings
    core.collect()
    core.write(OUT/'response-ledger.json', metadata_correction(core.read(OUT/'response-ledger.json')))
    core.evaluate()
    core.write(OUT/'summary.json', metadata_correction(core.read(OUT/'summary.json')))
    ev = core.read(OUT/'evaluation-verification.json'); ev['hashes'][str(OUT/'summary.json')] = core.digest(OUT/'summary.json')
    core.write(OUT/'evaluation-verification.json', ev)
    from scripts.report_functional_response_profile import make_report
    core.write(OUT/'delivery-verification.json', make_report(OUT, ROOT/core.CFG['notebook']))
    core.status('numerically-complete-awaiting-QA', physical_SSL=9632, image_forwards=161680)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('engineering', 'freeze', 'run'))
    stage = parser.parse_args().stage; core.setup()
    with threadpool_limits(limits=4):
        {'engineering': engineering, 'freeze': freeze, 'run': run}[stage]()

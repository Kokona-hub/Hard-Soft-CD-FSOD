#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Validate the builder's rank / context-subspace math without mmcv.

``build_s2h_knowledge.py`` imports mmcv and mmdet, which exist on the training
server; a laptop or the analysis machine usually has torch but not mmcv, so the
builder's own arithmetic could not be checked before a multi-hour rebuild was
launched.  This harness stubs ``mmengine.registry``, imports the *real*
``mmdet.models.utils.s2h_knowledge`` helpers, and then ``exec``s the two
functions straight out of the builder source - the shipped code is what is
exercised, not a copy.

It also asserts that the two CHSD formulas introduced for the reliability fix
are present in the source, so a later edit cannot silently undo them.

Usage
-----
    python tools/s2h/analysis/check_builder_math.py
"""
import importlib.util
import os.path as osp
import re
import sys
import types
from typing import Optional

import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
BUILDER = osp.join(ROOT, 'tools', 's2h', 'build_s2h_knowledge.py')
UTILS = osp.join(ROOT, 'mmdet', 'models', 'utils', 's2h_knowledge.py')


def stub_mmengine() -> None:
    """``mmengine.registry`` is only needed for the import to succeed."""
    engine = types.ModuleType('mmengine')
    registry = types.ModuleType('mmengine.registry')

    class _Registry:

        def register_module(self, *args, **kwargs):

            def decorate(obj):
                return obj

            return decorate

    registry.MODELS = _Registry()
    sys.modules['mmengine'] = engine
    sys.modules['mmengine.registry'] = registry


def grab(source: str, name: str) -> str:
    match = re.search(rf'^def {name}\(.*?(?=^def |\Z)', source, re.S | re.M)
    if not match:
        raise SystemExit(f'{name} not found in {BUILDER}')
    return match.group(0)


def main() -> int:
    stub_mmengine()
    spec = importlib.util.spec_from_file_location('s2h_utils', UTILS)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    with open(BUILDER, encoding='utf-8') as handle:
        source = handle.read()

    namespace = dict(torch=torch, Optional=Optional,
                     svd_context_basis=utils.svd_context_basis,
                     context_leakage_energy=utils.context_leakage_energy)
    exec(grab(source, 'energy_rank'), namespace)            # noqa: S102
    exec(grab(source, 'compute_context_subspace'), namespace)  # noqa: S102
    energy_rank = namespace['energy_rank']
    compute_context_subspace = namespace['compute_context_subspace']

    spectrum = torch.tensor([5.0, 3.0, 1.0, 0.4, 0.2, 0.1])
    total = float(spectrum.pow(2).sum())
    print('spectrum:', [round(float(v), 2) for v in spectrum])
    for target in (0.5, 0.8, 0.9, 0.99, 1.0):
        chosen = energy_rank(spectrum, target)
        held = float(spectrum[:chosen].pow(2).sum()) / total
        print(f'  energy {target:.2f} -> rank {chosen} (holds {held:.3f})')
        assert held >= min(target, 1.0) - 1e-6, (target, chosen, held)
        if chosen > 1:
            previous = float(spectrum[:chosen - 1].pow(2).sum()) / total
            assert previous < target + 1e-6, (target, chosen, previous)
    assert energy_rank(spectrum, 0.9, max_rank=2) == 2
    assert energy_rank(torch.zeros(3), 0.9) == 0

    drift = torch.randn(15, 8)
    automatic, kappa = compute_context_subspace(drift, 0)
    print('auto rank ->', tuple(automatic.shape), 'kappa', round(float(kappa), 4))
    assert automatic.shape[1] == 8 and 1 <= automatic.shape[0] <= 8
    assert 0.0 <= float(kappa) <= 1.0
    for fixed_rank in (1, 3, 4):
        basis, _ = compute_context_subspace(drift, fixed_rank)
        assert basis.shape == (fixed_rank, 8), basis.shape
    empty, _ = compute_context_subspace(torch.zeros(0, 8), 0)
    assert empty.numel() == 0

    # the two formulas the reliability fix depends on must stay in the source
    assert 'bounded = args.rel_floor + (1.0 - args.rel_floor) * soft_coverage' \
        in source, 'the floored continuous coverage is missing from the builder'
    assert 'mixed = args.soft_beta * s_ffcp + (1.0 - args.soft_beta) * s_res' \
        in source, 'the FFCP / CHSD blend is missing from the builder'
    print('[ok] builder math validated locally')
    return 0


if __name__ == '__main__':
    sys.exit(main())

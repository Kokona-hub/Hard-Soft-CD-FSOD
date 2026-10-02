# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Fast, dependency-light sanity checks for the S2H implementation.

Runs without any dataset or checkpoint -- only ``torch`` and the mmdet
registry (for module construction). It verifies

* the FFCP/CHSD numerical operators behave as expected,
* the ATAR correction is **zero at initialization** and respects its bound,
* the class-token broadcast is consistent with the class-logit averaging,
* the support-context interventions produce valid images.

Usage::

    python tools/s2h/selfcheck.py
"""
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from mmdet.models.utils.s2h_knowledge import (  # noqa: E402
    S2HKnowledgeBank, context_leakage_energy, l2_normalize, orthonormalize,
    project_l2_ball, project_out, svd_context_basis)
from support_interventions import (protection_mask,  # noqa: E402
                                   generate_interventions)


def check(cond, msg):
    status = 'ok  ' if cond else 'FAIL'
    print(f'[{status}] {msg}')
    if not cond:
        raise AssertionError(msg)


def test_operators():
    torch.manual_seed(0)
    x = torch.randn(5, 16)
    basis = orthonormalize(x)
    orth_error = (basis @ basis.t() -
                  torch.eye(basis.shape[0], device=basis.device)).abs().max()
    check(float(orth_error) < 1e-5,
          'orthonormalize returns an orthonormal basis')
    residual = project_out(x, basis)
    check(float(residual.norm()) < 1e-4,
          'project_out removes the whole span')

    drift = torch.randn(8, 16)
    u_ctx, _ = svd_context_basis(drift, rank=3)
    # Eq. (5) is normalized by the original foreground feature, not by the
    # drift itself. Keep a separate reference tensor to catch regressions in
    # the support-side knowledge builder.
    foreground = torch.randn_like(drift)
    energy = context_leakage_energy(drift, u_ctx, reference=foreground)
    check(torch.isfinite(energy) and float(energy) >= 0.0,
          f'context leakage energy is finite and non-negative (got {float(energy):.4f})')

    v = torch.randn(16) * 100
    p = project_l2_ball(v, radius=1.0)
    check(float(p.norm()) <= 1.0 + 1e-6, 'project_l2_ball respects the radius')


def test_knowledge_bank():
    torch.manual_seed(0)
    c, d = 4, 32
    bank = S2HKnowledgeBank(embed_dims=d, classes=[f'c{i}' for i in range(c)],
                            gamma=1.0, alpha_max=0.5, g_max=0.5, theta=0.5,
                            T_u=0.1)
    token_ids = [[1, 2], [3], [4, 5, 6], [7]]
    bank.set_knowledge(
        hard=l2_normalize(torch.randn(c, d), dim=-1),
        soft=l2_normalize(torch.randn(c, d), dim=-1),
        kappa=torch.rand(c),
        reliability=torch.rand(c),
        classes=[f'c{i}' for i in range(c)],
        token_ids=token_ids)
    check(bank.is_ready(), 'knowledge bank is ready after set_knowledge')

    hidden = torch.randn(2, 12, d)
    token_logits = torch.randn(2, 12, 16)
    class_logits = bank.class_logits_from_tokens(token_logits)
    check(class_logits.shape == (2, 12, c),
          'class logits have shape [B, N, C]')
    # averaging consistency: mean of tokens == class logit
    manual_prob = token_logits[..., token_ids[0]].sigmoid().mean(-1)
    manual = torch.logit(manual_prob.clamp(bank.eps, 1.0 - bank.eps))
    check(torch.allclose(class_logits[..., 0], manual, atol=1e-6),
          'class logit equals the token average')

    out = bank.atar(hidden, class_logits)
    delta = out['delta']
    bound = bank.gamma * bank.g_max * bank.alpha_max
    check(float(delta.abs().max()) <= bound + 1e-6,
          f'correction bounded by gamma*g_max*alpha_max (={bound})')
    check(float(delta.abs().max()) < 1e-3,
          f'correction ~0 at initialization (max={float(delta.abs().max()):.2e})')

    corrected = bank.broadcast_delta(token_logits, delta)
    for i, idxs in enumerate(token_ids):
        changed = corrected[..., idxs] - token_logits[..., idxs]
        if i == 0:
            check(torch.allclose(changed, delta[..., i:i + 1].expand_as(changed),
                                 atol=1e-6),
                  'broadcast adds the same delta to every token of a class')
    untouched = corrected[..., 9:10] - token_logits[..., 9:10]
    check(float(untouched.abs().max()) < 1e-6,
          'tokens of unrelated classes stay untouched')

    # gradient flows even though delta ~ 0 at init
    loss = delta.pow(2).sum()
    loss.backward()
    check(bank.raw_alpha.grad is not None and
          float(bank.raw_alpha.grad.abs().sum()) > 0,
          'alpha receives a non-zero gradient at initialization')
    check(bank.raw_g_cls.grad is not None and
          float(bank.raw_g_cls.grad.abs().sum()) > 0,
          'g_cls receives a non-zero gradient at initialization')


def test_stage_semantics():
    """Check that the three non-baseline stages are genuinely distinct."""
    torch.manual_seed(1)
    c, d = 3, 16
    hard = l2_normalize(torch.randn(c, d), dim=-1)
    soft = l2_normalize(torch.randn(c, d), dim=-1)
    kappa = torch.rand(c)
    reliability = torch.rand(c)
    token_ids = [[1], [2], [3]]
    hidden = torch.randn(1, 5, d)
    logits = torch.randn(1, 5, c)
    for stage in ('ffcp', 'ffcp_chsd', 'full'):
        bank = S2HKnowledgeBank(embed_dims=d, classes=[f'c{i}' for i in range(c)],
                                stage=stage)
        bank.set_knowledge(hard, soft, kappa, reliability,
                           [f'c{i}' for i in range(c)], token_ids)
        out = bank.atar(hidden, logits)
        if stage == 'ffcp':
            check(torch.allclose(out['agree'], torch.ones_like(out['agree'])),
                  'FFCP stage excludes the CHSD hard anchor')
        if stage != 'full':
            check(torch.allclose(out['u'], torch.ones_like(out['u'])),
                  f'{stage} stage disables ATAR uncertainty routing')
        else:
            check(torch.isfinite(out['u']).all(),
                  'full stage produces finite ATAR uncertainty')


def test_interventions():
    from PIL import Image
    img = Image.fromarray(
        (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8))
    boxes = [[10, 10, 30, 30]]
    mask = protection_mask(img.size, boxes, dilate=1.2)
    check(mask.shape == (64, 64) and mask.sum() > 0,
          'protection mask is non-empty')

    views = generate_interventions(
        img, boxes,
        same_domain_src=img.transpose(Image.FLIP_LEFT_RIGHT),
        class_mismatch_src=img.transpose(Image.FLIP_TOP_BOTTOM),
        dilate=1.2, rng=np.random.RandomState(0))
    check(set(views) == {'same_domain', 'class_mismatch', 'neutralized',
                         'reconstruction'},
          'all four support-context views are produced')
    for name, view in views.items():
        check(view.size == img.size and view.mode == 'RGB',
              f'{name} view keeps size/mode')

    # foreground must be preserved
    a = np.asarray(img.convert('RGB'), dtype=np.float32)
    b = np.asarray(views['class_mismatch'].convert('RGB'), dtype=np.float32)
    inside = mask
    check(np.abs(a[inside] - b[inside]).max() < 1e-6,
          'foreground pixels are frozen inside the protection mask')
    check(np.abs(a[~inside] - b[~inside]).max() > 0,
          'background pixels actually change')


def main():
    print('=== S2H self-check ===')
    test_operators()
    test_knowledge_bank()
    test_stage_semantics()
    test_interventions()
    print('\nAll S2H self-checks passed.')


if __name__ == '__main__':
    main()

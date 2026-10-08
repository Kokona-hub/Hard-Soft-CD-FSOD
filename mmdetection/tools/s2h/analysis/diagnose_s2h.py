#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Measure what the S2H correction actually does, without training anything.

The ablation says "the four stages are within +-0.5 mAP of the baseline", but
that only tells us the *outcome*.  This script opens the correction itself: for
one checkpoint and several knowledge configurations it records, over the test
images,

* ``mean|delta|`` / ``max|delta|``  - is the correction non-zero at all?
* ``mean alpha``                    - the learned class-wise strength
* ``mean u`` / ``active ratio``     - how many queries the router grants authority
* ``mean r_bar``                    - the reliability the knowledge file carries

and turns the numbers into one of three verdicts:

    delta ~ 0                     -> S2H never acted (the initial alpha of the
                                     earlier runs was 1.7e-4, and Eq. (13) keeps
                                     it there)
    delta non-zero, mAP flat      -> the knowledge / routing direction is wrong
    delta non-zero, mAP improves  -> the fix is in the training loop (loss or
                                     optimizer), not in the formulation

Pass the *same* checkpoint to every stage (``--checkpoint``): that isolates the
injection (knowledge + stage) from the detector weights, which is exactly the
first experiment the diagnosis asks for.  ``--checkpoint-map`` pairs every stage
with its *own* run instead (``baseline=a.pth,full=b.pth``), which is the faithful
comparison - a ``full`` checkpoint evaluated with the bank switched off is *not*
the baseline row of the ablation, because its weights were trained with the
correction active.  ``alpha`` and ``g`` are recomputed from the returned ``u`` /
``r_bar`` and the bank's own parameters, so nothing inside the model has to
change.

``baseline`` is a *reference* row: the bank is switched off for it by
construction, so it has no correction to measure and only ``det/img`` is
meaningful.  For the bank-carrying stages the script also compares the prompt
token table of the knowledge file with the one the checkpoint was trained with:
``_token_flat`` is a variable length buffer (one entry per prompt token of every
category), and ``load_state_dict`` silently skips it when the sizes differ, so a
knowledge file built from another prompt would leave the run measuring something
else without any visible error.

Usage
-----
    python tools/s2h/analysis/diagnose_s2h.py \\
        --config configs/s2h_dino/s2h_grounding_dino_swin-b_clipart1k_5shot.py \\
        --checkpoint cat_work_dir/ablation/clipart1k/full/seed3407/5shot/best_*.pth \\
        --knowledge-map "ffcp=work_dirs/s2h_knowledge/ablation/clipart1k/ffcp/seed3407/5shot.pth,\\
ffcp_chsd=work_dirs/s2h_knowledge/ablation/clipart1k/ffcp_chsd/seed3407/5shot.pth,\\
full=work_dirs/s2h_knowledge/ablation/clipart1k/full/seed3407/5shot.pth" \\
        --num-images 100 --json work_dirs/s2h_analysis/diagnose/clipart1k_5shot.json
"""
import argparse
import copy
import json
import os.path as osp
import sys

import numpy as np
from typing import Optional, Tuple

# The script is commonly launched as ``python tools/s2h/analysis/...`` from
# the mmdetection root.  Add that root explicitly; adding only ``analysis/``
# makes ``import mmdet`` fail on a clean server environment.
ANALYSIS_DIR = osp.abspath(osp.dirname(__file__))
MMDET_ROOT = osp.abspath(osp.join(ANALYSIS_DIR, '..', '..', '..'))
sys.path.insert(0, MMDET_ROOT)
sys.path.insert(0, ANALYSIS_DIR)


def assert_analysis_path(path: str) -> str:
    """Same guard as the other analysis scripts (never the ablation tree)."""
    parts = osp.abspath(path).replace('\\', '/').split('/')
    if 'ablation' in parts or not any(
            r in parts for r in ('s2h_analysis', 'sweep')):
        raise SystemExit(f'refusing to write to {path}')
    return path


# ---------------------------------------------------------------------------
# pure helpers (exercised by --self-check)
# ---------------------------------------------------------------------------
def parse_stage_map(text: str, fallback: str = None) -> dict:
    """``"ffcp=a.pth,full=b.pth"`` -> ``{'ffcp': 'a.pth', 'full': 'b.pth'}``.

    Shared by ``--knowledge-map`` and ``--checkpoint-map``.  A stage that is
    missing from the map falls back to ``fallback`` (the single ``--knowledge``
    / ``--checkpoint``), which is what a quick four-stage probe wants;
    ``baseline`` never needs a knowledge file because the bank is switched off
    for it.
    """
    mapping = {}
    for item in (text or '').replace(' ', '').split(','):
        if not item:
            continue
        if '=' not in item:
            raise SystemExit(
                f'--knowledge-map / --checkpoint-map expect stage=path, '
                f'got {item!r}')
        stage, path = item.split('=', 1)
        mapping[stage] = path
    if fallback:
        for stage in ('ffcp', 'ffcp_chsd', 'full'):
            mapping.setdefault(stage, fallback)
    return mapping


def parse_cfg_options(items) -> dict:
    """``['a.b=0.25', 'a.c=xy']`` -> ``{'a.b': 0.25, 'a.c': 'xy'}``.

    The values are cast the way ``tools/train.py`` casts ``--cfg-options``
    (number, then bool, then plain string): leaving them as strings would put a
    ``'0.25'`` into a float field, and the bank arithmetic would fail far away
    from the CLI.  Only ``key=value`` items are accepted, so a typo is reported
    here instead of turning into a silently ignored override.
    """
    options = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f'--cfg-options expects key=value, got {item!r}')
        key, raw = item.split('=', 1)
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                if raw.lower() in ('true', 'false'):
                    value = raw.lower() == 'true'
                elif raw.lower() in ('none', 'null'):
                    value = None
                else:
                    value = raw
        options[key] = value
    return options


def stage_options(stage: str, knowledge: str = None) -> dict:
    """``cfg_options`` for ``init_detector`` that put the head into ``stage``.

    The API wants a *dict* (``Config.merge_from_dict``), unlike the
    ``--cfg-options`` list of ``key=value`` strings that ``tools/train.py``
    takes; passing a list raises ``'list' object has no attribute 'items'``.
    """
    if stage == 'baseline':
        # a vanilla detector: no bank, no knowledge, no routing
        return {'model.bbox_head.s2h_cfg.enabled': False}
    options = {'model.bbox_head.s2h_cfg.enabled': True,
               'model.bbox_head.s2h_cfg.stage': stage}
    if knowledge:
        options['model.bbox_head.s2h_cfg.knowledge_path'] = knowledge
    return options


def expected_bank_sizes(shapes: Optional[dict]) -> Tuple[Optional[int],
                                                         Optional[int]]:
    """``(prompt tokens, classes)`` read off a checkpoint bank.

    ``_token_flat`` holds one index per prompt token of every category, so its
    length is the size of the *prompt*, not of the model; ``raw_alpha`` is the
    per-category strength.  ``(None, None)`` when the shapes are unknown, which
    disables the comparison instead of failing the run.
    """
    if not shapes:
        return None, None
    tokens = classes = None
    flat = shapes.get('_token_flat')
    if flat:
        tokens = int(flat[0])
    alpha = shapes.get('raw_alpha')
    if alpha:
        classes = int(alpha[0])
    return tokens, classes


def peek_checkpoint_bank(path: str) -> Optional[dict]:
    """Shape-only view of the S2H bank inside a checkpoint.

    Only the shapes are wanted, so the payload is dropped again right away; a
    checkpoint trained without the bank simply reports ``tokens=0``.  Every
    failure (no torch, unreadable file, unusual container) returns ``None``:
    this is a diagnostic of the diagnostic, and it must never be the reason a
    run dies.
    """
    import torch
    try:
        try:
            # torch>=2.6 defaults to weights_only=True, which can reject an
            # mmengine checkpoint container; the explicit retry keeps the check
            # working there and stays harmless where the flag does not exist
            payload = torch.load(path, map_location='cpu')
        except Exception:
            payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload.get('state_dict', payload)
        shapes = {key.split('s2h_bank.')[-1]: tuple(value.shape)
                  for key, value in state.items()
                  if 's2h_bank.' in key and hasattr(value, 'shape')}
        del payload, state
    except Exception:
        return None
    return shapes or None


DET_THRESHOLDS = (0.3, 0.5)


def prediction_scores(result) -> np.ndarray:
    """Scores of the boxes in one ``inference_detector`` result.

    The API returns a ``DetDataSample`` (or a one element ``SampleList``) whose
    ``pred_instances`` may be empty; the tensors may still live on the GPU, so
    the conversion is guarded.  These are the "boxes per image" numbers the
    DeepFish review asks for, and the only numbers a ``baseline`` row can
    contribute, because the bank is off for it.
    """
    sample = result[0] if isinstance(result, (list, tuple)) and result \
        else result
    instances = getattr(sample, 'pred_instances', None)
    scores = getattr(instances, 'scores', None)
    if scores is None:
        return np.zeros(0, np.float64)
    if hasattr(scores, 'detach'):                       # a torch tensor
        scores = scores.detach().cpu().numpy()
    return np.asarray(scores, np.float64).reshape(-1)


class Stats:
    """Accumulates the per-call statistics of the ATAR correction."""

    def __init__(self, stage: str):
        self.stage = stage
        self.calls = 0
        self.queries = 0
        self.delta_sum = 0.0
        self.delta_count = 0
        self.delta_max = 0.0
        self.u_sum = 0.0
        self.u_count = 0
        self.active = 0
        self.r_bar_sum = 0.0
        self.r_bar_count = 0
        self.alpha_sum = 0.0
        self.alpha_count = 0
        self.g_sum = 0.0
        self.g_count = 0
        self.det_sum = 0
        self.det_images = 0
        self.det_over = {threshold: 0 for threshold in DET_THRESHOLDS}

    def add(self, delta, u, r_bar, alpha, g_max, p0=None,
            g_values=None) -> None:
        delta = np.asarray(delta, np.float64)
        u = np.asarray(u, np.float64).reshape(-1)
        self.calls += 1
        self.queries += int(u.size)
        self.delta_sum += float(np.abs(delta).sum())
        self.delta_count += int(delta.size)
        self.delta_max = max(self.delta_max, float(np.abs(delta).max()))
        self.u_sum += float(u.sum())
        self.u_count += int(u.size)
        self.active += int((u > 0.5).sum())
        r_bar = np.asarray(r_bar, np.float64).reshape(-1)
        self.r_bar_sum += float(r_bar.sum())
        self.r_bar_count += int(r_bar.size)
        alpha = np.asarray(alpha, np.float64).reshape(-1)
        self.alpha_sum += float(alpha.sum())
        self.alpha_count += int(alpha.size)
        # Prefer the actual gate returned by ATAR.  It includes the configured
        # reliability exponent, which cannot be reconstructed from r_bar alone.
        if g_values is None:
            g_values = np.clip(np.asarray(u, np.float64)[:, None]
                               * np.asarray(r_bar, np.float64)[None, :],
                               0.0, g_max)
        gate = np.asarray(g_values, np.float64)
        self.g_sum += float(gate.sum())
        self.g_count += int(gate.size)
        if not hasattr(self, 'p0_values'):
            self.p0_values = []
            self.g_values = []
        if p0 is not None:
            self.p0_values.extend(np.asarray(p0, np.float64).reshape(-1).tolist())
        if g_values is not None:
            self.g_values.extend(gate.reshape(-1).tolist())

    def add_detections(self, scores, n_images: int = 1) -> None:
        """Accumulate the boxes-per-image profile of one image.

        The raw count is not enough: the Grounding DINO test pipeline runs at a
        very low score threshold behind a ``max_per_img`` cap, so it saturates
        (every image of the first run returned exactly the cap) and cannot show
        whether the correction changed anything.  The counts above a few score
        levels do not saturate, which is the false-positive profile the DeepFish
        review asks for.
        """
        scores = np.asarray(scores, np.float64).reshape(-1)
        self.det_sum += int(scores.size)
        self.det_images += int(n_images)
        for threshold in DET_THRESHOLDS:
            self.det_over[threshold] += int((scores >= threshold).sum())

    def summary(self) -> dict:
        # delta has one entry per (query, class) per call; the count is
        # accumulated directly, because multiplying the per-call totals would
        # square the number of calls
        delta_mean = (self.delta_sum / self.delta_count
                      if self.delta_count else 0.0)
        p0 = np.asarray(getattr(self, 'p0_values', []), np.float64)
        gv = np.asarray(getattr(self, 'g_values', []), np.float64)
        return dict(
            stage=self.stage, calls=self.calls, queries=int(self.queries),
            delta_mean_abs=delta_mean, delta_max_abs=self.delta_max,
            u_mean=(self.u_sum / self.u_count) if self.u_count else None,
            active_ratio=(self.active / self.u_count) if self.u_count else None,
            r_bar_mean=(self.r_bar_sum / self.r_bar_count)
            if self.r_bar_count else None,
            alpha_mean=(self.alpha_sum / self.alpha_count)
            if self.alpha_count else None,
            g_mean=(self.g_sum / self.g_count) if self.g_count else None,
            p0_mean=float(p0.mean()) if p0.size else None,
            p0_q05=float(np.quantile(p0, .05)) if p0.size else None,
            p0_q50=float(np.quantile(p0, .50)) if p0.size else None,
            p0_q95=float(np.quantile(p0, .95)) if p0.size else None,
            g_q05=float(np.quantile(gv, .05)) if gv.size else None,
            g_q50=float(np.quantile(gv, .50)) if gv.size else None,
            g_q95=float(np.quantile(gv, .95)) if gv.size else None,
            det_per_image=(self.det_sum / self.det_images)
            if self.det_images else None,
            det_over_image={f'det>{threshold:.1f}':
                            (self.det_over[threshold] / self.det_images)
                            if self.det_images else None
                            for threshold in DET_THRESHOLDS})

    def verdict(self, mAP: float = None, base_mAP: float = None) -> str:
        """One of the three outcomes the diagnosis distinguishes."""
        stats = self.summary()
        if self.stage == 'baseline':
            # the bank is switched off for the reference row by construction, so
            # an empty delta is the expected reading and not a failure: the row
            # exists to name what "no correction" looks like
            return ('reference: the bank is switched off for this stage, so '
                    'there is no correction to measure - only det/img is '
                    'meaningful, and only against the bank-carrying rows')
        if stats['delta_mean_abs'] < 1e-3 and stats['delta_max_abs'] < 1e-2:
            return ('delta ~ 0: the S2H correction never acted - check '
                    'init_alpha_bias / the regularizers, not FFCP or CHSD')
        if mAP is not None and base_mAP is not None:
            if mAP > base_mAP + 0.002:
                return ('delta is non-zero and the mAP improves: the injection '
                        'works on this checkpoint')
            return ('delta is non-zero but the mAP does not move: the knowledge '
                    'or the routing direction is the problem')
        return ('delta is non-zero: compare the mAP of the same checkpoint with '
                'tools/test.py to decide between "direction" and "training loop"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        'measure the S2H correction without training')
    parser.add_argument('--config', default=None)
    parser.add_argument('--checkpoint', default=None,
                        help='one checkpoint, used for every stage')
    parser.add_argument('--knowledge', default=None,
                        help='knowledge file used for every non-baseline stage')
    parser.add_argument('--knowledge-map', default='',
                        help="per-stage knowledge, e.g. "
                             "'ffcp=a.pth,ffcp_chsd=b.pth,full=c.pth'")
    parser.add_argument('--checkpoint-map', default='',
                        help="per-stage checkpoint, e.g. "
                             "'baseline=a.pth,ffcp=b.pth,full=c.pth'; stages "
                             'missing from the map use --checkpoint.  Pairing '
                             'every stage with its own run is the faithful '
                             'comparison, because a full-stage checkpoint '
                             'evaluated with the bank off is not the baseline '
                             'run of the ablation')
    parser.add_argument('--stages', nargs='+',
                        default=['baseline', 'ffcp', 'ffcp_chsd', 'full'])
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--cfg-options', nargs='*', default=None,
                        metavar='KEY=VALUE',
                        help='extra overrides merged into every stage, e.g. '
                             'model.bbox_head.s2h_cfg.g_max=0.25 - this is what '
                             'lets a hyper-parameter panel be measured without '
                             'retraining anything')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--json', default=None)
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def self_check() -> int:
    """Exercise the map parser, the option builder and the statistics."""
    mapping = parse_stage_map('ffcp=a.pth, full=b.pth', 'fallback.pth')
    assert mapping['ffcp'] == 'a.pth' and mapping['full'] == 'b.pth'
    assert mapping['ffcp_chsd'] == 'fallback.pth', mapping
    assert parse_stage_map('', 'x.pth')['full'] == 'x.pth'
    assert parse_stage_map('ffcp=a.pth') == {'ffcp': 'a.pth'}
    # the same parser backs --checkpoint-map: a per-stage baseline is kept, and
    # the single checkpoint is the fallback for everything else
    both = parse_stage_map('baseline=mine.pth', 'fallback.pth')
    assert both['baseline'] == 'mine.pth', both
    assert both['full'] == 'fallback.pth', both
    try:
        parse_stage_map('broken')
    except SystemExit:
        pass
    else:
        raise AssertionError('a malformed map was accepted')
    # (prompt tokens, categories) of a checkpoint bank, and the graceful
    # degradation that keeps a diagnostic from killing the run
    assert expected_bank_sizes(None) == (None, None)
    assert expected_bank_sizes({'_token_flat': (29,), 'raw_alpha': (20,)}) == \
        (29, 20)
    assert expected_bank_sizes({'_token_flat': (0,), 'raw_alpha': (1,)}) == (0, 1)
    assert expected_bank_sizes({'raw_alpha': (20,)}) == (None, 20)
    assert peek_checkpoint_bank('no-such-checkpoint.pth') is None
    # boxes per image, from either shape the inference API returns, and from a
    # GPU tensor (the scored must be moved off the device before numpy)
    class _Sample:
        def __init__(self, scores):
            self.pred_instances = type('Instances', (), {'scores': scores})()

    class _FakeTensor(list):
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self, np.float64)

    assert prediction_scores(_Sample([0.9, 0.4, 0.1])).size == 3
    assert prediction_scores(_Sample(_FakeTensor([0.9]))).size == 1
    assert prediction_scores([_Sample([])]).size == 0
    assert prediction_scores(None).size == 0
    # the raw count saturates on the max_per_img cap, so the score levels are
    # the part that can actually differ between stages
    profile = Stats('full')
    profile.add_detections(prediction_scores(_Sample([0.9, 0.4, 0.1])))
    profile.add_detections(prediction_scores(_Sample([0.35])))
    profile_summary = profile.summary()
    assert abs(profile_summary['det_per_image'] - 2.0) < 1e-9, profile_summary
    assert abs(profile_summary['det_over_image']['det>0.3'] - 1.5) < 1e-9, \
        profile_summary            # (0.9, 0.4, 0.35) over two images
    assert abs(profile_summary['det_over_image']['det>0.5'] - 0.5) < 1e-9, \
        profile_summary            # only 0.9
    # baseline is a reference row: an empty delta is the expected reading, not
    # a "the correction never acted" failure
    reference = Stats('baseline')
    reference.add_detections(prediction_scores(_Sample([0.9, 0.9])))
    reference.add_detections(prediction_scores(_Sample([0.9, 0.9])))
    assert reference.verdict().startswith('reference'), reference.verdict()
    assert abs(reference.summary()['det_per_image'] - 2.0) < 1e-9, \
        reference.summary()
    # init_detector wants a dict, not the list that --cfg-options takes
    assert stage_options('baseline') == {
        'model.bbox_head.s2h_cfg.enabled': False}
    full = stage_options('full', 'k.pth')
    assert full['model.bbox_head.s2h_cfg.knowledge_path'] == 'k.pth'
    assert full['model.bbox_head.s2h_cfg.stage'] == 'full'
    assert full['model.bbox_head.s2h_cfg.enabled'] is True

    stats = Stats('full')
    # two calls, three queries each, one class: delta = 0.1 exactly
    for _ in range(2):
        stats.add(delta=np.full((3, 1), 0.1), u=np.array([1.0, 0.0, 0.5]),
                  r_bar=np.array([0.8]), alpha=np.array([0.05]), g_max=0.5)
    summary = stats.summary()
    assert summary['calls'] == 2 and summary['queries'] == 6, summary
    assert abs(summary['delta_mean_abs'] - 0.1) < 1e-9, summary
    assert abs(summary['delta_max_abs'] - 0.1) < 1e-9, summary
    assert abs(summary['u_mean'] - 0.5) < 1e-9, summary
    assert abs(summary['active_ratio'] - 1.0 / 3.0) < 1e-9, summary
    assert abs(summary['alpha_mean'] - 0.05) < 1e-9, summary
    # g = clamp(u * r_bar, max=g_max): 1.0 * 0.8 is clipped to g_max = 0.5,
    # 0.0 stays 0.0, 0.5 * 0.8 = 0.4, so the mean is (0.5 + 0.0 + 0.4) / 3
    assert abs(summary['g_mean'] - (0.5 + 0.0 + 0.4) / 3.0) < 1e-9, summary
    assert stats.verdict().startswith('delta is non-zero'), stats.verdict()
    assert stats.verdict(mAP=0.60, base_mAP=0.55).startswith(
        'delta is non-zero and the mAP improves')
    assert stats.verdict(mAP=0.55, base_mAP=0.55).startswith(
        'delta is non-zero but the mAP does not move')

    silent = Stats('full')
    silent.add(delta=np.zeros((4, 2)), u=np.ones(4), r_bar=np.ones(1),
               alpha=np.zeros(1), g_max=0.5)
    assert silent.verdict().startswith('delta ~ 0'), silent.verdict()
    try:
        assert_analysis_path('cat_work_dir/ablation/x.json')
    except SystemExit:
        pass
    else:
        raise AssertionError('the guard let an ablation path pass')
    # --cfg-options values are cast, so a float field never receives a string
    assert parse_cfg_options(['a.b=0.25', 'a.c=2', 'a.d=true', 'a.e=xy']) == \
        {'a.b': 0.25, 'a.c': 2, 'a.d': True, 'a.e': 'xy'}
    assert parse_cfg_options(None) == {}
    try:
        parse_cfg_options(['broken'])
    except SystemExit:
        pass
    else:
        raise AssertionError('a malformed cfg option was accepted')
    # --checkpoint and --checkpoint-map are alternatives; requiring the former
    # unconditionally is what stopped a valid four-stage run before it started
    import argparse
    assert validate_inputs(argparse.Namespace(
        config='c.py', checkpoint='k.pth', checkpoint_map='')) is None
    assert validate_inputs(argparse.Namespace(
        config='c.py', checkpoint=None, checkpoint_map='full=k.pth')) is None
    assert validate_inputs(argparse.Namespace(
        config='c.py', checkpoint=None, checkpoint_map='')) == \
        '--checkpoint or --checkpoint-map is required'
    assert validate_inputs(argparse.Namespace(
        config=None, checkpoint='k.pth', checkpoint_map='')) == \
        '--config is required'
    print('[self-check] ok')
    return 0


def print_stage(summary: dict) -> None:
    """One line per stage, plus its verdict on the next line.

    Every field is read through ``get`` because a ``baseline`` row has no
    correction statistics (``None`` instead of a number), which ``_fmt``
    renders as ``None`` rather than crashing the table.
    """
    print(f"[diag] {summary['stage']:<9s} calls={summary['calls']:<4d} "
          f"queries={summary['queries']:<6d} "
          f"mean|delta|={_fmt(summary.get('delta_mean_abs'))} "
          f"max|delta|={_fmt(summary.get('delta_max_abs'))} "
          f"alpha={_fmt(summary.get('alpha_mean'))} "
          f"u={_fmt(summary.get('u_mean'))} "
          f"active={_fmt(summary.get('active_ratio'))} "
          f"r_bar={_fmt(summary.get('r_bar_mean'))} "
          f"det/img={_fmt(summary.get('det_per_image'))} "
          + ' '.join(f'{key}/img={_fmt(value)}'
                     for key, value in
                     (summary.get('det_over_image') or {}).items()))
    print(f'        -> {summary["verdict"]}')


def run(args) -> int:
    print(f'[diag] repo root: {MMDET_ROOT}')
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import inference_detector, init_detector
    from mmdet.registry import DATASETS

    knowledge_map = parse_stage_map(args.knowledge_map, args.knowledge)
    checkpoint_map = parse_stage_map(args.checkpoint_map, args.checkpoint)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset_cfg['lazy_init'] = False
    dataset = DATASETS.build(dataset_cfg)

    records = []
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        if not [i for i in info.get('instances', [])
                if not i.get('ignore_flag', 0)]:
            continue
        records.append(dict(img_path=info['img_path'],
                            text=info.get('text')
                            or tuple(dataset.metainfo['classes']),
                            custom_entities=bool(
                                info.get('custom_entities', True))))
        if args.num_images > 0 and len(records) >= args.num_images:
            break
    if not records:
        raise SystemExit('no test image with ground truth was found')
    print(f'[diag] {len(records)} images, stages '
          f'{", ".join(args.stages)}')

    report = {}
    for stage in args.stages:
        knowledge = knowledge_map.get(stage) if stage != 'baseline' else None
        if stage != 'baseline' and not knowledge:
            print(f'[warn] no knowledge for stage {stage}; skipping it')
            continue
        checkpoint = checkpoint_map.get(stage) or args.checkpoint
        if not checkpoint:
            # init_detector accepts checkpoint=None and would silently build a
            # randomly initialised detector, whose delta would be meaningless:
            # a stage that is in neither map has to be skipped out loud
            print(f'[warn] no checkpoint for stage {stage} (absent from '
                  '--checkpoint-map and --checkpoint was not given); '
                  'skipping it')
            continue
        tokens, classes = expected_bank_sizes(peek_checkpoint_bank(checkpoint))
        if tokens is not None:
            print(f'[diag] {stage:<9s} checkpoint bank: prompt tokens={tokens} '
                  f'categories={classes}')
        # a fresh Config per stage: init_detector() merges cfg_options into the
        # object it is handed, and MODELS.build() writes share_pred_layer /
        # num_pred_layer / as_two_stage into config.model.bbox_head *in place*
        # (deformable_detr.py).  Reusing one Config therefore trips the detector
        # assertion on the second stage, and would also carry the previous
        # stage's knowledge_path over into the next one.
        stage_cfg = Config.fromfile(args.config)
        options = stage_options(stage, knowledge)
        options.update(parse_cfg_options(args.cfg_options))
        model = init_detector(stage_cfg, checkpoint, device=args.device,
                              cfg_options=options)
        model.eval()
        bank = getattr(model.bbox_head, 's2h_bank', None)
        if stage == 'baseline':
            # `enabled=False` switches the bank off *on purpose*: baseline is the
            # no-correction reference, so demanding an active bank would make the
            # reference unmeasurable.  A baseline checkpoint carries an empty
            # token table, hence the size mismatch note that init_detector may
            # print when the checkpoint comes from a bank-carrying run - that is
            # the expected consequence of switching the bank off, not corruption.
            if bank is not None and model.bbox_head.s2h_active():
                raise SystemExit(
                    '[baseline] the bank is active although the stage switches '
                    'it off - the cfg option did not reach the head')
        else:
            if bank is None or not model.bbox_head.s2h_active():
                raise SystemExit(
                    f'stage {stage}: the S2H bank is not active.  The knowledge '
                    f'file ({knowledge}) is missing, unreadable or empty, or it '
                    'carries no category of this dataset')
            loaded = int(bank._token_flat.numel())
            if tokens is not None and loaded != tokens:
                print(f'[warn] {stage}: the knowledge file carries {loaded} '
                      f'prompt tokens but the checkpoint was trained with '
                      f'{tokens}; load_state_dict keeps the knowledge version, '
                      'so this row measures a different prompt than the '
                      'checkpoint was trained for - rebuild the knowledge file '
                      'with the same classes / attribute vocabulary')
            if classes is not None and bank.num_classes != classes:
                print(f'[warn] {stage}: the bank holds {bank.num_classes} '
                      f'categories, the checkpoint {classes}; alpha / r_bar '
                      'cannot be paired across that mismatch')

        stats = Stats(stage)
        original = bank.atar if (bank is not None and stage != 'baseline') \
            else None

        def probe(hidden_states, class_logits, _bank=bank, _stats=stats,
                  _original=original):
            out = _original(hidden_states, class_logits)
            with torch.no_grad():
                # Older server copies of s2h_knowledge.py do not return the
                # actual gate yet. Reconstruct it from the public ATAR values
                # so diagnosis remains usable while the code is synchronized.
                gate = out.get('g')
                if gate is None:
                    gate = (out['u'].unsqueeze(-1) *
                            out['r_bar'].pow(max(
                                float(getattr(_bank, 'reliability_power', 0.0)),
                                0.0)))
                    gate = gate.clamp(max=float(_bank.g_max))
                _stats.add(delta=out['delta'].detach().cpu().numpy(),
                           u=out['u'].detach().cpu().numpy(),
                           r_bar=out['r_bar'].detach().cpu().numpy(),
                           p0=out['p0'].detach().cpu().numpy(),
                           g_values=gate.detach().cpu().numpy(),
                           alpha=_bank.current_alpha().detach().cpu().numpy(),
                           g_max=_bank.g_max)
            return out

        if original is not None:
            bank.atar = probe
        try:
            with torch.no_grad():
                for record in records:
                    detections = inference_detector(
                        model, record['img_path'], text_prompt=record['text'],
                        custom_entities=record['custom_entities'])
                    stats.add_detections(prediction_scores(detections))
        finally:
            if original is not None:
                bank.atar = original

        summary = stats.summary()
        summary['knowledge'] = knowledge
        summary['checkpoint'] = checkpoint
        summary['verdict'] = stats.verdict()
        report[stage] = summary
        print_stage(summary)
        del model

    if args.json:
        path = assert_analysis_path(args.json)
        import os
        os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(dict(config=args.config, checkpoint=args.checkpoint,
                           images=len(records), stages=report),
                      handle, indent=2, ensure_ascii=False, default=float)
        print(f'[write] {path}')
    return 0


def _fmt(value):
    return 'None' if value is None else f'{value:.4f}'


def validate_inputs(args) -> Optional[str]:
    """``None`` when the CLI is usable, otherwise the message to raise.

    ``--checkpoint`` and ``--checkpoint-map`` are *alternatives*: the first is a
    single checkpoint used for every stage, the second names one per stage.
    Demanding the former unconditionally is what killed a valid four-stage run
    before it started.
    """
    if not args.config:
        return '--config is required'
    if not (args.checkpoint or args.checkpoint_map):
        return '--checkpoint or --checkpoint-map is required'
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    problem = validate_inputs(args)
    if problem:
        raise SystemExit(problem)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())

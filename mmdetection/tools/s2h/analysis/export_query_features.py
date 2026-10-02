#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Export the decoder query embeddings behind Figure 3 (t-SNE panels).

What it produces
----------------
One ``.npz`` per model variant, holding the final-decoder-layer query
embeddings together with everything the t-SNE figure needs to label them:

===========================  =================================================
``z``                        ``[N, D]`` last-layer query embedding (the same
                             tensor ATAR compares against the Soft prototype)
``z_clean``                  ``[N, D]`` the same embedding with the class
                             context subspace removed (panel 3); only present
                             when a context basis could be obtained
``label``                    ``[N]`` predicted category index of the query
``matched``                  ``[N]`` 1 when the query hits a ground truth of
                             the same category at IoU >= 0.5
``score``                    ``[N]`` category score of the query
``scores``                   ``[N, C]`` full per-category score vector of the
                             query, i.e. the distribution the detector acts on;
                             it yields the score margin
                             ``s_gt - max_{c != gt} s_c`` of a matched query
``query``                    ``[N]`` query slot inside its image, so panels of
                             different models can be restricted to the *same*
                             queries before they are compared
``box``                      ``[N, 4]`` query box in *original* image pixels
``image``                    ``[N]`` index into ``images`` (below)
``images``                   list of image paths that were scanned
``classes``                  list of category names (index order of ``label``)
===========================  =================================================

Design constraints (please keep them)
-------------------------------------
* **No pipeline file is modified.** The embeddings are captured with runtime
  hooks that are installed on the *instantiated model object* inside this
  process.  ``tools/train.py``, ``tools/test.py``, ``run_ablation.sh``,
  ``summarize_ablation.py`` and every config are untouched, so the reported
  mAP numbers cannot change.
* **No ablation artifact is written.** ``--out`` must live under
  ``s2h_analysis/`` or ``sweep/``; the script refuses anything else (see
  ``assert_analysis_path``), so a typo cannot overwrite a trained run.
* Inference goes through ``mmdet.apis.inference_detector`` with the config's
  own test pipeline, i.e. exactly the path used by ``tools/test.py``.

Usage
-----
::

    CFG=configs/s2h_dino/s2h_grounding_dino_swin-b_clipart1k_5shot.py

    # panel 1: vanilla Grounding DINO (the ablation "baseline" run)
    python tools/s2h/analysis/export_query_features.py --config $CFG \\
        --stage baseline --checkpoint $BASE_CKPT --num-images 100 \\
        --out work_dirs/s2h_analysis/tsne/clipart1k_5shot_seed3407/baseline.npz

    # panels 2+3: the full S2H-CD-FSOD model
    python tools/s2h/analysis/export_query_features.py --config $CFG \\
        --stage full --checkpoint $FULL_CKPT --knowledge $FULL_KNOW \\
        --context-basis auto --num-images 100 \\
        --out work_dirs/s2h_analysis/tsne/clipart1k_5shot_seed3407/full.npz

Both exports must use the same ``--num-images``/``--images`` so the t-SNE
panels share one feature subset.
"""
import argparse
import copy
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# tools/s2h/analysis/ -> the mmdet source tree, so that `import mmdet` works
# when the script is started straight from the repository (the same effect the
# other tools/s2h scripts get with their own two-level insertion).
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..',
                                        '..')))

MATCH_IOU = 0.5
# writing outside these roots is refused (see assert_analysis_path)
SAFE_ROOTS = ('s2h_analysis', 'sweep')


# ---------------------------------------------------------------------------
# small helpers (pure numpy -> exercised by --self-check)
# ---------------------------------------------------------------------------
def assert_analysis_path(path: str) -> str:
    """Refuse to write anywhere but the analysis / sweep roots.

    This is the guard that keeps the mAP experiments intact: the ablation
    knowledge lives in ``work_dirs/s2h_knowledge/ablation`` and every trained
    run in ``cat_work_dir/ablation``.
    """
    normalised = osp.abspath(path).replace('\\', '/')
    if 'ablation' in normalised.split('/'):
        raise SystemExit(
            f'refusing to write into an ablation directory: {path}\n'
            'use work_dirs/s2h_analysis/... or an explicit sweep/ path')
    if not any(root in normalised.split('/') for root in SAFE_ROOTS):
        raise SystemExit(
            f'{path} is not under one of {SAFE_ROOTS}; pass an analysis path '
            'such as work_dirs/s2h_analysis/tsne/...')
    return path


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``[N, 4]`` / ``[M, 4]`` xyxy boxes."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), np.float32)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * \
        np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * \
        np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_queries(query_boxes: np.ndarray, query_labels: np.ndarray,
                  gt_boxes: np.ndarray, gt_labels: np.ndarray,
                  iou_thr: float = MATCH_IOU) -> np.ndarray:
    """Class-aware one-to-one matching, highest IoU first.

    A query counts as matched only when it overlaps a still unmatched ground
    truth box *of the same category* -- the same rule the qualitative figures
    use, so "matched" means the same thing in both figures.
    """
    n_query = len(query_boxes)
    n_gt = len(gt_boxes)
    matched = np.zeros(n_query, dtype=bool)
    if n_query == 0 or n_gt == 0:
        return matched
    ious = iou_matrix(query_boxes, gt_boxes)
    same = np.asarray(query_labels, np.int64)[:, None] == \
        np.asarray(gt_labels, np.int64)[None, :]
    ious = np.where(same, ious, -1.0)
    gt_taken = np.zeros(n_gt, dtype=bool)
    flat = ious.reshape(-1)
    for index in np.argsort(flat)[::-1]:
        if flat[index] < iou_thr:
            break
        query_index, gt_index = divmod(int(index), n_gt)
        if matched[query_index] or gt_taken[gt_index]:
            continue
        matched[query_index] = True
        gt_taken[gt_index] = True
    return matched


def stage_cfg_options(stage: str, knowledge: Optional[str]) -> Dict[str, object]:
    """The same ``--cfg-options`` the ablation launcher uses."""
    if stage == 'baseline':
        return {'model.bbox_head.s2h_cfg.enabled': False}
    options = {
        'model.bbox_head.s2h_cfg.enabled': True,
        'model.bbox_head.s2h_cfg.stage': stage,
    }
    if knowledge:
        options['model.bbox_head.s2h_cfg.knowledge_path'] = knowledge
    return options


# ---------------------------------------------------------------------------
# capture: runtime hooks on the instantiated model (no source edits)
# ---------------------------------------------------------------------------
class QueryCapture:
    """Collect ``hidden_states`` / token logits / boxes of one forward pass.

    ``bbox_head.forward`` is a plain module ``__call__``, so ordinary module
    hooks are enough for the embeddings and the prediction tensors.  The token
    positive map lives on the *data sample*, so ``detector.predict`` is wrapped
    on the instance to stash it -- again in-process only.
    """

    def __init__(self, model):
        self.model = model
        self.head = model.bbox_head
        self.hidden: Optional[object] = None
        self.classes: Optional[object] = None
        self.coords: Optional[object] = None
        self.data_samples = None
        self._handles = [
            self.head.register_forward_pre_hook(self._pre),
            self.head.register_forward_hook(self._post),
        ]
        self._original_predict = model.predict
        capture = self

        def _predict(*args, **kwargs):
            # ``GroundingDINO.predict(batch_inputs, batch_data_samples)``: keep
            # the samples so the token -> category reduction can reuse the
            # detector's own map.  The map is written *inside* ``predict``
            # (``data_samples.token_positive_map = ...``), so the samples have to
            # be recognised by a field they already carry and the map is read
            # only after the call returned.  Assigned on the *instance*, so it
            # is called without an implicit `self` - hence the closure.
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, (list, tuple)) and value and \
                        hasattr(value[0], 'metainfo'):
                    capture.data_samples = value
                    break
            return capture._original_predict(*args, **kwargs)

        model.predict = _predict

    def _pre(self, module, args):
        self.hidden = args[0]

    def _post(self, module, args, output):
        self.classes, self.coords = output[0], output[1]

    def close(self):
        for handle in self._handles:
            handle.remove()
        self.model.predict = self._original_predict
        self.hidden = self.classes = self.coords = None
        self.data_samples = None


def query_boxes_original(coords, img_shape, scale_factor) -> np.ndarray:
    """Normalized cxcywh -> xyxy in original image pixels.

    Mirrors ``GroundingDINOHead._predict_by_feat_single`` (multiply by the
    resized ``img_shape``, then divide by ``scale_factor``).  The S2H test
    pipeline is ``FixScaleResize`` only -- there is no padding step, so this
    reproduces the localization path exactly.
    """
    import torch
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    boxes = bbox_cxcywh_to_xyxy(coords.float())
    boxes[:, 0::2] *= float(img_shape[1])
    boxes[:, 1::2] *= float(img_shape[0])
    scale = torch.as_tensor(scale_factor).float().reshape(-1)
    if scale.numel() == 4:          # some versions store (w, h, w, h)
        boxes /= scale.repeat(2)
    else:
        boxes[:, 0::2] /= scale[0]
        boxes[:, 1::2] /= scale[1]
    return boxes.cpu().numpy().astype(np.float32)


def per_query_category_scores(token_logits, token_positive_map):
    """Token logits -> per-query category probabilities, as the head does it."""
    import torch
    from mmdet.models.dense_heads.grounding_dino_head import (
        convert_grounding_to_cls_scores)

    with torch.no_grad():
        scores = convert_grounding_to_cls_scores(
            logits=token_logits[None].sigmoid(),
            positive_maps=[token_positive_map])[0]
    return scores.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# context subspace (only needed for the third panel)
# ---------------------------------------------------------------------------
def _load_builder_module():
    """Import ``tools/s2h/build_s2h_knowledge.py`` without executing ``main``.

    The builder imports its own siblings flat (``from support_interventions
    import generate_interventions``), which Python resolves only when the script
    is started as ``python tools/s2h/build_s2h_knowledge.py``.  Loading it by
    path therefore requires its directory on ``sys.path`` first, so it sees
    exactly what it would see when run directly.  Its source is not modified.
    """
    import importlib.util
    path = osp.abspath(osp.join(osp.dirname(__file__), '..',
                                'build_s2h_knowledge.py'))
    builder_dir = osp.dirname(path)
    if builder_dir not in sys.path:
        sys.path.insert(0, builder_dir)
    spec = importlib.util.spec_from_file_location('s2h_builder', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['s2h_builder'] = module
    spec.loader.exec_module(module)
    return module


def context_basis_for_classes(config: str, checkpoint: str, knowledge: str,
                              rank: Optional[int], device: str) -> Dict:
    """Re-derive ``U_c^ctx`` for every class of a support split.

    FFCP is deterministic given ``(support split, seed, rank, dilation)``, so
    re-running the builder's own helpers with the *same* arguments recorded in
    the knowledge file reproduces the subspace CHSD actually removed.  Nothing
    is written: this is a read-only recomputation used for the third panel.
    """
    import torch
    from mmengine.config import Config
    from mmdet.apis import init_detector

    builder = _load_builder_module()
    payload = torch.load(knowledge, map_location='cpu')
    meta = payload.get('meta', {}) or {}
    build_args = dict(meta.get('args', {}) or {})

    cfg = Config.fromfile(config)
    if 'bbox_head' in cfg.model:
        s2h = cfg.model.bbox_head.get('s2h_cfg', None)
        if s2h is not None:
            s2h.enabled = False
    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    classes, per_class = builder.build_dataset(cfg, 0)
    non_empty = [c for c in per_class if per_class[c]]
    pipe = builder.build_preprocess_pipeline()

    class _Args:
        pass

    args = _Args()
    args.dilate = float(build_args.get('dilate', 1.2))
    args.neutral_mode = str(build_args.get('neutral_mode', 'gray'))
    args.seed = int(build_args.get('seed', 0))
    args.ffcp_rank = int(rank if rank is not None
                         else build_args.get('ffcp_rank', 4))

    basis = {}
    for c in range(len(classes)):
        samples = per_class[c]
        if not samples:
            continue
        mism = None
        for other in non_empty:
            if other != c:
                mism = per_class[other][0]['img_path']
                break
        desc = dict(class_id=c, samples=samples, mismatch_src=mism,
                    same_domain_src=mism)
        try:
            u_ctx = _context_basis_single(builder, model, desc, args, pipe,
                                         device)
        except Exception as error:      # pragma: no cover - defensive
            print(f'[warn] context basis for {classes[c]} failed: {error}')
            continue
        if u_ctx is not None and len(u_ctx):
            basis[c] = np.asarray(u_ctx, dtype=np.float32)
    del model
    return dict(basis=basis, classes=list(classes),
                build_args=build_args, rank=args.ffcp_rank)


def _context_basis_single(builder, model, desc, args, pipe, device):
    """Eqs. (4)-(5) for one class, mirroring ``build_class_knowledge``."""
    import torch
    from PIL import Image

    c = desc['class_id']
    samples = desc['samples']
    rng = np.random.RandomState(args.seed + c)
    f_fg, f_rec, tilde = [], [], {'same_domain': [], 'class_mismatch': [],
                                  'neutralized': []}
    for k, sample in enumerate(samples):
        img = Image.open(sample['img_path']).convert('RGB')
        boxes = sample['boxes']
        same_src = None
        if len(samples) > 1:
            same_src = Image.open(samples[(k + 1) % len(samples)]['img_path']
                                  ).convert('RGB')
        elif desc.get('same_domain_src') is not None:
            same_src = Image.open(desc['same_domain_src']).convert('RGB')
        diff_src = desc.get('mismatch_src')
        if diff_src is not None:
            diff_src = Image.open(diff_src).convert('RGB')
        views = builder.generate_interventions(
            img, boxes, same_domain_src=same_src, class_mismatch_src=diff_src,
            dilate=args.dilate, neutral_mode=args.neutral_mode, rng=rng)
        f0 = builder.extract_box_features(model, img, boxes, img.size, pipe,
                                          device)
        if f0 is None:
            continue
        f_fg.append(f0)
        for key in ('same_domain', 'class_mismatch', 'neutralized'):
            ft = builder.extract_box_features(model, views[key], boxes,
                                              img.size, pipe, device)
            if ft is not None:
                tilde[key].append(ft)
        fr = builder.extract_box_features(model, views['reconstruction'],
                                          boxes, img.size, pipe, device,
                                          dilate=1.0)
        if fr is not None:
            f_rec.append(fr)
    if not f_fg:
        return None
    f_fg = torch.cat(f_fg, 0)
    f_rec = torch.cat(f_rec, 0) if f_rec else f_fg.clone()
    n_base = min(f_fg.shape[0], f_rec.shape[0])
    f_fg, f_rec = f_fg[:n_base], f_rec[:n_base]
    drifts, aligned_fg = [], []
    for key in ('same_domain', 'class_mismatch', 'neutralized'):
        ft = torch.cat(tilde[key], 0)[:n_base] if tilde[key] else f_fg.clone()
        n = min(f_fg.shape[0], ft.shape[0], f_rec.shape[0])
        drifts.append((f_fg[:n] - ft[:n]) - (f_fg[:n] - f_rec[:n]))
        aligned_fg.append(f_fg[:n])
    u_ctx, _ = builder.compute_context_subspace(
        torch.cat(drifts, 0), args.ffcp_rank, reference=torch.cat(aligned_fg, 0))
    return u_ctx.cpu().numpy() if u_ctx is not None else None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _require(args, *names) -> None:
    """Validate arguments *after* the ``--self-check`` shortcut."""
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise SystemExit('missing required argument(s): '
                         + ', '.join('--' + n.replace('_', '-')
                                     for n in missing))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        'Export decoder query embeddings for the Fig. 3 t-SNE panels')
    parser.add_argument('--config', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--knowledge', default=None,
                        help='cached knowledge of a non-baseline stage')
    parser.add_argument('--stage', default='full',
                        choices=['baseline', 'ffcp', 'ffcp_chsd', 'full'])
    parser.add_argument('--out', default=None, help='output .npz path')
    parser.add_argument('--images', default=None,
                        help='optional text file with one image path per line')
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='a query below this score is never exported')
    parser.add_argument('--max-per-class', type=int, default=200,
                        help='cap of matched queries kept per category')
    parser.add_argument('--max-unmatched-per-class', type=int, default=60)
    parser.add_argument('--context-basis', default='auto',
                        help="'auto' re-derives U_ctx from the support split "
                             "(third panel), 'off' skips it, or a .npz/.pth "
                             'holding a precomputed basis')
    parser.add_argument('--rank', type=int, default=None,
                        help='FFCP rank used when re-deriving the basis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def self_check() -> int:
    """CPU-only checks of the geometry and the path guard."""
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    labels = np.array([0, 1], np.int64)
    queries = np.array([[0, 0, 10, 10], [100, 100, 110, 110],
                        [300, 300, 310, 310]], np.float32)
    matched = match_queries(queries, np.array([0, 1, 0], np.int64), gt, labels)
    assert matched.tolist() == [True, True, False], matched
    # wrong-class overlap must not count as a match
    wrong = match_queries(queries[:1], np.array([1], np.int64), gt, labels)
    assert wrong.tolist() == [False], wrong
    # a second box on an already matched object is not a second match
    dup = match_queries(np.array([[0, 0, 10, 10], [1, 1, 9, 9]], np.float32),
                        np.array([0, 0], np.int64), gt[:1], labels[:1])
    assert dup.tolist() == [True, False], dup
    assert abs(iou_matrix(gt[:1], gt[:1])[0, 0] - 1.0) < 1e-6
    # the writer guard must reject ablation paths and accept analysis ones
    for bad in ('cat_work_dir/ablation/clipart1k/full/seed3407/5shot/x.npz',
                'work_dirs/s2h_knowledge/ablation/clipart1k/x.npz'):
        try:
            assert_analysis_path(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f'guard let {bad} through')
    ok = assert_analysis_path('work_dirs/s2h_analysis/tsne/a.npz')
    assert ok.endswith('a.npz')

    # the query capture has to cope with two facts that broke an earlier
    # version: the samples must be recognised *before* ``predict`` is called,
    # while ``token_positive_map`` only appears *inside* it
    class _Sample:
        """Stand-in for ``DetDataSample``: unknown attributes go to metainfo."""

        def __init__(self, **metainfo):
            object.__setattr__(self, 'metainfo', dict(metainfo))

        def __setattr__(self, name, value):
            self.metainfo[name] = value

    class _Handle:
        def remove(self):
            self.removed = True

    class _Head:
        def register_forward_pre_hook(self, hook):
            return _Handle()

        def register_forward_hook(self, hook):
            return _Handle()

    class _Detector:
        def __init__(self):
            self.bbox_head = _Head()

        def predict(self, batch_inputs, batch_data_samples):
            # GroundingDINO writes the prompt -> token map at this point, i.e.
            # after the capture has already inspected its arguments
            assert 'token_positive_map' not in batch_data_samples[0].metainfo
            batch_data_samples[0].token_positive_map = {0: (0, 2), 1: (3, 5)}
            return batch_data_samples

    detector = _Detector()
    capture = QueryCapture(detector)
    probes = [_Sample(img_shape=(800, 1333), scale_factor=(1.0, 1.0))]
    detector.predict([[0]], probes)
    assert capture.data_samples is probes, capture.data_samples
    assert probes[0].metainfo['token_positive_map'] == {0: (0, 2), 1: (3, 5)}
    capture.close()
    assert getattr(detector.predict, '__self__', None) is detector
    print('[self-check] ok')
    return 0


def collect_images(dataset, args) -> List[dict]:
    """Fixed image list: sorted stems, optionally overridden by --images."""
    wanted = None
    if args.images:
        with open(args.images, encoding='utf-8') as handle:
            wanted = [line.strip() for line in handle if line.strip()]
    records = []
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        instances = [ins for ins in info.get('instances', [])
                     if not ins.get('ignore_flag', 0)]
        if not instances:
            continue
        records.append(dict(
            index=index, img_path=info['img_path'],
            name=osp.splitext(osp.basename(info['img_path']))[0],
            text=info.get('text') or tuple(dataset.metainfo['classes']),
            custom_entities=bool(info.get('custom_entities', True)),
            gt_boxes=np.asarray([ins['bbox'] for ins in instances],
                                np.float32),
            gt_labels=np.asarray([int(ins['bbox_label']) for ins in instances],
                                 np.int64)))
    if wanted is not None:
        keep = {osp.splitext(osp.basename(w))[0] for w in wanted}
        records = [r for r in records if r['name'] in keep]
        if not records:
            raise SystemExit('--images matched no image of the test split')
    records.sort(key=lambda r: r['name'])
    if args.num_images > 0:
        records = records[:args.num_images]
    if not records:
        raise SystemExit('no test image with ground truth was found')
    return records


def run(args) -> int:
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import inference_detector, init_detector
    from mmdet.registry import DATASETS

    _require(args, 'config', 'checkpoint', 'out')
    assert_analysis_path(args.out)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset_cfg['lazy_init'] = False
    dataset = DATASETS.build(dataset_cfg)
    classes = [str(name) for name in dataset.metainfo['classes']]

    records = collect_images(dataset, args)
    print(f'[data] {args.stage}: {len(records)} query images, '
          f'{len(classes)} categories')

    model = init_detector(cfg, args.checkpoint, device=args.device,
                          cfg_options=stage_cfg_options(args.stage,
                                                        args.knowledge))
    model.dataset_meta = {'classes': classes}
    model.eval()
    capture = QueryCapture(model)

    basis = {}
    want_basis = args.context_basis not in (None, '', 'off')
    if want_basis and args.stage == 'baseline':
        # the vanilla detector carries no S2H bank, so there is no context
        # subspace to remove: the third panel only exists for the full model
        print('[basis] stage=baseline has no S2H bank - the context panel is '
              'only computed for a non-baseline stage, exporting one panel')
        want_basis = False
    if want_basis and args.context_basis == 'auto':
        if not args.knowledge:
            print('[warn] --context-basis auto needs --knowledge; skipping '
                  'the third panel')
        else:
            print('[basis] re-deriving the FFCP context subspace ...')
            info = context_basis_for_classes(args.config, args.checkpoint,
                                             args.knowledge, args.rank,
                                             args.device)
            basis = info['basis']
            print(f'[basis] {len(basis)}/{len(classes)} classes, '
                  f'rank={info["rank"]}, args={info["build_args"]}')
    elif want_basis:
        payload = np.load(args.context_basis, allow_pickle=True)
        basis = {int(k): np.asarray(v, np.float32)
                 for k, v in dict(payload['basis'].item()).items()}

    rng = np.random.RandomState(args.seed)
    rows = {k: [] for k in ('z', 'z_clean', 'label', 'matched', 'score',
                            'scores', 'query', 'box', 'image')}
    kept = {c: [0, 0] for c in range(len(classes))}

    for position, record in enumerate(records):
        result = inference_detector(model, record['img_path'],
                                    text_prompt=record['text'],
                                    custom_entities=record['custom_entities'])
        if capture.hidden is None or capture.coords is None:
            raise SystemExit('hooks captured nothing - is this an S2H '
                             'Grounding DINO checkpoint?')
        samples = capture.data_samples
        meta = samples[0].metainfo if samples else {}
        # the detector stores the prompt -> token map on the sample; it lives in
        # ``metainfo``, a plain dict, so no attribute access is involved here
        positive_map = meta.get('token_positive_map')
        if positive_map is None and samples:
            positive_map = getattr(samples[0], 'token_positive_map', None)
        if positive_map is None:
            raise SystemExit('the detector produced no token_positive_map for '
                             'this sample; cannot reduce tokens to categories')

        img_shape = tuple(int(v) for v in meta['img_shape'][:2])
        scale_factor = meta['scale_factor']
        hidden = capture.hidden[-1][0].float()          # [num_queries, D]
        coords = capture.coords[-1][0].float()
        tokens = capture.classes[-1][0].float()

        boxes = query_boxes_original(coords, img_shape, scale_factor)
        scores = per_query_category_scores(tokens, positive_map)
        labels = scores.argmax(-1).astype(np.int64)
        top = scores.max(-1)
        matched = match_queries(boxes, labels, record['gt_boxes'],
                                record['gt_labels'])
        hidden_np = hidden.cpu().numpy()

        order = np.argsort(-top)
        for index in order:
            score = float(top[index])
            if score < args.score_thr:
                break
            category = int(labels[index])
            is_matched = bool(matched[index])
            budget = args.max_per_class if is_matched else \
                args.max_unmatched_per_class
            if kept[category][int(not is_matched)] >= budget:
                continue
            kept[category][int(not is_matched)] += 1
            rows['z'].append(hidden_np[index])
            rows['label'].append(category)
            rows['matched'].append(is_matched)
            rows['score'].append(score)
            # the full per-class distribution of this query: the score margin
            # `s_gt - max_{c != gt} s_c` is what the detector actually acts on,
            # and it is computed from these numbers rather than from a cosine
            rows['scores'].append(scores[index].astype(np.float32))
            # query slot inside the image, so panels of different models can be
            # restricted to the *same* queries before they are compared
            rows['query'].append(int(index))
            rows['box'].append(boxes[index])
            rows['image'].append(position)
            if category in basis and basis[category].size:
                vector = hidden_np[index]
                for normal in basis[category]:
                    vector = vector - float(vector @ normal) * normal
                rows['z_clean'].append(vector)
            else:
                rows['z_clean'].append(np.full_like(hidden_np[index], np.nan))
        if (position + 1) % 25 == 0:
            print(f'[scan] {position + 1}/{len(records)} images, '
                  f'{len(rows["z"])} queries kept')

    capture.close()
    if not rows['z']:
        raise SystemExit('no query passed --score-thr; lower it and retry')

    payload = dict(
        z=np.stack(rows['z']), label=np.asarray(rows['label'], np.int64),
        matched=np.asarray(rows['matched'], bool),
        score=np.asarray(rows['score'], np.float32),
        scores=np.stack(rows['scores']),
        query=np.asarray(rows['query'], np.int64),
        box=np.stack(rows['box']), image=np.asarray(rows['image'], np.int64),
        classes=np.asarray(classes, dtype=object),
        images=np.asarray([r['img_path'] for r in records], dtype=object),
        stage=args.stage, score_thr=args.score_thr, match_iou=MATCH_IOU,
        has_context=bool(basis),
        meta=json.dumps(dict(config=args.config, checkpoint=args.checkpoint,
                             knowledge=args.knowledge, seed=args.seed,
                             num_images=len(records))))
    clean = np.stack(rows['z_clean'])
    if not basis:
        clean = np.full_like(payload['z'], np.nan)
    payload['z_clean'] = clean

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    n_matched = int(payload['matched'].sum())
    print(f'[write] {args.out}  queries={len(payload["z"])} '
          f'(matched={n_matched}, unmatched={len(payload["z"]) - n_matched}) '
          f'context_panel={"on" if basis else "off"}')
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    torch_missing = False
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_missing = True
    if torch_missing:
        raise SystemExit('PyTorch is required; run inside the training env '
                         '(PYTHONNOUSERSITE=1)')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Qualitative ``Ground Truth | Baseline | FFCP | FFCP+CHSD | Full`` figures.

This script is **read-only with respect to the trained runs**: it loads the
checkpoints and the cached knowledge produced by ``tools/s2h/run_ablation.sh``,
runs the exact same inference path as ``tools/test.py`` (config test pipeline +
``custom_entities`` prompt from the dataset) and renders the paper figure.
Nothing in the evaluation / metric code is touched, so the reported COCO
numbers stay exactly reproducible.

For every dataset/shot it emits

* ``<out>/<dataset>_<shot>shot.png``          all selected rows in one figure
* ``<out>/<dataset>_<shot>shot_row<k>.png``   one sample per figure (paper use)
* ``<out>/<dataset>_<shot>shot.json``         every score behind the captions
* ``<out>/<dataset>_<s1>shot_<s2>shot.png``  one dataset across shot budgets
* ``<out>/all_datasets_<shot>shot.png``      every dataset stacked

Rows are ranked so that samples with an **ascending** chain come first: a
ground-truth box whose score grows stage after stage
(``baseline <= ffcp <= ffcp_chsd <= full``) is the strongest signal, ordered by
how much it climbs, and the caption of every row reports exactly that box.
``--require-rising`` turns the preference into a filter; if a dataset has no
ascending chain the level is relaxed step by step down to
``--min-requirement`` (default ``full>baseline``, i.e. a row is never shown
when Full is not better than the baseline).  Each run also prints ``[stats]``
lines with the best stage over the scanned candidates and how many of them have
an ascending chain (``--num-images 0`` prints only the statistics).

A figure should not showcase wrong detections either, so every candidate is
scored against the ground truth with a class-aware, one-to-one matching
(IoU >= 0.5 on the boxes above ``--score-thr``, i.e. on the boxes that are
drawn): a wrong-class hit or a duplicate box on a real object counts as a false
positive.  ``--require-clean-rising`` then keeps only the samples whose last
drawn stage adds **no** wrong box (``--max-fp-delta``, default 0) and loses
none of the detections the baseline already had, ``--min-precision`` adds a
precision floor, and ``--rank-by clean`` orders the survivors by new wrong
boxes first and gained true positives second.  ``[stats]`` prints the summed
``TP``/``FP`` of both stages so the selection is auditable.

By default a figure only carries the text that identifies what is drawn: the
dataset name at the left of every row, the stage name above every column, and
the box chips (the ground truth shows its class name, a prediction shows
``class score``).  The per-row score chain, the mAP values and the figure title
stay off - switch them on with ``--show-caption``, ``--show-map`` (reads each
run's ``vis_data/scalars.json`` at the epoch of the checkpoint that is actually
loaded) and ``--show-subtitle``.  ``--no-show-labels`` draws the boxes without
any text, ``--gt-header ''`` drops the first column header.

Usage
-----
::

    # the paper figure: the three datasets, one row each (5-shot by default)
    python tools/s2h/visualize_ablation.py --datasets clipart1k deepfish NEU-DET

    # check the 5-shot runs exist before spending GPU time
    python tools/s2h/visualize_ablation.py --list-checkpoints \
        --datasets clipart1k deepfish NEU-DET

    # other shot budgets, or several at once (rows are labelled with the shot)
    python tools/s2h/visualize_ablation.py --shots 10
    python tools/s2h/visualize_ablation.py --shots 1 5 --align-shots

    # only samples whose score climbs baseline -> ffcp -> ffcp+chsd -> full
    python tools/s2h/visualize_ablation.py --datasets clipart1k deepfish NEU-DET \
        --max-candidates 150 --require-rising --min-lift 0.05 --list-top 20

    # ... and when a dataset has no ascending chain, fall back to the floor
    # "Full must beat Baseline" instead of dropping the dataset.  `>` would be
    # read as a shell redirection, hence either quote it or use the alias.
    python tools/s2h/visualize_ablation.py --datasets clipart1k deepfish NEU-DET \
        --max-candidates 150 --require-rising --min-requirement improvement
    # python tools/s2h/visualize_ablation.py --min-requirement 'full>baseline'

    # tighter: both an ascending chain and Full as the best stage
    python tools/s2h/visualize_ablation.py --datasets NEU-DET \
        --max-candidates 150 --require-rising --require-full-best

    # compact three column version (Ground Truth | Baseline | Full)
    python tools/s2h/visualize_ablation.py --columns baseline,full

    # scan more candidates / force a specific class or image
    python tools/s2h/visualize_ablation.py --datasets NEU-DET \
        --max-candidates 80 --classes crazing
    python tools/s2h/visualize_ablation.py --datasets clipart1k \
        --img-names bike_001,bench_022

    # first inspect the ranking, then render the samples you liked
    python tools/s2h/visualize_ablation.py --datasets clipart1k \
        --max-candidates 150 --list-top 20 --num-images 0

    # check which checkpoint every stage resolves to (no dataset, no model)
    python tools/s2h/visualize_ablation.py --list-checkpoints --datasets NEU-DET

    # only rank rows where some stage actually detects something
    python tools/s2h/visualize_ablation.py --datasets NEU-DET \
        --max-candidates 120 --min-best-score 0.3 --list-top 10

    # cheap smoke test that touches no GPU and no checkpoint
    python tools/s2h/visualize_ablation.py --self-check

The script re-launches itself with ``PYTHONNOUSERSITE=1`` when a user-site
PyTorch would shadow the conda one (see ``tools/s2h/WEEKLY_REPORT.md``), so it
can be started like the other tools on the shared server.
"""
import argparse
import copy
import json
import os
import os.path as osp
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# make the mmdet source tree importable when running from the repo root
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))


# ---------------------------------------------------------------------------
# environment isolation (the same requirement as tools/dist_train.sh)
# ---------------------------------------------------------------------------
def shadowing_user_site() -> str:
    """Path of a user site-packages that holds a second PyTorch, else ``''``.

    ``tools/s2h/WEEKLY_REPORT.md`` documents this on the shared server: a
    user-site PyTorch (2.8 / CUDA 12.8) overrides the conda one and mmcv's
    compiled extensions then abort with::

        mmcv/_ext...so: undefined symbol: ..._ops...zeros_like...
    """
    try:
        import site
    except ImportError:  # pragma: no cover - always available in CPython
        return ''
    user_site = site.getusersitepackages()
    if not user_site or not osp.isdir(user_site):
        return ''
    for package in ('torch', 'torchvision'):
        if osp.isdir(osp.join(user_site, package)):
            return user_site
    return ''


def ensure_conda_torch(argv: Sequence[str]) -> None:
    """Re-launch with ``PYTHONNOUSERSITE=1`` when ~/.local shadows PyTorch.

    ``PYTHONNOUSERSITE`` is only honoured at interpreter start-up, so it has to
    be fixed by starting a fresh process (exactly what ``dist_train.sh`` does
    through the environment).
    """
    if os.environ.get('PYTHONNOUSERSITE') == '1' or \
            os.environ.get('S2H_VIS_REEXEC') == '1':
        return
    user_site = shadowing_user_site()
    if not user_site:
        return
    env = dict(os.environ)
    env['PYTHONNOUSERSITE'] = '1'
    env['S2H_VIS_REEXEC'] = '1'
    print(f'[env] {user_site} contains another PyTorch; re-launching with '
          'PYTHONNOUSERSITE=1 so that the conda build of mmcv is used')
    try:
        os.execve(sys.executable, [sys.executable] + list(argv), env)
    except OSError as exc:  # pragma: no cover - platform dependent
        print(f'[env] re-launch failed ({exc}); continuing in this process')


def check_framework_imports() -> None:
    """Import the compiled stack once, with an actionable error message."""
    try:
        import mmcv
        from mmcv import ops  # noqa: F401  (loads mmcv._ext)
        import mmengine
        import mmdet
        import torch
    except ImportError as exc:
        if 'undefined symbol' in str(exc):
            raise SystemExit(
                'mmcv was built against a different PyTorch than the one that '
                'is imported now:\n  ' + str(exc).strip().splitlines()[-1] +
                '\n\nRe-run inside the conda environment with user '
                'site-packages disabled:\n'
                '  PYTHONNOUSERSITE=1 python tools/s2h/visualize_ablation.py '
                '...\n'
                '(see tools/s2h/WEEKLY_REPORT.md - always keep '
                'PYTHONNOUSERSITE=1 on this server)')
        raise
    print(f'[env] python {sys.version.split()[0]} | torch {torch.__version__} '
          f'({osp.dirname(torch.__file__)}) | mmcv {mmcv.__version__} | '
          f'mmengine {mmengine.__version__} | mmdet {mmdet.__version__}')


# ---------------------------------------------------------------------------
# ablation layout (must mirror tools/s2h/run_ablation.sh)
# ---------------------------------------------------------------------------
STAGES = ('baseline', 'ffcp', 'ffcp_chsd', 'full')
STAGE_TITLES = {
    'baseline': 'Baseline',
    'ffcp': 'FFCP',
    'ffcp_chsd': 'FFCP+CHSD',
    'full': 'Full S2H-CD-FSOD',
}
def column_titles(stages: Sequence[str],
                  gt_header: str = 'Ground Truth') -> Tuple[str, ...]:
    """Header of every drawn column (the ground truth is always first).

    ``gt_header=''`` leaves the first column unlabelled, which is useful when
    the figure should only show the dataset name and the stage names
    (pass ``--gt-header ''``).
    """
    return (gt_header,) + tuple(STAGE_TITLES[s] for s in stages)


STAGE_ALIASES = {
    'baseline': 'baseline', 'base': 'baseline',
    'ffcp': 'ffcp',
    'ffcp_chsd': 'ffcp_chsd', 'ffcp+chsd': 'ffcp_chsd', 'ffcp-chsd': 'ffcp_chsd',
    'chsd': 'ffcp_chsd',
    'full': 'full', 's2h': 'full',
}


def parse_stages(text: str) -> Tuple[str, ...]:
    """``'baseline,ffcp+chsd,full'`` -> ``('baseline', 'ffcp_chsd', 'full')``.

    The ground-truth column is implicit and always drawn.  The canonical
    ``STAGES`` order is kept so that different figures stay comparable.
    """
    wanted: List[str] = []
    for token in str(text).replace(' ', '').split(','):
        if not token or token in ('gt', 'ground_truth', 'groundtruth'):
            continue
        key = token.lower()
        if key not in STAGE_ALIASES:
            raise SystemExit(
                f'unknown column {token!r}; known: baseline, ffcp, ffcp_chsd, '
                'full (plus gt, which is always drawn)')
        stage = STAGE_ALIASES[key]
        if stage not in wanted:
            wanted.append(stage)
    if not wanted:
        raise SystemExit('--columns selected no stage column')
    return tuple(stage for stage in STAGES if stage in wanted)

# category index -> colour so that a category keeps its colour in every panel
PRED_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
               '#393b79', '#843c0c')
GT_COLOR = '#FFD400'          # amber, clearly outside the prediction palette
GT_TEXT_COLOR = '#7a5c00'     # dark amber so the GT chip stays readable
MATCH_IOU = 0.5               # standard COCO matching threshold
# validation-best key used by tools/s2h/run_ablation.sh to pick the checkpoint
CKPT_PATTERN = 'best_coco_bbox_mAP_epoch_*.pth'

DATASET_ALIASES = {
    'clipart1k': 'clipart1k', 'clipart': 'clipart1k', 'clip': 'clipart1k',
    'deepfish': 'FISH', 'fish': 'FISH',
    'neu-det': 'NEU-DET', 'neudet': 'NEU-DET', 'neu': 'NEU-DET',
    'artaxor': 'ArTaxOr', 'dior': 'DIOR', 'uodd': 'UODD',
}
DISPLAY_NAMES = {
    'clipart1k': 'Clipart1k', 'FISH': 'DeepFish', 'NEU-DET': 'NEU-DET',
    'ArTaxOr': 'ArTaxOr', 'DIOR': 'DIOR', 'UODD': 'UODD',
}
DEFAULT_DATASETS = ('clipart1k', 'deepfish', 'NEU-DET')


def canonical_dataset(name: str) -> str:
    """Map a user friendly name onto the name used by the configs."""
    key = name.strip().lower().replace('_', '-')
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(
        f'unknown dataset {name!r}; known: {sorted(set(DATASET_ALIASES))}')


def display_name(dataset: str) -> str:
    return DISPLAY_NAMES.get(dataset, dataset)


# ---------------------------------------------------------------------------
# artifact lookup
# ---------------------------------------------------------------------------
def _epoch_of(path: Path) -> int:
    for token in path.stem.split('_'):
        if token.isdigit():
            return int(token)
    return -1


def resolve_checkpoint(work_dir: Path, pattern: str = CKPT_PATTERN
                       ) -> Tuple[Optional[Path], str, List[Path]]:
    """Validation-best checkpoint of one ablation run.

    Returns ``(path, matched_pattern, all_best_candidates)``.

    The preferred pattern mirrors ``tools/s2h/run_ablation.sh``
    (``best_coco_bbox_mAP_epoch_*.pth``).  ``save_best='auto'`` on a classwise
    ``CocoMetric`` writes one ``best_*.pth`` per metric key, so the fallback is
    reported explicitly instead of being used silently.
    """
    if not work_dir.is_dir():
        return None, '', []
    bests = sorted(work_dir.glob('best_*.pth'), key=_epoch_of)
    for candidate_pattern in (pattern, 'best_*.pth'):
        matches = sorted(work_dir.glob(candidate_pattern), key=_epoch_of)
        if matches:
            return matches[-1], candidate_pattern, bests
    latest = work_dir / 'latest.pth'
    if latest.is_file():
        return latest, 'latest.pth', bests
    return None, '', bests


def _epoch_from_name(name: str) -> Optional[int]:
    for token in Path(name).stem.split('_'):
        if token.isdigit():
            return int(token)
    return None


def read_scalars(work_dir: Path) -> List[Tuple[Optional[int], float]]:
    """``[(epoch, bbox_mAP), ...]`` from the run's ``vis_data/scalars.json``."""
    rows: List[Tuple[Optional[int], float]] = []
    for path in sorted(Path(work_dir).glob('**/scalars.json')):
        try:
            with path.open('r', encoding='utf-8') as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            value = None
            for key, raw in item.items():
                if key == 'bbox_mAP' or key.endswith('/bbox_mAP'):
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
            if value is None:
                continue
            epoch = item.get('epoch')
            rows.append((int(epoch) if epoch is not None else None, value))
    return rows


def best_map_for(work_dir: Path, epoch: Optional[int] = None
                 ) -> Optional[float]:
    """Validation mAP of the selected checkpoint (falls back to the best)."""
    rows = read_scalars(work_dir)
    if not rows:
        return None
    if epoch is not None:
        for row_epoch, value in rows:
            if row_epoch == int(epoch):
                return value
    return max(value for _, value in rows)


def resolve_knowledge(knowledge_root: str, dataset: str, stage: str,
                      seed: str, shot: str, config_default: str = '') -> Optional[Path]:
    """Cached FFCP/CHSD knowledge of one ablation run."""
    candidates = []
    if knowledge_root:
        root = Path(knowledge_root)
        candidates += [
            root / dataset / stage / f'seed{seed}' / f'{shot}shot.pth',
            root / dataset / f'seed{seed}' / stage / f'{shot}shot.pth',
            root / f'{dataset}_{stage}_seed{seed}_{shot}shot.pth',
        ]
    candidates.append(Path(f'work_dirs/s2h_knowledge/{dataset}_{shot}shot.pth'))
    if config_default:
        candidates.append(Path(str(config_default)))
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def stage_cfg_options(stage: str, knowledge: Optional[Path]) -> Dict[str, object]:
    """``--cfg-options`` equivalent of the ablation launcher."""
    if stage == 'baseline':
        return {'model.bbox_head.s2h_cfg.enabled': False}
    options = {
        'model.bbox_head.s2h_cfg.enabled': True,
        'model.bbox_head.s2h_cfg.stage': stage,
    }
    if knowledge is not None:
        options['model.bbox_head.s2h_cfg.knowledge_path'] = str(knowledge)
    return options


# ---------------------------------------------------------------------------
# geometry / scoring helpers (pure numpy, exercised by --self-check)
# ---------------------------------------------------------------------------
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` xyxy boxes."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * \
        np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * \
        np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_score(gt_box: np.ndarray, gt_label: int, pred: Optional[dict]
                ) -> Tuple[float, float]:
    """Best same-class ``(iou, score)`` for one ground-truth box."""
    if not pred:
        return 0.0, 0.0
    labels = pred['labels']
    keep = labels == gt_label
    if not keep.any():
        return 0.0, 0.0
    ious = iou_matrix(gt_box[None, :], pred['bboxes'][keep])[0]
    best = int(np.argmax(ious))
    return float(ious[best]), float(pred['scores'][keep][best])


def build_chains(gt_boxes: np.ndarray, gt_labels: np.ndarray,
                 predictions: Dict[str, Optional[dict]],
                 stages: Sequence[str] = STAGES
                 ) -> List[Tuple[int, np.ndarray, List[Optional[float]]]]:
    """Per-GT score chain over ``stages`` (``None`` = stage unavailable)."""
    chains = []
    for box, label in zip(gt_boxes, gt_labels):
        chain: List[Optional[float]] = []
        for stage in stages:
            pred = predictions.get(stage)
            if pred is None:
                chain.append(None)
                continue
            iou, score = match_score(box, int(label), pred)
            chain.append(score if iou >= MATCH_IOU else 0.0)
        chains.append((int(label), box, chain))
    return chains


def chain_steps(chain, tolerance: float = 1e-9):
    """Adjacent steps of a chain plus the climb / fall summaries.

    Returns ``(steps, climb, fall, monotone)`` where ``steps`` only contains
    the stages that are available, and ``monotone`` means no step goes down.
    """
    seq = [v for v in chain if v is not None]
    steps = [b - a for a, b in zip(seq, seq[1:])]
    climb = float(sum(max(0.0, s) for s in steps))
    fall = float(sum(max(0.0, -s) for s in steps))
    monotone = bool(steps) and all(s >= -tolerance for s in steps)
    return steps, climb, fall, monotone


def chain_stats(chains, stages: Sequence[str] = STAGES) -> dict:
    """Per-sample statistics shared by the ranking and the printed summary.

    ``climb`` is how much the chains rise in total (positive steps only) and
    ``fall`` how much they drop back, so ``climb - fall`` rewards the chains
    that really improve stage after stage.  ``best_climb`` is the climb of the
    best *monotone* chain, i.e. a ground-truth box whose score goes
    ``baseline <= ffcp <= ffcp_chsd <= full`` - exactly the qualitative example
    a results figure wants.  ``mono_ratio`` is the fraction of such boxes.
    """
    stages = list(stages)
    full_index = stages.index('full') if 'full' in stages else len(stages) - 1
    net = 0
    gains, bests, per_stage = [], [], [0.0] * len(stages)
    mono = total = 0
    climb = fall = best_climb = 0.0
    full_scores, full_improved, full_is_best = [], False, False
    for _, _, chain in chains:
        available = [v for v in chain if v is not None]
        if not available:
            continue
        baseline = chain[0] if chain[0] is not None else 0.0
        later = [v for v in chain[1:] if v is not None]
        best_later = max(later) if later else 0.0
        net += int(baseline <= 0.0 and best_later > 0.0)
        net -= int(baseline > 0.0 and best_later <= 0.0)
        gains.append(best_later - baseline)
        bests.append(max(available))
        for i, value in enumerate(chain):
            per_stage[i] = max(per_stage[i], value or 0.0)
        _, box_climb, box_fall, monotone = chain_steps(chain)
        if len(available) > 1:
            total += 1
            mono += int(monotone)
            climb += box_climb
            fall += box_fall
            if monotone:
                best_climb = max(best_climb, box_climb)
        full_score = chain[full_index]
        if full_score is not None:
            full_scores.append(full_score)
            if chain[0] is not None and full_score > chain[0] + 1e-9:
                full_improved = True
            if full_score >= max(available) - 1e-9:
                full_is_best = True
    best_stage = stages[int(np.argmax(per_stage))] if per_stage and \
        max(per_stage) > 0 else '-'
    return dict(
        net=net,
        gain=float(sum(gains)),
        climb=float(climb),
        fall=float(fall),
        rise=float(climb - fall),
        best_climb=float(best_climb),
        best=float(max(bests)) if bests else 0.0,
        full=float(np.mean(full_scores)) if full_scores else 0.0,
        mean_full=float(np.mean(full_scores)) if full_scores else 0.0,
        mono_ratio=(mono / total) if total else 0.0,
        per_stage=per_stage,
        best_stage=best_stage,
        full_improved=full_improved,
        full_is_best=full_is_best,
    )


# ---------------------------------------------------------------------------
# detection bookkeeping (true / false positives)
# ---------------------------------------------------------------------------
def match_detections(gt_boxes: np.ndarray, gt_labels: np.ndarray,
                     pred: Optional[dict],
                     iou_thr: float = MATCH_IOU,
                     class_aware: bool = True) -> dict:
    """One-to-one greedy matching of one stage's predictions to the ground truth.

    A prediction is a true positive only when it overlaps a *still unmatched*
    ground-truth box (``IoU >= iou_thr``) **and** carries the same category, so a
    wrong-class hit or a duplicate box on a real object counts as a false
    positive - which is exactly what a reader flags as a wrong detection in a
    qualitative figure.  Every other prediction is a false positive, every
    unmatched box a false negative.
    """
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    n_gt = int(gt_boxes.shape[0])
    n_pred = int(len(pred['bboxes'])) if pred else 0
    tp_mask = np.zeros(n_pred, dtype=bool)
    gt_hit = np.zeros(n_gt, dtype=bool)
    if n_pred and n_gt:
        ious = iou_matrix(pred['bboxes'], gt_boxes)
        if class_aware:
            same = np.asarray(pred['labels'], dtype=np.int64)[:, None] == \
                gt_labels[None, :]
            # a wrong-class overlap must never win the assignment
            ious = np.where(same, ious, -1.0)
        flat_ious = ious.reshape(-1)
        for flat in np.argsort(flat_ious)[::-1]:
            if flat_ious[flat] < iou_thr:
                break
            pred_index, gt_index = divmod(int(flat), n_gt)
            if tp_mask[pred_index] or gt_hit[gt_index]:
                continue
            tp_mask[pred_index] = True
            gt_hit[gt_index] = True
    return dict(tp=int(tp_mask.sum()), fp=int(n_pred - int(tp_mask.sum())),
                fn=int(n_gt - int(gt_hit.sum())), n_gt=n_gt, n_pred=n_pred,
                tp_mask=tp_mask, fp_mask=~tp_mask, gt_hit=gt_hit)


def detection_stats(gt_boxes: np.ndarray, gt_labels: np.ndarray,
                    predictions: Dict[str, Optional[dict]],
                    stages: Sequence[str] = STAGES) -> dict:
    """True / false positive summary of one sample and its drawn stages.

    ``fp_delta`` / ``tp_delta`` compare the last drawn stage with the reference
    stage (``baseline``, or the first drawn column when ``baseline`` is hidden),
    ``recovered`` counts the objects the reference misses and the last stage
    finds, and ``precision`` is ``TP / (TP + FP)``.  The counts use the boxes
    that survived ``--score-thr`` - the very same boxes the figure draws - and
    matching is class-aware, see :func:`match_detections`.
    """
    stages = list(stages)
    per_stage = [match_detections(gt_boxes, gt_labels, predictions.get(stage))
                 for stage in stages]
    reference = stages.index('baseline') if 'baseline' in stages else 0
    final = stages.index('full') if 'full' in stages else len(stages) - 1
    base, last = per_stage[reference], per_stage[final]

    def precision(entry: dict) -> float:
        counted = entry['tp'] + entry['fp']
        return (entry['tp'] / counted) if counted else 1.0

    return dict(
        ref_stage=stages[reference], last_stage=stages[final],
        n_gt=int(base['n_gt']),
        tp_base=base['tp'], fp_base=base['fp'], fn_base=base['fn'],
        tp_full=last['tp'], fp_full=last['fp'], fn_full=last['fn'],
        tp_delta=last['tp'] - base['tp'],
        fp_delta=last['fp'] - base['fp'],
        recovered=int((last['gt_hit'] & ~base['gt_hit']).sum()),
        lost=int((base['gt_hit'] & ~last['gt_hit']).sum()),
        precision_base=precision(base), precision_full=precision(last),
        per_stage=[dict(tp=e['tp'], fp=e['fp'], fn=e['fn'])
                   for e in per_stage])


def clean_ok(det: dict, max_fp_delta: int = 0) -> bool:
    """Whether a sample improves *without* introducing wrong detections.

    ``max_fp_delta`` tolerates that many extra false positives between the
    reference stage and the last drawn stage; the default ``0`` demands that the
    method does not add a single wrong box.  Losing a true positive also counts
    as unclean, so the rise can never come from trading detections away.
    """
    return det['fp_delta'] <= int(max_fp_delta) and det['tp_delta'] >= 0


# quality levels used by ``--require-rising`` (strictest first)
LEVELS = ('rising', 'full>baseline', 'recovery', 'detected')

# `>` would be swallowed by the shell as a redirection, so accept plain aliases
LEVEL_ALIASES = {
    'rising': 'rising', 'ascending': 'rising', 'mono': 'rising',
    'full>baseline': 'full>baseline', 'fullbaseline': 'full>baseline',
    'fullabovebaseline': 'full>baseline', 'improvement': 'full>baseline',
    'improve': 'full>baseline', 'full': 'full>baseline',
    'recovery': 'recovery', 'recover': 'recovery', 'net': 'recovery',
    'detected': 'detected', 'any': 'detected',
}


def parse_level(text: str) -> str:
    """Normalise ``--min-requirement``, so ``improvement`` == ``full>baseline``."""
    key = str(text).strip().lower().replace(' ', '')
    key = key.replace('_', '').replace('-', '').replace('>=', '>')
    if key in LEVEL_ALIASES:
        return LEVEL_ALIASES[key]
    raise SystemExit(
        f'unknown requirement {text!r}; use one of {list(LEVELS)}.  Quote it '
        "in the shell (--min-requirement 'full>baseline') or simply write "
        '--min-requirement improvement')


def level_ok(level: str, stats: dict, min_lift: float) -> bool:
    """Whether a sample reaches one quality level."""
    if level == 'rising':
        return stats['best_climb'] >= min_lift
    if level == 'full>baseline':
        return bool(stats['full_improved'])
    if level == 'recovery':
        return stats['net'] > 0
    return stats['best'] > 0


def rank_key(chains, rank_by: str, stages: Sequence[str] = STAGES,
             det: Optional[dict] = None):
    """Sort key so that the most convincing samples come first.

    ``rising`` (default) ground-truth boxes whose score climbs monotonically
                        ``baseline -> ffcp -> ffcp_chsd -> full``, biggest
                        climb first - the story a results figure has to tell,
    ``mono``            same, but ignoring the size of the climb,
    ``recovery``        objects the baseline misses that the method finds,
    ``gain``            largest confidence gain of the method over the baseline,
    ``confidence``      highest score at the last stage,
    ``ffcp``            the baseline -> FFCP step only,
    ``clean``           fewest *new* wrong detections first, then the largest
                        true-positive gain and the best precision - needs ``det``
                        from :func:`detection_stats`.
    """
    stats = chain_stats(chains, stages)
    if rank_by == 'clean' and det is not None:
        return (-float(det['fp_delta']), float(det['tp_delta']),
                float(det['precision_full']), stats['best_climb'],
                stats['rise'])
    if rank_by == 'confidence':
        return (stats['full'], stats['net'], stats['gain'], stats['best'])
    if rank_by == 'gain':
        return (stats['gain'], stats['rise'], stats['net'], stats['best'])
    if rank_by == 'ffcp':
        return (stats['per_stage'][1] - stats['per_stage'][0]
                if len(stats['per_stage']) > 1 else 0.0,
                stats['rise'], stats['net'], stats['best'])
    if rank_by == 'mono':
        return (stats['mono_ratio'], stats['best_climb'], stats['rise'],
                stats['best'])
    if rank_by == 'recovery':
        return (stats['net'], stats['rise'], stats['best'], stats['full'])
    return (stats['best_climb'], stats['mono_ratio'], stats['rise'],
            stats['best'])


def format_caption(chains, classes: Sequence[str], pretty: bool = True,
                   stages: Sequence[str] = STAGES) -> str:
    """One-line caption under a row, e.g. ``bicycle: 0.11 -> 0.12 -> ...``."""
    stages = list(stages)
    if not chains:
        return 'no ground-truth instance in this sample'

    def name_of(label: int) -> str:
        if 0 <= label < len(classes):
            text = str(classes[label])
        else:
            text = f'class {label}'
        return text.replace('_', ' ') if pretty else text

    def key(item):
        _, _, chain = item
        _, climb, fall, monotone = chain_steps(chain)
        final = chain[-1] if chain[-1] is not None else 0.0
        # report the box that climbs stage after stage, then the biggest climb
        return (int(monotone), climb - fall, climb, final)

    label, _, chain = max(chains, key=key)
    cells = ['  --  ' if v is None else f'{v:.2f}' for v in chain]
    arrow = ' -> '.join(cells)
    confident = [i for i, v in enumerate(chain) if v is not None and v > 0]
    if not confident:
        note = 'not detected by any stage'
    elif confident[0] == 0:
        note = 'detected by every drawn stage'
    elif confident[0] == len(stages) - 1:
        note = f'missed by {STAGE_TITLES[stages[confident[0] - 1]]} -> ' \
               f'recovered by {STAGE_TITLES[stages[-1]]}'
    else:
        note = f'first detected by {STAGE_TITLES[stages[confident[0]]]}'
    last = stages.index('full') if 'full' in stages else len(stages) - 1
    finals = [c[last] for _, _, c in chains if c[last] is not None]
    hits = sum(1 for v in finals if v > 0)
    return (f'{name_of(label)}: {arrow}   [{note}]   '
            f'GT hit@{STAGE_TITLES[stages[last]]}: {hits}/{len(chains)}')


# ---------------------------------------------------------------------------
# dataset / model plumbing
# ---------------------------------------------------------------------------
def resolve_image_path(img_path: str, data_root: Optional[str]) -> str:
    candidates = [img_path]
    if data_root:
        candidates.append(osp.join(data_root, img_path))
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return img_path


def collect_candidates(dataset, args) -> List[dict]:
    """Ground-truth records of the test split (no model needed)."""
    wanted = {w.strip().lower()
              for w in (args.img_names or '').split(',') if w.strip()}
    wanted_classes = {c.strip().lower().replace('_', ' ')
                      for c in (args.classes or [])}
    data_root = getattr(dataset, 'data_root', None)
    records = []
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        instances = [ins for ins in info.get('instances', [])
                     if not ins.get('ignore_flag', 0)]
        if not instances:
            continue
        boxes = np.asarray([ins['bbox'] for ins in instances], dtype=np.float32)
        labels = np.asarray([int(ins['bbox_label']) for ins in instances],
                            dtype=np.int64)
        basename = osp.basename(info['img_path'])
        stem = osp.splitext(basename)[0]
        classes = [str(dataset.metainfo['classes'][int(l)]).replace('_', ' ')
                   for l in labels if int(l) < len(dataset.metainfo['classes'])]
        if wanted and stem.lower() not in wanted and basename.lower() not in wanted:
            continue
        if wanted_classes and not wanted_classes & {c.lower() for c in classes}:
            continue
        records.append(dict(
            index=index,
            name=stem,
            img_path=resolve_image_path(info['img_path'], data_root),
            text=info.get('text') or tuple(dataset.metainfo['classes']),
            custom_entities=bool(info.get('custom_entities', True)),
            gt_boxes=boxes,
            gt_labels=labels,
            gt_names=classes))
    return records


def subsample(records: List[dict], args) -> List[dict]:
    """Deterministic subsample when the test split is large."""
    limit = int(args.max_candidates)
    if args.img_names or limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(args.sample_seed)
    picked = sorted(rng.sample(range(len(records)), limit))
    return [records[i] for i in picked]


def load_stage_model(cfg, stage: str, checkpoint: Path, knowledge: Optional[Path],
                     classes: Sequence[str], device: str):
    """Build one ablation model with the same options as the launcher."""
    from mmdet.apis import init_detector
    options = stage_cfg_options(stage, knowledge)
    model = init_detector(copy.deepcopy(cfg), str(checkpoint), device=device,
                          cfg_options=options)
    model.dataset_meta = {'classes': list(classes)}
    return model


def predict_image(model, record: dict, score_thr: float) -> dict:
    """Predictions of one image, in original-image coordinates."""
    from mmdet.apis import inference_detector
    result = inference_detector(model, record['img_path'],
                                text_prompt=record.get('text'),
                                custom_entities=record.get('custom_entities',
                                                           True))
    instances = result.pred_instances
    if len(instances) == 0:
        return dict(bboxes=np.zeros((0, 4), dtype=np.float32),
                    scores=np.zeros(0, dtype=np.float32),
                    labels=np.zeros(0, dtype=np.int64))
    keep = instances.scores.cpu().numpy() >= float(score_thr)
    return dict(bboxes=instances.bboxes.cpu().numpy()[keep].astype(np.float32),
                scores=instances.scores.cpu().numpy()[keep].astype(np.float32),
                labels=instances.labels.cpu().numpy()[keep].astype(np.int64))


def read_image(path: str) -> np.ndarray:
    """Image as RGB for matplotlib."""
    import mmcv
    img = mmcv.imread(path, channel_order='rgb')
    if img is None:
        raise FileNotFoundError(f'cannot read image: {path}')
    return img


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _draw_panel(ax, img, gt_boxes, gt_labels, classes, pred, score_thr,
                max_per_class, only_gt_classes, pretty, note=None,
                label_font: float = 8.0, box_width: float = 2.0,
                draw_labels: bool = False):
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Rectangle

    def class_name(label: int) -> str:
        name = str(classes[label]) if 0 <= label < len(classes) else f'#{label}'
        return name.replace('_', ' ') if pretty else name

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    img_h, img_w = img.shape[:2]

    def label(x, y, text, color, va):
        """Readable chip: white background, coloured text + white outline."""
        ax.text(x, y, text, fontsize=label_font, va=va, ha='left', color=color,
                clip_on=True, zorder=6,
                path_effects=[path_effects.withStroke(linewidth=2.2,
                                                      foreground='white')],
                bbox=dict(facecolor='white', alpha=0.92, edgecolor=color,
                          linewidth=1.0, boxstyle='square,pad=0.16'))

    gt_labels_present = set(int(l) for l in gt_labels)
    for box, label_ in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        ax.add_patch(Rectangle((x1, y1), max(x2 - x1, 1e-3), max(y2 - y1, 1e-3),
                               fill=False, edgecolor=GT_COLOR,
                               linewidth=box_width, linestyle='--', zorder=4))
        if draw_labels:
            # above the box when there is room, otherwise just inside it
            inside = y1 < 0.06 * img_h
            label(x1, y1 + 0.012 * img_h if inside else y1 - 0.012 * img_h,
                  class_name(int(label_)), GT_TEXT_COLOR,
                  'top' if inside else 'bottom')

    if note:
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha='center',
                va='center', fontsize=label_font + 1, color='0.35', zorder=7)
    if pred is None:
        return

    order = np.argsort(-pred['scores']) if len(pred['scores']) else []
    per_class: Dict[int, int] = {}
    for i in order:
        label_id = int(pred['labels'][i])
        if only_gt_classes and label_id not in gt_labels_present:
            continue
        per_class[label_id] = per_class.get(label_id, 0) + 1
        if per_class[label_id] > max_per_class:
            continue
        x1, y1, x2, y2 = [float(v) for v in pred['bboxes'][i][:4]]
        color = PRED_COLORS[label_id % len(PRED_COLORS)]
        ax.add_patch(Rectangle((x1, y1), max(x2 - x1, 1e-3), max(y2 - y1, 1e-3),
                               fill=False, edgecolor=color, linewidth=box_width,
                               zorder=5))
        if draw_labels:
            # inside the top-left corner: never overlaps the ground-truth chip
            tall = (y2 - y1) > 0.08 * img_h
            figure_y = y1 + 0.012 * img_h if tall else y2 + 0.004 * img_h
            label(x1, figure_y,
                  f'{class_name(label_id)} {pred["scores"][i]:.2f}',
                  color, 'top' if tall else 'bottom')


def render_rows(rows: List[dict], out_dir: Path, tag: str, args,
                subtitle: str = '', write_single_rows: bool = True) -> List[Path]:
    """Write one combined figure plus one figure per row."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    panel_w, panel_h = args.panel_size
    if write_single_rows:
        for row_idx, row in enumerate(rows):
            fig = _figure_for_rows([row], panel_w, panel_h, args, subtitle)
            stem = f'{tag}_row{row_idx + 1}'
            for ext in args.formats:
                path = out_dir / f'{stem}.{ext}'
                fig.savefig(path, dpi=args.dpi, bbox_inches='tight',
                            facecolor='white')
                written.append(path)
            plt.close(fig)

    fig = _figure_for_rows(rows, panel_w, panel_h, args, subtitle)
    for ext in args.formats:
        path = out_dir / f'{tag}.{ext}'
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        written.append(path)
    plt.close(fig)
    return written


def _figure_for_rows(rows, panel_w, panel_h, args, subtitle):
    import matplotlib.pyplot as plt

    stages = list(getattr(args, 'stage_list', STAGES))
    titles = list(column_titles(stages, getattr(args, 'gt_header',
                                                'Ground Truth')))
    n_rows = len(rows)
    n_cols = len(titles)
    if not getattr(args, 'show_subtitle', True):
        subtitle = ''

    # a figure that holds a single dataset can print the validation mAP of every
    # stage in its column title; a stacked figure needs one line per row instead
    show_map = bool(getattr(args, 'show_map', False))
    show_caption = bool(getattr(args, 'show_caption', False))
    single_dataset = len({row['dataset'] for row in rows}) == 1
    row_maps = (show_map and bool(rows) and not single_dataset
                and any(row.get('maps') for row in rows))
    if show_map and single_dataset and rows and rows[0].get('maps'):
        maps = rows[0]['maps']
        titles = [titles[0]] + [
            titles[index + 1] + (
                f'\nmAP {maps[stage] * 100:.1f}'
                if maps.get(stage) is not None else '')
            for index, stage in enumerate(stages)]

    fig_w = panel_w * n_cols + 1.0
    caption_h = 1.0 if row_maps else (0.5 if show_caption else 0.08)
    fig_h = (panel_h + caption_h) * n_rows + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        2 * n_rows, n_cols,
        height_ratios=[panel_h, caption_h] * n_rows,
        hspace=0.08, wspace=0.03,
        top=0.97 - (0.05 if subtitle else 0.0), bottom=0.02,
        left=0.06, right=0.995)

    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = fig.add_subplot(grid[2 * r, c])
            if c == 0:
                _draw_panel(ax, row['image'], row['gt_boxes'], row['gt_labels'],
                            row['classes'], None, args.score_thr,
                            args.max_per_class, args.only_gt_classes,
                            args.pretty_names, label_font=args.label_font,
                            box_width=args.box_width,
                            draw_labels=args.show_labels)
                ax.set_title(titles[c], fontsize=9.5, pad=3)
                ax.text(-0.04, 0.5,
                        row.get('label') or display_name(row['dataset']),
                        transform=ax.transAxes, rotation=90, va='center',
                        ha='right', fontsize=9.5)
            else:
                stage = stages[c - 1]
                pred = row['predictions'].get(stage)
                _draw_panel(ax, row['image'], row['gt_boxes'], row['gt_labels'],
                            row['classes'], pred, args.score_thr,
                            args.max_per_class, args.only_gt_classes,
                            args.pretty_names,
                            note=None if pred is not None
                            else 'checkpoint missing',
                            label_font=args.label_font,
                            box_width=args.box_width,
                            draw_labels=args.show_labels)
                if r == 0:
                    ax.set_title(titles[c], fontsize=9.5, pad=3)

        cap_ax = fig.add_subplot(grid[2 * r + 1, :])
        cap_ax.axis('off')
        if row_maps:
            maps = row.get('maps') or {}
            values = ['--' if maps.get(stage) is None
                      else f'{maps[stage] * 100:.1f}' for stage in stages]
            map_line = 'mAP: ' + '   |   '.join(
                f'{STAGE_TITLES[stage]} {value}'
                for stage, value in zip(stages, values))
            cap_ax.text(0.0, 0.74, row['caption'], transform=cap_ax.transAxes,
                        fontsize=args.caption_font, va='center', ha='left',
                        family='monospace')
            cap_ax.text(0.0, 0.22, map_line, transform=cap_ax.transAxes,
                        fontsize=args.caption_font, va='center', ha='left',
                        family='monospace')
        elif show_caption:
            cap_ax.text(0.0, 0.5, row['caption'], transform=cap_ax.transAxes,
                        fontsize=args.caption_font, va='center', ha='left',
                        family='monospace')

    if subtitle:
        fig.suptitle(subtitle, fontsize=10, y=0.995, va='top')
    return fig


# ---------------------------------------------------------------------------
# self check (no checkpoint / no GPU needed)
# ---------------------------------------------------------------------------
def self_check() -> int:
    """Exercise the pure-python parts of the pipeline."""
    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    boxes_b = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    ious = iou_matrix(boxes_a, boxes_b)
    assert abs(ious[0, 0] - 1.0) < 1e-6, ious
    assert abs(ious[0, 1]) < 1e-6, ious
    assert iou_matrix(np.zeros((0, 4)), boxes_b).shape == (0, 2)

    pred = dict(bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
                scores=np.array([0.9], dtype=np.float32),
                labels=np.array([0], dtype=np.int64))
    iou, score = match_score(np.array([0, 0, 10, 10], dtype=np.float32), 0, pred)
    assert abs(iou - 1.0) < 1e-6 and abs(score - 0.9) < 1e-6
    assert match_score(np.array([0, 0, 10, 10], np.float32), 1, pred) == (0.0, 0.0)
    assert match_score(np.array([0, 0, 10, 10], np.float32), 0, None) == (0.0, 0.0)

    predictions = {
        'baseline': dict(bboxes=np.zeros((0, 4), np.float32),
                         scores=np.zeros(0, np.float32),
                         labels=np.zeros(0, np.int64)),
        'ffcp': dict(bboxes=np.array([[0, 0, 10, 10]], np.float32),
                     scores=np.array([0.2], np.float32),
                     labels=np.array([0], np.int64)),
        'ffcp_chsd': None,
        'full': dict(bboxes=np.array([[0, 0, 10, 10]], np.float32),
                     scores=np.array([0.7], np.float32),
                     labels=np.array([0], np.int64)),
    }
    chains = build_chains(np.array([[0, 0, 10, 10]], np.float32),
                          np.array([0], np.int64), predictions)
    chain = chains[0][2]
    assert chain[0] == 0.0 and chain[2] is None, chain
    assert abs(chain[1] - 0.2) < 1e-6 and abs(chain[3] - 0.7) < 1e-6, chain
    key = rank_key(chains, 'recovery')
    assert key[0] == 1, key

    # a box the baseline finds but the S2H stages lose must rank below zero
    empty = dict(bboxes=np.zeros((0, 4), np.float32),
                 scores=np.zeros(0, np.float32),
                 labels=np.zeros(0, np.int64))
    hit = dict(bboxes=np.array([[0, 0, 10, 10]], np.float32),
               scores=np.array([0.8], np.float32),
               labels=np.array([0], np.int64))
    lost_chains = build_chains(
        np.array([[0, 0, 10, 10]], np.float32), np.array([0], np.int64),
        {'baseline': hit, 'ffcp': empty, 'ffcp_chsd': None, 'full': empty})
    assert rank_key(lost_chains, 'recovery')[0] == -1, \
        rank_key(lost_chains, 'recovery')
    lost_stats = chain_stats(lost_chains)
    assert not lost_stats['full_improved'] and not lost_stats['full_is_best'], \
        lost_stats
    assert lost_stats['best_stage'] == 'baseline', lost_stats

    # column selection (`--columns`) and progression ranking
    assert parse_stages('baseline,ffcp+chsd,full') == \
        ('baseline', 'ffcp_chsd', 'full')
    assert parse_stages('full,baseline,gt') == ('baseline', 'full')
    assert column_titles(('baseline', 'full')) == \
        ('Ground Truth', 'Baseline', 'Full S2H-CD-FSOD')

    def boxed(score):
        return dict(bboxes=np.array([[0, 0, 10, 10]], np.float32),
                    scores=np.array([score], np.float32),
                    labels=np.array([0], np.int64))

    monotone = build_chains(
        np.array([[0, 0, 10, 10]], np.float32), np.array([0], np.int64),
        {'baseline': boxed(0.1), 'ffcp': boxed(0.3), 'ffcp_chsd': None,
         'full': boxed(0.6)})
    mono_stats = chain_stats(monotone)
    assert mono_stats['mono_ratio'] == 1.0, mono_stats
    assert mono_stats['full_improved'] and mono_stats['full_is_best'], mono_stats
    assert mono_stats['best_stage'] == 'full', mono_stats
    # ... and hidden columns are simply not part of the chain any more
    short_chains = build_chains(
        np.array([[0, 0, 10, 10]], np.float32), np.array([0], np.int64),
        {'baseline': boxed(0.1), 'full': boxed(0.6)},
        ('baseline', 'full'))
    assert len(short_chains[0][2]) == 2, short_chains

    # ascending chains must outrank bumpy ones and own the caption
    assert abs(mono_stats['best_climb'] - 0.5) < 1e-6, mono_stats
    assert rank_key(monotone, 'rising')[0] == mono_stats['best_climb']
    bumpy = build_chains(
        np.array([[0, 0, 10, 10]], np.float32), np.array([0], np.int64),
        {'baseline': boxed(0.0), 'ffcp': boxed(0.5), 'ffcp_chsd': boxed(0.1),
         'full': boxed(0.4)})
    bumpy_stats = chain_stats(bumpy)
    assert bumpy_stats['mono_ratio'] == 0.0 and bumpy_stats['best_climb'] == 0.0, \
        bumpy_stats
    assert rank_key(monotone, 'rising') > rank_key(bumpy, 'rising')
    mixed = [(0, np.zeros(4), [0.0, 0.5, 0.1, 0.4]),
             (1, np.zeros(4), [0.10, 0.20, 0.30, 0.90])]
    assert format_caption(mixed, ['bumpy', 'climbing']).startswith('climbing:'), \
        format_caption(mixed, ['bumpy', 'climbing'])

    caption = format_caption(chains, ['bicycle'])
    assert 'bicycle' in caption and '0.70' in caption, caption
    assert '->' in caption and caption.count('--') == 1, caption

    # detection bookkeeping: class-aware one-to-one matching of TP / FP
    gt_pairs = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    gt_pairs_cls = np.array([0, 1], np.int64)

    def stage_pred(boxes, scores, labels):
        return dict(bboxes=np.array(boxes, np.float32).reshape(-1, 4),
                    scores=np.array(scores, np.float32),
                    labels=np.array(labels, np.int64))

    hits = match_detections(gt_pairs, gt_pairs_cls,
                            stage_pred([[0, 0, 10, 10], [100, 100, 110, 110]],
                                       [0.9, 0.8], [0, 1]))
    assert (hits['tp'], hits['fp'], hits['fn']) == (2, 0, 0), hits
    # a wrong-class hit on a real object is a wrong detection, not a hit
    mislabelled = match_detections(gt_pairs, gt_pairs_cls,
                                   stage_pred([[0, 0, 10, 10]], [0.9], [1]))
    assert (mislabelled['tp'], mislabelled['fp'], mislabelled['fn']) == \
        (0, 1, 2), mislabelled
    # a second box on an already matched object is a duplicate, not a hit
    duplicate = match_detections(gt_pairs[:1], gt_pairs_cls[:1],
                                 stage_pred([[0, 0, 10, 10], [1, 1, 9, 9]],
                                            [0.9, 0.85], [0, 0]))
    assert (duplicate['tp'], duplicate['fp']) == (1, 1), duplicate
    nothing = match_detections(gt_pairs, gt_pairs_cls, None)
    assert (nothing['tp'], nothing['fp'], nothing['fn']) == (0, 0, 2), nothing

    # `clean_ok` / `--rank-by clean`: fewer wrong boxes wins over a raw gain
    noisy = detection_stats(
        gt_pairs, gt_pairs_cls,
        {'baseline': stage_pred([[0, 0, 10, 10], [300, 300, 310, 310],
                                 [0, 0, 10, 10]], [0.9, 0.4, 0.8], [0, 0, 0]),
         'full': stage_pred([[0, 0, 10, 10]], [0.9], [0])},
        ('baseline', 'full'))
    assert (noisy['tp_base'], noisy['fp_base']) == (1, 2), noisy
    assert (noisy['tp_full'], noisy['fp_full']) == (1, 0), noisy
    assert noisy['fp_delta'] == -2 and clean_ok(noisy), noisy
    added_fp = detection_stats(
        gt_pairs, gt_pairs_cls,
        {'baseline': stage_pred([[0, 0, 10, 10]], [0.9], [0]),
         'full': stage_pred([[0, 0, 10, 10], [300, 300, 310, 310]], [0.9, 0.35],
                            [0, 0])},
        ('baseline', 'full'))
    assert added_fp['fp_delta'] == 1, added_fp
    assert not clean_ok(added_fp) and clean_ok(added_fp, max_fp_delta=1), added_fp
    won = detection_stats(
        gt_pairs, gt_pairs_cls,
        {'baseline': stage_pred([[0, 0, 10, 10]], [0.9], [0]),
         'full': stage_pred([[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.8],
                            [0, 1])},
        ('baseline', 'full'))
    assert (won['recovered'], won['tp_delta'], won['fp_delta']) == (1, 1, 0), won
    assert clean_ok(won) and abs(won['precision_full'] - 1.0) < 1e-9, won
    missed = detection_stats(
        gt_pairs, gt_pairs_cls,
        {'baseline': stage_pred([[0, 0, 10, 10], [100, 100, 110, 110]],
                                [0.9, 0.8], [0, 1]),
         'full': stage_pred([[0, 0, 10, 10]], [0.9], [0])},
        ('baseline', 'full'))
    assert (missed['lost'], missed['tp_delta']) == (1, -1), missed
    assert not clean_ok(missed), missed
    # the ranking sorts the raw key descending, so `-fp_delta` first means: a
    # sample that adds a wrong box always ranks below one that removes two,
    # even when the latter gains no extra true positive.
    assert rank_key(monotone, 'clean', det=won) > \
        rank_key(monotone, 'clean', det=added_fp)
    assert rank_key(monotone, 'clean', det=noisy) > \
        rank_key(monotone, 'clean', det=added_fp)
    only_full = detection_stats(gt_pairs, gt_pairs_cls,
                                {'full': stage_pred([[0, 0, 10, 10]], [0.9],
                                                    [0])}, ('full',))
    assert only_full['ref_stage'] == 'full', only_full

    # user-site detection (the failure mode documented in WEEKLY_REPORT.md)
    import site as _site
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        original = _site.getusersitepackages
        try:
            _site.getusersitepackages = lambda: tmp
            assert shadowing_user_site() == '', 'empty user site must be ignored'
            os.makedirs(osp.join(tmp, 'torch'))
            assert shadowing_user_site() == tmp, shadowing_user_site()
        finally:
            _site.getusersitepackages = original

    # checkpoint resolution (same priority as tools/s2h/run_ablation.sh)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / 'latest.pth').write_bytes(b'x')
        (work / 'best_coco_crazing_precision_epoch_24.pth').write_bytes(b'x')
        path, matched, bests = resolve_checkpoint(work, CKPT_PATTERN)
        assert path is not None and \
            path.name == 'best_coco_crazing_precision_epoch_24.pth', path
        assert matched == 'best_*.pth' and len(bests) == 1, (matched, bests)
        (work / 'best_coco_bbox_mAP_epoch_7.pth').write_bytes(b'x')
        (work / 'best_coco_bbox_mAP_epoch_19.pth').write_bytes(b'x')
        path, matched, bests = resolve_checkpoint(work, CKPT_PATTERN)
        assert path is not None and path.name == 'best_coco_bbox_mAP_epoch_19.pth', path
        assert matched == CKPT_PATTERN, matched
        assert not resolve_checkpoint(Path(tmp) / 'nope')[0]

    # relaxation ladder (--require-rising --min-requirement) and mAP reading
    import tempfile
    assert LEVELS[0] == 'rising' and LEVELS.index('full>baseline') == 1
    assert parse_level('full>baseline') == 'full>baseline'
    assert parse_level('improvement') == 'full>baseline'
    assert parse_level('full_baseline') == 'full>baseline'
    assert parse_level('  Rising ') == 'rising'
    assert level_ok('rising', mono_stats, 0.05)
    assert not level_ok('rising', bumpy_stats, 0.05)
    assert level_ok('full>baseline', mono_stats, 0.05)
    assert level_ok('recovery', chain_stats(chains), 0.05)
    assert level_ok('detected', mono_stats, 0.05)
    assert not level_ok('detected', chain_stats([]), 0.05)
    assert _epoch_from_name('best_coco_bbox_mAP_epoch_19.pth') == 19
    assert _epoch_from_name('latest.pth') is None

    with tempfile.TemporaryDirectory() as tmp:
        vis = Path(tmp) / 'vis_data'
        vis.mkdir()
        (vis / 'scalars.json').write_text(
            '{"epoch": 4, "coco/bbox_mAP": 0.285}\n'
            '{"epoch": 19, "coco/bbox_mAP": 0.312}\n', encoding='utf-8')
        assert abs(best_map_for(Path(tmp), 19) - 0.312) < 1e-9
        assert abs(best_map_for(Path(tmp), 99) - 0.312) < 1e-9
        assert best_map_for(Path(tmp) / 'nope') is None

    print('[self-check] ok')
    print('  caption example:', caption)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Qualitative GT/Baseline/FFCP/FFCP+CHSD/Full figures '
                    'for the S2H-CD-FSOD ablation.')
    parser.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS),
                        help='clipart1k, deepfish, NEU-DET, ArTaxOr, DIOR, UODD')
    parser.add_argument('--shots', nargs='+', default=['5'],
                        help='shot budget of the ablation runs to render '
                             '(default: 5)')
    parser.add_argument('--seed', default='3407',
                        help='training seed of the ablation runs')
    parser.add_argument('--config-dir', default='configs/s2h_dino')
    parser.add_argument('--work-root', default='cat_work_dir/ablation')
    parser.add_argument('--ckpt-pattern', default=CKPT_PATTERN,
                        help='glob of the validation-best checkpoint; keep the '
                             'default to match tools/s2h/run_ablation.sh')
    parser.add_argument('--knowledge-root',
                        default='work_dirs/s2h_knowledge/ablation')
    parser.add_argument('--out', default='work_dirs/s2h_vis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-images', type=int, default=1,
                        help='rows per dataset (0 = render nothing, only list)')
    parser.add_argument('--columns', default='baseline,ffcp,ffcp_chsd,full',
                        help="stage columns to draw, e.g. 'baseline,full' or "
                             "'baseline,ffcp+chsd,full'; the ground-truth "
                             'column is always drawn')
    parser.add_argument('--list-top', type=int, default=0,
                        help='print the N best ranked samples '
                             '(names usable with --img-names)')
    parser.add_argument('--max-candidates', type=int, default=40,
                        help='test images scored before picking the rows '
                             '(0 = scan the whole split, slow)')
    parser.add_argument('--sample-seed', type=int, default=0,
                        help='sampling seed for the candidate subset')
    parser.add_argument('--rank-by', default='rising',
                        choices=['rising', 'mono', 'recovery', 'gain',
                                 'confidence', 'ffcp', 'clean'],
                        help="'rising' (default) prefers ground-truth boxes "
                             'whose score climbs baseline -> ffcp -> '
                             'ffcp_chsd -> full, biggest climb first; '
                             "'clean' prefers the fewest new wrong detections "
                             'and then the largest true-positive gain')
    parser.add_argument('--classes', nargs='*', default=None,
                        help='keep only candidate images containing these '
                             'categories (e.g. --classes bicycle)')
    parser.add_argument('--img-names', default='',
                        help='comma separated image stems to render instead '
                             'of the automatic ranking')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='minimum score drawn in the prediction panels')
    parser.add_argument('--max-per-class', type=int, default=3)
    parser.add_argument('--only-gt-classes', action='store_true',
                        help='hide predictions whose category is not in the GT')
    parser.add_argument('--min-final-score', type=float, default=0.0,
                        help='drop candidates whose best Full score is lower')
    parser.add_argument('--min-best-score', type=float, default=0.0,
                        help='drop candidates that no stage detects above this '
                             'score (removes all-zero rows from the ranking)')
    parser.add_argument('--require-improvement', action='store_true',
                        help='keep only samples where Full beats Baseline on at '
                             'least one ground-truth box')
    parser.add_argument('--require-full-best', action='store_true',
                        help='keep only samples where Full is the best stage for '
                             'at least one ground-truth box (avoids rows where '
                             'an earlier stage outscores the final method)')
    parser.add_argument('--require-clean-rising', action='store_true',
                        help='keep only samples that improve *without* adding '
                             'wrong detections: the number of false positives '
                             'must not grow beyond --max-fp-delta and no true '
                             'positive may be lost, comparing the last drawn '
                             'stage with the baseline.  Matching is class-aware '
                             'at IoU>=0.5 on the boxes above --score-thr, i.e. '
                             'on exactly the boxes the figure draws')
    parser.add_argument('--max-fp-delta', type=int, default=0,
                        help='tolerated increase of wrong detections between the '
                             'baseline and the last drawn stage; used by '
                             '--require-clean-rising (default: 0 = not a single '
                             'extra wrong box)')
    parser.add_argument('--min-precision', type=float, default=0.0,
                        help='drop candidates whose precision TP/(TP+FP) at the '
                             'last drawn stage is lower (default: 0 = off; e.g. '
                             '0.6 keeps only samples without obvious wrong '
                             'boxes)')
    parser.add_argument('--require-rising', action='store_true',
                        help='prefer samples with a ground-truth box whose score '
                             'increases stage after stage (>= --min-lift); when '
                             'none exists the level is relaxed down to '
                             '--min-requirement')
    parser.add_argument('--min-requirement', default='full>baseline',
                        metavar='LEVEL',
                        help='hard floor of the relaxation ladder used by '
                             "--require-rising: rising | full>baseline "
                             '(alias improvement) | recovery | detected '
                             '(default: full>baseline, i.e. Full must at '
                             'least beat Baseline)')
    parser.add_argument('--min-lift', type=float, default=0.05,
                        help='minimum climb of an ascending chain')
    parser.add_argument('--align-shots', action='store_true',
                        help='with several --shots, re-use the images picked for '
                             'the first shot, so the rows compare the same '
                             'sample across shot budgets')
    parser.add_argument('--pretty-names', action=argparse.BooleanOptionalAction,
                        default=True, help='render `_` as a space')
    parser.add_argument('--panel-size', nargs=2, type=float,
                        default=(2.6, 2.2), metavar=('W', 'H'))
    parser.add_argument('--caption-font', type=float, default=7.5)
    parser.add_argument('--show-labels', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='draw the class name / score chips on the boxes '
                             '(default: on; --no-show-labels draws boxes only)')
    parser.add_argument('--show-caption', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='draw the per-row score chain under each row '
                             '(default: off)')
    parser.add_argument('--show-map', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='draw the validation mAP of each stage '
                             '(default: off, read from scalars.json)')
    parser.add_argument('--show-subtitle', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='draw the figure title (default: off)')
    parser.add_argument('--gt-header', default='Ground Truth',
                        help="header of the first column; pass '' to leave it "
                             'empty so that only the stage names are shown')
    parser.add_argument('--label-font', type=float, default=8.5,
                        help='font size of the box labels (class + score)')
    parser.add_argument('--box-width', type=float, default=2.0,
                        help='line width of the drawn boxes')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--formats', nargs='+', default=['png'],
                        choices=['png', 'pdf', 'svg'])
    parser.add_argument('--allow-missing', action='store_true',
                        help='render rows even if some stage checkpoint is '
                             'absent (the panel shows `checkpoint missing`)')
    parser.add_argument('--self-check', action='store_true',
                        help='run the CPU-only helper checks and exit')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='print the resolved checkpoint of every stage and '
                             'exit (no dataset, no model)')
    parser.add_argument('--no-reexec', action='store_true',
                        help='do not re-launch with PYTHONNOUSERSITE=1 when a '
                             'user-site PyTorch shadows the conda build')
    return parser.parse_args(argv)


def list_checkpoints(dataset: str, shot: str, args) -> None:
    """Print the checkpoint of every stage (fast, no dataset, no model)."""
    print(f'[ckpt] {dataset} {shot}-shot  (preferred pattern: '
          f'{args.ckpt_pattern})')
    for stage in STAGES:
        work_dir = Path(args.work_root) / dataset / stage / \
            f'seed{args.seed}' / f'{shot}shot'
        path, matched, best_candidates = resolve_checkpoint(
            work_dir, args.ckpt_pattern)
        status = path.name if path is not None else 'MISSING'
        print(f'  {stage:<10s} -> {status}   [{matched or "-"}]')
        for candidate in best_candidates:
            mark = ' <- uses' if path is not None and candidate == path else ''
            print(f'      {candidate.name}{mark}')


def run_one_dataset(dataset: str, shot: str, args) -> List[Path]:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS

    config_path = Path(args.config_dir) / \
        f's2h_grounding_dino_swin-b_{dataset}_{shot}shot.py'
    if not config_path.is_file():
        print(f'[skip] {dataset} {shot}-shot: missing config {config_path}')
        return [], []

    cfg = Config.fromfile(str(config_path))
    # mmdet's pipeline transforms (e.g. `FixScaleResize`) are registered in the
    # `mmdet::transform` scope, so the scope has to be active *before* the
    # dataset is built.  `init_detector` would do it, but it runs later; the
    # value comes from the config chain (`_base_/default_runtime.py`).
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset_cfg['lazy_init'] = False
    test_dataset = DATASETS.build(dataset_cfg)
    classes = list(test_dataset.metainfo['classes'])

    records = collect_candidates(test_dataset, args)
    if not records:
        print(f'[skip] {dataset} {shot}-shot: no candidate images '
              '(check --classes / --img-names / the test split)')
        return [], []
    candidates = subsample(records, args)
    print(f'[data] {dataset} {shot}-shot: {len(records)} test images with GT, '
          f'{len(candidates)} scanned')

    # ---- one stage at a time keeps the GPU memory flat -------------------
    predictions: Dict[str, Dict[str, Optional[dict]]] = {
        stage: {} for stage in STAGES}
    available = {}
    used_checkpoints: Dict[str, str] = {}
    stage_maps: Dict[str, Optional[float]] = {}
    s2h_default = ''
    model_cfg = cfg.get('model', {}) or {}
    if isinstance(model_cfg, dict):
        bbox_head_cfg = model_cfg.get('bbox_head', {}) or {}
        s2h_cfg = bbox_head_cfg.get('s2h_cfg', {}) or {}
        s2h_default = s2h_cfg.get('knowledge_path', '') or ''

    for stage in STAGES:
        work_dir = Path(args.work_root) / dataset / stage / \
            f'seed{args.seed}' / f'{shot}shot'
        checkpoint, matched, best_candidates = resolve_checkpoint(
            work_dir, args.ckpt_pattern)
        knowledge = resolve_knowledge(args.knowledge_root, dataset, stage,
                                      args.seed, shot, s2h_default) \
            if stage != 'baseline' else None
        if checkpoint is None or (stage != 'baseline' and knowledge is None):
            missing = 'checkpoint' if checkpoint is None else 'knowledge'
            print(f'[warn] {dataset} {shot}-shot {stage}: missing {missing} '
                  f'({work_dir})')
            available[stage] = False
            continue
        print(f'[model] {dataset} {shot}-shot {stage}: '
              f'ckpt={checkpoint.name} [{matched}]'
              + (f' knowledge={knowledge}' if knowledge else ''))
        used_checkpoints[stage] = f'{checkpoint} [{matched}]'
        if args.show_map:
            epoch = _epoch_from_name(checkpoint.name)
            stage_maps[stage] = best_map_for(work_dir, epoch)
            if stage_maps[stage] is not None:
                print(f'[mAP] {dataset} {shot}-shot {stage}: '
                      f'mAP={stage_maps[stage]:.4f}'
                      + (f' (epoch {epoch})' if epoch is not None else ''))
        if matched != args.ckpt_pattern:
            preview = ', '.join(p.name for p in best_candidates[:6])
            if len(best_candidates) > 6:
                preview += f', ... (+{len(best_candidates) - 6})'
            print(f'[warn] no `{args.ckpt_pattern}` in {work_dir}')
            print(f'       using `{checkpoint.name}`; best_*.pth present: '
                  f'{preview or "none"}')
            print('       this may differ from the checkpoint behind the '
                  'reported metrics - pass --ckpt-pattern, or run with '
                  '--list-checkpoints to inspect the directory')
        model = load_stage_model(cfg, stage, checkpoint, knowledge, classes,
                                 args.device)
        for record in candidates:
            predictions[stage][record['name']] = predict_image(
                model, record, args.score_thr)
        available[stage] = True
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - CPU only runs
            pass

    if not any(available.values()):
        print(f'[skip] {dataset} {shot}-shot: no stage could be loaded')
        return [], []
    if not all(available.values()) and not args.allow_missing:
        missing = [s for s in STAGES if not available[s]]
        print(f'[skip] {dataset} {shot}-shot: missing stages {missing}. '
              'Re-run with --allow-missing to render the remaining panels.')
        return [], []

    # ---- rank the candidates --------------------------------------------
    stages = list(getattr(args, 'stage_list', STAGES))
    dropped = {'min-best-score': 0, 'min-final-score': 0,
               'require-improvement': 0, 'require-full-best': 0,
               'require-clean-rising': 0, 'min-precision': 0,
               'require-rising': 0}
    best_stage_count: Dict[str, int] = {}
    improved = full_best = rising = clean = 0
    tp_base_sum = fp_base_sum = tp_full_sum = fp_full_sum = 0
    entries = []
    for record in candidates:
        preds = {stage: predictions[stage].get(record['name'])
                 if available.get(stage) else None for stage in stages}
        chains = build_chains(record['gt_boxes'], record['gt_labels'], preds,
                              stages)
        stats = chain_stats(chains, stages)
        det = detection_stats(record['gt_boxes'], record['gt_labels'], preds,
                              stages)
        best_stage_count[stats['best_stage']] = \
            best_stage_count.get(stats['best_stage'], 0) + 1
        improved += int(stats['full_improved'])
        full_best += int(stats['full_is_best'])
        rising += int(stats['best_climb'] >= args.min_lift)
        clean += int(clean_ok(det, args.max_fp_delta))
        tp_base_sum += det['tp_base']
        fp_base_sum += det['fp_base']
        tp_full_sum += det['tp_full']
        fp_full_sum += det['fp_full']
        if args.min_best_score > 0 and stats['best'] < args.min_best_score:
            dropped['min-best-score'] += 1
            continue
        if args.min_final_score > 0 and stats['full'] < args.min_final_score:
            dropped['min-final-score'] += 1
            continue
        if args.require_improvement and not stats['full_improved']:
            dropped['require-improvement'] += 1
            continue
        if args.require_full_best and not stats['full_is_best']:
            dropped['require-full-best'] += 1
            continue
        if args.min_precision > 0 and det['precision_full'] < args.min_precision:
            dropped['min-precision'] += 1
            continue
        if args.require_clean_rising and not clean_ok(det, args.max_fp_delta):
            dropped['require-clean-rising'] += 1
            continue
        entries.append((record, preds, chains, stats, det))

    ranking = sorted(best_stage_count.items(), key=lambda item: -item[1])
    print(f'[stats] {dataset} {shot}-shot: best stage over '
          f'{len(candidates)} candidates -> '
          + ', '.join(f'{name}={count}' for name, count in ranking)
          + f' | Full > Baseline on {improved}/{len(candidates)} samples, '
          f'Full is the best stage on {full_best}/{len(candidates)}')
    print(f'[stats] {dataset} {shot}-shot: ascending chains '
          f'(baseline->ffcp->ffcp+chsd->full, one GT box climbs >= '
          f'{args.min_lift:.2f}) on {rising}/{len(candidates)} candidates')
    ref_stage = stages[0] if 'baseline' not in stages else 'baseline'
    last_stage = stages[-1] if 'full' not in stages else 'full'
    print(f'[stats] {dataset} {shot}-shot: wrong detections @score-thr '
          f'{args.score_thr:g} (class-aware, IoU>={MATCH_IOU:g}) -> TP '
          f'{tp_base_sum}->{tp_full_sum}, FP {fp_base_sum}->{fp_full_sum} '
          f'summed over {len(candidates)} candidates; '
          f'{STAGE_TITLES[last_stage]} vs {STAGE_TITLES[ref_stage]} adds no '
          f'wrong box and keeps every hit on {clean}/{len(candidates)} samples')
    if 'baseline' not in stages:
        print(f'[note] {dataset} {shot}-shot: `baseline` is not drawn, so the '
              f'wrong-detection comparison uses {STAGE_TITLES[ref_stage]} as '
              'the reference')
    if any(dropped.values()):
        print('[stats] dropped by filters: '
              + ', '.join(f'{name}={count}' for name, count in dropped.items()
                          if count))

    # ---- quality level, relaxed automatically down to --min-requirement ---
    if args.require_rising:
        floor = LEVELS.index(args.min_requirement)
        chosen = None
        for name in LEVELS[:floor + 1]:
            keep = [entry for entry in entries
                    if level_ok(name, entry[3], args.min_lift)]
            if keep:
                chosen, entries = name, keep
                break
        if chosen is None:
            print(f'[skip] {dataset} {shot}-shot: no candidate reaches '
                  f'`{args.min_requirement}`; lower --min-requirement or check '
                  'the checkpoints')
            return [], []
        if chosen != 'rising':
            print(f'[relax] {dataset} {shot}-shot: no ascending chain -> using '
                  f'`{chosen}` ({len(entries)}/{len(candidates)} candidates)')

    scored = [(rank_key(chains, args.rank_by, stages, det), record, preds,
               chains, stats, det)
              for record, preds, chains, stats, det in entries]
    if not scored:
        print(f'[skip] {dataset} {shot}-shot: no candidate survived the filters '
              '(relax --min-best-score / --require-full-best)')
        return [], []
    scored.sort(key=lambda item: tuple(-float(v) for v in item[0])
                + (item[1]['name'],))

    # ranking table: use the printed stems with --img-names to fix the samples
    if args.list_top > 0:
        print(f'[rank] {dataset} {shot}-shot top '
              f'{min(args.list_top, len(scored))} of {len(scored)} '
              '(TP/FP = matched/wrong boxes at the last drawn stage vs '
              'baseline):')
        for position, (key, record, _, chains, _, det) in \
                enumerate(scored[:args.list_top], start=1):
            print(f'  {position:2d}. {record["name"]:<24s} '
                  f'{format_caption(chains, classes, args.pretty_names, stages)}'
                  f'   [TP {det["tp_base"]}->{det["tp_full"]}, '
                  f'FP {det["fp_base"]}->{det["fp_full"]}]')
    if args.num_images <= 0:
        print(f'[info] --num-images 0: nothing rendered for '
              f'{dataset} {shot}-shot')
        return [], []

    chosen = scored[:max(int(args.num_images), 1)]
    rows, dump = [], []
    # with several shots the row label has to say which one it is
    label = display_name(dataset) + (
        f' {shot}-shot' if len(args.shots) > 1 else '')
    for rank, (key, record, preds, chains, stats, det) in \
            enumerate(chosen, start=1):
        rows.append(dict(dataset=dataset, shot=shot, label=label,
                         name=record['name'], classes=classes, maps=stage_maps,
                         image=read_image(record['img_path']),
                         gt_boxes=record['gt_boxes'],
                         gt_labels=record['gt_labels'],
                         predictions=preds,
                         caption=format_caption(chains, classes,
                                                args.pretty_names, stages)))
        dump.append(dict(
            image=record['img_path'], name=record['name'], rank=rank,
            rank_key=list(key), stats=stats, detection=det,
            gt=[dict(label=int(l), name=classes[int(l)],
                     box=[float(v) for v in b], chain=c)
                for l, b, c in chains],
            caption=rows[-1]['caption']))

    tag = f'{dataset}_{shot}shot'
    subtitle = (f'{display_name(dataset)} {shot}-shot   |   '
                f'validation-best checkpoints (seed {args.seed})')
    written = render_rows(rows, Path(args.out), tag, args, subtitle)

    json_path = Path(args.out) / f'{tag}.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(dict(dataset=dataset, shot=shot, seed=args.seed,
                       columns=list(stages), available=available,
                       checkpoints=used_checkpoints,
                       best_stage_counts=best_stage_count,
                       full_improved=improved, full_best=full_best,
                       clean_samples=clean,
                       tp_fp_sums=dict(tp_base=tp_base_sum,
                                       fp_base=fp_base_sum,
                                       tp_full=tp_full_sum,
                                       fp_full=fp_full_sum),
                       require_clean_rising=args.require_clean_rising,
                       max_fp_delta=args.max_fp_delta,
                       min_precision=args.min_precision,
                       dropped=dropped, rank_by=args.rank_by,
                       score_thr=args.score_thr, rows=dump),
                  handle, indent=2, ensure_ascii=False)
    written.append(json_path)
    for path in written:
        print(f'[write] {path}')
    return written, rows


def main(argv=None) -> int:
    raw = list(sys.argv[1:]) if argv is None else list(argv)
    args = parse_args(raw)
    if args.self_check:
        return self_check()
    args.stage_list = parse_stages(args.columns)
    args.min_requirement = parse_level(args.min_requirement)

    # match the training launcher: never let ~/.local shadow the conda stack
    if not args.no_reexec:
        ensure_conda_torch([sys.argv[0]] + raw)

    os.makedirs(args.out, exist_ok=True)
    datasets = [canonical_dataset(name) for name in args.datasets]
    if args.list_checkpoints:
        for dataset in datasets:
            for shot in args.shots:
                list_checkpoints(dataset, str(shot), args)
        return 0

    check_framework_imports()

    combined: Dict[str, List[dict]] = {}
    per_dataset: Dict[str, List[dict]] = {}
    aligned: Dict[str, str] = {}
    for dataset in datasets:
        for shot in args.shots:
            shot = str(shot)
            local_args = args
            if args.align_shots and dataset in aligned:
                local_args = copy.copy(args)
                local_args.img_names = aligned[dataset]
            _, rows = run_one_dataset(dataset, shot, local_args)
            if args.align_shots and dataset not in aligned and rows:
                aligned[dataset] = ','.join(row['name'] for row in rows)
            combined.setdefault(shot, []).extend(rows)
            per_dataset.setdefault(dataset, []).extend(rows)

    # one dataset across shots: shows whether 5/10-shot repairs a stage
    if len(args.shots) > 1:
        shot_tag = '_'.join(f'{shot}shot' for shot in args.shots)
        for dataset, rows in per_dataset.items():
            if not rows:
                continue
            subtitle = (f'{display_name(dataset)}   |   '
                        + ', '.join(f'{shot}-shot' for shot in args.shots)
                        + f'   |   validation-best checkpoints '
                          f'(seed {args.seed})')
            written = render_rows(rows, Path(args.out),
                                  f'{dataset}_{shot_tag}', args, subtitle,
                                  write_single_rows=False)
            for path in written:
                print(f'[write] {path}')

    # a single figure stacking every dataset, like the qualitative table
    if len(datasets) > 1:
        names = ' / '.join(display_name(d) for d in datasets)
        for shot, rows in combined.items():
            if not rows:
                continue
            subtitle = (f'{names}   |   {shot}-shot   |   '
                        f'validation-best checkpoints (seed {args.seed})')
            written = render_rows(rows, Path(args.out),
                                  f'all_datasets_{shot}shot', args, subtitle,
                                  write_single_rows=False)
            for path in written:
                print(f'[write] {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

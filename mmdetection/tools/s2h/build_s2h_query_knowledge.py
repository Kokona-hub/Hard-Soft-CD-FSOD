#!/usr/bin/env python
"""Build S2H knowledge whose Soft prototype lives in decoder-query space.

The original builder pools backbone/neck features inside support boxes.  ATAR
actually compares the prototype with the final Grounding-DINO decoder query,
so this builder extracts those queries from the same frozen checkpoint and
matches them to support annotations by IoU.

The default ``--base-knowledge`` mode is deliberately conservative: it keeps
the existing Hard prototype, reliability and context statistics and replaces
only ``soft``.  This is the controlled experiment needed to isolate a
representation-space mismatch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Make the repository importable when launched from tools/s2h.
ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
S2H_DIR = osp.dirname(__file__)
if S2H_DIR not in sys.path:
    sys.path.insert(0, S2H_DIR)

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise xyxy IoU, returned as [len(a), len(b)]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    tl = np.maximum(a[..., :2], b[..., :2])
    br = np.minimum(a[..., 2:], b[..., 2:])
    wh = np.maximum(br - tl, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(boxes_a[:, 2] - boxes_a[:, 0], 0) * \
        np.maximum(boxes_a[:, 3] - boxes_a[:, 1], 0)
    area_b = np.maximum(boxes_b[:, 2] - boxes_b[:, 0], 0) * \
        np.maximum(boxes_b[:, 3] - boxes_b[:, 1], 0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter,
                              1e-8)


def query_boxes_original(coords, img_shape, scale_factor) -> np.ndarray:
    """Mirror GroundingDINO postprocessing for normalized cxcywh boxes."""
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    boxes = bbox_cxcywh_to_xyxy(coords.float())
    boxes[:, 0::2] *= float(img_shape[1])
    boxes[:, 1::2] *= float(img_shape[0])
    scale = torch.as_tensor(scale_factor).float().reshape(-1)
    if scale.numel() == 4:
        boxes /= scale.repeat(2)
    else:
        boxes[:, 0::2] /= scale[0]
        boxes[:, 1::2] /= scale[1]
    return boxes.cpu().numpy().astype(np.float32)


def resolve_img_path(path: str, data_root: str) -> Optional[str]:
    candidates = [path]
    if data_root:
        candidates.extend([
            osp.join(data_root, path),
            osp.join(data_root, *path.replace('\\', '/').split('/')),
        ])
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    return None


def dataset_from_cfg(cfg):
    """Build the train dataset without using its augmentation pipeline."""
    from mmdet.registry import DATASETS

    node = copy.deepcopy(cfg.train_dataloader.dataset)
    while isinstance(node, dict) and 'dataset' in node and \
            node.get('type') in ('MultiImageMixDataset', 'RepeatDataset',
                                 'ClassBalancedDataset'):
        node = node['dataset']
    node.pop('pipeline', None)
    node['lazy_init'] = False
    return DATASETS.build(node)


def records_from_dataset(dataset) -> Tuple[List[str], List[dict]]:
    classes = [str(x) for x in dataset.metainfo['classes']]
    data_root = str(getattr(dataset, 'data_root', '') or '')
    records = []
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        instances = [x for x in info.get('instances', [])
                     if not x.get('ignore_flag', 0)]
        if not instances:
            continue
        path = resolve_img_path(info['img_path'], data_root)
        if path is None:
            raise FileNotFoundError(
                f"cannot resolve support image {info['img_path']!r} "
                f"under data_root={data_root!r}")
        records.append(dict(
            index=index,
            img_path=path,
            text=info.get('text') or tuple(classes),
            custom_entities=bool(info.get('custom_entities', True)),
            boxes=np.asarray([x['bbox'] for x in instances], np.float32),
            labels=np.asarray([int(x['bbox_label']) for x in instances],
                              np.int64)))
    if not records:
        raise RuntimeError('the train/support split contains no annotations')
    return classes, records


class QueryCapture:
    """Capture final decoder hidden states and boxes around inference."""

    def __init__(self, model):
        self.model = model
        self.hidden = None
        self.coords = None
        self.tokens = None
        self.samples = None
        head = model.bbox_head
        self.handles = [
            head.register_forward_pre_hook(self._pre),
            head.register_forward_hook(self._post),
        ]
        original = model.predict
        self.original_predict = original
        capture = self

        def wrapped(*args, **kwargs):
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, (list, tuple)) and value and \
                        hasattr(value[0], 'metainfo'):
                    capture.samples = value
                    break
            return original(*args, **kwargs)

        model.predict = wrapped

    def _pre(self, module, args):
        self.hidden = args[0]

    def _post(self, module, args, output):
        self.tokens = output[0]
        self.coords = output[1]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model.predict = self.original_predict


def category_scores(token_logits, positive_map):
    from mmdet.models.dense_heads.grounding_dino_head import \
        convert_grounding_to_cls_scores

    with torch.no_grad():
        scores = convert_grounding_to_cls_scores(
            logits=token_logits[None].sigmoid(), positive_maps=[positive_map])[0]
    return scores.detach().cpu().numpy().astype(np.float32)


def match_support_queries(hidden: np.ndarray, boxes: np.ndarray,
                          scores: np.ndarray, gt_boxes: np.ndarray,
                          gt_labels: np.ndarray, iou_thr: float,
                          score_thr: float) -> Tuple[Dict[int, List[np.ndarray]],
                                                     Dict[str, int]]:
    """Greedy one-query-per-GT matching with strict class agreement."""
    labels = scores.argmax(-1).astype(np.int64)
    top_scores = scores.max(-1)
    ious = iou_matrix(boxes, gt_boxes)
    out: Dict[int, List[np.ndarray]] = {}
    used = set()
    strict = 0
    candidate = 0
    for gt_index, gt_label in enumerate(gt_labels.tolist()):
        order = np.argsort(-ious[:, gt_index])
        chosen = None
        for query_index in order.tolist():
            if ious[query_index, gt_index] < iou_thr:
                break
            if query_index in used:
                continue
            candidate += 1
            if (labels[query_index] == int(gt_label) and
                    top_scores[query_index] >= score_thr):
                chosen = query_index
                break
        if chosen is None:
            continue
        used.add(chosen)
        strict += 1
        out.setdefault(int(gt_label), []).append(hidden[chosen])
    return out, dict(gt=len(gt_labels), matched=strict, candidates=candidate)


def text_embeddings(model, classes: Sequence[str], device):
    from mmdet.models.utils.s2h_knowledge import l2_normalize
    from tools.s2h.build_s2h_knowledge import get_token_ids

    token_ids, caption = get_token_ids(model, classes)
    text_dict = model.language_model([caption])
    emb = text_dict['embedded']
    if model.text_feat_map is not None:
        emb = model.text_feat_map(emb)
    emb = emb[0]
    hard = []
    for ids in token_ids:
        hard.append(emb[ids].mean(0) if ids else emb.new_zeros(emb.shape[-1]))
    return l2_normalize(torch.stack(hard), dim=-1).detach().cpu(), token_ids


def save_payload(path: str, payload: dict):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    print(f'[query-knowledge] saved -> {path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build decoder-query-space S2H Soft prototypes')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--out')
    parser.add_argument('--base-knowledge', default=None,
                        help='keep hard/kappa/reliability from this old file')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--shrink-xi', type=float, default=5.0)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--stage', choices=['ffcp', 'ffcp_chsd', 'full'],
                        default='full')
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args()


def self_check():
    boxes = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    gt = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    scores = np.asarray([[.9, .1], [.8, .2]], np.float32)
    hidden = np.eye(2, dtype=np.float32)
    out, stats = match_support_queries(
        hidden, boxes, scores, gt, np.asarray([0, 0]), .5, 0.0)
    assert stats['matched'] == 2 and len(out[0]) == 2, (out, stats)
    # A wrong predicted class must never become a support prototype.
    wrong, wrong_stats = match_support_queries(
        hidden, boxes, scores[:, ::-1], gt, np.asarray([0, 0]), .5, 0.0)
    assert wrong_stats['matched'] == 0 and not wrong, (wrong, wrong_stats)
    assert iou_matrix(gt, gt).diagonal().tolist() == [1.0, 1.0]
    print('[query-knowledge] self-check ok')


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return
    for name in ('config', 'checkpoint', 'out'):
        if not getattr(args, name):
            raise SystemExit(f'--{name} is required unless --self-check is used')
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import inference_detector, init_detector
    from mmdet.models.utils.s2h_knowledge import l2_normalize, robust_mean

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    # The builder must never apply an old S2H correction while extracting
    # support queries.  The checkpoint remains the user's frozen detector.
    if 'bbox_head' in cfg.model and cfg.model.bbox_head.get('s2h_cfg'):
        cfg.model.bbox_head.s2h_cfg.enabled = False
    dataset = dataset_from_cfg(cfg)
    classes, records = records_from_dataset(dataset)
    if args.max_images > 0:
        records = records[:args.max_images]
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = init_detector(cfg, args.checkpoint, device=device)
    model.eval()
    hard_text, token_ids = text_embeddings(model, classes, device)
    capture = QueryCapture(model)

    per_class: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(classes))}
    stats = dict(images=0, gt=0, matched=0, candidates=0)
    for position, record in enumerate(records):
        inference_detector(model, record['img_path'],
                           text_prompt=record['text'],
                           custom_entities=record['custom_entities'])
        if capture.hidden is None or capture.coords is None or not capture.samples:
            raise RuntimeError('query capture produced no decoder output')
        sample = capture.samples[0]
        meta = sample.metainfo
        positive_map = getattr(sample, 'token_positive_map', None)
        if positive_map is None:
            positive_map = meta.get('token_positive_map')
        if positive_map is None:
            raise RuntimeError('support inference produced no token_positive_map')
        hidden = capture.hidden[-1][0].detach().float()
        coords = capture.coords[-1][0].detach().float()
        boxes = query_boxes_original(coords, tuple(meta['img_shape'][:2]),
                                     meta['scale_factor'])
        if capture.tokens is None:
            raise RuntimeError('token logits were not captured')
        scores = category_scores(capture.tokens[-1][0], positive_map)
        matched, row_stats = match_support_queries(
            hidden.cpu().numpy(), boxes, scores, record['boxes'],
            record['labels'], args.iou_threshold, args.score_threshold)
        for cls, vectors in matched.items():
            per_class[cls].extend(torch.from_numpy(np.stack(vectors)))
        stats['images'] += 1
        for key, value in row_stats.items():
            stats[key] += value
        if (position + 1) % 10 == 0:
            print(f'[query-knowledge] {position + 1}/{len(records)} images, '
                  f'matched={stats["matched"]}/{stats["gt"]}')
    capture.close()

    base = torch.load(args.base_knowledge, map_location='cpu') \
        if args.base_knowledge else None
    if base is not None and list(base['classes']) != classes:
        raise ValueError('base knowledge classes do not match config classes')

    soft = []
    reliability = []
    counts = {}
    for cls in range(len(classes)):
        vectors = per_class[cls]
        counts[classes[cls]] = len(vectors)
        if vectors:
            x = l2_normalize(torch.stack(vectors), dim=-1)
            mean = robust_mean(x, dim=0)
            # Shot-aware shrinkage is applied in the same query space, toward
            # the text identity, and is not erased by a later normalization.
            alpha = args.shrink_xi / (len(vectors) + args.shrink_xi)
            proto = (1.0 - alpha) * mean + alpha * hard_text[cls]
            soft.append(l2_normalize(proto.unsqueeze(0), dim=-1)[0])
            # A conservative support-side confidence: more matched queries and
            # more stable query features both improve reliability.
            spread = float((x @ l2_normalize(mean.unsqueeze(0), dim=-1)[0]).mean())
            reliability.append(float(np.clip(spread * min(1.0, len(vectors) / 3.0), 0, 1)))
        else:
            soft.append(hard_text[cls])
            reliability.append(0.0)
    soft = torch.stack(soft).float()
    if base is None:
        hard = hard_text.float()
        kappa = torch.tensor([1.0 - x for x in reliability])
        rel = torch.tensor(reliability)
    else:
        hard = base['hard'].float()
        kappa = base['kappa'].float()
        rel = base['reliability'].float()
    payload = dict(
        hard=hard, soft=soft, kappa=kappa, reliability=rel,
        classes=classes, token_ids=token_ids,
        meta=dict(
            stage=args.stage,
            representation='decoder_query',
            soft_source='matched_final_decoder_queries',
            config=args.config, checkpoint=args.checkpoint,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            shrink_xi=args.shrink_xi,
            support_stats=stats, matched_per_class=counts,
            base_knowledge=args.base_knowledge,
        ))
    save_payload(args.out, payload)


if __name__ == '__main__':
    main()

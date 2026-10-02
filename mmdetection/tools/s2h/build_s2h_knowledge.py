# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Offline builder for the S2H-CD-FSOD Hard-Soft support knowledge.

This script runs the two support-side stages of the paper and caches the
resulting knowledge tuple ``K_c = (h_c, s_c, kappa_c)`` into a ``.pth`` file
that is later consumed by :class:`S2HKnowledgeBank`:

* **FFCP** -- renders foreground-frozen support interventions, extracts the
  visual descriptors of the original / intervened / reconstruction views and
  estimates the class-wise context-sensitive subspace ``U_ctx`` and the
  context dependence ``kappa_ctx`` (Eqs. (3)-(5)).
* **CHSD** -- builds the bounded Hard identity anchor ``h_c`` and the
  context-cleaned, shot-aware Soft prototype ``s_c`` (Eqs. (6)-(9)).

Usage
-----
::

    python tools/s2h/build_s2h_knowledge.py \
        --config configs/s2h_dino/s2h_grounding_dino_swin-b_ArTaxOr_1shot.py \
        --checkpoint https://download.openmmlab.com/.../groundingdino_swinb_cogcoor_mmdet-55949c9c.pth \
        --out work_dirs/s2h_knowledge/ArTaxOr_1shot.pth \
        --stage full

For a quick integration smoke test (random knowledge with a correct token map)
add ``--dry-run``.
"""
import argparse
import copy
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# make the mmdet source tree importable when running from the repo root
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from mmcv.transforms import Compose  # noqa: E402
from mmdet.models.utils.s2h_knowledge import (  # noqa: E402
    context_leakage_energy, get_attribute_candidates, l2_normalize,
    orthonormalize, pool_features_in_boxes, project_l2_ball, project_out,
    robust_mean, svd_context_basis)
from support_interventions import generate_interventions  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def build_preprocess_pipeline() -> Compose:
    """Pipeline that mirrors the Grounding DINO test-time preprocessing."""
    return Compose([
        dict(type='LoadImageFromNDArray'),
        dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor')),
    ])


def preprocess_pil(model, pil_img: Image.Image, pipe: Compose, device):
    """Preprocess a PIL image exactly like the training/validation pipeline."""
    bgr = np.asarray(pil_img.convert('RGB'))[:, :, ::-1].copy()
    data = pipe(dict(img=bgr, img_id=0, img_path=None))
    batch = model.data_preprocessor(
        dict(inputs=[data['inputs']], data_samples=[data['data_samples']]),
        training=False)
    return batch['inputs'].to(device), batch['data_samples'][0]


@torch.no_grad()
def extract_box_features(model, pil_img, boxes, img_shape, pipe, device,
                         dilate: float = 1.0) -> Optional[torch.Tensor]:
    """Pool visual descriptors inside boxes for a single PIL image."""
    if len(boxes) == 0:
        return None
    imgs, data_sample = preprocess_pil(model, pil_img, pipe, device)
    feat_shape = tuple(data_sample.metainfo['img_shape'])
    sf = data_sample.metainfo['scale_factor']
    sf = torch.as_tensor(sf, dtype=torch.float32, device=device)
    boxes_t = torch.as_tensor(np.asarray(boxes), dtype=torch.float32,
                              device=device)
    boxes_t = boxes_t * sf.repeat(2) if sf.numel() == 2 else boxes_t * sf
    feats = model.extract_feat(imgs)
    pooled = pool_features_in_boxes(feats, [boxes_t], feat_shape,
                                    dilate=dilate)[0]
    return pooled.detach().float().cpu()


@torch.no_grad()
def text_embedding(model, phrases: Sequence[str], device) -> torch.Tensor:
    """Embed phrases into the Grounding DINO alignment space ``[N, D]``."""
    if len(phrases) == 0:
        return torch.zeros(0, 256)
    text_dict = model.language_model(list(phrases))
    emb = text_dict['embedded']
    if model.text_feat_map is not None:
        emb = model.text_feat_map(emb)
    mask = text_text_mask(text_dict, emb)
    emb = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return emb.detach().float().cpu()


def text_text_mask(text_dict, emb):
    if 'text_token_mask' in text_dict:
        mask = text_dict['text_token_mask'].to(emb.device).float()
    else:
        mask = text_dict['masks']
        if mask.dim() > 2:
            mask = mask.mean(-1) if mask.shape[-1] == emb.shape[1] else \
                mask.any(-1).float()
        mask = mask.to(emb.device).float()
    if mask.shape[1] != emb.shape[1]:
        mask = mask[:, :emb.shape[1]]
    return mask


def class_text_embeddings(model, caption_string, token_ids, device):
    """Text embedding of every category, taken from its own prompt tokens."""
    text_dict = model.language_model([caption_string])
    emb = text_dict['embedded']
    if model.text_feat_map is not None:
        emb = model.text_feat_map(emb)
    emb = emb[0]                                       # [L, D]
    out = []
    for idxs in token_ids:
        if len(idxs) == 0:
            out.append(torch.zeros(emb.shape[-1]))
        else:
            out.append(emb[idxs].mean(0))
    return torch.stack(out, dim=0).detach().float().cpu()


def get_token_ids(model, classes) -> Tuple[List[List[int]], str]:
    """Class -> text-token index map using the exact Grounding DINO prompt."""
    tokenized, caption_string, tokens_positive, _ = \
        model.get_tokens_and_prompts(list(classes), True)
    _, positive_map = model.get_positive_map(tokenized, tokens_positive)
    token_ids = []
    for c in range(len(classes)):
        idxs = torch.nonzero(positive_map[c]).flatten().tolist()
        token_ids.append([int(i) for i in idxs])
    return token_ids, caption_string


def resolve_img_path(img_path: str, data_root: str) -> Optional[str]:
    """Resolve an image path against the dataset root (mmengine versions
    differ in whether ``data_root`` is already joined)."""
    candidates = [img_path]
    if data_root:
        candidates.append(osp.join(data_root, img_path))
        candidates.append(osp.join(data_root, *img_path.split('/')))
    for cand in candidates:
        if osp.exists(cand):
            return cand
    return None


def build_dataset(config: dict, max_samples: int = 0):
    """Return ``(classes, per_class_samples)`` from the training dataloader."""
    from mmengine.registry import DATASETS
    ds_cfg = copy.deepcopy(config['train_dataloader']['dataset'])
    dataset_type = ds_cfg.pop('type')
    # Config-only directives must not reach the dataset constructor.
    for key in ('_delete_', '_scope_'):
        ds_cfg.pop(key, None)
    dataset = DATASETS.build(dict(type=dataset_type, **ds_cfg))
    classes = list(dataset.metainfo['classes'])
    data_root = ds_cfg.get('data_root', '') or ''

    per_class: Dict[int, List[dict]] = {i: [] for i in range(len(classes))}
    limit = max_samples if max_samples > 0 else len(dataset)
    n_used = 0
    for i in range(min(limit, len(dataset))):
        info = dataset.get_data_info(i)
        instances = info.get('instances', [])
        if not instances:
            continue
        keep = [ins for ins in instances if not ins.get('ignore_flag', 0)]
        if not keep:
            continue
        bboxes = np.asarray([ins['bbox'] for ins in keep], dtype=np.float32)
        labels = np.asarray([int(ins['bbox_label']) for ins in keep])
        img_path = resolve_img_path(info['img_path'], data_root)
        if img_path is None:
            print(f"warning: cannot resolve image path {info['img_path']}")
            continue
        n_used += 1
        for label in np.unique(labels):
            if int(label) >= len(classes):
                continue
            sel = bboxes[labels == label]
            if sel.shape[0] == 0:
                continue
            # one descriptor per (image, category)
            per_class[int(label)].append(
                dict(img_path=img_path, boxes=sel.tolist()))
    print(f'[S2H] dataset: {n_used} support images, {len(classes)} categories')
    return classes, per_class


# ---------------------------------------------------------------------------
# FFCP + CHSD
# ---------------------------------------------------------------------------
def compute_context_subspace(drifts: torch.Tensor, rank: int,
                             reference: Optional[torch.Tensor] = None):
    """FFCP: ``U_ctx`` + ``e_ctx`` + ``kappa_ctx`` (Eqs. (4)-(5))."""
    if drifts.numel() == 0:
        return drifts.new_zeros((0, 0)), drifts.new_zeros(())
    u_ctx, _ = svd_context_basis(drifts, rank)
    e_ctx = context_leakage_energy(drifts, u_ctx, reference=reference)
    kappa = 1.0 - torch.exp(-e_ctx)
    return u_ctx, kappa.clamp(0.0, 1.0 - 1e-6)


def build_class_knowledge(desc: dict, classes, token_ids, text_emb, args,
                          model, pipe, device) -> dict:
    """Run FFCP + CHSD for the class described by ``desc``."""
    c = desc['class_id']
    samples = desc['samples']
    # Keep the sample/view axes until after Eq. (4). Flattening all views into
    # one list loses the correspondence between a foreground and its probes
    # and can silently mix different boxes or intervention types.
    f_fg, f_tilde_views, f_rec = [], {k: [] for k in (
        'same_domain', 'class_mismatch', 'neutralized')}, []
    rng = np.random.RandomState(args.seed + c)

    for k, sample in enumerate(samples):
        img = Image.open(sample['img_path']).convert('RGB')
        boxes = sample['boxes']
        # background sources: another support image of the same class and one
        # of a different class
        same_src = None
        if len(samples) > 1:
            other = samples[(k + 1) % len(samples)]
            same_src = Image.open(other['img_path']).convert('RGB')
        elif desc.get('same_domain_src') is not None:
            # In 1-shot-per-class settings there is no second image of the
            # same category. Use another target-domain support image so the
            # same-domain probe still changes context without adding labels.
            same_src = Image.open(desc['same_domain_src']).convert('RGB')
        diff_src = desc.get('mismatch_src')
        if diff_src is not None:
            diff_src = Image.open(diff_src).convert('RGB')

        views = generate_interventions(
            img, boxes, same_domain_src=same_src,
            class_mismatch_src=diff_src, dilate=args.dilate,
            neutral_mode=args.neutral_mode, rng=rng)

        f0 = extract_box_features(model, img, boxes, img.size, pipe, device)
        if f0 is None:
            continue
        f_fg.append(f0)
        for key in ('same_domain', 'class_mismatch', 'neutralized'):
            ft = extract_box_features(model, views[key], boxes, img.size,
                                      pipe, device)
            if ft is not None:
                f_tilde_views[key].append(ft)
        fr = extract_box_features(model, views['reconstruction'], boxes,
                                  img.size, pipe, device, dilate=1.0)
        if fr is not None:
            f_rec.append(fr)

    if not f_fg:
        return None

    f_fg = torch.cat(f_fg, 0)                      # [K, D]
    f_rec = torch.cat(f_rec, 0) if f_rec else f_fg.clone()
    n_base = min(f_fg.shape[0], f_rec.shape[0])
    f_fg = f_fg[:n_base]
    f_rec = f_rec[:n_base]

    # Eq. (4), evaluated independently for each intervention r. A missing
    # source (common in 1-shot) is represented by the neutralized view rather
    # than changing the number of rows for only one intervention type.
    view_features = []
    aligned_fg, aligned_tilde = [], []
    for key in ('same_domain', 'class_mismatch', 'neutralized'):
        if f_tilde_views[key]:
            ft = torch.cat(f_tilde_views[key], 0)[:n_base]
        else:
            ft = f_fg.clone()
        n = min(f_fg.shape[0], ft.shape[0], f_rec.shape[0])
        fg_n, ft_n, rec_n = f_fg[:n], ft[:n], f_rec[:n]
        view_features.append((fg_n - ft_n) - (fg_n - rec_n))
        aligned_fg.append(fg_n)
        aligned_tilde.append(ft_n)
    drift = torch.cat(view_features, 0)              # [R*K, D]
    f_fg_cf = torch.cat(aligned_fg, 0)
    f_tilde = torch.cat(aligned_tilde, 0)

    # ---- FFCP -------------------------------------------------------
    u_ctx, kappa = compute_context_subspace(
        drift, args.ffcp_rank, reference=f_fg_cf)

    # FFCP-only stops here: no structural-attribute scoring, Hard anchor or
    # CHSD projection is allowed to leak into this ablation.
    t0 = l2_normalize(text_emb[c].unsqueeze(0), dim=-1)[0]
    if args.stage == 'ffcp':
        s_c = robust_mean(project_out(f_fg, u_ctx), dim=0)
        reliability = float(np.clip(1.0 - float(kappa), 0.0, 1.0))
        return dict(
            hard=t0,
            soft=l2_normalize(s_c, dim=-1),
            kappa=float(kappa), reliability=reliability,
            coverage=0.0, n_attr_kept=0, n_attr=0,
            n_support=int(f_fg.shape[0]),
            u_ctx_rank=int(u_ctx.shape[0]) if u_ctx.numel() else 0,
            stage='ffcp')

    # ---- CHSD: Hard identity anchor -------------------------------
    attrs = get_attribute_candidates(classes[c])
    a_emb = text_embedding(model, attrs, device)                # [L, D]
    a_hat = l2_normalize(a_emb, dim=-1)
    # Counterfactual instability is evaluated on the same repeated foreground
    # rows as the three intervention families.
    f_hat = l2_normalize(f_fg_cf, dim=-1)
    ft_hat = l2_normalize(f_tilde, dim=-1)

    f_bar_fg = f_fg.mean(0)
    f_bar_ctx = f_tilde.mean(0)
    sim_fg = a_hat @ l2_normalize(f_bar_fg, dim=-1)
    sim_ctx = a_hat @ l2_normalize(f_bar_ctx, dim=-1)
    # counterfactual instability
    diff = (a_hat @ f_hat.t()) - (a_hat @ ft_hat.t())          # [L, n]
    delta_cf = diff.abs().mean(-1)

    q = sim_fg - args.lambda_b * sim_ctx - args.lambda_cf * delta_cf \
        - args.lambda_kappa * kappa
    keep = q >= args.theta_h
    if keep.any():
        w = torch.softmax(q[keep] / args.attr_temp, dim=0)
        delta_h = project_l2_ball((a_emb[keep] * w[:, None]).sum(0),
                                  args.hard_radius)
        h_c = l2_normalize(t0 + delta_h, dim=-1)
        u_h = orthonormalize(torch.cat([t0[None], a_emb[keep]], dim=0))
        coverage = float(keep.sum()) / max(len(attrs), 1)
    else:
        h_c = t0.clone()
        u_h = orthonormalize(t0[None])
        coverage = 0.0

    # ---- CHSD: Soft appearance prototype --------------------------
    u_bar_ctx = orthonormalize(project_out(u_ctx, u_h)) if u_ctx.numel() else \
        u_ctx.new_zeros((0, f_fg.shape[1]))
    r_app = project_out(project_out(f_fg, u_h), u_bar_ctx)
    k_support = float(f_fg.shape[0])
    alpha_k = args.shrink_xi / (k_support + args.shrink_xi)
    s_c = (1.0 - alpha_k) * robust_mean(r_app, dim=0)

    reliability = float(np.clip((1.0 - float(kappa)) * coverage, 0.0, 1.0))

    return dict(
        hard=l2_normalize(h_c, dim=-1),
        soft=l2_normalize(s_c, dim=-1),
        kappa=float(kappa),
        reliability=reliability,
        coverage=coverage,
        n_attr_kept=int(keep.sum()),
        n_attr=len(attrs),
        n_support=int(f_fg.shape[0]),
        u_ctx_rank=int(u_ctx.shape[0]) if u_ctx.numel() else 0,
        stage=args.stage,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser('Build S2H Hard-Soft support knowledge')
    parser.add_argument('--config', required=True,
                        help='mmdet config of the target few-shot dataset')
    parser.add_argument('--checkpoint', default=None,
                        help='detector checkpoint; defaults to config load_from')
    parser.add_argument('--out', required=True, help='output .pth path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dilate', type=float, default=1.2)
    parser.add_argument('--neutral-mode', default='gray',
                        choices=['gray', 'noise'])
    parser.add_argument('--ffcp-rank', type=int, default=4)
    parser.add_argument('--hard-radius', type=float, default=1.0)
    parser.add_argument('--attr-temp', type=float, default=0.5)
    parser.add_argument('--theta-h', type=float, default=0.3)
    parser.add_argument('--shrink-xi', type=float, default=5.0)
    parser.add_argument('--lambda-b', type=float, default=0.5)
    parser.add_argument('--lambda-cf', type=float, default=1.0)
    parser.add_argument('--lambda-kappa', type=float, default=1.0)
    parser.add_argument('--max-samples', type=int, default=0,
                        help='limit images per dataset (0 = all)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stage', choices=['ffcp', 'ffcp_chsd', 'full'],
                        default='full',
                        help='knowledge ablation stage; baseline needs no file')
    parser.add_argument('--dry-run', action='store_true',
                        help='emit random knowledge with a correct token map')
    return parser.parse_args()


def main():
    args = parse_args()
    from mmengine.config import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(args.config)
    # The builder itself produces the knowledge file, so that file cannot be
    # loaded while constructing the detector.  Disable only the in-model S2H
    # bank during extraction; training configs still enable it and load the
    # completed file afterward.  This also makes ``--dry-run`` independent of
    # any pre-existing cache while preserving the real token map.
    model_cfg = cfg.model
    if 'bbox_head' in model_cfg:
        s2h_cfg = model_cfg.bbox_head.get('s2h_cfg', None)
        if s2h_cfg is not None:
            s2h_cfg.enabled = False
    device = args.device if torch.cuda.is_available() else 'cpu'
    # ``init_detector(..., checkpoint=None)`` builds random weights and does
    # not consult ``cfg.load_from``.  FFCP/CHSD must use the same pretrained
    # detector as fine-tuning, so honor the config fallback explicitly.
    checkpoint = args.checkpoint or cfg.get('load_from')
    if not checkpoint:
        raise ValueError(
            'no detector checkpoint was provided and config.load_from is empty')
    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    classes, _ = _safe_classes(cfg)
    token_ids, caption_string = get_token_ids(model, classes)
    print(f'[S2H] {len(classes)} categories, '
          f'{sum(len(t) for t in token_ids)} text tokens')

    if args.dry_run:
        print('[S2H] dry-run: emitting random Hard/Soft knowledge')
        n = len(classes)
        d = model.bbox_head.embed_dims if hasattr(model.bbox_head, 'embed_dims') \
            else cfg.model.get('embed_dims', 256)
        torch.manual_seed(args.seed)
        hard = l2_normalize(torch.randn(n, d), dim=-1)
        soft = l2_normalize(torch.randn(n, d), dim=-1)
        kappa = torch.zeros(n)
        reliability = torch.ones(n)
        _save(args.out, hard, soft, kappa, reliability, classes, token_ids,
              dict(dry_run=True, config=args.config, stage=args.stage))
        return

    classes, per_class = build_dataset(cfg, args.max_samples)
    text_emb = class_text_embeddings(model, caption_string, token_ids, device)
    pipe = build_preprocess_pipeline()

    # a mismatched background source per class (the first sample of another class)
    non_empty = [c for c in per_class if per_class[c]]
    hard_list, soft_list, kappa_list, rel_list, metas = [], [], [], [], []
    for c in range(len(classes)):
        samples = per_class[c]
        if not samples:
            print(f'[S2H] skip class {classes[c]}: no support samples')
            hard_list.append(torch.zeros(model.embed_dims))
            soft_list.append(torch.zeros(model.embed_dims))
            kappa_list.append(torch.tensor(0.0))
            rel_list.append(torch.tensor(0.0))
            continue
        mism = None
        for other in non_empty:
            if other != c:
                mism = per_class[other][0]['img_path']
                break
        same_domain = None
        for other in non_empty:
            if other != c:
                same_domain = per_class[other][0]['img_path']
                break
        out = build_class_knowledge(
            dict(class_id=c, samples=samples, mismatch_src=mism,
                 same_domain_src=same_domain),
            classes, token_ids, text_emb, args, model, pipe, device)
        if out is None:
            print(f'[S2H] warning: class {classes[c]} produced no knowledge')
            hard_list.append(torch.zeros(model.embed_dims))
            soft_list.append(torch.zeros(model.embed_dims))
            kappa_list.append(torch.tensor(0.0))
            rel_list.append(torch.tensor(0.0))
            continue
        hard_list.append(out['hard'])
        soft_list.append(out['soft'])
        kappa_list.append(torch.tensor(out['kappa']))
        rel_list.append(torch.tensor(out['reliability']))
        metas.append(dict(cls=classes[c], **{k: v for k, v in out.items()
                                             if k not in ('hard', 'soft')}))
        print(f"[S2H] {classes[c]}: kappa={out['kappa']:.3f} "
              f"rel={out['reliability']:.3f} attrs={out['n_attr_kept']}/"
              f"{out['n_attr']} rank={out['u_ctx_rank']}")

    _save(args.out, torch.stack(hard_list), torch.stack(soft_list),
          torch.stack(kappa_list), torch.stack(rel_list), classes, token_ids,
          dict(config=args.config, args=vars(args), stage=args.stage,
               per_class=metas))


def _classes_from_node(node) -> Optional[List[str]]:
    """Return ``metainfo.classes`` from a dataloader/dataset config node.

    Wrapper datasets (e.g. ``MultiImageMixDataset``, used by the UODD configs
    for mosaic/mixup) keep the real dataset - and therefore the ``metainfo`` -
    in a nested ``dataset`` key, so descend through the wrapper chain instead
    of assuming the top level carries ``metainfo``.
    """
    if not isinstance(node, dict):
        return None
    metainfo = node.get('metainfo')
    if isinstance(metainfo, dict) and metainfo.get('classes'):
        return list(metainfo['classes'])
    if 'dataset' in node:
        return _classes_from_node(node['dataset'])
    return None


def _safe_classes(cfg) -> Tuple[List[str], None]:
    for split in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dataloader = cfg.get(split)
        if isinstance(dataloader, dict):
            classes = _classes_from_node(dataloader.get('dataset', {}))
            if classes:
                return classes, None
    # some configs declare the classes only once at module level ...
    classes = _classes_from_node(cfg)
    if classes:
        return classes, None
    # ... or inside the S2H head config.
    s2h_cfg = cfg.get('model', {}).get('bbox_head', {}).get('s2h_cfg', {})
    if isinstance(s2h_cfg, dict) and s2h_cfg.get('classes'):
        return list(s2h_cfg['classes']), None
    raise KeyError(
        'could not find `metainfo.classes` in the train/val/test dataloaders, '
        'in a module-level `metainfo`, or in `model.bbox_head.s2h_cfg.classes`')


def _save(path, hard, soft, kappa, reliability, classes, token_ids, meta):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    torch.save(dict(
        hard=hard.float(), soft=soft.float(), kappa=kappa.float(),
        reliability=reliability.float(), classes=list(classes),
        token_ids=[list(map(int, t)) for t in token_ids], meta=meta), path)
    print(f'[S2H] saved knowledge -> {path}')


if __name__ == '__main__':
    main()

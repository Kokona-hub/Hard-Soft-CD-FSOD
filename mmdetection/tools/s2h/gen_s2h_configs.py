# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Generate S2H-CD-FSOD configs that reuse Domain-RAG's dataset settings.

For every ``(dataset, shot)`` pair the script copies the *verbatim* dataset
block of Domain-RAG's few-shot config
(``configs/grounding_dino/CDFSOD_detection_few-shot_<DS>_<shot>shot.py``),
inherits the Grounding DINO Swin-B fine-tuning recipe and swaps in the
knowledge-enhanced detector/head. This guarantees that the comparison against
Domain-RAG / GroundingDINO uses **identical** data, augmentation, optimizer and
schedule settings -- the only difference is the S2H correction.

Usage::

    python tools/s2h/gen_s2h_configs.py
    python tools/s2h/gen_s2h_configs.py --datasets ArTaxOr NEU-DET --shots 1
"""
import argparse
import os
import os.path as osp
import re
import sys

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

# dataset name -> number of fine-tuning epochs, following the paper:
# "Clipart1k and DeepFish are fine-tuned for five epochs, while ArTaxOr, DIOR,
#  NEU-DET, and UODD are fine-tuned for 30 epochs."
DEFAULT_DATASETS = ['ArTaxOr', 'clipart1k', 'DIOR', 'FISH', 'NEU-DET', 'UODD']
DEFAULT_SHOTS = ['1', '5', '10']
FIVE_EPOCH_DATASETS = {'clipart1k', 'FISH'}

MODEL_TEMPLATE = '''

# ---------------------------------------------------------------------------
# S2H-CD-FSOD: knowledge-enhanced Hard-Soft routing on top of the exact
# Grounding DINO Swin-B recipe reproduced by Domain-RAG.
# ---------------------------------------------------------------------------
model = dict(
    type='S2HGroundingDINO',
    bbox_head=dict(
        type='S2HGroundingDINOHead',
        s2h_cfg=dict(
            enabled=True,
            stage='full',
            classes=metainfo['classes'],
            knowledge_path={knowledge_path!r},
            gamma={gamma},
            alpha_max={alpha_max},
            g_max={g_max},
            theta={theta},
            T_u={t_u},
            init_alpha_bias={init_alpha_bias},
            init_alpha_from_reliability=True,
            positive_only=False,
            center_delta=True,
            correction_mode='contrastive',
            evidence_temperature=0.20,
            routing_confidence='sigmoid',
            soft_mix=0.35,
            reliability_power={reliability_power},
            alpha_rel_floor={alpha_rel_floor},
            alpha_rel_scale={alpha_rel_scale},
            train_injection=False,
            res_weight=0.0,
            con_weight=0.0)))

# training loop / schedule (mirrors the Domain-RAG launch script)
max_epochs = {epochs}
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones={milestones},
        gamma=0.1)
]

# Save checkpoints during training, but do not select on the query set.  The
# launcher evaluates the fixed final epoch because this protocol has no
# independent validation split.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best=None,
                    max_keep_ckpts=3))
'''


def build_one(dataset: str, shot: str, args) -> str:
    src = osp.join(args.src_dir,
                   f'CDFSOD_detection_few-shot_{dataset}_{shot}shot.py')
    if not osp.exists(src):
        raise FileNotFoundError(f'missing Domain-RAG config: {src}')
    with open(src, 'r', encoding='utf-8') as f:
        dataset_block = f.read()
    if args.data_root_prefix:
        root = args.data_root_prefix.rstrip('/') + '/' + dataset + '/'
        dataset_block, replaced = re.subn(
            r'^data_root\s*=\s*[^\n]+$',
            f'data_root = {root!r}', dataset_block, count=1,
            flags=re.MULTILINE)
        if not replaced:
            raise ValueError(f'{src} does not define data_root')

    # DIOR and UODD use multi-image transforms (CachedMosaic/Mosaic and
    # CachedMixUp).  Those transforms require MultiImageMixDataset to provide
    # ``mix_results``.  Keep image loading/annotation transforms in the inner
    # dataset and run the multi-image augmentation pipeline in the wrapper.
    if dataset in {'DIOR', 'UODD'}:
        prefix = dataset.lower().replace('-', '_')
        dataset_block += f'''

# {dataset} multi-image augmentation requires a MultiImageMixDataset wrapper.
_{prefix}_base_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=train_ann_file,
    metainfo=metainfo,
    data_prefix=dict(img='train/'),
    pipeline=train_pipeline[:2],
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True)
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=_{prefix}_base_train_dataset,
        pipeline=train_pipeline[2:]))
'''

    epochs = 5 if dataset in FIVE_EPOCH_DATASETS else args.epochs
    milestones = '[3]' if epochs <= 5 else f'[{max(epochs - 19, 1)}]'
    knowledge_path = args.knowledge_dir.format(
        dataset=dataset, shot=shot) if args.knowledge_dir else ''

    header = ("# Auto-generated by tools/s2h/gen_s2h_configs.py -- DO NOT EDIT "
              "MANUALLY.\n"
              "# Dataset settings below are copied verbatim from Domain-RAG's "
              "reproduction config,\n"
              "# so that the comparison is apples-to-apples.\n"
              "_base_ = [\n"
              "    '../grounding_dino/"
              "grounding_dino_swin-b_finetune_16xb2_1x_coco.py',\n"
              "]\n\n")

    body = MODEL_TEMPLATE.format(
        knowledge_path=knowledge_path, gamma=args.gamma,
        alpha_max=args.alpha_max, g_max=args.g_max, theta=args.theta,
        t_u=args.t_u, init_alpha_bias=args.init_alpha_bias,
        reliability_power=args.reliability_power,
        alpha_rel_floor=args.alpha_rel_floor,
        alpha_rel_scale=args.alpha_rel_scale,
        res_weight=args.res_weight, con_weight=args.con_weight,
        epochs=epochs, milestones=milestones)

    return header + dataset_block + body


def parse_args():
    parser = argparse.ArgumentParser('Generate S2H configs')
    parser.add_argument(
        '--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--shots', nargs='+', default=DEFAULT_SHOTS)
    parser.add_argument(
        '--src-dir', default='configs/grounding_dino',
        help='directory holding the Domain-RAG few-shot configs')
    parser.add_argument(
        '--out-dir', default='configs/s2h_dino')
    parser.add_argument(
        '--knowledge-dir',
        default='work_dirs/s2h_knowledge/{dataset}_{shot}shot.pth',
        help='path template of the cached knowledge file '
             '(empty string disables the default path)')
    parser.add_argument(
        '--data-root-prefix', default='',
        help='optional dataset root prefix; generated configs read '
             '<prefix>/<dataset>/ instead of data/<dataset>/')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha-max', type=float, default=0.5)
    parser.add_argument('--g-max', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--t-u', type=float, default=0.1)
    parser.add_argument('--res-weight', type=float, default=0.0)
    parser.add_argument('--con-weight', type=float, default=0.0)
    parser.add_argument('--init-alpha-bias', type=float, default=-2.5)
    parser.add_argument('--reliability-power', type=float, default=1.0)
    parser.add_argument('--alpha-rel-floor', type=float, default=0.10)
    parser.add_argument('--alpha-rel-scale', type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for dataset in args.datasets:
        for shot in args.shots:
            content = build_one(dataset, shot, args)
            name = f's2h_grounding_dino_swin-b_{dataset}_{shot}shot.py'
            out = osp.join(args.out_dir, name)
            with open(out, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'generated {out}')


if __name__ == '__main__':
    main()

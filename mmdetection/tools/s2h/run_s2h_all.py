# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""One-command launcher for the S2H-CD-FSOD experiments.

Mirrors ``auto_modify_swin_t_config.py`` of Domain-RAG: for every
``(dataset, shot)`` pair it

1. optionally builds / refreshes the cached Hard-Soft knowledge with
   ``build_s2h_knowledge.py``, then
2. launches fine-tuning with ``tools/dist_train.sh`` into a dedicated work dir.

Usage::

    # full pipeline
    python tools/s2h/run_s2h_all.py --datasets ArTaxOr NEU-DET --shots 1 5 10

    # skip the support-side build (knowledge already cached / dry-run)
    python tools/s2h/run_s2h_all.py --datasets ArTaxOr --shots 1 --skip-build

    # baseline for an apples-to-apples comparison: keep the S2H head but
    # disable the correction (identical to the plain GroundingDINO recipe)
    python tools/s2h/run_s2h_all.py --datasets ArTaxOr --shots 1 --disable-s2h
"""
import argparse
import os
import os.path as osp
import subprocess
import sys

DEFAULT_DATASETS = ['ArTaxOr', 'clipart1k', 'DIOR', 'FISH', 'NEU-DET', 'UODD']
DEFAULT_SHOTS = ['1', '5', '10']


def run(cmd: str) -> None:
    print(f'\n$ {cmd}')
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f'command failed (exit {ret}): {cmd}')


def parse_args():
    parser = argparse.ArgumentParser('Run S2H-CD-FSOD experiments')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--shots', nargs='+', default=DEFAULT_SHOTS)
    parser.add_argument('--config-dir', default='configs/s2h_dino')
    parser.add_argument('--knowledge-dir', default='work_dirs/s2h_knowledge')
    parser.add_argument('--work-dir', default='cat_work_dir/s2h')
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument(
        '--checkpoint', default=None,
        help=('local detector checkpoint used both to build support knowledge '
              'and as load_from for fine-tuning'))
    parser.add_argument('--build-knowledge', action='store_true',
                        help='(re)build the cached support knowledge first')
    parser.add_argument('--skip-build', action='store_true',
                        help='never build knowledge, only train')
    parser.add_argument('--dry-run-knowledge', action='store_true',
                        help='build random knowledge (integration smoke test)')
    parser.add_argument('--disable-s2h', action='store_true',
                        help='train the baseline (S2H correction disabled)')
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    for dataset in args.datasets:
        for shot in args.shots:
            cfg = osp.join(
                args.config_dir,
                f's2h_grounding_dino_swin-b_{dataset}_{shot}shot.py')
            if not osp.exists(cfg):
                print(f'[skip] missing config {cfg}; run '
                      f'gen_s2h_configs.py first')
                continue
            knowledge = osp.join(args.knowledge_dir,
                                 f'{dataset}_{shot}shot.pth')

            if args.build_knowledge and not args.skip_build:
                if not osp.exists(knowledge) or args.dry_run_knowledge:
                    cmd = (f'{args.python} tools/s2h/build_s2h_knowledge.py '
                           f'--config {cfg} --out {knowledge}')
                    if args.checkpoint:
                        cmd += f' --checkpoint {args.checkpoint}'
                    if args.dry_run_knowledge:
                        cmd += ' --dry-run'
                    run(cmd)

            work_dir = osp.join(args.work_dir, f'{dataset}_{shot}shot')
            os.makedirs(work_dir, exist_ok=True)

            cfg_options = []
            # Keep support knowledge extraction and fine-tuning on precisely
            # the same detector initialization.  Without this override,
            # training falls back to the remote URL in the base config.
            if args.checkpoint:
                cfg_options.append(f'load_from={args.checkpoint}')
            if args.disable_s2h:
                # the S2H head falls back to the plain path when disabled
                cfg_options.append('model.bbox_head.s2h_cfg.enabled=False')
            extra = (' --cfg-options ' + ' '.join(cfg_options)
                     if cfg_options else '')
            cmd = (f'bash tools/dist_train.sh {cfg} {args.gpus} '
                   f'--work-dir {work_dir}{extra}')
            run(cmd)


if __name__ == '__main__':
    main()

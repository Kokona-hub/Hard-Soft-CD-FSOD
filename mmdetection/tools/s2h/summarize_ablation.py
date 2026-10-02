#!/usr/bin/env python
"""Summarize the four-stage, three-seed S2H ablation.

The script reads MMEngine ``vis_data/scalars.json`` files written by each
training run and writes one row per dataset/stage/shot plus a mean/std
aggregation over seeds. The default ``final`` selection matches the
Domain-RAG protocol; ``best`` is retained for exploratory diagnosis only.
Missing runs are reported instead of being silently dropped from the expected
216-run matrix.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


DATASETS = ('ArTaxOr', 'clipart1k', 'DIOR', 'FISH', 'NEU-DET', 'UODD')
STAGES = ('baseline', 'ffcp', 'ffcp_chsd', 'full')
SHOTS = ('1', '5', '10')
SEEDS = ('3407', '3408', '3409')


def _metric(record, suffixes):
    for key, value in record.items():
        if any(key == suffix or key.endswith('/' + suffix)
               for suffix in suffixes):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def read_scalars(path):
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            epoch = item.get('epoch')
            m_ap = _metric(item, ('bbox_mAP', 'coco/bbox_mAP'))
            if m_ap is None:
                continue
            rows.append({
                'epoch': int(epoch) if epoch is not None else '',
                'mAP': m_ap,
                'mAP50': _metric(item, ('bbox_mAP_50', 'coco/bbox_mAP_50')),
                'mAP75': _metric(item, ('bbox_mAP_75', 'coco/bbox_mAP_75')),
            })
    return rows


def select_run(work, selection='final'):
    candidates = sorted(work.glob('**/scalars.json'))
    if not candidates:
        return None
    records = []
    for path in candidates:
        records.extend(read_scalars(path))
    if not records:
        return None
    if selection == 'best':
        return max(records, key=lambda item: item['mAP'])
    # Some MMEngine scalar writers omit epoch. JSONL order is chronological,
    # so use the last evaluation in that case.
    with_epoch = [r for r in records if r['epoch'] != '']
    if with_epoch:
        return max(with_epoch, key=lambda item: item['epoch'])
    return records[-1]


def fmt(value):
    return '' if value is None else f'{value:.6f}'


def parse_args():
    parser = argparse.ArgumentParser('Summarize S2H ablation results')
    parser.add_argument('--root', default='cat_work_dir/ablation')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--stages', nargs='+', default=STAGES)
    parser.add_argument('--shots', nargs='+', default=SHOTS)
    parser.add_argument('--seeds', nargs='+', default=SEEDS)
    parser.add_argument('--selection', choices=('final', 'best'), default='final',
                        help='metric selection; final is the paper protocol')
    parser.add_argument('--out', default='work_dirs/s2h_ablation_summary.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    missing = []
    for dataset in args.datasets:
        for stage in args.stages:
            for shot in args.shots:
                values = []
                for seed in args.seeds:
                    work = root / dataset / stage / f'seed{seed}' / f'{shot}shot'
                    result = select_run(work, args.selection)
                    if result is None:
                        missing.append(str(work))
                        continue
                    row = dict(dataset=dataset, stage=stage, shot=shot,
                               seed=seed, **result)
                    rows.append(row)
                    values.append(row)
                if values:
                    aggregate = dict(dataset=dataset, stage=stage, shot=shot,
                                     seed='mean_std', epoch='')
                    for metric in ('mAP', 'mAP50', 'mAP75'):
                        nums = [v[metric] for v in values if v[metric] is not None]
                        aggregate[metric] = mean(nums) if nums else None
                        aggregate[f'{metric}_std'] = (
                            stdev(nums) if len(nums) > 1 else 0.0
                        ) if nums else None
                    rows.append(aggregate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ('dataset', 'stage', 'shot', 'seed', 'epoch',
              'mAP', 'mAP_std', 'mAP50', 'mAP50_std', 'mAP75', 'mAP75_std')
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if key in (
                'mAP', 'mAP_std', 'mAP50', 'mAP50_std', 'mAP75', 'mAP75_std')
                             else value for key, value in row.items()})
    print(f'wrote {len(rows)} rows -> {out}')
    if missing:
        print(f'missing/unreadable runs: {len(missing)}')
        for path in missing:
            print(f'  [MISS] {path}')
    expected = len(args.datasets) * len(args.stages) * len(args.shots) * len(args.seeds)
    observed = sum(1 for row in rows if row['seed'] != 'mean_std')
    print(f'observed runs: {observed}/{expected}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Select the checkpoint at the highest recorded bbox mAP.

Checkpoint filenames encode the epoch at which a new best was observed, not
the value of that best metric.  Picking the lexicographically last filename
can therefore select a later, weaker model.  This helper reads MMEngine's
``vis_data/scalars.json`` and resolves the epoch with the largest
``coco/bbox_mAP`` value.
"""
import argparse
import json
from pathlib import Path
import re


METRIC_KEYS = ('coco/bbox_mAP', 'bbox_mAP')


def best_epoch(work_dir: Path):
    candidates = [work_dir / 'vis_data' / 'scalars.json']
    candidates.extend(work_dir.glob('**/vis_data/scalars.json'))
    records = []
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = next((record.get(key) for key in METRIC_KEYS
                              if record.get(key) is not None), None)
                epoch = record.get('epoch')
                if value is not None and epoch is not None:
                    records.append((float(value), int(epoch)))
    if not records:
        return None
    return max(records, key=lambda item: (item[0], item[1]))[1]


def resolve(work_dir: Path, epoch=None):
    explicit_epoch = epoch is not None
    if epoch is None:
        epoch = best_epoch(work_dir)
    if epoch is not None:
        patterns = (
            f'best_coco_bbox_mAP_epoch_{epoch}.pth',
            f'epoch_{epoch}.pth',
        )
        for pattern in patterns:
            path = work_dir / pattern
            if path.is_file() and path.stat().st_size:
                return path
        if explicit_epoch:
            return None
    for pattern in ('latest.pth', 'last_checkpoint'):
        path = work_dir / pattern
        if path.is_file() and path.stat().st_size:
            if path.name == 'last_checkpoint':
                target = Path(path.read_text(encoding='utf-8').strip())
                if not target.is_absolute():
                    target = work_dir / target
                if target.is_file():
                    return target
            else:
                return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--epoch', type=int,
                        help='select epoch_N.pth explicitly instead of mAP-best')
    args = parser.parse_args()
    path = resolve(args.work_dir, args.epoch)
    if path is None:
        raise SystemExit(f'no usable checkpoint found under {args.work_dir}')
    print(path)


if __name__ == '__main__':
    main()

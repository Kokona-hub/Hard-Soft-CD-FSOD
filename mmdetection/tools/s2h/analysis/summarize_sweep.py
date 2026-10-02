#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Aggregate the Figure 4 sweep and draw its five panels.

Reads the ``vis_data/scalars.json`` that ``tools/test.py`` / ``tools/train.py``
already write for every sweep run (same file the ablation summary uses), takes
the best validation / test mAP of each run, averages the seeds and writes

* ``<out>.runs.csv``    one row per (parameter, value, seed),
* ``<out>.summary.csv`` mean +/- std per (parameter, value),
* ``<out>.pdf/.png``    one panel per parameter, default marked, stable
                        region shaded.

The script never writes into an ablation directory (same guard as
``export_query_features.py``) and only reads results, so the mAP experiments
stay untouched.

Usage
-----
::

    python tools/s2h/analysis/summarize_sweep.py \
        --root cat_work_dir/sweep --dataset clipart1k \
        --params g_max theta con_weight ffcp_rank hard_radius \
        --seeds 3407 3408 3409 \
        --out work_dirs/s2h_analysis/fig4_hyperparam_clipart1k_5shot
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional

import numpy as np

# tools/s2h/analysis/ -> the mmdet source tree (see export_query_features.py)
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..',
                                        '..')))

# panel -> (default value, log x axis, human label)
PARAMS = {
    'ffcp_rank': (4, True, 'FFCP context rank'),
    'hard_radius': (1.0, True, r'Hard update radius $\rho_H$'),
    'g_max': (0.5, False, r'Authority ceiling $g_{\max}$'),
    'theta': (0.5, False, r'Uncertainty threshold $\theta$'),
    'con_weight': (0.1, False, r'Consistency weight $\lambda_{\mathrm{con}}$'),
}
METRIC_KEYS = ('coco/bbox_mAP', 'bbox_mAP')


def assert_analysis_path(path: str) -> str:
    try:
        from export_query_features import assert_analysis_path as guard
        return guard(path)
    except SystemExit:
        raise
    except Exception:
        parts = osp.abspath(path).replace('\\', '/').split('/')
        if 'ablation' in parts or not any(
                r in parts for r in ('s2h_analysis', 'sweep')):
            raise SystemExit(f'refusing to write to {path}')
        return path


def read_scalars(path: str) -> Optional[float]:
    """Best metric in one ``scalars.json`` (mirrors summarize_ablation)."""
    best = None
    try:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in METRIC_KEYS:
                    if key in entry and entry[key] is not None:
                        value = float(entry[key])
                        best = value if best is None else max(best, value)
                        break
    except OSError:
        return None
    return best


def find_run(directory: str) -> Optional[str]:
    """The scalars.json of a run, tolerating mmengine's timestamped subdirs."""
    if not osp.isdir(directory):
        return None
    direct = osp.join(directory, 'vis_data', 'scalars.json')
    if osp.isfile(direct):
        return direct
    for root, _dirs, files in os.walk(directory):
        if 'scalars.json' in files and osp.basename(root) == 'vis_data':
            return osp.join(root, 'scalars.json')
    return None


def pick_run(root: str, dataset: str, param: str, value: str, seed: str,
             shot: str) -> Optional[Dict]:
    """Prefer the evaluated test run, fall back to the training run."""
    base = osp.join(root, dataset, param, value, f'seed{seed}')
    for suffix in (f'{shot}shot_test', f'{shot}shot'):
        directory = osp.join(base, suffix)
        scalars = find_run(directory)
        if scalars:
            return dict(metric=read_scalars(scalars),
                        source=osp.basename(directory), path=scalars)
    return None


def aggregate(rows: List[Dict]) -> List[Dict]:
    """Mean / std per (parameter, value); ``rows`` may hold several seeds."""
    grouped: Dict[tuple, List[float]] = {}
    for row in rows:
        if row['metric'] is None:
            continue
        grouped.setdefault((row['param'], row['value']), []).append(row['metric'])
    summary = []
    for (param, value), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        summary.append(dict(param=param, value=value, n=len(values),
                            mean=float(array.mean()),
                            std=float(array.std(ddof=1)) if len(values) > 1
                            else 0.0))
    return summary


def stable_region(summary: List[Dict], param: str, tolerance: float
                  ) -> Optional[List[float]]:
    """Contiguous stretch of values within ``tolerance`` of the best mean.

    This is what the paper calls the "central stable region"; reporting it as a
    data-derived interval keeps the claim honest.
    """
    entries = [row for row in summary if row['param'] == param]
    if not entries:
        return None
    entries.sort(key=lambda row: row['value'])
    best = max(row['mean'] for row in entries)
    index = [i for i, row in enumerate(entries)
             if row['mean'] >= best - tolerance]
    if not index:
        return None
    # longest contiguous run of qualifying indices
    runs, current = [], [index[0]]
    for i in index[1:]:
        if i == current[-1] + 1:
            current.append(i)
        else:
            runs.append(current)
            current = [i]
    runs.append(current)
    run = max(runs, key=len)
    # No fabricated widening: a single qualifying value is reported as the
    # degenerate interval [v, v] so the figure cannot claim a plateau that the
    # sweep did not measure.
    return [float(entries[run[0]]['value']), float(entries[run[-1]]['value'])]


def self_check() -> int:
    rows = [dict(param='g_max', value=0.5, metric=0.30, source='t', path=''),
            dict(param='g_max', value=0.5, metric=0.32, source='t', path=''),
            dict(param='g_max', value=1.0, metric=0.20, source='t', path='')]
    summary = aggregate(rows)
    assert len(summary) == 2, summary
    assert abs(summary[0]['mean'] - 0.31) < 1e-9, summary
    assert abs(summary[0]['std'] - np.std([0.30, 0.32], ddof=1)) < 1e-9
    assert summary[1]['std'] == 0.0
    region = stable_region(summary, 'g_max', 0.005)
    assert region == [0.5, 0.5], region     # degenerate interval, not widened
    long_rows = [dict(param='theta', value=v, metric=m, source='t', path='')
                 for v, m in ((0.3, 0.10), (0.4, 0.29), (0.5, 0.30),
                              (0.6, 0.295), (0.7, 0.20))]
    region = stable_region(aggregate(long_rows), 'theta', 0.02)
    assert region == [0.4, 0.6], region
    print('[self-check] ok')
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser('Figure 4: hyper-parameter sensitivity')
    parser.add_argument('--root', default='cat_work_dir/sweep')
    parser.add_argument('--dataset', default='clipart1k')
    parser.add_argument('--shot', default='5')
    parser.add_argument('--params', nargs='+', default=list(PARAMS))
    parser.add_argument('--seeds', nargs='+', default=['3407', '3408', '3409'])
    parser.add_argument('--tolerance', type=float, default=0.005,
                        help='width of the stable band in mAP units '
                             '(0.005 = 0.5 mAP points)')
    parser.add_argument('--out', default=None,
                        help='output stem; defaults to work_dirs/s2h_analysis/'
                             'fig4_hyperparam_<dataset>_<shot>shot')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg'])
    parser.add_argument('--panel-size', nargs=2, type=float,
                        default=(2.4, 2.0), metavar=('W', 'H'))
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def draw(summary: List[Dict], params: List[str], tolerance: float, args
         ) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    rows = (len(params) + 2) // 3
    width, height = args.panel_size
    fig, axes = plt.subplots(rows, 3, figsize=(width * 3, height * rows),
                             squeeze=False)
    flat = [axis for row in axes for axis in row]
    for axis in flat[len(params):]:
        axis.axis('off')
    report = {}
    for axis, param in zip(flat, params):
        entries = sorted([r for r in summary if r['param'] == param],
                         key=lambda r: r['value'])
        default, log_axis, label = PARAMS.get(param, (None, False, param))
        if not entries:
            axis.set_title(f'{label} (no runs)', fontsize=8)
            axis.axis('off')
            continue
        values = np.asarray([r['value'] for r in entries], dtype=float)
        means = np.asarray([r['mean'] for r in entries]) * 100.0
        errors = np.asarray([r['std'] for r in entries]) * 100.0
        axis.errorbar(values, means, yerr=errors, marker='o', markersize=3.5,
                      linewidth=1.2, capsize=2, color='#1f4e79')
        region = stable_region(summary, param, tolerance)
        if region:
            axis.axvspan(region[0], region[1], color='#1f4e79', alpha=0.08,
                         zorder=0)
        if default is not None:
            axis.axvline(default, color='0.35', linestyle='--', linewidth=0.9)
        if log_axis:
            axis.set_xscale('log')
            axis.set_xticks(list(values))
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
        axis.set_title(label, fontsize=8.5)
        axis.set_xlabel('value', fontsize=8)
        axis.set_ylabel('mAP (%)', fontsize=8)
        axis.tick_params(labelsize=7.5)
        axis.grid(alpha=0.18, linewidth=0.6)
        report[param] = dict(
            values=[float(v) for v in values],
            mean=[float(m) for m in means], std=[float(e) for e in errors],
            default=default, stable_region=region,
            n_seeds=entries[0]['n'])
    fig.tight_layout()
    written = []
    for extension in args.formats:
        path = f'{args.out}.{extension}'
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        written.append(path)
    plt.close(fig)
    path = f'{args.out}.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    written.append(path)
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    out = args.out or (f'work_dirs/s2h_analysis/fig4_hyperparam_'
                       f'{args.dataset}_{args.shot}shot')
    args.out = out
    assert_analysis_path(out)

    rows: List[Dict] = []
    missing: List[str] = []
    for param in args.params:
        values = sorted({entry.name for entry in os.scandir(
            osp.join(args.root, args.dataset, param))}) \
            if osp.isdir(osp.join(args.root, args.dataset, param)) else []
        if not values:
            missing.append(f'{param}: no run directory under '
                           f'{args.root}/{args.dataset}/{param}')
            continue
        for value in values:
            for seed in args.seeds:
                found = pick_run(args.root, args.dataset, param, value, seed,
                                 args.shot)
                if not found or found['metric'] is None:
                    missing.append(f'{param}={value} seed{seed}')
                    continue
                rows.append(dict(param=param, value=float(value), seed=seed,
                                 metric=found['metric'],
                                 source=found['source'], path=found['path']))

    if not rows:
        raise SystemExit('no sweep run found; run run_sweep.sh first')

    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    runs_csv = f'{out}.runs.csv'
    with open(runs_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['param', 'value', 'seed',
                                                    'metric', 'source', 'path'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['param'], r['value'],
                                                     r['seed'])))
    summary = aggregate(rows)
    summary_csv = f'{out}.summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['param', 'value', 'n',
                                                    'mean', 'std'])
        writer.writeheader()
        writer.writerows(summary)

    printed = {param: [r for r in summary if r['param'] == param]
               for param in args.params}
    for param, entries in printed.items():
        for entry in entries:
            print(f'[sweep] {param}={entry["value"]}: '
                  f'{entry["mean"] * 100:.2f} +/- {entry["std"] * 100:.2f} '
                  f'(n={entry["n"]})')
    for warning in missing:
        print(f'[missing] {warning}')

    written = [runs_csv, summary_csv]
    try:
        written += draw(summary, args.params, args.tolerance, args)
    except ImportError:
        print('[warn] matplotlib unavailable: CSV written, figure skipped')
    for path in written:
        print(f'[write] {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
"""Aggregate and plot the two-parameter, three-shot Clipart1k Figure 4.

The figure has two panels (one per parameter). Each panel uses the parameter
values on the x-axis and has three lines for 1/5/10-shot mAP. This script only
reads sweep ``scalars.json`` files and writes date-scoped analysis artifacts;
it does not touch training, evaluation, checkpoints, or ablation outputs.
"""
import argparse
import csv
import json
import os
import os.path as osp
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np


PARAM_LABELS = {
    'g_max': r'Authority ceiling $g_{\max}$',
    'theta': r'Uncertainty threshold $\theta$',
}
PARAM_DEFAULTS = {'g_max': 0.5, 'theta': 0.5}
SHOT_COLORS = {'1': '#2563eb', '5': '#db2777', '10': '#16a34a'}
SHOT_MARKERS = {'1': 'o', '5': 's', '10': '^'}
METRIC_KEYS = ('coco/bbox_mAP', 'bbox_mAP')


def assert_output_path(path: str) -> None:
    normalized = osp.abspath(path).replace('\\', '/').split('/')
    if 'ablation' in normalized:
        raise SystemExit(f'refusing to write into an ablation directory: {path}')
    if not any(part in normalized for part in ('s2h_analysis', 'fig4')):
        raise SystemExit(
            f'refusing to write outside an analysis/fig4 directory: {path}')


def read_scalars(path: str) -> Optional[float]:
    best = None
    try:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                value = None
                for key in METRIC_KEYS:
                    if key in entry and entry[key] is not None:
                        value = float(entry[key])
                        break
                if value is not None:
                    best = value if best is None else max(best, value)
    except OSError:
        return None
    return best


def read_log_metric(path: str) -> Optional[float]:
    """Read the final mAP from a MMEngine text log when scalars are absent."""
    best = None
    patterns = (
        re.compile(r'coco/bbox_mAP:\s*([0-9]*\.?[0-9]+)'),
        re.compile(r'coco/bbox_mAP\s*=\s*([0-9]*\.?[0-9]+)'),
    )
    try:
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        value = float(match.group(1))
                        best = value if best is None else max(best, value)
                        break
    except OSError:
        return None
    return best


def find_scalars(directory: str) -> Optional[str]:
    direct = osp.join(directory, 'vis_data', 'scalars.json')
    if osp.isfile(direct):
        return direct
    if not osp.isdir(directory):
        return None
    for root, _dirs, files in os.walk(directory):
        if 'scalars.json' in files and osp.basename(root) == 'vis_data':
            return osp.join(root, 'scalars.json')
    return None


def find_metric(directory: str) -> Optional[tuple]:
    """Return ``(metric, source_file)`` from a run directory.

    Test runs normally expose ``vis_data/scalars.json``. The log fallback is
    useful for older MMEngine runs where LoggerHook emitted the metric but did
    not flush a scalars file before shutdown.
    """
    scalars = find_scalars(directory)
    if scalars:
        metric = read_scalars(scalars)
        if metric is not None:
            return metric, scalars
    if not osp.isdir(directory):
        return None
    candidates = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(('.log', '.txt')):
                candidates.append(osp.join(root, name))
    for path in sorted(candidates):
        metric = read_log_metric(path)
        if metric is not None:
            return metric, path
    return None


def run_row(root: str, dataset: str, param: str, value: str,
            shot: str, seed: str) -> Optional[Dict]:
    base = osp.join(root, dataset, param, value, f'seed{seed}')
    for suffix in (f'{shot}shot_test', f'{shot}shot'):
        directory = osp.join(base, suffix)
        found = find_metric(directory)
        if found:
            metric, source_path = found
            return dict(param=param, value=float(value), shot=str(shot),
                        seed=str(seed), metric=metric,
                        source=osp.basename(directory), path=source_path)
    return None


def values_for(root: str, dataset: str, param: str) -> List[str]:
    directory = osp.join(root, dataset, param)
    if not osp.isdir(directory):
        return []
    values = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            try:
                float(entry.name)
            except ValueError:
                continue
            values.append(entry.name)
    return sorted(values, key=float)


def aggregate(rows: Iterable[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['param'], row['value'], row['shot'])].append(
            float(row['metric']))
    summary = []
    for (param, value, shot), values in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1],
                                                int(item[0][2]))):
        array = np.asarray(values, dtype=np.float64)
        summary.append(dict(param=param, value=float(value), shot=shot,
                            n=len(values), mean=float(array.mean()),
                            std=float(array.std(ddof=1)) if len(values) > 1
                            else 0.0))
    return summary


def draw(summary: List[Dict], params: List[str], shots: List[str],
         out: str, dpi: int, formats: List[str]) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(params), figsize=(4.6 * len(params), 3.6),
                             squeeze=False)
    axes = list(axes[0])
    for axis, param in zip(axes, params):
        entries = [row for row in summary if row['param'] == param]
        values = sorted({row['value'] for row in entries})
        if not values:
            axis.text(0.5, 0.5, 'no completed runs', ha='center', va='center')
            axis.set_axis_off()
            continue
        for shot in shots:
            by_value = {row['value']: row for row in entries
                        if row['shot'] == str(shot)}
            xs = [value for value in values if value in by_value]
            if not xs:
                continue
            ys = [100.0 * by_value[value]['mean'] for value in xs]
            errors = [100.0 * by_value[value]['std'] for value in xs]
            color = SHOT_COLORS.get(str(shot), '#4b5563')
            marker = SHOT_MARKERS.get(str(shot), 'D')
            axis.errorbar(xs, ys, yerr=errors, color=color, marker=marker,
                          linewidth=1.8, markersize=5.5, capsize=2,
                          label=f'{shot}-shot', zorder=3)
            for x, y in zip(xs, ys):
                axis.annotate(f'{y:.1f}', (x, y), xytext=(0, 7),
                              textcoords='offset points', ha='center',
                              fontsize=8, color=color)
        default = PARAM_DEFAULTS.get(param)
        if default is not None and min(values) <= default <= max(values):
            axis.axvline(default, color='#6b7280', linestyle='--',
                         linewidth=0.9, zorder=1)
        axis.set_title(PARAM_LABELS.get(param, param), fontsize=11)
        axis.set_xlabel('parameter value', fontsize=9)
        axis.set_ylabel('mAP (%)', fontsize=9)
        axis.set_xticks(values)
        axis.grid(axis='y', alpha=0.22, linewidth=0.7)
        axis.tick_params(labelsize=8.5)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(shots),
                   frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle('Clipart1k: S2H sensitivity across shot budgets',
                 fontsize=12, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    written = []
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    for extension in formats:
        path = f'{out}.{extension}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        written.append(path)
    plt.close(fig)
    return written


def self_check() -> int:
    rows = [dict(param='g_max', value=0.1, shot='1', metric=0.2),
            dict(param='g_max', value=0.1, shot='5', metric=0.3),
            dict(param='g_max', value=0.1, shot='10', metric=0.4),
            dict(param='theta', value=0.5, shot='1', metric=0.25)]
    result = aggregate(rows)
    assert len(result) == 4 and result[0]['n'] == 1
    print('[self-check] ok')
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Draw two-parameter Clipart1k three-shot Figure 4')
    parser.add_argument('--tag', default='20260930_clipart1k_shots')
    parser.add_argument('--root', default=None)
    parser.add_argument('--dataset', default='clipart1k')
    parser.add_argument('--params', nargs=2, default=['g_max', 'theta'])
    parser.add_argument('--shots', nargs='+', default=['1', '5', '10'])
    parser.add_argument('--seeds', nargs='+', default=['3407'])
    parser.add_argument('--out', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                        choices=['png', 'pdf', 'svg'])
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    root = args.root or f'cat_work_dir/fig4/{args.tag}'
    out = args.out or (
        f'work_dirs/s2h_analysis/fig4/{args.tag}/clipart1k_three_shot')
    assert_output_path(out)
    rows, missing = [], []
    for param in args.params:
        values = values_for(root, args.dataset, param)
        if not values:
            missing.append(f'{param}: no runs under {root}/{args.dataset}')
            continue
        for value in values:
            for shot in args.shots:
                for seed in args.seeds:
                    row = run_row(root, args.dataset, param, value, shot, seed)
                    if row is None:
                        missing.append(f'{param}={value} {shot}-shot seed{seed}')
                    else:
                        rows.append(row)
    if not rows:
        raise SystemExit('no completed sweep runs found; run the Fig.4 wrapper first')
    summary = aggregate(rows)
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    runs_csv, summary_csv = f'{out}.runs.csv', f'{out}.summary.csv'
    with open(runs_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['param'], r['value'],
                                                      int(r['shot']), r['seed'])))
    with open(summary_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    written = [runs_csv, summary_csv]
    written += draw(summary, list(args.params), list(args.shots), out,
                    args.dpi, args.formats)
    metadata = dict(dataset=args.dataset, params=args.params, shots=args.shots,
                    seeds=args.seeds, root=root, runs=len(rows), missing=missing,
                    source_protocol='20260930 ablation checkpoints/knowledge')
    json_path = f'{out}.json'
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    written.append(json_path)
    for row in summary:
        print(f"[fig4] {row['param']}={row['value']} {row['shot']}-shot: "
              f"{row['mean'] * 100:.2f} +/- {row['std'] * 100:.2f} "
              f"(n={row['n']})")
    for warning in missing:
        print(f'[missing] {warning}')
    if missing:
        print('[hint] inspect actual files with:')
        print(f'  find {root} -type f \\( -name scalars.json -o -name "*.log" \\) | head')
    for path in written:
        print(f'[write] {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

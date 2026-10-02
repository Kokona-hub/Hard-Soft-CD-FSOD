#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Estimate the routing threshold ``theta`` on the support set, and audit the
correction it controls.

Two questions are answered here, both of which a reviewer will ask.

1. **``theta`` should not be a constant across domains.**  Eq. (11) routes a
   query through ``u = sigmoid((theta - p0) / T_u)``: one reference confidence is
   compared with the baseline confidence ``p0`` of every query.  DeepFish is a
   one-class dataset, UODD is low contrast and Clipart1k has 20 categories, so
   their ``p0`` distributions differ by construction and a single absolute 0.5
   cannot be right for all of them.  ``theta`` is therefore read off the support
   set:

       theta = quantile(p0 of the positive queries, --quantile)     (default 0.35)

   ``p0`` is taken in the bank's own scale (``sigmoid(mean token logit).max_c``),
   which is why the export must come from the current
   ``export_query_features.py``.  The estimate has to come from the *support*
   split (``--split train``); tuning it on test would be tuning on the test set.

2. **Is the correction targeted, or a near-constant offset?**  A large gain on a
   single dataset can be nothing but a global confidence shift.  Comparing the
   same ``(image, query)`` slot of a baseline and a full export gives the
   per-query distribution of ``delta p0`` plus ``corr(delta p0, -p0_baseline)``:
   a constant offset has ``std/|mean| -> 0`` and no correlation, a targeted
   correction concentrates on the queries the router is asked to fix.  The
   per-image prediction profile (preds / matched / unmatched) shows whether a
   gain comes from recall or from extra low-threshold false positives.

Usage
-----
    # 1. support-set export of the *baseline* model (no bank, so no tuning leak)
    python tools/s2h/analysis/export_query_features.py --config "$CFG" \\
        --stage baseline --checkpoint "$BASE" --split train --num-images 0 \\
        --out "$OUT/theta/<ds>_<shot>shot_baseline.npz"

    # 2. the threshold, plus the audit against the test export
    python tools/s2h/analysis/estimate_theta.py \\
        --baseline "$OUT/theta/<ds>_<shot>shot_baseline.npz" \\
        --full "$OUT/<ds>_<shot>shot_full.npz" \\
        --json work_dirs/s2h_analysis/theta/<ds>_<shot>shot.json

Nothing is trained, nothing is written except the optional ``--json`` (guarded
like every other analysis artifact), and no GPU is needed.
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.abspath(osp.dirname(__file__)))


def assert_analysis_path(path: str) -> str:
    """Reuse the exporter's guard, falling back to the same rule by hand."""
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


# ---------------------------------------------------------------------------
# loading + statistics (pure functions, exercised by --self-check)
# ---------------------------------------------------------------------------
QUANTILES = (0.1, 0.25, 0.35, 0.5, 0.75, 0.9)


def load_panel(path: str) -> dict:
    """Load the columns this script needs, with a message that says what to do.

    ``p0`` is the only mandatory column: without it there is no quantity in the
    bank's scale to compare ``theta`` with, and substituting ``scores`` (a mean
    of sigmoid outputs) would produce a threshold that means nothing.
    """
    payload = np.load(path, allow_pickle=True)
    if 'p0' not in payload:
        raise SystemExit(
            f'{path} carries no `p0` column; re-run export_query_features.py '
            "with the current revision (p0 was added so that theta can be "
            "estimated in the bank's own scale)")
    panel = {'path': path, 'p0': np.asarray(payload['p0'], np.float64),
             'matched': np.asarray(payload['matched'], bool)}
    for optional in ('label', 'score', 'image', 'query'):
        if optional in payload:
            panel[optional] = np.asarray(payload[optional])
    panel['stage'] = str(payload['stage']) if 'stage' in payload else '?'
    panel['score_thr'] = float(payload['score_thr']) if 'score_thr' in payload \
        else 0.0
    return panel


def quantiles(values, probabilities=QUANTILES) -> dict:
    """``{p: value}``; an empty array yields an empty mapping."""
    values = np.asarray(values, np.float64).reshape(-1)
    if values.size == 0:
        return {}
    return {p: float(np.quantile(values, p)) for p in probabilities}


def theta_summary(panel: dict) -> dict:
    """Positive-query confidence quantiles of one baseline panel."""
    p0 = panel['p0']
    matched = panel['matched']
    return dict(n=int(p0.size), n_matched=int(matched.sum()),
                n_unmatched=int((~matched).sum()),
                positive=quantiles(p0[matched]),
                all=quantiles(p0),
                unmatched=quantiles(p0[~matched]))


def adaptive_theta(panel: dict, quantile: float) -> float:
    """``quantile`` of the positive-query confidence of a baseline panel.

    Eq. (11) is a *relative* rule - "less confident than this reference" - so the
    reference has to follow the confidence scale of the domain.  Using the
    positives (matched queries, i.e. queries that did hit a ground-truth box of
    their own category) keeps the threshold tied to the cases the router is
    supposed to reinforce, and it is estimated on the support split only.
    """
    values = panel['p0'][panel['matched']]
    if values.size == 0:
        raise SystemExit('the baseline export contains no matched (positive) '
                         'query, so theta cannot be estimated from it')
    return float(np.quantile(values, quantile))


def pair_by_query(base: dict, full: dict):
    """Row indices of the two panels describing the same ``(image, query)``.

    Panels keep a different number of queries above the score threshold, so an
    unpaired comparison would mix different query sets; ``None`` means the export
    predates the ``query`` column.
    """
    if 'query' not in base or 'query' not in full or \
            'image' not in base or 'image' not in full:
        return None
    where = {}
    for row, key in enumerate(zip(base['image'], base['query'])):
        where[(int(key[0]), int(key[1]))] = row
    left, right = [], []
    for row, key in enumerate(zip(full['image'], full['query'])):
        slot = where.get((int(key[0]), int(key[1])))
        if slot is not None:
            left.append(int(slot))
            right.append(int(row))
    if not left:
        return None
    return np.asarray(left, np.int64), np.asarray(right, np.int64)


def bias_summary(base: dict, full: dict, shift=None) -> dict:
    """Per-query ``delta p0`` of the correction, and whether it is targeted.

    ``corr(delta p0, -p0_baseline)`` is the decisive number: the correction is
    supposed to act where the baseline is unsure, which makes ``delta p0`` grow
    as ``p0`` falls, i.e. a positive correlation with ``-p0``.  A near-constant
    offset shows up as ``std/|mean| -> 0`` and a correlation near zero.
    """
    if shift is None:
        left = np.arange(base['p0'].size, dtype=np.int64)
        right = np.arange(full['p0'].size, dtype=np.int64)
        paired = False
    else:
        left, right = shift
        paired = True
    before = base['p0'][left]
    delta = full['p0'][right] - before
    matched = base['matched'][left]

    def describe(values):
        if values.size == 0:
            return None
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        return dict(n=int(values.size), mean=mean, std=std,
                    ratio=(abs(std / mean) if mean else None),
                    quantiles=quantiles(values, (0.1, 0.5, 0.9)))

    correlation = None
    if delta.size > 1 and delta.std() > 0 and before.std() > 0:
        correlation = float(np.corrcoef(delta, -before)[0, 1])
    overall = describe(delta)
    verdict = 'undetermined'
    if overall is not None:
        ratio = overall['ratio']
        if ratio is not None and ratio <= 0.5 and \
                (correlation is None or correlation <= 0.2):
            verdict = 'near-constant offset'
        elif correlation is not None and correlation >= 0.35:
            verdict = 'targeted at the low-confidence queries'
        else:
            verdict = 'mixed: partly targeted, partly a global shift'
    return dict(paired=paired, n_paired=int(delta.size),
                overall=overall,
                matched=describe(delta[matched]),
                unmatched=describe(delta[~matched]),
                corr_with_negative_p0=correlation, verdict=verdict)


def profile(panel: dict, score_thr: float = None) -> dict:
    """Per-image prediction counts, split into matched and unmatched (≈FP).

    ``matched`` means IoU>=0.5 against a ground-truth box of the *same* category,
    so the unmatched part is the high-confidence false-positive count that has to
    be reported next to the mAP.
    """
    p0 = panel['p0']
    matched = panel['matched']
    threshold = panel['score_thr'] if score_thr is None else score_thr
    if 'score' in panel:
        keep = panel['score'] >= threshold
        p0, matched = p0[keep], matched[keep]
    if 'image' in panel:
        images = panel['image']
    else:
        images = np.zeros(p0.size, dtype=np.int64)
    n_images = int(np.unique(images).size)
    return dict(n_images=n_images, n_queries=int(p0.size),
                n_matched=int(matched.sum()),
                n_unmatched=int((~matched).sum()),
                preds_per_image=(p0.size / n_images if n_images else 0.0),
                matched_per_image=(float(matched.sum()) / n_images
                                   if n_images else 0.0),
                unmatched_per_image=(float((~matched).sum()) / n_images
                                     if n_images else 0.0),
                score_thr=float(threshold))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def format_quantiles(values: dict) -> str:
    return ' '.join(f'q{int(p * 100)}={v:.3f}' for p, v in values.items())


def report_theta(summary: dict, theta: float, quantile: float) -> float:
    print(f'[theta] {summary["n"]} queries (matched {summary["n_matched"]}, '
          f'unmatched {summary["n_unmatched"]})')
    print(f'        p0 matched   : {format_quantiles(summary["positive"])}')
    print(f'        p0 unmatched : {format_quantiles(summary["unmatched"])}')
    print(f'        p0 all       : {format_quantiles(summary["all"])}')
    print(f'[theta] positive-query quantile {quantile:.2f} -> theta = {theta:.3f}')
    print(f'        about {100.0 * quantile:.0f}% of the positive support '
          'queries fall below it and receive ATAR authority')
    print(f'        --cfg-options model.bbox_head.s2h_cfg.theta={theta:.3f}')
    return theta


def report_profile(panel: dict) -> dict:
    stats = profile(panel)
    print(f'[profile] {panel["path"]} ({panel["stage"]}): '
          f'{stats["n_images"]} images -> {stats["preds_per_image"]:.2f} '
          f'preds/image, {stats["matched_per_image"]:.2f} matched/image, '
          f'{stats["unmatched_per_image"]:.2f} unmatched/image '
          f'(score>={stats["score_thr"]:.2f})')
    return stats


def report_bias(summary: dict) -> dict:
    if not summary['paired']:
        print('[warn] the two panels carry no `query` column, so the per-query '
              'comparison mixes different query sets; re-export with the '
              'current export_query_features.py for a paired audit')
    print(f'[bias] full - baseline on {summary["n_paired"]} '
          f'{"paired " if summary["paired"] else ""}queries')
    for name, entry in (('matched', summary['matched']),
                        ('unmatched', summary['unmatched'])):
        if entry is None:
            print(f'        {name:<9s}: none')
            continue
        ratio = entry['ratio']
        ratio_text = f'std/|mean|={ratio:.2f}' if ratio is not None else 'mean=0'
        print(f'        {name:<9s}: n={entry["n"]:<5d} delta p0 mean '
              f'{entry["mean"]:+.3f} std {entry["std"]:.3f} ({ratio_text}) '
              f'q10={entry["quantiles"][0.1]:+.3f} '
              f'q50={entry["quantiles"][0.5]:+.3f} '
              f'q90={entry["quantiles"][0.9]:+.3f}')
    correlation = summary['corr_with_negative_p0']
    if correlation is not None:
        print(f'        corr(delta p0, -p0_baseline) = {correlation:+.3f}')
    print(f'[verdict] the correction is a {summary["verdict"]}')
    if summary['verdict'] == 'near-constant offset':
        print('        a constant shift raises confidence almost uniformly, so '
              'the gain has to be corroborated by AP50/AP75 and by the number of '
              'false positives per image before it can be called genuine')
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        'estimate the S2H routing threshold and audit the correction')
    parser.add_argument('--baseline', default=None,
                        help='support-set export of the baseline model '
                             '(export_query_features.py --split train)')
    parser.add_argument('--full', default=None,
                        help='optional test export of the full model, for the '
                             'per-query bias audit')
    parser.add_argument('--quantile', type=float, default=0.35,
                        help='quantile of the positive support confidences '
                             'taken as theta (default 0.35)')
    parser.add_argument('--score-thr', type=float, default=None,
                        help='override the --score-thr stored in the export')
    parser.add_argument('--json', default=None,
                        help='optional sidecar JSON (analysis directory only)')
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def self_check() -> int:
    """Exercise the estimator and the audit on synthetic exports."""
    import tempfile

    def write(path, p0, matched, image, query, score=None, thr=0.3):
        payload = dict(p0=np.asarray(p0, np.float32),
                       matched=np.asarray(matched, bool),
                       label=np.zeros(len(p0), np.int64),
                       image=np.asarray(image, np.int64),
                       query=np.asarray(query, np.int64),
                       score=np.asarray(score if score is not None
                                        else np.ones(len(p0)), np.float32),
                       stage='baseline', score_thr=thr)
        np.savez_compressed(path, **payload)
        return path

    with tempfile.TemporaryDirectory() as folder:
        p0 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90]
        matched = [True, True, True, True, True, True, True, False]
        image = [0, 0, 0, 0, 1, 1, 1, 1]
        query = [0, 1, 2, 3, 0, 1, 2, 3]
        base = load_panel(write(osp.join(folder, 'base.npz'), p0, matched,
                                image, query))

        summary = theta_summary(base)
        assert summary['n'] == 8 and summary['n_matched'] == 7
        assert abs(summary['positive'][0.5] - 0.40) < 1e-6, summary['positive']
        # theta follows the confidence scale of the panel it is estimated from
        assert abs(adaptive_theta(base, 0.5) - 0.40) < 1e-6
        assert 0.30 < adaptive_theta(base, 0.35) < 0.40, adaptive_theta(base, 0.35)
        # the extremes are the smallest / largest *positive* value (float32 in
        # the export, hence the tolerance)
        assert abs(adaptive_theta(base, 0.0) - 0.10) < 1e-6
        assert abs(adaptive_theta(base, 1.0) - 0.80) < 1e-6, \
            adaptive_theta(base, 1.0)

        stats = profile(base)
        assert stats['n_images'] == 2 and stats['n_queries'] == 8
        assert abs(stats['preds_per_image'] - 4.0) < 1e-9, stats
        assert abs(stats['unmatched_per_image'] - 0.5) < 1e-9, stats
        dropped = profile(load_panel(write(
            osp.join(folder, 'scored.npz'), p0, matched, image, query,
            score=[0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])), score_thr=0.3)
        assert dropped['n_queries'] == 7, dropped

        # (a) a constant offset -> near-constant verdict, no correlation
        flat = load_panel(write(osp.join(folder, 'flat.npz'),
                                np.asarray(p0) + 0.1, matched, image, query))
        shift = pair_by_query(base, flat)
        assert shift is not None and shift[0].size == 8
        flat_summary = bias_summary(base, flat, shift)
        assert flat_summary['paired']
        assert abs(flat_summary['overall']['mean'] - 0.1) < 1e-6, flat_summary
        assert flat_summary['overall']['ratio'] < 1e-6, flat_summary['overall']
        assert flat_summary['verdict'] == 'near-constant offset', flat_summary

        # (b) a targeted correction: delta grows as p0 falls
        targeted = load_panel(write(
            osp.join(folder, 'targeted.npz'),
            [0.45, 0.45, 0.45, 0.45, 0.55, 0.60, 0.80, 0.90],
            matched, image, query))
        targeted_summary = bias_summary(base, targeted,
                                        pair_by_query(base, targeted))
        assert targeted_summary['corr_with_negative_p0'] > 0.35, targeted_summary
        assert targeted_summary['verdict'].startswith('targeted'), \
            targeted_summary
        assert targeted_summary['matched']['n'] == 7

        # an export without `p0` must be refused, not guessed
        np.savez_compressed(osp.join(folder, 'old.npz'),
                            matched=np.ones(3, bool), image=np.zeros(3, int))
        try:
            load_panel(osp.join(folder, 'old.npz'))
        except SystemExit:
            pass
        else:
            raise AssertionError('a panel without p0 was accepted')

        # the json guard has to keep the estimator out of the ablation tree
        try:
            assert_analysis_path('cat_work_dir/ablation/clipart1k/x.json')
        except SystemExit:
            pass
        else:
            raise AssertionError('the analysis guard let an ablation path pass')
        assert assert_analysis_path(
            'work_dirs/s2h_analysis/theta/x.json').endswith('x.json')
    print('[self-check] ok')
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.baseline:
        raise SystemExit('--baseline is required (the support-set export)')
    if not 0.0 < args.quantile < 1.0:
        raise SystemExit('--quantile must be inside (0, 1)')

    baseline = load_panel(args.baseline)
    if args.score_thr is not None:
        baseline['score_thr'] = float(args.score_thr)
    payload = dict(baseline=osp.abspath(args.baseline),
                   stage=baseline['stage'], quantile=args.quantile)
    theta = report_theta(theta_summary(baseline),
                         adaptive_theta(baseline, args.quantile),
                         args.quantile)
    payload['theta'] = theta
    payload['profile_baseline'] = report_profile(baseline)

    if args.full:
        full = load_panel(args.full)
        if args.score_thr is not None:
            full['score_thr'] = float(args.score_thr)
        payload['full'] = osp.abspath(args.full)
        payload['profile_full'] = report_profile(full)
        payload['bias'] = report_bias(
            bias_summary(baseline, full, pair_by_query(baseline, full)))

    if args.json:
        path = assert_analysis_path(args.json)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False,
                      default=float)
        print(f'[write] {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

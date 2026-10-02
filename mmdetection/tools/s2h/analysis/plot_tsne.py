#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Draw Figure 3: joint t-SNE of the exported decoder query embeddings.

The three panels are ``baseline`` / ``full`` / ``full, context removed``.  The
single most important implementation detail is that **all panels are embedded
with one joint fit**: t-SNE has no ``transform``, so fitting each panel
separately would produce three unrelated coordinate systems and the comparison
would be meaningless.  Points are stacked, embedded once, and then split again
by panel.

``sklearn.manifold.TSNE`` is used when it is installed (it is the reference
implementation the paper cites).  If it is missing, ``--tsne-backend auto``
falls back to an exact, self-contained t-SNE of the same objective (no new
dependency, no change to any environment); the implementation actually used is
printed and stored in the sidecar JSON, so a figure can always be reproduced.

Markers follow the paper caption: filled triangles are queries matched to a
ground truth box, hollow circles are high-confidence unmatched queries, and
colours denote categories.

Three deliberate choices make the figure defensible:

* **Every panel shows the same queries** (``--shared-queries``, default on).  The
  models put different queries above ``--score-thr`` (188 vs 169 in practice),
  so without the intersection a reader comparing panels compares different
  points.
* **The evidence metric is the score margin** ``s_gt - max_{c != gt} s_c``
  (``--evidence margin``, default): it is the quantity the detector acts on, it
  needs no prototypes, and unlike the cosine to the Soft prototypes - which is
  ~0 by construction, because ATAR consumes those prototypes through a scaled
  dot product - it can actually separate the variants.
* **The claim is tested paired** (last panel and ``paired_shifts`` in the JSON):
  the per-query change of the *same* queries, with a bootstrap CI and the
  fraction of queries that improved.  Two independent confidence intervals
  cannot resolve a shift that is an order of magnitude smaller than the spread.
  A pair whose two panels share the detector scores - a pure feature projection
  such as ``z_clean`` - is measured with the feature-space cosine instead,
  because a score margin is blind to a projection of the embedding; exports that
  predate the ``scores`` column fall back to the cosine for the same reason.

The Soft prototypes can still be drawn as stars with ``--anchors``, but they are
off by default: in one joint fit they collapse into a single blob and a joint
t-SNE cannot express the distance between an isolated cluster and a cloud.

Usage
-----
::

    python tools/s2h/analysis/plot_tsne.py \\
        --features work_dirs/s2h_analysis/tsne/clipart1k_5shot_seed3407/baseline.npz \\
                   work_dirs/s2h_analysis/tsne/clipart1k_5shot_seed3407/full.npz \\
        --labels 'Baseline' 'Full S2H-CD-FSOD' 'Full (context removed)' \\
        --out work_dirs/s2h_analysis/fig3_tsne

``--features`` may hold one file per panel; the third panel is used when the
last file carries a usable context-removed embedding (see
``export_query_features.py``), otherwise it is skipped with a warning.  The
figure then holds the t-SNE panels plus one panel with the paired per-query
change (``--with-delta-panel``), and the sidecar ``.json`` records the evidence
metric, the shared-query flag, the paired shifts and the md5 of the prototypes.
"""
import argparse
import hashlib
import json
import math
import os
import os.path as osp
import sys
from typing import Dict, List, Optional

import numpy as np

# tools/s2h/analysis/ -> the mmdet source tree (see export_query_features.py)
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..',
                                        '..')))

DEFAULT_LABELS = ('Baseline', 'Full S2H-CD-FSOD', 'Full (context removed)')
MATCHED = 'matched'
UNMATCHED = 'unmatched'


# ---------------------------------------------------------------------------
# helpers (pure numpy -> exercised by --self-check)
# ---------------------------------------------------------------------------
def load_panel(path: str) -> Dict:
    payload = np.load(path, allow_pickle=True)
    panel = {key: payload[key] for key in
             ('z', 'label', 'matched', 'score', 'image')}
    for optional in ('scores', 'query'):
        # written by the score-margin / shared-query revision of the exporter
        if optional in payload:
            panel[optional] = payload[optional]
    panel['classes'] = [str(c) for c in payload['classes']]
    clean = payload['z_clean'] if 'z_clean' in payload else None
    has_context = bool(payload['has_context']) if 'has_context' in payload \
        else False
    if clean is not None and has_context and np.isfinite(clean).all():
        panel['z_clean'] = clean
    panel['stage'] = str(payload['stage']) if 'stage' in payload else \
        osp.splitext(osp.basename(path))[0]
    panel['score_thr'] = float(payload['score_thr']) if 'score_thr' in payload \
        else 0.0
    return panel


def select_rows(panel: Dict, keep: np.ndarray) -> Dict:
    """The panel restricted to ``keep`` rows, with every column kept aligned."""
    rows = {key: panel[key][keep] for key in
            ('z', 'label', 'matched', 'score', 'image') if key in panel}
    for optional in ('scores', 'query', 'z_clean'):
        if optional in panel:
            rows[optional] = panel[optional][keep]
    return {**panel, **rows}


def balanced_keep(panel: Dict, limit: Optional[int], seed: int) -> np.ndarray:
    """Row indices of the balanced per-category / per-flag cap (see subsample).

    Returned separately from :func:`subsample` because the shared-query mode has
    to apply *one* mask to every panel: subsampling each panel on its own would
    cut a different subset of the shared queries and destroy the pairing.
    """
    if not limit:
        return np.arange(len(panel['z']), dtype=np.int64)
    rng = np.random.RandomState(seed)
    keep: List[int] = []
    for category in np.unique(panel['label']):
        for flag in (True, False):
            index = np.where((panel['label'] == category) &
                             (panel['matched'] == flag))[0]
            if index.size > limit:
                index = rng.choice(index, limit, replace=False)
            keep.extend(int(i) for i in np.sort(index))
    return np.asarray(sorted(keep), dtype=np.int64)


def subsample(panel: Dict, limit: Optional[int], seed: int) -> Dict:
    """Balanced cap: same number of matched / unmatched points per category."""
    if not limit:
        return panel
    return select_rows(panel, balanced_keep(panel, limit, seed))


def shared_query_subset(panels: List[Dict]) -> Optional[List[np.ndarray]]:
    """Per-panel row *indices* of the queries every panel contains.

    Different models put a different set of queries above ``--score-thr``, so
    the raw panels hold different point sets (188 vs 169 in practice) and a
    comparison of the clouds would compare different things.  Restricting every
    panel to the intersection on ``(image, query)`` makes all comparisons
    query-for-query - and the indices are returned in a **common canonical
    order**, so row *n* of every panel is the same query.  Keeping each panel in
    its own order would leave the paired tests comparing unrelated rows (they
    verify the pairing and refuse it).  ``None`` means the exports predate the
    ``query`` column.
    """
    if any('query' not in panel for panel in panels):
        return None
    keys = [[(int(i), int(q)) for i, q in zip(panel['image'], panel['query'])]
            for panel in panels]
    common = sorted(set.intersection(*[set(k) for k in keys]))
    indices = []
    for panel_keys in keys:
        position = {key: row for row, key in enumerate(panel_keys)}
        indices.append(np.asarray([position[key] for key in common], np.int64))
    return indices


def matched_cosines(panel: Dict,
                    prototypes: Optional[np.ndarray]) -> np.ndarray:
    """Cosine of every *matched* query against its own class evidence.

    The vector stays aligned with the panel's rows (NaN where a query is
    unmatched, its label has no prototype or a norm vanishes), which is what
    lets two panels describing the same queries be compared entry by entry.
    """
    cosines = np.full(len(panel['z']), np.nan, np.float64) \
        if 'z' in panel else np.zeros(0, np.float64)
    if prototypes is None or not cosines.size:
        return cosines
    for index in np.where(panel['matched'].astype(bool))[0]:
        label = int(panel['label'][index])
        if label >= len(prototypes):
            continue
        vector = panel['z'][index]
        prototype = prototypes[label]
        denominator = float(np.linalg.norm(vector) * np.linalg.norm(prototype))
        if denominator > 0:
            cosines[index] = float(vector @ prototype / denominator)
    return cosines


def matched_margins(panel: Dict) -> np.ndarray:
    """Score margin of every matched query: ``s_gt - max_{c != gt} s_c``.

    This is the quantity the detector actually acts on - the correct category
    against its strongest competitor - and it needs no prototypes, because
    matching is class-aware and a matched query's own label *is* the ground
    truth category.  NaN where the query is unmatched or the export predates
    the ``scores`` column.
    """
    margins = np.full(len(panel['z']), np.nan, np.float64)
    if 'scores' not in panel:
        return margins
    for index in np.where(panel['matched'].astype(bool))[0]:
        row = np.asarray(panel['scores'][index], np.float64)
        label = int(panel['label'][index])
        if row.size < 2 or label >= row.size:
            continue
        margins[index] = float(row[label] - np.delete(row, label).max())
    return margins


def evidence_row(panel: Dict, prototypes: Optional[np.ndarray],
                 metric: str = 'margin') -> np.ndarray:
    """Row-aligned per-query evidence, NaN where it is undefined."""
    return matched_cosines(panel, prototypes) if metric == 'cosine' \
        else matched_margins(panel)


def class_evidence(panel: Dict, prototypes: Optional[np.ndarray],
                   metric: str = 'margin') -> np.ndarray:
    """The usable per-query evidence values, i.e. without the NaN entries.

    This is the number the caption's claim rests on, so it is printed and
    written to JSON next to the figure.
    """
    values = evidence_row(panel, prototypes, metric)
    return values[np.isfinite(values)]


def panel_metric(metas: List[Dict], order: int, requested: str) -> str:
    """Metric to report for one panel.

    Two situations force the feature-space cosine instead of the score margin:
    an export that predates the ``scores`` column (the margin simply cannot be
    computed), and a panel whose classification scores repeat the previous
    panel's - a pure feature projection such as ``z`` vs ``z_clean``, which a
    score-based margin is blind to and would silently report as unchanged.
    """
    if requested != 'margin':
        return requested
    current = metas[order]
    if 'scores' not in current:
        return 'cosine'
    if order and 'scores' not in metas[order - 1]:
        return 'cosine'
    if order and np.array_equal(metas[order - 1]['scores'], current['scores']):
        return 'cosine'
    return requested


def same_queries(panel_a: Dict, panel_b: Dict) -> bool:
    """Whether two panels describe the same queries in the same row order.

    Identity is ``(image, query)`` and never the predicted ``label``: two models
    predict different categories on the same query - that is the difference the
    figure is about - so validating the pairing through the labels rejects
    exactly the pairs that matter.  (An earlier version did that and silently
    dropped the baseline -> full comparison.)
    """
    if len(panel_a['z']) != len(panel_b['z']):
        return False
    if 'query' in panel_a and 'query' in panel_b:
        return np.array_equal(panel_a['image'], panel_b['image']) and \
            np.array_equal(panel_a['query'], panel_b['query'])
    # exports without a `query` column: only `z` vs `z_clean` of one file can be
    # paired, and those rows are identical by construction
    return np.array_equal(panel_a['label'], panel_b['label']) and \
        np.array_equal(panel_a['image'], panel_b['image'])


def paired_alignment(panel_a: Dict, panel_b: Dict,
                     prototypes: Optional[np.ndarray],
                     metric: str = 'margin') -> Optional[tuple]:
    """Per-query evidence of two panels that describe the *same* queries.

    A shift of a few hundredths cannot be read off two independent confidence
    intervals, because the per-query scatter is an order of magnitude larger
    than the shift; only a paired comparison of the very same queries can
    resolve it.  Rows are paired positionally - true for any two panels that
    were restricted to the shared queries, and for ``z`` vs ``z_clean`` of one
    export, which are appended together - and the pairing is validated through
    the query identity before it is trusted.
    """
    if not same_queries(panel_a, panel_b):
        print('[warn] the two panels are not the same queries (different '
              'counts / images / slots); the paired test is skipped')
        return None
    before = evidence_row(panel_a, prototypes, metric)
    after = evidence_row(panel_b, prototypes, metric)
    usable = np.isfinite(before) & np.isfinite(after)
    if not usable.any():
        return None
    return before[usable], after[usable]


def mcnemar_exact(newly: int, lost: int) -> Optional[float]:
    """Two-sided exact McNemar test over the discordant pairs.

    ``newly`` queries turn from unmatched into matched between two panels of the
    same queries, ``lost`` the other way round.  The exact binomial version is
    used rather than the chi-square approximation because a 100-image figure has
    few discordant pairs, and it needs no scipy.
    """
    total = int(newly) + int(lost)
    if total == 0:
        return None
    tail = sum(math.comb(total, i) for i in range(min(newly, lost) + 1)) / \
        float(2 ** total)
    return min(1.0, 2.0 * tail)


def match_flips(panel_a: Dict, panel_b: Dict) -> Optional[Dict]:
    """How many of the *same* queries change match status between two panels.

    This is the figure's strongest and least assailable statement: on identical
    queries, the method turns high-confidence unmatched responses into matches.
    It is a paired binary outcome, so the test is McNemar's exact one.
    """
    if not same_queries(panel_a, panel_b):
        print('[warn] the two panels are not the same queries; the '
              'match-status flips are skipped')
        return None
    before = panel_a['matched'].astype(bool)
    after = panel_b['matched'].astype(bool)
    newly = int((~before & after).sum())
    lost = int((before & ~after).sum())
    return dict(newly_matched=newly, lost_matched=lost, net=newly - lost,
                n=int(before.size), matched_before=int(before.sum()),
                matched_after=int(after.sum()), discordant=newly + lost,
                mcnemar_p=mcnemar_exact(newly, lost))


def prototype_fingerprint(path: Optional[str]) -> Optional[Dict]:
    """Path, size and md5 of the knowledge file behind the anchors.

    Two runs of the same figure must use the *same* prototypes, but nothing in
    the coordinates says which file was loaded; this makes it auditable.
    """
    if not path:
        return None
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return dict(path=osp.abspath(path), bytes=osp.getsize(path),
                md5=digest.hexdigest())


def bootstrap_ci(values: np.ndarray, samples: int = 2000, seed: int = 0
                 ) -> Optional[List[float]]:
    """Percentile bootstrap CI of the mean (reported in the caption)."""
    if values.size == 0:
        return None
    rng = np.random.RandomState(seed)
    means = [values[rng.randint(0, values.size, values.size)].mean()
             for _ in range(samples)]
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def self_check() -> int:
    """Exercise the balanced cap, the evidence metric and the bootstrap."""
    panel = dict(
        z=np.array([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]], np.float32),
        label=np.array([0, 0, 1, 1], np.int64),
        matched=np.array([True, False, True, False]),
        score=np.ones(4, np.float32), image=np.zeros(4, np.int64))
    capped = subsample(panel, 1, 0)
    assert len(capped['z']) == 4, capped['z']           # nothing to cut yet
    bigger = {**panel, 'z': np.concatenate([panel['z']] * 3),
              'label': np.tile(panel['label'], 3),
              'matched': np.tile(panel['matched'], 3),
              'score': np.ones(12, np.float32),
              'image': np.zeros(12, np.int64)}
    capped = subsample(bigger, 2, 0)
    assert len(capped['z']) == 8, len(capped['z'])      # 2 per class/flag
    prototypes = np.array([[1.0, 0], [0, 1.0]], np.float32)
    evidence = class_evidence(panel, prototypes, 'cosine')
    assert evidence.size == 2 and np.allclose(evidence, 1.0), evidence
    ci = bootstrap_ci(evidence, samples=200, seed=0)
    assert ci and abs(ci[0] - ci[1]) < 1e-6, ci
    assert bootstrap_ci(np.zeros(0)) is None

    # the built-in backend must actually separate two well-separated clusters,
    # otherwise the fallback would silently draw a meaningless figure
    rng = np.random.RandomState(0)
    cloud = np.concatenate([rng.normal(-4.0, 0.25, size=(40, 8)),
                            rng.normal(4.0, 0.25, size=(40, 8))]).astype(
                                np.float32)
    embedded = tsne_numpy(cloud, perplexity=10.0, seed=0, n_iter=300)
    assert embedded.shape == (80, 2) and np.isfinite(embedded).all()
    spread = np.linalg.norm(embedded[:40].mean(0) - embedded[:20].mean(0))
    gap = np.linalg.norm(embedded[:40].mean(0) - embedded[40:].mean(0))
    assert gap > 4.0 * spread, (spread, gap)

    # the dispatcher reports which implementation ran, and an explicit
    # `sklearn` request must fail loudly instead of being substituted
    coordinates, name = embed_2d(cloud, 10.0, 0, 'numpy', 300, 5000)
    assert name == 'numpy-exact' and coordinates.shape == (80, 2), name
    try:
        _, requested = embed_2d(cloud, 10.0, 0, 'sklearn', 300, 5000)
    except SystemExit:
        requested = None            # scikit-learn missing -> refused
    else:
        assert requested == 'sklearn', requested
    try:
        embed_2d(cloud, 10.0, 0, 'numpy', 300, 10)
    except SystemExit:
        pass                        # the O(n^2) guard has to stop it
    else:
        raise AssertionError('--numpy-max-points was ignored')

    # the third panel must report the *projected* evidence: an earlier version
    # copied panel 2's `z`, so the JSON printed identical numbers twice while
    # the scatter was drawn from `z_clean`
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        for name, shifted in (('baseline', False), ('full', True)):
            z = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
            np.savez_compressed(
                osp.join(folder, f'{name}.npz'), z=z,
                z_clean=z if name == 'baseline' else z - 1.0,
                label=np.array([0, 1], np.int64),
                matched=np.array([True, True]),
                score=np.ones(2, np.float32), image=np.zeros(2, np.int64),
                classes=np.asarray(['a', 'b'], dtype=object), stage=name,
                score_thr=0.3, has_context=shifted)
        stub = argparse.Namespace(
            features=[osp.join(folder, 'baseline.npz'),
                      osp.join(folder, 'full.npz')],
            max_per_class=0, sample_seed=0, labels=None,
            with_context_panel=True, shared_queries=False)
        matrices, metas, _ = build_panels(stub)
    assert len(matrices) == 3, len(matrices)
    assert np.allclose(matrices[2], metas[2]['z']), metas[2]['z']
    assert not np.allclose(metas[2]['z'], metas[1]['z']), 'panel 3 reused z'

    # the paired test must align the two panels row by row and refuse a pair
    # that does not describe the same queries
    prototype = np.array([[1.0, 0.0]], np.float32)
    before = dict(z=np.array([[1.0, 0.5], [0.5, 1.0]], np.float32),
                  label=np.array([0, 0], np.int64),
                  matched=np.array([True, True]),
                  image=np.array([0, 1], np.int64),
                  query=np.array([3, 4], np.int64))
    after = {**before,
             'z': np.array([[1.0, 0.1], [1.0, 0.3]], np.float32)}
    pair = paired_alignment(before, after, prototype, 'cosine')
    assert pair is not None
    assert pair[0].shape == (2,) and np.all(pair[1] > pair[0]), pair
    assert abs(pair[0][0] - 1.0 / np.sqrt(1.25)) < 1e-6, pair[0]
    assert paired_alignment(before, after, None, 'cosine') is None
    # different predictions on the same query are NOT a different query set:
    # identity is (image, query), and rejecting a pair on the `label` alone is
    # what silently dropped the baseline -> full comparison
    relabelled = paired_alignment(before, {**after, 'label': np.array([0, 1])},
                                  prototype, 'cosine')
    assert relabelled is not None, 'a label mismatch must not break the pairing'
    assert paired_alignment(before, {**after, 'query': np.array([3, 5])},
                            prototype, 'cosine') is None
    assert paired_alignment(before, {**after, 'image': np.array([0, 0])},
                            prototype, 'cosine') is None
    row = matched_cosines({'z': before['z'], 'label': before['label'],
                           'matched': np.array([True, False])}, prototype)
    assert np.isnan(row[1]) and abs(row[0] - 1.0 / np.sqrt(1.25)) < 1e-6, row
    assert class_evidence({'z': before['z'], 'label': before['label'],
                           'matched': np.array([True, False])},
                          prototype, 'cosine').size == 1

    # score margin: the quantity the detector acts on, computed from the
    # exported per-class distribution and needing no prototypes
    margins = matched_margins(dict(
        z=np.zeros((3, 2), np.float32), label=np.array([0, 1, 2], np.int64),
        matched=np.array([True, True, False]),
        scores=np.array([[0.8, 0.2, 0.1], [0.3, 0.5, 0.4], [0.0, 0.0, 0.0]],
                        np.float32)))
    assert abs(margins[0] - 0.6) < 1e-6, margins
    assert abs(margins[1] - 0.1) < 1e-6, margins
    assert np.isnan(margins[2]), margins
    assert np.isnan(matched_margins({'z': np.zeros((1, 2), np.float32),
                                     'label': np.zeros(1, np.int64),
                                     'matched': np.array([True])})).all()
    # a paired margin comparison on the very same queries
    base = dict(z=np.zeros((2, 2), np.float32),
                label=np.array([0, 0], np.int64),
                matched=np.array([True, True]), image=np.array([0, 1], np.int64),
                query=np.array([0, 1], np.int64),
                scores=np.array([[0.60, 0.40], [0.55, 0.45]], np.float32))
    better = {**base,
              'scores': np.array([[0.90, 0.10], [0.70, 0.30]], np.float32)}
    shift = paired_alignment(base, better, None, 'margin')
    assert shift is not None and np.all(shift[1] > shift[0]), shift
    assert abs((shift[1] - shift[0]).mean() - 0.45) < 1e-6, shift

    # shared queries: only the intersection of (image, query) survives and the
    # panels stay row-aligned afterwards
    left = dict(z=np.zeros((4, 2), np.float32),
                label=np.array([0, 0, 0, 1], np.int64),
                matched=np.array([True, True, True, True]),
                image=np.array([0, 0, 1, 1], np.int64),
                query=np.array([1, 2, 3, 4], np.int64),
                scores=np.arange(8, dtype=np.float32).reshape(4, 2))
    right = {**left, 'image': np.array([0, 1, 1, 2], np.int64),
             'z': np.ones((4, 2), np.float32)}
    masks = shared_query_subset([left, right])
    # (0, 1) and (1, 3) are in both panels; (0, 2) / (1, 4) are not, and the
    # indices come back in the canonical key order, not in each panel's own
    assert masks is not None and masks[0].size == 2, masks
    assert masks[0].tolist() == [0, 2] and masks[1].tolist() == [0, 2], masks
    trimmed = [select_rows(panel, mask)
               for panel, mask in zip([left, right], masks)]
    assert trimmed[0]['query'].tolist() == trimmed[1]['query'].tolist() == \
        [1, 3], trimmed[0]['query']
    assert np.array_equal(trimmed[0]['scores'], left['scores'][[0, 2]])
    assert shared_query_subset([{'z': left['z']}, left]) is None
    # one shared keep mask keeps the panels pairable
    keep = balanced_keep(trimmed[0], 10, 0)
    assert np.array_equal(keep, np.arange(2)), keep

    # match-status flips: the figure's strongest claim, tested exactly
    assert abs(mcnemar_exact(10, 0) - 2.0 / 1024.0) < 1e-12, mcnemar_exact(10, 0)
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(7, 3) == mcnemar_exact(3, 7), 'must be two-sided'
    assert mcnemar_exact(5, 5) == 1.0, mcnemar_exact(5, 5)
    flip_before = dict(z=np.zeros((4, 2), np.float32),
                       label=np.zeros(4, np.int64),
                       matched=np.array([False, False, True, True]),
                       image=np.arange(4, dtype=np.int64))
    flip_after = {**flip_before,
                  'matched': np.array([True, True, True, False])}
    entry = match_flips(flip_before, flip_after)
    assert entry is not None
    assert (entry['newly_matched'], entry['lost_matched']) == (2, 1), entry
    assert entry['net'] == 1 and entry['matched_before'] == 2, entry
    assert entry['matched_after'] == 3 and entry['discordant'] == 3, entry
    assert abs(entry['mcnemar_p'] - 1.0) < 1e-12, entry
    assert match_flips(flip_before, {**flip_after,
                                     'image': np.arange(4, 8, dtype=np.int64)}) \
        is None

    # the same queries in a different row order must be re-aligned: keeping each
    # panel in its own order is what silently dropped the baseline -> full pair
    rolled = select_rows(left, np.array([2, 3, 0, 1]))
    ordered = shared_query_subset([left, rolled])
    assert ordered is not None
    aligned = [select_rows(panel, mask)
               for panel, mask in zip([left, rolled], ordered)]
    assert aligned[0]['query'].tolist() == aligned[1]['query'].tolist() == \
        [1, 2, 3, 4], [a['query'].tolist() for a in aligned]
    assert np.array_equal(aligned[0]['image'], aligned[1]['image'])
    assert paired_alignment(aligned[0], aligned[1], None, 'margin') is not None, \
        'rows of the same queries were not re-aligned'
    print('[self-check] ok')
    return 0


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser('Figure 3: t-SNE of decoder queries')
    parser.add_argument('--features', nargs='+', default=None,
                        help='one .npz per panel (see export_query_features)')
    parser.add_argument('--labels', nargs='*', default=None)
    parser.add_argument('--prototypes', default=None,
                        help='optional knowledge .pth providing the Soft '
                             'prototypes drawn as stars')
    parser.add_argument('--max-per-class', type=int, default=150,
                        help='balanced cap per panel/class (0 = keep all)')
    parser.add_argument('--context-panel', dest='with_context_panel',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='add the panel from the context-removed embedding '
                             'of the last --features file; disable it with '
                             '--no-context-panel')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-seed', type=int, default=0)
    parser.add_argument('--tsne-backend', default='auto',
                        choices=['auto', 'sklearn', 'numpy'],
                        help="'auto' (default) uses scikit-learn when it is "
                             "installed and the built-in exact t-SNE "
                             "otherwise, 'numpy' forces the built-in one")
    parser.add_argument('--tsne-iter', type=int, default=1000,
                        help='iterations of the built-in t-SNE (ignored by '
                             'scikit-learn, which has its own schedule)')
    parser.add_argument('--numpy-max-points', type=int, default=5000,
                        help='refuse the built-in O(n^2) t-SNE above this many '
                             'points (its time and memory grow quadratically)')
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--evidence', default='margin',
                        choices=['margin', 'cosine'],
                        help="'margin' (default) reports the score margin "
                             "s_gt - max_{c != gt} s_c of the matched queries, "
                             "i.e. the quantity the detector acts on; 'cosine' "
                             "reports the cosine to the Soft prototypes, which "
                             "needs --prototypes and is near zero by "
                             "construction")
    parser.add_argument('--shared-queries',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='restrict every panel to the queries all panels '
                             'have in common, so every comparison is '
                             'query-for-query (needs the `query` column '
                             'written by export_query_features.py)')
    parser.add_argument('--delta-panel', dest='with_delta_panel',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='add a panel with the paired per-query change of '
                             'neighbouring panels (disable with '
                             '--no-delta-panel)')
    parser.add_argument('--flip-panel', dest='with_flip_panel',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='add a panel with the matched <-> unmatched flips '
                             'of the same queries and the exact McNemar p '
                             '(disable with --no-flip-panel)')
    parser.add_argument('--delta-bins', type=int, default=24)
    parser.add_argument('--rows', type=int, default=2,
                        help='rows of panels; 2 (default) keeps a five-panel '
                             'figure readable when it is scaled to \\textwidth, '
                             '1 puts everything in a single strip')
    parser.add_argument('--anchors',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='draw the Soft class prototypes as stars; off by '
                             'default because one joint t-SNE cannot express '
                             'their distance to the query cloud and they '
                             'collapse into a single blob')
    parser.add_argument('--panel-size', nargs=2, type=float,
                        default=(2.5, 2.5), metavar=('W', 'H'))
    parser.add_argument('--point-size', type=float, default=9.0)
    parser.add_argument('--legend-classes', type=int, default=12,
                        help='how many categories to name in the legend')
    parser.add_argument('--out', default=None,
                        help='output stem, e.g. work_dirs/s2h_analysis/fig3_tsne')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg'])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def _guard(path: str) -> str:
    """Reuse the writer guard of the export script (kept in one place)."""
    try:
        from export_query_features import assert_analysis_path
    except Exception:                                  # pragma: no cover
        def assert_analysis_path(target):
            parts = osp.abspath(target).replace('\\', '/').split('/')
            if 'ablation' in parts or not any(
                    r in parts for r in ('s2h_analysis', 'sweep')):
                raise SystemExit(f'refusing to write to {target}')
            return target
    return assert_analysis_path(path)


def load_prototypes(path: str) -> Optional[np.ndarray]:
    import torch
    payload = torch.load(path, map_location='cpu')
    soft = payload.get('soft')
    if soft is None:
        print(f'[warn] {path} has no `soft` tensor; drawing without anchors')
        return None
    return soft.float().numpy()


def build_panels(args) -> tuple:
    """Return ``(matrices, metas, titles)``.

    With ``--shared-queries`` every panel is first restricted to the queries all
    panels have in common and then capped with **one** shared keep mask, so the
    panels stay row-aligned and any pair of them can be compared query by query.
    Panel 3 reuses panel 2's rows by construction.
    """
    loaded = [load_panel(p) for p in args.features]
    masks = shared_query_subset(loaded) if args.shared_queries else None
    if masks is None and args.shared_queries:
        print('[warn] the exports carry no `query` column, so the panels keep '
              'their own point sets (re-run export_query_features.py to '
              'compare the same queries)')
    elif masks is not None:
        if not masks[0].size:
            raise SystemExit('the panels share no query at all; check '
                             '--features')
        counts = ', '.join(f'{len(mask)}/{len(panel["z"])}'
                           for mask, panel in zip(masks, loaded))
        print(f'[queries] shared across {len(loaded)} panels: {counts} '
              '(kept/total, identical row order in every panel)')
        loaded = [select_rows(panel, mask)
                  for panel, mask in zip(loaded, masks)]

    limit = args.max_per_class or None
    if limit and masks is not None:
        # one mask for every panel: cutting each panel on its own would keep a
        # different subset of the shared queries and break the pairing
        keep = balanced_keep(loaded[0], limit, args.sample_seed)
        panels = [select_rows(panel, keep) for panel in loaded]
    else:
        panels = [subsample(panel, limit, args.sample_seed)
                  for panel in loaded]

    titles = list(args.labels) if args.labels else list(DEFAULT_LABELS)
    matrices = [np.asarray(p['z'], np.float32) for p in panels]
    metas = list(panels)
    last = panels[-1]
    if args.with_context_panel and 'z_clean' in last:
        clean = np.asarray(last['z_clean'], np.float32)
        matrices.append(clean)
        # two different consumers: the scatter is drawn from `matrices` while
        # the evidence report reads `z`, so the third panel has to expose the
        # *projected* vectors - otherwise it silently reprints panel 2
        metas.append({**last, 'z': clean, 'matched': last['matched']})
        while len(titles) < len(matrices):
            # index by the number of panels, so the third one is not named after
            # the second (`--labels` may name fewer panels than are drawn)
            titles.append(DEFAULT_LABELS[len(titles)]
                          if len(titles) < len(DEFAULT_LABELS)
                          else f'panel {len(titles) + 1}')
    elif args.with_context_panel:
        print('[warn] no context-removed embedding in '
              f'{args.features[-1]}; the third panel is skipped')
    titles = titles[:len(matrices)]
    return matrices, metas, titles


# ---------------------------------------------------------------------------
# embedding backends
# ---------------------------------------------------------------------------
def tsne_numpy(matrix: np.ndarray, perplexity: float, seed: int,
               n_iter: int = 1000, exaggeration: float = 12.0,
               exag_iters: int = 250) -> np.ndarray:
    """Exact (O(n^2)) t-SNE, used when scikit-learn is not installed.

    Same objective and schedule as ``sklearn.manifold.TSNE``: conditional
    Gaussians tuned to the target perplexity, Student-t low dimensional
    similarities, early exaggeration at an automatic learning rate
    (``n / exaggeration / 4``, the value behind ``learning_rate='auto'``), a
    momentum ramp at ``exag_iters`` and a PCA initialisation.  Only the
    optimisation is different - the gradient is evaluated exactly instead of
    with a Barnes-Hut tree, which costs ``O(n^2)`` memory and time per step and
    is why ``--numpy-max-points`` guards the entry.
    """
    n = len(matrix)
    x = np.asarray(matrix, np.float64)
    x = x - x.mean(0)
    squared = (x ** 2).sum(1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2.0 * x @ x.T,
                           0.0)

    # binary search for the precision beta that gives each row the target
    # perplexity, exactly as the reference implementation does
    conditional = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        row = distances[i]
        beta, lo, hi = 1.0, -np.inf, np.inf
        total = 0.0
        probabilities = None
        for _ in range(60):
            probabilities = np.exp(-row * beta)
            probabilities[i] = 0.0
            total = float(probabilities.sum())
            if total <= 0:
                break
            entropy = np.log(total) + beta * float((row * probabilities).sum()) \
                / total
            if abs(entropy - target) <= 1e-5:
                break
            if entropy > target:
                lo = beta
                beta = beta * 2.0 if hi == np.inf else 0.5 * (beta + hi)
            else:
                hi = beta
                beta = beta * 0.5 if lo == -np.inf else 0.5 * (beta + lo)
        if total > 0:
            conditional[i] = probabilities / total
    joint = np.maximum(conditional + conditional.T, 1e-12) / (2.0 * n)

    # PCA init (`init='pca'`), tiny scale so the early phase can still move
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    y = u[:, :2] * singular[:2] * 1e-4
    learning_rate = 0.25 * n / exaggeration
    gains = np.ones_like(y)
    velocity = np.zeros_like(y)
    momentum = 0.5
    for step in range(max(1, n_iter)):
        row_norm = (y ** 2).sum(1)
        attraction = 1.0 / (1.0 + row_norm[:, None] + row_norm[None, :]
                            - 2.0 * y @ y.T)
        np.fill_diagonal(attraction, 0.0)
        similarity = np.maximum(attraction / attraction.sum(), 1e-12)
        weight = (joint * (exaggeration if step < exag_iters else 1.0)
                  - similarity) * attraction
        gradient = 4.0 * (weight.sum(1, keepdims=True) * y - weight @ y)
        if step == exag_iters:
            momentum = 0.8
        gains = np.maximum(np.where(np.sign(velocity) != np.sign(gradient),
                                    gains + 0.2, gains * 0.8), 0.01)
        velocity = momentum * velocity - learning_rate * gains * gradient
        y = y + velocity
        y = y - y.mean(0)
        if (step + 1) % 100 == 0:
            print(f'[tsne]   built-in backend: {step + 1}/{n_iter} iterations')
    return y.astype(np.float32)


def embed_2d(matrix: np.ndarray, perplexity: float, seed: int,
             backend: str = 'auto', n_iter: int = 1000,
             max_points: int = 5000) -> tuple:
    """Embed ``matrix`` and report which backend produced the coordinates.

    scikit-learn is preferred whenever it can be imported, because it is the
    implementation the paper's protocol refers to; ``--tsne-backend numpy``
    forces the built-in one (useful to check that a conclusion does not depend
    on the backend).
    """
    if backend in ('auto', 'sklearn'):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            if backend == 'sklearn':
                raise SystemExit('--tsne-backend sklearn was requested but '
                                 'scikit-learn is not installed in this '
                                 'environment')
            print('[warn] scikit-learn is not installed; using the built-in '
                  'exact t-SNE (same objective, slower optimisation)')
        else:
            embedded = TSNE(n_components=2, perplexity=perplexity, init='pca',
                            learning_rate='auto',
                            random_state=seed).fit_transform(matrix)
            return np.asarray(embedded, np.float32), 'sklearn'
    if len(matrix) > max_points:
        raise SystemExit(
            f'the built-in t-SNE is O(n^2) and received {len(matrix)} points '
            f'(> --numpy-max-points {max_points}): install scikit-learn or '
            'lower --max-per-class')
    return tsne_numpy(matrix, perplexity, seed, n_iter), 'numpy-exact'


DELTA_COLOURS = ('#1f77b4', '#d62728', '#2ca02c')


def draw_delta_panel(axis, pairs: List[Dict], titles: List[str], args,
                     line2d) -> None:
    """Histogram of the paired per-query change of adjacent panel pairs.

    A mean shift of a few hundredths is invisible in a scatter plot, but the
    distribution of the per-query differences makes it readable and also shows
    how many queries moved the right way - which is the sentence the caption
    should quote instead of a difference of two means.
    """
    handles = []
    for order, pair in enumerate(pairs):
        delta = pair['after'] - pair['before']
        colour = DELTA_COLOURS[order % len(DELTA_COLOURS)]
        axis.hist(delta, bins=args.delta_bins, histtype='step', lw=1.4,
                  color=colour, density=True, zorder=3)
        mean = float(delta.mean())
        axis.axvline(mean, color=colour, ls='--', lw=1.0, zorder=2)
        interval = bootstrap_ci(delta, seed=args.tsne_seed)
        text = f"[{pair.get('metric', 'margin')}] " \
               f"{titles[pair['right']]} - {titles[pair['left']]}"
        if interval:
            # two lines: the pair name is long and the panel is only ~2.5in wide
            text += (f'\nmean {mean:+.3f} [{interval[0]:+.3f}, '
                     f'{interval[1]:+.3f}], '
                     f'{100.0 * (delta > 0.0).mean():.0f}% up')
        else:
            text += f'  mean {mean:+.3f}'
        handles.append(line2d([], [], color=colour, lw=1.4, label=text))
    axis.axvline(0.0, color='0.45', lw=0.8, zorder=1)
    # no x-label: the metric is already named in every legend entry below, and a
    # label here collides with the figure-level legend
    axis.set_title('paired change, same queries', fontsize=8)
    axis.tick_params(axis='x', labelsize=6)
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    # a light frame keeps the numbers readable even when the bars run under it
    axis.legend(handles=handles, fontsize=6, loc='upper left',
                frameon=True, framealpha=0.85, edgecolor='none')


def draw_flip_panel(axis, flips: List[Dict], titles: List[str],
                    pairs: List[Dict], line2d) -> None:
    """Matched <-> unmatched flips of the *same* queries, with McNemar's p.

    Up bars: high-confidence unmatched responses the method turns into matches.
    Down bars: matches it loses.  Both counts come from the identical query set,
    so the comparison cannot be dismissed as "a different set of queries was
    scored" - which is exactly the objection a curated qualitative figure
    invites.
    """
    positions = np.arange(len(flips), dtype=np.float64)
    width = 0.36
    gain = np.asarray([entry['newly_matched'] for entry in flips], np.float64)
    drop = np.asarray([entry['lost_matched'] for entry in flips], np.float64)
    gained = axis.bar(positions - width / 2, gain, width, color='#1f77b4',
                      zorder=3, label='unmatched $\\rightarrow$ matched')
    lost = axis.bar(positions + width / 2, -drop, width, color='#d62728',
                    zorder=3, label='matched $\\rightarrow$ unmatched')
    for position, entry in zip(positions, flips):
        axis.annotate(f"+{entry['newly_matched']}",
                      (position - width / 2, entry['newly_matched']),
                      ha='center', va='bottom', fontsize=6)
        axis.annotate(f"-{entry['lost_matched']}",
                      (position + width / 2, -entry['lost_matched']),
                      ha='center', va='top', fontsize=6)
    axis.axhline(0.0, color='0.45', lw=0.8, zorder=2)
    axis.set_xticks(positions)
    axis.set_xticklabels([f'#{index + 1}' for index in range(len(flips))],
                         fontsize=7)
    axis.tick_params(axis='y', labelsize=6)
    axis.set_title('match-status flips, same queries', fontsize=8)
    # the pair names are long, so the legend carries them instead of the ticks
    handles = [gained, lost]
    for index, (entry, pair) in enumerate(zip(flips, pairs)):
        p_value = entry['mcnemar_p']
        text = f"#{index + 1} {titles[pair['right']]} - {titles[pair['left']]}"
        text += f', McNemar p = {p_value:.2g}' if p_value is not None \
            else ', p = n/a'
        handles.append(line2d([], [], linestyle='none',
                              label=text))
    axis.legend(handles=handles, fontsize=5, loc='upper left', frameon=True,
                framealpha=0.85, edgecolor='none')


def draw(matrices: List[np.ndarray], metas: List[Dict], titles: List[str],
         prototypes: Optional[np.ndarray], args) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blocks, slices, cursor = [], [], 0
    for matrix in matrices:
        blocks.append(matrix)
        slices.append((cursor, cursor + len(matrix)))
        cursor += len(matrix)
    anchor = None
    if args.anchors and prototypes is not None and len(prototypes):
        blocks.append(np.asarray(prototypes, np.float32))
        anchor = (cursor, cursor + len(prototypes))
    stacked = np.concatenate(blocks, 0)
    total = len(stacked)
    perplexity = args.perplexity
    if total <= perplexity * 3:
        perplexity = max(5.0, (total - 1) / 3.0)
        print(f'[warn] {total} points: perplexity lowered to {perplexity:.1f}')
    print(f'[tsne] joint fit of {total} points (all panels together), '
          f'perplexity={perplexity:.1f}, seed={args.tsne_seed}')
    embedded, backend = embed_2d(stacked, perplexity, args.tsne_seed,
                                 args.tsne_backend, args.tsne_iter,
                                 args.numpy_max_points)
    print(f'[tsne] backend={backend}')

    cmap = plt.get_cmap('tab20')
    colours = {int(c): cmap(i % 20) for i, c in enumerate(
        sorted({int(v) for m in metas for v in np.unique(m['label'])}))}
    names = metas[0]['classes']
    from matplotlib.lines import Line2D

    # paired, per-query comparison of neighbouring panels; only identical query
    # sets can be paired, which `--shared-queries` guarantees
    pairs, flips = [], []
    for left in range(len(metas) - 1):
        metric = panel_metric(metas, left + 1, args.evidence)
        if metric != args.evidence and 'scores' not in metas[left]:
            print('[paired] the exports carry no `scores` column, so the paired '
                  'tests use the cosine to the class evidence (re-export to get '
                  'the score margin)')
        elif metric != args.evidence:
            print(f'[paired] {titles[left + 1]} shares the detector scores of '
                  f'{titles[left]}, so it is measured with the cosine to the '
                  'class evidence instead of the score margin')
        pair = paired_alignment(metas[left], metas[left + 1], prototypes,
                                metric)
        flip = match_flips(metas[left], metas[left + 1]) if pair else None
        if pair is None or flip is None:
            continue
        pairs.append(dict(left=left, right=left + 1, before=pair[0],
                          after=pair[1], metric=metric))
        flips.append(flip)
    show_delta = bool(pairs) and args.with_delta_panel
    show_flips = bool(flips) and args.with_flip_panel
    width, height = args.panel_size
    panels_width = len(matrices) + (1 if show_delta else 0) \
        + (1 if show_flips else 0)
    # a single row of five panels is unusable in a two-column paper (shrinking it
    # to \textwidth leaves ~5pt labels), so the default is two rows
    rows = max(1, min(args.rows, panels_width))
    cols = int(math.ceil(panels_width / rows))
    fig, grid = plt.subplots(rows, cols, figsize=(width * cols, height * rows),
                             squeeze=False)
    axes = list(grid.ravel())
    handles = None
    for axis, matrix, meta, title, (start, stop) in zip(
            axes[:len(matrices)], matrices, metas, titles, slices):
        xy = embedded[start:stop]
        labels = meta['label']
        matched = meta['matched'].astype(bool)
        for flag, marker, filled in ((True, '^', True), (False, 'o', False)):
            for category in np.unique(labels):
                select = (labels == category) & (matched == flag)
                if not select.any():
                    continue
                colour = colours[int(category)]
                axis.scatter(xy[select, 0], xy[select, 1], s=args.point_size,
                             marker=marker, linewidths=0.6,
                             facecolors=colour if filled else 'none',
                             edgecolors=colour, alpha=0.85, zorder=3)
        if anchor is not None:
            star = embedded[anchor[0]:anchor[1]]
            axis.scatter(star[:, 0], star[:, 1], s=args.point_size * 14,
                         marker='*', facecolors='none', edgecolors='black',
                         linewidths=0.9, zorder=4)
        n_matched = int(meta['matched'].astype(bool).sum())
        axis.set_title(f'{title}\n{n_matched} matched, '
                       f'{len(meta["matched"]) - n_matched} unmatched',
                       fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    if show_delta:
        draw_delta_panel(axes[len(matrices)], pairs, titles, args, Line2D)
    if show_flips:
        draw_flip_panel(axes[panels_width - 1], flips, titles, pairs, Line2D)
    for axis in axes[panels_width:]:
        axis.set_visible(False)
    if handles is None:
        handles = [
            Line2D([], [], marker='^', linestyle='none', color='0.25',
                   label='matched query'),
            Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
                   color='0.25', label='high-confidence unmatched'),
        ]
        if anchor is not None:
            handles.append(Line2D([], [], marker='*', linestyle='none',
                                  markerfacecolor='none', color='black',
                                  markeredgecolor='black',
                                  label='class prototype'))
        named = list(colours.items())[:args.legend_classes]
        handles += [Line2D([], [], marker='o', linestyle='none',
                           color=colours[c],
                           label=names[c] if c < len(names) else f'class {c}')
                    for c, _ in named]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 7), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    written = []
    for extension in args.formats:
        path = f'{args.out}.{extension}'
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        written.append(path)
    plt.close(fig)

    # the caption claims the matched queries line up better with their class
    # evidence; report the numbers instead of only showing the picture
    report = {'tsne_seed': args.tsne_seed, 'perplexity': perplexity,
              'tsne_backend': backend, 'n_points': int(total),
              'evidence': args.evidence,
              'shared_queries': bool(all('query' in m for m in metas)),
              'prototypes': prototype_fingerprint(args.prototypes),
              'panels': {}, 'paired_shifts': [], 'match_flips': []}
    for order, (title, meta) in enumerate(zip(titles, metas)):
        metric = panel_metric(metas, order, args.evidence)
        values = class_evidence(meta, prototypes, metric)
        n_matched = int(meta['matched'].astype(bool).sum())
        report['panels'][title] = dict(
            metric=metric,
            n_matched=n_matched,
            n_unmatched=int(len(meta['matched']) - n_matched),
            mean_evidence=(float(values.mean()) if values.size else None),
            evidence_ci95=bootstrap_ci(values, seed=args.tsne_seed))
    # the claim is about the *same* queries, so test it that way: with
    # `--shared-queries` every pair of panels is paired row by row
    for pair in pairs:
        delta = pair['after'] - pair['before']
        report['paired_shifts'].append(dict(
            panels=[titles[pair['left']], titles[pair['right']]],
            metric=pair.get('metric', args.evidence),
            n=int(delta.size), mean_delta=float(delta.mean()),
            ci95=bootstrap_ci(delta, seed=args.tsne_seed),
            fraction_improved=float((delta > 0.0).mean())))
    for entry, pair in zip(flips, pairs):
        report['match_flips'].append(dict(
            panels=[titles[pair['left']], titles[pair['right']]],
            newly_matched=entry['newly_matched'],
            lost_matched=entry['lost_matched'], net=entry['net'],
            n=entry['n'], matched_before=entry['matched_before'],
            matched_after=entry['matched_after'],
            discordant=entry['discordant'], mcnemar_p=entry['mcnemar_p']))
    path = f'{args.out}.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    written.append(path)
    for title, entry in report['panels'].items():
        print(f'[evidence] {title} ({entry["metric"]}): '
              f'matched={entry["n_matched"]} '
              f'unmatched={entry["n_unmatched"]} '
              f'mean={entry["mean_evidence"]} '
              f'ci95={entry["evidence_ci95"]}')
    if report['prototypes']:
        print(f'[anchors] {report["prototypes"]["path"]} '
              f'md5={report["prototypes"]["md5"]}')
    for entry in report['paired_shifts']:
        print(f'[paired] {entry["panels"][1]} - {entry["panels"][0]}: '
              f'n={entry["n"]} queries, mean {entry["metric"]} shift '
              f'{entry["mean_delta"]:+.4f} ci95={entry["ci95"]}, improved on '
              f'{100.0 * entry["fraction_improved"]:.0f}% of them')
    for entry in report['match_flips']:
        p_value = entry['mcnemar_p']
        print(f'[flips] {entry["panels"][1]} - {entry["panels"][0]}: '
              f'{entry["matched_before"]} -> {entry["matched_after"]} matched '
              f'of {entry["n"]} same queries; +{entry["newly_matched"]} gained, '
              f'-{entry["lost_matched"]} lost, net {entry["net"]:+d}'
              + (f', McNemar exact p={p_value:.3g}' if p_value is not None
                 else ''))
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    for name in ('features', 'out'):
        if not getattr(args, name):
            raise SystemExit(f'missing required argument: --{name}')
    _guard(args.out)
    matrices, metas, titles = build_panels(args)
    prototypes = load_prototypes(args.prototypes) if args.prototypes else None
    if prototypes is not None:
        classes = metas[0]['classes']
        if len(prototypes) != len(classes):
            print(f'[warn] prototypes={len(prototypes)} classes={len(classes)}'
                  '; drawing the panels without anchors')
            prototypes = None
    if args.anchors and prototypes is None:
        print('[warn] --anchors needs --prototypes with a `soft` tensor; '
              'drawing the panels without stars')
    if args.evidence == 'cosine' and prototypes is None:
        print('[warn] --evidence cosine needs --prototypes; falling back to '
              'the score margin')
        args.evidence = 'margin'
    written = draw(matrices, metas, titles, prototypes, args)
    for path in written:
        print(f'[write] {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

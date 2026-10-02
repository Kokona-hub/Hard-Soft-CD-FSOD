#!/usr/bin/env python
# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Check a few-shot split for image leakage between train and test.

Motivated by DeepFish, whose mAP jumps far above the reported baselines: if the
few-shot support images are frames of the same videos as the test images - or
literal copies of them under another name - the gain is an artefact, not a
result, and a reviewer will find it.  The script is read-only and needs no GPU.

It answers, in order of decisiveness:

1. **Byte-identical files** (md5 of the file content): the same picture stored
   twice, possibly under a different name or in another directory.
2. **Filename / image-id overlap**: the obvious bookkeeping leak.
3. **Shared frame stems**: ``videoA_frame0012.jpg`` and
   ``videoA_frame0013.jpg`` are neighbouring frames; a random frame split of one
   video leaks almost the whole video.
4. **Near-duplicate content** (8x8 average hash, Hamming distance): catches
   re-encoded or resized copies and, together with (3), quantifies how much of
   the test set is visually adjacent to the support set.

Usage
-----
    python tools/s2h/analysis/check_split_leakage.py \
        --root data/FISH --train annotations/5_shot.json \
        --test annotations/test.json --train-img train --test-img test

Nothing is written anywhere; the exit status is 1 when any overlap is found, so
it can be used as a gate in the ablation launcher.
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import re
import sys

try:                                                    # optional, only for (4)
    from PIL import Image
except ImportError:                                     # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# helpers (pure functions, exercised by --self-check)
# ---------------------------------------------------------------------------
def frame_stem(name: str) -> str:
    """Filename without directory, extension and trailing frame number.

    ``videoA_frame0012.jpg`` -> ``videoa_frame`` so that neighbouring frames of
    one video share a stem.  Files without a trailing number keep their stem.
    """
    stem = osp.splitext(osp.basename(name))[0].lower()
    return re.sub(r'[\s_\-]*\d+$', '', stem)


def average_hash(path: str, size: int = 8):
    """8x8 average hash of an image; ``None`` when PIL is unavailable."""
    if Image is None:
        return None
    try:
        with Image.open(path) as handle:
            grey = handle.convert('L').resize((size, size))
            pixels = list(grey.tobytes())        # 'L' = one byte per pixel
    except Exception:
        return None
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_split(ann_file: str, image_dir: str) -> list:
    """Load a COCO annotation file into a list of image records."""
    with open(ann_file, encoding='utf-8') as handle:
        payload = json.load(handle)
    records = []
    for image in payload.get('images', []):
        name = image.get('file_name') or image.get('flickr_url') or ''
        path = osp.join(image_dir, name) if image_dir else name
        records.append(dict(id=image.get('id'), name=name, path=path,
                            stem=frame_stem(name)))
    return records


def compare_splits(train: list, test: list, md5_limit: int = 400,
                   hash_limit: int = 400, max_distance: int = 4) -> dict:
    """All overlap statistics between two splits (no printing)."""
    train_names = {r['name'].lower() for r in train}
    test_names = {r['name'].lower() for r in test}
    train_ids = {r['id'] for r in train}
    test_ids = {r['id'] for r in test}
    train_stems = {r['stem'] for r in train}
    test_stems = {r['stem'] for r in test}

    def digests(records, limit):
        out = {}
        for record in records[:limit]:
            if osp.isfile(record['path']):
                out.setdefault(file_md5(record['path']), []).append(record)
        return out

    def hashes(records, limit):
        out = {}
        for record in records[:limit]:
            if osp.isfile(record['path']):
                value = average_hash(record['path'])
                if value is not None:
                    out.setdefault(value, []).append(record)
        return out

    train_md5, test_md5 = digests(train, md5_limit), digests(test, md5_limit)
    duplicate_bytes = []
    for digest in set(train_md5) & set(test_md5):
        duplicate_bytes.append((train_md5[digest][0]['name'],
                                test_md5[digest][0]['name']))

    train_hash, test_hash = hashes(train, hash_limit), hashes(test, hash_limit)
    near_duplicates = []
    train_items = [(value, recs[0]) for value, recs in train_hash.items()]
    test_items = [(value, recs[0]) for value, recs in test_hash.items()]
    for value_a, record_a in train_items:
        for value_b, record_b in test_items:
            if hamming(value_a, value_b) <= max_distance:
                near_duplicates.append((record_a['name'], record_b['name'],
                                        hamming(value_a, value_b)))

    return dict(
        n_train=len(train), n_test=len(test),
        name_overlap=sorted(train_names & test_names),
        id_overlap=sorted(x for x in train_ids & test_ids if x is not None),
        stem_overlap=sorted(train_stems & test_stems),
        duplicate_bytes=duplicate_bytes,
        near_duplicates=near_duplicates[:20],
        n_near_duplicates=len(near_duplicates),
        images_checked=(min(len(train), md5_limit), min(len(test), md5_limit)))


def report(stats: dict, max_stem_examples: int = 12) -> bool:
    """Print the findings; return ``True`` when the split looks clean."""
    clean = True
    print(f'[split] train={stats["n_train"]} test={stats["n_test"]} images')
    print(f'[hash ] content compared on {stats["images_checked"][0]} train / '
          f'{stats["images_checked"][1]} test images')

    def section(title, items, limit=max_stem_examples):
        nonlocal clean
        count = len(items)
        mark = 'OK  ' if count == 0 else 'LEAK'
        clean = clean and count == 0
        print(f'[{mark}] {title}: {count}')
        for item in items[:limit]:
            print(f'        {item}')
        if count > limit:
            print(f'        ... (+{count - limit} more)')

    section('byte-identical files across the splits', stats['duplicate_bytes'])
    section('identical file names', stats['name_overlap'])
    section('identical image ids', stats['id_overlap'])
    section('shared frame stems (same video / sequence)',
            [f'{s}.*' for s in stats['stem_overlap']])
    section('near-duplicate content (aHash distance <= 4)',
            [f'{a}  ~  {b}  (d={d})' for a, b, d in stats['near_duplicates']],
            limit=8)
    if stats['n_near_duplicates']:
        print('       the support set and the test set contain visually '
              'adjacent frames; report the split as-is only if the protocol '
              'defines it that way (e.g. an official benchmark split), and '
              'state it explicitly')
    return clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser('few-shot split leakage check')
    parser.add_argument('--root', default='data/FISH',
                        help='dataset root holding the annotations')
    parser.add_argument('--train', default='annotations/5_shot.json')
    parser.add_argument('--test', default='annotations/test.json')
    parser.add_argument('--train-img', default='train',
                        help='image directory of the training split')
    parser.add_argument('--test-img', default='test')
    parser.add_argument('--md5-limit', type=int, default=400)
    parser.add_argument('--hash-limit', type=int, default=400)
    parser.add_argument('--max-distance', type=int, default=4)
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args(argv)


def self_check() -> int:
    """Exercise the stem / hash helpers on synthetic records."""
    assert frame_stem('videoA_frame0012.jpg') == 'videoa_frame'
    assert frame_stem('IMG_0042.JPG') == 'img'
    assert frame_stem('plain-name.png') == 'plain-name'
    assert hamming(0b1010, 0b1010) == 0 and hamming(0b1010, 0b0101) == 4
    train = [dict(id=1, name='videoA_frame0010.jpg', path='a.jpg',
                  stem=frame_stem('videoA_frame0010.jpg'))]
    test = [dict(id=1, name='videoA_frame0011.jpg', path='b.jpg',
                 stem=frame_stem('videoA_frame0011.jpg')),
            dict(id=7, name='videoB_frame0003.jpg', path='c.jpg',
                 stem=frame_stem('videoB_frame0003.jpg'))]
    stats = compare_splits(train, test, md5_limit=0, hash_limit=0)
    assert stats['id_overlap'] == [1], stats['id_overlap']
    assert stats['stem_overlap'] == ['videoa_frame'], stats['stem_overlap']
    assert not stats['duplicate_bytes'] and not stats['near_duplicates']
    print('[self-check] ok')
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    train = read_split(osp.join(args.root, args.train),
                       osp.join(args.root, args.train_img))
    test = read_split(osp.join(args.root, args.test),
                      osp.join(args.root, args.test_img))
    stats = compare_splits(train, test, md5_limit=args.md5_limit,
                           hash_limit=args.hash_limit,
                           max_distance=args.max_distance)
    clean = report(stats)
    print('[verdict] ' + ('no overlap found' if clean else
                          'overlap found - investigate before trusting the '
                          'numbers'))
    return 0 if clean else 1


if __name__ == '__main__':
    sys.exit(main())

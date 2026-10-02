#!/usr/bin/env python
"""Prepare reproducible COCO layouts for the S2H CD-FSOD experiments.

The original Domain-RAG configs expect every target domain under
``data/<dataset>`` with ``train/``, ``test/`` and COCO annotations.  Several
of the supplied source datasets instead provide a full train/val COCO split.
This tool creates non-destructive image symlinks and derives nested 1/5/10
*image-shot* support splits with a recorded random seed.

It intentionally does not modify source annotations.  DIOR and UODD category
names are normalized in the copied annotations because MMDetection matches
COCO categories against the exact strings declared in the existing configs.
"""
import argparse
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence


DATASETS = {
    'ArTaxOr': dict(
        train='derived/artaxor_7class/baseline/train.json',
        test='derived/artaxor_7class/baseline/val.json',
        image_root='ArTaxOr',
        category_map={}),
    'FISH': dict(
        train='derived/deepfish/baseline/train.json',
        test='derived/deepfish/baseline/val.json',
        image_root='DeepFish/Segmentation/images',
        category_map={}),
    'UODD': dict(
        train='derived/uodd/baseline/train.json',
        test='derived/uodd/baseline/val.json',
        image_root='Underwater-object-detection-dataset-main/imgs',
        category_map={
            'holothurian': 'seacucumber',
            'echinus': 'seaurchin',
        }),
    'DIOR': dict(
        train='DIOR/coco_annotations/trainval.json',
        test='DIOR/coco_annotations/test.json',
        train_image_root='DIOR/JPEGImages-trainval',
        test_image_root='DIOR/JPEGImages-test',
        category_map={
            'expressway service area': 'Expressway-Service-area',
            'expressway toll station': 'Expressway-toll-station',
            'baseball field': 'baseballfield',
            'basketball court': 'basketballcourt',
            'golf field': 'golffield',
            'ground track field': 'groundtrackfield',
            'storage tank': 'storagetank',
            'tennis court': 'tenniscourt',
            'train station': 'trainstation',
        }),
    'NEU-DET': dict(
        train='raw/NEU-DET/annotations/train.json',
        test='raw/NEU-DET/annotations/val.json',
        train_image_root='raw/NEU-DET/train/images',
        test_image_root='raw/NEU-DET/val/images',
        category_map={}),
}

EXPECTED_CLASSES = {
    'ArTaxOr': {
        'Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera',
        'Lepidoptera', 'Odonata'},
    'FISH': {'fish'},
    'UODD': {'seacucumber', 'seaurchin', 'scallop'},
    'DIOR': {
        'Expressway-Service-area', 'Expressway-toll-station', 'airplane',
        'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass',
        'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation',
        'vehicle', 'windmill'},
    'NEU-DET': {
        'crazing', 'inclusion', 'patches', 'pitted_surface',
        'rolled-in_scale', 'scratches'},
}


def load_coco(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    required = {'images', 'annotations', 'categories'}
    missing = required.difference(data)
    if missing:
        raise ValueError(f'{path} is not a COCO detection annotation: missing {missing}')
    return data


def normalize_categories(coco: Mapping, category_map: Mapping[str, str]) -> dict:
    out = deepcopy(coco)
    for category in out['categories']:
        category['name'] = category_map.get(category['name'], category['name'])
    return out


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f'image directory does not exist: {target}')
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f'{link} already points to {link.resolve()}, expected {target}')
        return
    if link.exists():
        raise RuntimeError(f'{link} already exists and is not a symlink; refusing to replace it')
    link.symlink_to(target, target_is_directory=True)


def validate_images(coco: Mapping, image_root: Path) -> None:
    missing = [x['file_name'] for x in coco['images']
               if not (image_root / x['file_name']).is_file()]
    if missing:
        preview = ', '.join(missing[:3])
        raise FileNotFoundError(
            f'{len(missing)} image files are missing beneath {image_root}; examples: {preview}')


def image_shot_split(coco: Mapping, shots: int, seed: int, dataset: str) -> dict:
    """Select nested per-class support images while retaining full labels.

    A selected support image retains all of its annotations.  This avoids
    treating visible but unlabelled objects as background during augmentation.
    The metadata records effective per-class image counts, which can exceed
    ``shots`` when a selected image contains multiple classes.
    """
    image_to_categories = defaultdict(set)
    category_to_images = defaultdict(set)
    for ann in coco['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_to_categories[image_id].add(category_id)
        category_to_images[category_id].add(image_id)

    chosen = set()
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    for category in categories:
        category_id = category['id']
        candidates = sorted(category_to_images[category_id])
        if len(candidates) < shots:
            raise ValueError(
                f'{dataset}/{category["name"]}: only {len(candidates)} images, need {shots}')

        # Prefer single-class images, then avoid reusing another class's
        # selected image where possible.  Both decisions make the support
        # budget closer to the intended K-image-per-class protocol.
        pure = [x for x in candidates if image_to_categories[x] == {category_id}]
        mixed = [x for x in candidates if image_to_categories[x] != {category_id}]
        rng = random.Random(f'{seed}:{dataset}:{category_id}')
        rng.shuffle(pure)
        rng.shuffle(mixed)
        ordered = pure + mixed
        fresh = [x for x in ordered if x not in chosen]
        selected = (fresh + [x for x in ordered if x in chosen])[:shots]
        chosen.update(selected)

    images = [x for x in coco['images'] if x['id'] in chosen]
    annotations = [x for x in coco['annotations'] if x['image_id'] in chosen]
    effective = {}
    for category in categories:
        category_id = category['id']
        effective[category['name']] = len({
            ann['image_id'] for ann in annotations if ann['category_id'] == category_id
        })

    out = deepcopy(coco)
    out['images'] = images
    out['annotations'] = annotations
    out['info'] = dict(out.get('info', {}))
    out['info'].update({
        'description': f'S2H {dataset} {shots}-shot image support split',
        's2h_split_protocol': 'nested class-balanced image-shot; full labels retained',
        's2h_split_seed': seed,
        's2h_requested_shots_per_class': shots,
        's2h_effective_images_per_class': effective,
    })
    return out


def instance_shot_split(coco: Mapping, shots: int, seed: int,
                        dataset: str) -> dict:
    """Select exactly ``shots`` annotated instances per category.

    This is the strict instance-shot protocol used by the non-DIOR benchmark
    domains.  Only selected annotations are retained.  The metadata records
    how many annotations were removed from selected images so the support
    manifest makes the labeling assumption explicit.
    """
    category_to_annotations = defaultdict(list)
    for ann in coco['annotations']:
        category_to_annotations[int(ann['category_id'])].append(ann)

    selected_ids = set()
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    for category in categories:
        category_id = int(category['id'])
        candidates = sorted(category_to_annotations[category_id],
                            key=lambda ann: int(ann['id']))
        if len(candidates) < shots:
            raise ValueError(
                f'{dataset}/{category["name"]}: only {len(candidates)} '
                f'instances, need {shots}')
        rng = random.Random(f'{seed}:{dataset}:{category_id}:instance')
        order = list(candidates)
        rng.shuffle(order)
        selected_ids.update(int(ann['id']) for ann in order[:shots])

    selected_annotations = [ann for ann in coco['annotations']
                            if int(ann['id']) in selected_ids]
    selected_images = {int(ann['image_id']) for ann in selected_annotations}
    images = [img for img in coco['images'] if int(img['id']) in selected_images]
    effective = {
        category['name']: sum(
            1 for ann in selected_annotations
            if int(ann['category_id']) == int(category['id']))
        for category in categories
    }
    selected_image_ids = {int(img['id']) for img in images}
    original_on_selected = [ann for ann in coco['annotations']
                            if int(ann['image_id']) in selected_image_ids]

    out = deepcopy(coco)
    out['images'] = images
    out['annotations'] = selected_annotations
    out['info'] = dict(out.get('info', {}))
    out['info'].update({
        'description': f'S2H {dataset} {shots}-shot instance support split',
        's2h_split_protocol': 'strict class-balanced instance-shot',
        's2h_split_seed': seed,
        's2h_requested_shots_per_class': shots,
        's2h_effective_instances_per_class': effective,
        's2h_dropped_annotations_on_selected_images': (
            len(original_on_selected) - len(selected_annotations)),
    })
    return out


def write_json(path: Path, data: Mapping) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def prepare_one(data_root: Path, output_root: Path, dataset: str,
                shots: Sequence[int], seed: int, protocol: str) -> None:
    spec = DATASETS[dataset]
    train_path = data_root / spec['train']
    test_path = data_root / spec['test']
    train = normalize_categories(load_coco(train_path), spec['category_map'])
    test = normalize_categories(load_coco(test_path), spec['category_map'])
    for split_name, coco in [('train', train), ('test', test)]:
        names = {x['name'] for x in coco['categories']}
        if names != EXPECTED_CLASSES[dataset]:
            raise ValueError(
                f'{dataset}/{split_name}: category names do not match the '
                f'corresponding Domain-RAG config; got {sorted(names)}')

    shared_image_root = spec.get('image_root')
    train_image_root = data_root / spec.get('train_image_root', shared_image_root)
    test_image_root = data_root / spec.get('test_image_root', shared_image_root)
    validate_images(train, train_image_root)
    validate_images(test, test_image_root)

    target = output_root / dataset
    ann_dir = target / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    ensure_symlink(target / 'train', train_image_root)
    ensure_symlink(target / 'test', test_image_root)
    write_json(ann_dir / 'test.json', test)
    for shot in shots:
        dataset_protocol = ('image' if dataset == 'DIOR' else 'instance') \
            if protocol == 'paper' else protocol
        if dataset_protocol == 'image':
            split = image_shot_split(train, shot, seed, dataset)
        else:
            split = instance_shot_split(train, shot, seed, dataset)
        actual_protocol = split['info']['s2h_split_protocol']
        if dataset_protocol == 'instance':
            counts = split['info']['s2h_effective_instances_per_class']
            invalid = {name: count for name, count in counts.items()
                       if count != shot}
            if invalid:
                raise RuntimeError(
                    f'{dataset}/{shot}-shot instance split is not exact: '
                    f'{invalid}')
        path = ann_dir / f'{shot}_shot.json'
        write_json(path, split)
        print(f'[OK] {dataset} {shot}-shot ({actual_protocol}): '
              f'{len(split["images"])} images, '
              f'{len(split["annotations"])} annotations -> {path}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='directory containing raw/, derived/, and DIOR/')
    parser.add_argument('--output-root', type=Path, default=Path('data'),
                        help='MMDetection data directory')
    parser.add_argument('--datasets', nargs='+', choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--seed', type=int, default=20250919)
    parser.add_argument('--protocol', choices=('image', 'instance', 'paper'),
                        default='image',
                        help='support split protocol; paper uses image-level '
                             'shots for DIOR and K-instance shots elsewhere; '
                             'image preserves all '
                             'annotations on selected images, instance keeps '
                             'exactly K selected annotations per class')
    return parser.parse_args()


def main():
    args = parse_args()
    if any(x <= 0 for x in args.shots):
        raise ValueError('--shots must be positive integers')
    for dataset in args.datasets:
        prepare_one(args.source_root, args.output_root, dataset,
                    sorted(set(args.shots)), args.seed, args.protocol)


if __name__ == '__main__':
    main()

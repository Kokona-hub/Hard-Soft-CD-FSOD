# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Foreground-frozen support-context interventions used by FFCP.

Implements Eq. (3) of "Trust What You Transfer":

    x~_(c,i)^(r) = M_(c,i) * x_(c,i) + (1 - M_(c,i)) * B_i^(r)

where ``M`` is a dilated protection mask derived from the ground-truth box and
``B`` is a background obtained through one of three recipes:

- ``same_domain``      : background of another support image of the same class
- ``class_mismatch``   : background of a support image of another class
- ``neutralized``      : constant / noise background (context neutralization)

A ``reconstruction`` control re-inserts the original context through the same
compositing / resampling pipeline, so that drift caused by resampling can be
subtracted in Eq. (4).

All pixel operations here are dependency-light (PIL + numpy) so that the
interventions can be rendered online from immutable support images, exactly as
described in the paper.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

__all__ = [
    'protection_mask', 'compose_with_background', 'reconstruction_control',
    'generate_interventions', 'neutral_background'
]


def protection_mask(size: Sequence[int],
                    boxes: Sequence[Sequence[float]],
                    dilate: float = 1.2) -> np.ndarray:
    """Dilated protection mask ``M_(c,i)`` in Eq. (3).

    Args:
        size (Sequence[int]): ``(width, height)`` of the image.
        boxes (Sequence[Sequence[float]]): Ground-truth boxes in ``xyxy``.
        dilate (float): Dilation ratio covering the object and its boundary.

    Returns:
        np.ndarray: Boolean mask of shape ``[H, W]`` (``True`` = protected).
    """
    width, height = int(size[0]), int(size[1])
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw = (x2 - x1) * dilate
        bh = (y2 - y1) * dilate
        xi1 = int(max(0, np.floor(cx - bw / 2)))
        yi1 = int(max(0, np.floor(cy - bh / 2)))
        xi2 = int(min(width, np.ceil(cx + bw / 2)))
        yi2 = int(min(height, np.ceil(cy + bh / 2)))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = True
    return mask


def neutral_background(size: Sequence[int],
                       mode: str = 'gray',
                       rng: Optional[np.random.RandomState] = None
                       ) -> Image.Image:
    """Build a context-neutralizing background ``B``."""
    width, height = int(size[0]), int(size[1])
    if mode == 'noise':
        rng = rng or np.random.RandomState(0)
        arr = rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    else:
        arr = np.full((height, width, 3), 127, dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


def compose_with_background(img: Image.Image,
                            mask: np.ndarray,
                            background: Image.Image) -> Image.Image:
    """``M * x + (1 - M) * B``."""
    if background.size != img.size:
        background = background.resize(img.size, Image.BILINEAR)
    x = np.asarray(img.convert('RGB'), dtype=np.float32)
    b = np.asarray(background.convert('RGB'), dtype=np.float32)
    m = mask[:, :, None].astype(np.float32)
    out = m * x + (1.0 - m) * b
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode='RGB')


def reconstruction_control(img: Image.Image,
                           resample: int = 512) -> Image.Image:
    """Reconstruction control of Eq. (4).

    Re-inserts the original context but passes the image through the same
    down-/up-sampling round trip used by the compositing pipeline, so that
    resampling-induced drift can be subtracted from the measured drift.
    """
    w, h = img.size
    small = img.resize((resample, resample), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def generate_interventions(img: Image.Image,
                           boxes: Sequence[Sequence[float]],
                           same_domain_src: Optional[Image.Image] = None,
                           class_mismatch_src: Optional[Image.Image] = None,
                           dilate: float = 1.2,
                           neutral_mode: str = 'gray',
                           rng: Optional[np.random.RandomState] = None
                           ) -> Dict[str, Image.Image]:
    """Render all support-context views for a single support image.

    Args:
        img (Image.Image): The original support image ``x``.
        boxes (Sequence[Sequence[float]]): Its ground-truth boxes.
        same_domain_src (Image.Image, optional): A support image of the *same*
            category, used as the in-domain background source.
        class_mismatch_src (Image.Image, optional): A support image of a
            *different* category, used as the mismatched background source.
        dilate (float): Protection-mask dilation ratio.
        neutral_mode (str): ``'gray'`` or ``'noise'``.
        rng (np.random.RandomState, optional): Random state for reproducibility.

    Returns:
        dict[str, Image.Image]: Keys ``same_domain``, ``class_mismatch``,
        ``neutralized`` and ``reconstruction``. Missing sources fall back to
        the neutralized background so the builder never crashes on 1-shot
        support sets.
    """
    mask = protection_mask(img.size, boxes, dilate=dilate)
    neutral = neutral_background(img.size, neutral_mode, rng)

    same_bg = same_domain_src if same_domain_src is not None else neutral
    diff_bg = class_mismatch_src if class_mismatch_src is not None else neutral

    return {
        'same_domain': compose_with_background(img, mask, same_bg),
        'class_mismatch': compose_with_background(img, mask, diff_bg),
        'neutralized': compose_with_background(img, mask, neutral),
        'reconstruction': reconstruction_control(img),
    }

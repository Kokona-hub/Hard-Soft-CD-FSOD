# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Grounding DINO detector with the knowledge-enhanced Hard-Soft head.

This detector is a *drop-in* replacement of ``GroundingDINO``: the backbone,
neck, transformer encoder/decoder and the localization path are identical to
the configuration reproduced by Domain-RAG. The only difference is that the
bounding-box head is :class:`S2HGroundingDINOHead`, which adds the bounded
Hard-Soft class-logit correction of the S2H-CD-FSOD method.

The detector additionally exposes ``extract_roi_features`` which is used by
``tools/s2h/build_s2h_knowledge.py`` to pool support foreground features from
the visual encoder -- the same features that FFCP probes and CHSD decomposes.
"""
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from ..utils import pool_features_in_boxes
from .grounding_dino import GroundingDINO


@MODELS.register_module()
class S2HGroundingDINO(GroundingDINO):
    """Grounding DINO + FFCP / CHSD / ATAR."""

    # strides of the four feature levels produced by the ChannelMapper neck
    LEVEL_STRIDES = (8, 16, 32, 64)

    def set_s2h_knowledge(self, knowledge_path: str) -> None:
        """Attach a cached knowledge file to the bounding-box head."""
        self.bbox_head.set_s2h_knowledge(knowledge_path)

    def s2h_active(self) -> bool:
        return bool(self.bbox_head.s2h_active())

    # ------------------------------------------------------------------
    # support-side feature extraction (used by the offline builder)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_roi_features(self,
                             imgs: Tensor,
                             boxes: Sequence[Tensor],
                             img_shape: Tuple[int, int],
                             dilate: float = 1.0) -> List[Tensor]:
        """Pool visual-encoder features inside the (dilated) boxes.

        Args:
            imgs (Tensor): Batch of preprocessed images, shape ``[B, 3, H, W]``.
            boxes (Sequence[Tensor]): ``B`` tensors of boxes in ``xyxy`` pixel
                coordinates of the *original* image.
            img_shape (Tuple[int, int]): ``(H, W)`` of the original image.
            dilate (float): Box dilation ratio used by the FFCP protection mask
                (``1.0`` means no dilation).

        Returns:
            list[Tensor]: One tensor per image with shape ``[num_boxes, C]``.
        """
        feats = self.extract_feat(imgs)
        return pool_features_in_boxes(feats, boxes, img_shape, dilate=dilate)


__all__ = ['S2HGroundingDINO']

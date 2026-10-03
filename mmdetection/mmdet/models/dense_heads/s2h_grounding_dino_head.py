# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Grounding DINO head with the Asymmetric Task-Authority Routing (ATAR).

The head keeps the original Grounding DINO classification / localization
branches completely unchanged. On top of the ordinary decoder logits it

1. converts the per-token phrase logits into per-category logits ``L0``,
2. computes the bounded Hard-Soft cosine correction ``delta`` (Eq. (12)) with
   the cached support knowledge, and
3. broadcasts ``delta`` back onto the text tokens of the corresponding
   category so that the standard phrase-level loss / post-processing works
   untouched.

The localization branch is never modified, matching the paper's claim that
the reliability mechanism is isolated from the localization adapter.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from ..layers import inverse_sigmoid
from ..utils import S2HKnowledgeBank
from .grounding_dino_head import GroundingDINOHead


@MODELS.register_module()
class S2HGroundingDINOHead(GroundingDINOHead):
    """Grounding DINO head with knowledge-enhanced Hard-Soft routing.

    Args:
        s2h_cfg (dict, optional): Configuration of the knowledge-enhanced
            routing. Supported keys:

            - ``enabled`` (bool): master switch (default ``True``)
            - ``res_weight`` (float): weight of the residual penalty ``lambda_res``
            - ``con_weight`` (float): weight of the consistency term ``lambda_con``
            - ``stage`` (str): ``ffcp``, ``ffcp_chsd`` or ``full``. The first
              two disable the corresponding later components for an honest
              cumulative ablation.
            - plus every argument accepted by
              :class:`~mmdet.models.utils.S2HKnowledgeBank`
              (``classes``, ``knowledge_path``, ``gamma``, ``alpha_max``,
              ``g_max``, ``theta``, ``T_u``, ``embed_dims``).
    """

    def __init__(self,
                 *args,
                 s2h_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        s2h_cfg = dict(s2h_cfg or {})
        self.s2h_enabled = s2h_cfg.pop('enabled', True)
        self.s2h_res_weight = s2h_cfg.pop('res_weight', 0.0)
        self.s2h_con_weight = s2h_cfg.pop('con_weight', 0.0)
        self.s2h_train_injection = s2h_cfg.get('train_injection', True)
        self.s2h_stage = s2h_cfg.get('stage', 'full')
        s2h_cfg.setdefault('embed_dims', self.embed_dims)
        # an empty / absent knowledge path simply leaves the bank inactive so
        # that the head reduces to vanilla Grounding DINO (baseline runs).
        if not self.s2h_enabled or not s2h_cfg.get('knowledge_path'):
            s2h_cfg['knowledge_path'] = None
        self.s2h_bank = S2HKnowledgeBank(**s2h_cfg)
        if not self.s2h_train_injection:
            # In inference-only mode the bank is a fixed support-side
            # calibrator.  Freezing its routing parameters avoids adding
            # unused trainable parameters to DDP/optimizer state.
            for parameter in self.s2h_bank.parameters():
                parameter.requires_grad_(False)
        self._s2h_aux: Optional[Dict[str, Tensor]] = None

    # ------------------------------------------------------------------
    # knowledge injection
    # ------------------------------------------------------------------
    def set_s2h_knowledge(self, knowledge_path: str) -> None:
        """Load a cached knowledge file into the bank."""
        self.s2h_bank.load_knowledge(knowledge_path)
        if not self.s2h_train_injection:
            for parameter in self.s2h_bank.parameters():
                parameter.requires_grad_(False)

    def s2h_active(self) -> bool:
        return bool(self.s2h_enabled and self.s2h_bank.is_ready())

    # ------------------------------------------------------------------
    # forward with ATAR injection
    # ------------------------------------------------------------------
    def forward(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                return_aux: bool = False,
                num_denoising_queries: int = 0):
        """Forward function, mirroring ``GroundingDINOHead.forward``.

        The only difference is that the *last* decoder layer logits are passed
        through the ATAR correction for the matching (non-denoising) queries.
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        aux = None
        num_layers = hidden_states.shape[0]
        bank = self.s2h_bank
        # Keep the few-shot optimization distribution identical to the
        # Domain-RAG baseline when requested.  ``train_injection=False``
        # means correction is disabled in ``model.train()`` but remains
        # enabled for validation/test (``model.eval()``).
        active = self.s2h_active() and (
            (not self.training) or self.s2h_train_injection)

        for layer_id in range(num_layers):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                       memory_text,
                                                       text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                tmp_reg_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()

            if active and layer_id == num_layers - 1:
                outputs_class, aux = self._apply_atar(hidden_state,
                                                      outputs_class, bank,
                                                      num_denoising_queries)

            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        self._s2h_aux = aux
        if return_aux:
            return all_layers_outputs_classes, all_layers_outputs_coords, aux
        return all_layers_outputs_classes, all_layers_outputs_coords

    def _apply_atar(self, hidden_state: Tensor, token_logits: Tensor,
                    bank: S2HKnowledgeBank,
                    num_denoising_queries: int = 0
                    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Apply Eq. (12) to the matching queries and broadcast to tokens."""
        # DINO places denoising queries first and matching queries last. The
        # head does not own ``num_queries``; use the value supplied by the
        # detector's ``dn_meta`` instead of silently applying ATAR to DN data.
        start = min(max(int(num_denoising_queries), 0), token_logits.shape[1])

        # split away the denoising queries (they are placed in front)
        match_hidden = hidden_state[:, start:, :]
        match_tokens = token_logits[:, start:, :]

        class_logits = bank.class_logits_from_tokens(match_tokens)
        aux = bank.atar(match_hidden, class_logits)
        delta = aux['delta']

        corrected = bank.broadcast_delta(match_tokens, delta)
        if start > 0:
            corrected = torch.cat([token_logits[:, :start, :], corrected],
                                  dim=1)
        return corrected, aux

    # ------------------------------------------------------------------
    # loss with auxiliary residual / consistency terms
    # ------------------------------------------------------------------
    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Loss function mirroring ``GroundingDINOHead.loss`` plus Eq. (13)."""
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        dn_queries = int((dn_meta or {}).get('num_denoising_queries', 0))
        outs = self(hidden_states, references, memory_text, text_token_mask,
                    num_denoising_queries=dn_queries)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        aux = self._s2h_aux
        # The residual/consistency regularizers belong to the complete ATAR
        # stage.  Leaving them on in the FFCP/CHSD ablations would introduce
        # an otherwise hidden ATAR training signal.
        if (self.s2h_active() and aux is not None and self.training and
                self.s2h_bank.stage == 'full'):
            delta = aux['delta']
            p0 = aux['class_logits']
            u = aux['u']

            # residual penalty: discourage unnecessary score changes
            if self.s2h_res_weight > 0:
                losses['loss_s2h_res'] = delta.abs().mean() * \
                    self.s2h_res_weight

            # consistency: preserve the baseline ranking on uncertain queries
            if self.s2h_con_weight > 0:
                prob0 = F.softmax(p0, dim=-1)
                prob1 = F.softmax(p0 + delta, dim=-1)
                kl = F.kl_div(prob1.clamp_min(1e-8).log(), prob0,
                              reduction='none').sum(-1)          # [B, N]
                kl = kl.reshape(-1)
                w = u.reshape(-1)
                w = w / (w.sum() + self.s2h_bank.eps)
                losses['loss_s2h_con'] = (w * kl).sum() * \
                    self.s2h_con_weight
        return losses

    # ------------------------------------------------------------------
    # inference: keep the ordinary Grounding DINO post-processing
    # ------------------------------------------------------------------
    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Identical to the parent ``predict`` (ATAR is applied inside
        :meth:`forward`)."""
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]
        outs = self(hidden_states, references, memory_text, text_token_mask)
        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions


__all__ = ['S2HGroundingDINOHead']

# Copyright (c) S2H-CD-FSOD. All rights reserved.
"""Knowledge-Enhanced Hard-Soft reasoning utilities for CD-FSOD.

This module implements the support-side machinery of the S2H-CD-FSOD method
("Trust What You Transfer") on top of the Grounding DINO detector used by
Domain-RAG:

1. ``FFCP`` (Foreground-Frozen Context Probing): estimate the class-wise
   context-sensitive subspace from label-preserving support interventions.
2. ``CHSD`` (Context-Constrained Hard-Soft Decomposition): build a bounded
   Hard identity anchor and a context-cleaned Soft appearance prototype.
3. ``ATAR`` (Asymmetric Task-Authority Routing): query-side, bounded cosine
   correction of the class logits (implemented in ``S2HKnowledgeBank.atar``).

The heavy support-side computation is done offline by
``tools/s2h/build_s2h_knowledge.py`` and cached into a ``.pth`` file. The
module below only stores the resulting knowledge and applies ATAR at
training / inference time.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

# ---------------------------------------------------------------------------
# Generic numerical operators
# ---------------------------------------------------------------------------


def robust_mean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Robust aggregation of support residuals.

    The paper denotes this operator as ``RobustMean``. We use the coordinate
    wise median, which is a standard robust estimator and is not affected by a
    single outlier support view.

    Args:
        x (Tensor): Tensor to aggregate.
        dim (int): Dimension to reduce. Defaults to 0.

    Returns:
        Tensor: Robust mean reduced along ``dim``.
    """
    if x.numel() == 0:
        return x
    return x.median(dim=dim).values


def l2_normalize(x: torch.Tensor, dim: int = -1,
                 eps: float = 1e-6) -> torch.Tensor:
    """Numerically safe L2 normalization."""
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def orthonormalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return an orthonormal basis spanning the rows of ``x``.

    Args:
        x (Tensor): Matrix of shape ``[n, d]`` holding candidate directions.
        eps (float): Tolerance used to drop (numerically) zero directions.

    Returns:
        Tensor: Orthonormal basis of shape ``[r, d]`` with ``r <= n``.
    """
    if x is None or x.numel() == 0:
        return x.new_zeros((0, x.shape[-1] if x is not None else 0))
    # QR on the transposed matrix gives an orthonormal basis of the column
    # space, i.e. of the span of the rows of ``x``.
    q, r = torch.linalg.qr(x.t())
    diag = r.diagonal().abs()
    keep = diag > eps
    if keep.sum() == 0:
        return q.new_zeros((0, x.shape[-1]))
    return q[:, keep].t().contiguous()


def svd_context_basis(drift: torch.Tensor,
                      rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Leading right singular vectors of the stacked drift matrix.

    Implements ``U_ctx = TopSVD(D_c, r)`` of Eq. (4)/(5).

    Args:
        drift (Tensor): Drift matrix of shape ``[n_views, d]``.
        rank (int): Number of leading right singular vectors to keep.

    Returns:
        Tuple[Tensor, Tensor]: ``(U_ctx, singular_values)`` where ``U_ctx`` has
        shape ``[min(rank, n_views), d]``.
    """
    if drift is None or drift.numel() == 0:
        return drift.new_zeros((0, 0)), drift.new_zeros((0, ))
    # ``torch.svd_lowrank`` is efficient and stable for the small support sets
    # used in few-shot detection.
    k = min(rank, min(drift.shape))
    u, s, v = torch.svd_lowrank(drift, q=max(k, 1))
    return v[:, :k].t().contiguous(), s[:k]


def context_leakage_energy(drift: torch.Tensor,
                           u_ctx: torch.Tensor,
                           reference: Optional[torch.Tensor] = None,
                           eps: float = 1e-6) -> torch.Tensor:
    """Normalized context leakage energy ``e_ctx`` of Eq. (5).

    ``e_ctx = mean_n || U_ctx^T d_n ||^2 / (||f_n||^2 + eps)``.

    The paper normalizes by the original foreground representation ``f_n``;
    callers that do not have it may omit ``reference`` and fall back to the
    drift norm for backwards compatibility.

    Args:
        drift (Tensor): Drift matrix of shape ``[n_views, d]``.
        u_ctx (Tensor): Context basis of shape ``[r, d]``.
        reference (Tensor, optional): Original foreground features aligned with
            ``drift``, shape ``[n, d]``.
        eps (float): Numerical stabilizer.

    Returns:
        Tensor: Scalar energy in ``[0, 1]``.
    """
    if drift is None or drift.numel() == 0 or u_ctx is None or u_ctx.numel() == 0:
        return drift.new_zeros(())
    proj = drift @ u_ctx.t()                      # [n, r]
    num = proj.pow(2).sum(-1)                     # [n]
    base = drift if reference is None else reference
    if base.shape != drift.shape:
        raise ValueError('reference and drift must have the same shape')
    den = base.pow(2).sum(-1) + eps               # [n]
    return (num / den).mean()


def project_out(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``x`` that lies in ``span(basis)``.

    Computes ``(I - U U^T) x`` without materializing the ``D x D`` matrix.

    Args:
        x (Tensor): Tensor of shape ``[..., D]``.
        basis (Tensor): Orthonormal rows of shape ``[n, D]``. May be empty.

    Returns:
        Tensor: Same shape as ``x`` with the basis component removed.
    """
    if basis is None or basis.numel() == 0:
        return x
    return x - (x @ basis.t()) @ basis


def pool_features_in_boxes(feats: Sequence[torch.Tensor],
                           boxes: Sequence[torch.Tensor],
                           img_shape,
                           dilate: float = 1.0) -> List[torch.Tensor]:
    """Pool multi-level visual-encoder features inside (dilated) boxes.

    The box size selects the pyramid level whose stride best matches the object
    scale; an average pooling then produces one ``C``-dimensional descriptor
    per box. This is the visual descriptor used by FFCP / CHSD.

    Args:
        feats (Sequence[Tensor]): Feature pyramid, each ``[B, C, h, w]``.
        boxes (Sequence[Tensor]): Per-image ``[n, 4]`` boxes in ``xyxy`` pixel
            coordinates of the original image.
        img_shape (tuple): ``(H, W)`` of the original image.
        dilate (float): Box dilation ratio (FFCP protection mask).

    Returns:
        list[Tensor]: Per-image descriptors of shape ``[n, C]``.
    """
    orig_h, orig_w = img_shape
    device = feats[0].device
    num_levels = len(feats)
    # assume a standard pyramid stride of 8 * 2**level
    strides = [8 * (2 ** i) for i in range(num_levels)]
    out = []
    for b in range(len(boxes)):
        per_img = []
        for box in boxes[b]:
            box = box.to(device).float()
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)
            w = float(box[2] - box[0]) * dilate
            h = float(box[3] - box[1]) * dilate
            side = max((w * h) ** 0.5, 1.0)
            level = int(torch.tensor(
                [abs(torch.log(torch.tensor(side / s))) for s in strides]
            ).argmin())
            feat = feats[level]
            _, c, fh, fw = feat.shape
            x1 = (cx - w / 2) / orig_w * fw
            y1 = (cy - h / 2) / orig_h * fh
            x2 = (cx + w / 2) / orig_w * fw
            y2 = (cy + h / 2) / orig_h * fh
            xi1 = max(int(torch.floor(torch.tensor(x1))), 0)
            yi1 = max(int(torch.floor(torch.tensor(y1))), 0)
            xi2 = min(max(int(torch.ceil(torch.tensor(x2))), xi1 + 1), fw)
            yi2 = min(max(int(torch.ceil(torch.tensor(y2))), yi1 + 1), fh)
            patch = feat[b:b + 1, :, yi1:yi2, xi1:xi2]
            pooled = torch.nn.functional.adaptive_avg_pool2d(patch, 1)
            per_img.append(pooled.flatten())
        if per_img:
            out.append(torch.stack(per_img, dim=0))
        else:
            out.append(feats[0].new_zeros((0, feats[0].shape[1])))
    return out


def project_l2_ball(delta: torch.Tensor,
                    radius: float,
                    eps: float = 1e-6) -> torch.Tensor:
    """Project a vector onto the closed L2 ball of the given radius.

    Implements the ``Π_{||.||_2 <= rho_H}`` projection of Eq. (7).

    Args:
        delta (Tensor): Vector to project.
        radius (float): Ball radius.
        eps (float): Numerical stabilizer.

    Returns:
        Tensor: Projected vector with norm ``<= radius``.
    """
    if delta.numel() == 0:
        return delta
    norm = delta.norm(p=2).clamp_min(eps)
    scale = torch.clamp(radius / norm, max=1.0)
    return delta * scale


# ---------------------------------------------------------------------------
# Structural attribute templates (Hard knowledge candidates)
# ---------------------------------------------------------------------------

# The paper builds Hard identity candidates from "predefined structural
# templates rather than free-form target-domain generation". The templates
# below are the *structural* parts of an object: parts, shape, texture and
# boundary terms -- never viewpoint / scale / background terms.
_GENERIC_TEMPLATES = [
    '{c}', 'the {c}', 'a photo of a {c}', 'the {c} body', 'the {c} shape',
    'the {c} outline', 'the {c} texture', 'the {c} surface',
]

# Curated structural parts for the CD-FSOD benchmark categories.
STRUCTURAL_ATTRIBUTE_TEMPLATES: Dict[str, List[str]] = {
    # ---- ArTaxOr (arthropods) ----
    'Araneae': ['the spider body', 'the spider legs', 'the cephalothorax',
                'the abdomen', 'the pedipalps'],
    'Coleoptera': ['the beetle body', 'the elytra', 'the beetle antennae',
                   'the beetle legs', 'the pronotum'],
    'Diptera': ['the fly body', 'the fly wings', 'the compound eye',
                'the fly antennae', 'the thorax'],
    'Hemiptera': ['the bug body', 'the true bug wings', 'the rostrum',
                  'the bug antennae'],
    'Hymenoptera': ['the wasp body', 'the hymenopteran wings', 'the waist',
                    'the antennae'],
    'Lepidoptera': ['the butterfly wing pattern', 'the butterfly body',
                    'the antennae', 'the wing veins'],
    'Odonata': ['the dragonfly body', 'the elongated wings', 'the compound eyes',
                'the long abdomen'],
    # ---- DIOR (aerial) ----
    'airplane': ['the wings', 'the fuselage', 'the tail', 'the engine nacelle'],
    'airport': ['the runway', 'the terminal building', 'the apron'],
    'baseballfield': ['the diamond', 'the infield', 'the outfield'],
    'basketballcourt': ['the court lines', 'the hoop', 'the key'],
    'bridge': ['the deck', 'the piers', 'the span'],
    'chimney': ['the chimney stack', 'the chimney base'],
    'dam': ['the dam wall', 'the reservoir', 'the spillway'],
    'golffield': ['the fairway', 'the green', 'the bunker'],
    'groundtrackfield': ['the track', 'the infield', 'the lanes'],
    'harbor': ['the quay', 'the docks', 'the berths'],
    'overpass': ['the roadway', 'the pillars', 'the ramp'],
    'ship': ['the hull', 'the bow', 'the superstructure'],
    'stadium': ['the stands', 'the pitch', 'the roof'],
    'storagetank': ['the cylindrical tank', 'the tank roof'],
    'tenniscourt': ['the court lines', 'the net'],
    'trainstation': ['the platforms', 'the station roof', 'the tracks'],
    'vehicle': ['the wheels', 'the chassis', 'the cabin'],
    'windmill': ['the tower', 'the blades', 'the nacelle'],
    # ---- NEU-DET (industrial defects) ----
    'crazing': ['the crazing crack network', 'the fine crack lines',
                'the surface crack texture'],
    'inclusion': ['the inclusion patch', 'the foreign particle',
                  'the inclusion boundary'],
    'patches': ['the patch region', 'the patch boundary',
                'the patch surface'],
    'pitted_surface': ['the pit marks', 'the pitted texture',
                       'the pit boundary'],
    'rolled-in_scale': ['the rolled-in scale streak', 'the scale stripe',
                        'the scale boundary'],
    'scratches': ['the scratch line', 'the scratch direction',
                  'the scratch boundary'],
    # ---- UODD (underwater) ----
    'seacucumber': ['the elongated body', 'the body spines',
                    'the sea cucumber outline'],
    'seaurchin': ['the spherical test', 'the spines', 'the sea urchin shell'],
    'scallop': ['the fan-shaped shell', 'the shell ridges', 'the shell outline'],
    # ---- Common / other ----
    'fish': ['the fish body', 'the fins', 'the tail', 'the head', 'the scales'],
}

# Synonym / alias table used to fetch the base category text embedding and to
# look up curated templates.
CLASS_ALIASES: Dict[str, List[str]] = {
    'aeroplane': ['airplane'],
    'airplane': ['aeroplane'],
    'motorbike': ['motorcycle'],
    'tvmonitor': ['tv', 'television'],
    'diningtable': ['dining table'],
    'pottedplant': ['potted plant'],
}


def get_attribute_candidates(class_name: str) -> List[str]:
    """Return the structural attribute candidates for a category name.

    Falls back to a generic structural template when the category is unknown.
    """
    key = class_name.strip().lower().replace('_', ' ').replace('-', ' ')
    for name, phrases in STRUCTURAL_ATTRIBUTE_TEMPLATES.items():
        normalized = name.strip().lower().replace('_', ' ').replace('-', ' ')
        if normalized == key:
            return list(phrases)
    # alias lookup
    for alias, targets in CLASS_ALIASES.items():
        if alias == key:
            for target in targets:
                if target in STRUCTURAL_ATTRIBUTE_TEMPLATES:
                    return list(STRUCTURAL_ATTRIBUTE_TEMPLATES[target])
    return [t.format(c=class_name) for t in _GENERIC_TEMPLATES]


# ---------------------------------------------------------------------------
# Knowledge bank (module)
# ---------------------------------------------------------------------------


@MODELS.register_module()
class S2HKnowledgeBank(nn.Module):
    """Container for the Hard-Soft support knowledge + ATAR routing.

    The knowledge tuple ``K_c = (h_c, s_c, kappa_c)`` described in Eq. (2) is
    produced offline and injected through :meth:`set_knowledge`. Only the
    bounded routing parameters ``alpha_c`` (class-wise strength) and ``g_cls``
    (learned sigmoid gate) are trainable.  The default generated configs keep
    the correction out of the training distribution and apply it at inference.

    Args:
        embed_dims (int): Dimension of the Grounding DINO alignment space.
            Defaults to 256.
        classes (list[str], optional): Ordered category names of the target
            dataset. Must match the ``metainfo['classes']`` order.
        gamma (float): Global correction scale. Defaults to 1.0.
        alpha_max (float): Upper bound of the class-wise strength. Defaults to 0.5.
        g_max (float): Authority ceiling ``g_max`` of Eq. (11). Defaults to 0.5.
        theta (float): Uncertainty threshold ``theta``. Defaults to 0.5.
        T_u (float): Routing temperature ``T_u``. Defaults to 0.1.
        init_alpha_bias (float): Pre-sigmoid bias for ``alpha``. A value of -8
            makes ``alpha`` (and therefore the correction) essentially zero at
            initialization while still allowing gradients to flow.
        init_alpha_from_reliability (bool): Initialize ``alpha`` monotonically
            from support reliability instead of using one global bias.
        positive_only (bool): Keep only positive class corrections.
        center_delta (bool): Remove the per-query mean correction before the
            optional positive-only projection.
        soft_mix (float): Hard/soft direction mixture for CHSD and ATAR.
        train_injection (bool): Whether to inject corrections during training.
        reliability_power (float): Exponent applied to the reliability gate.
        alpha_rel_floor (float): Reliability-aware alpha floor as a fraction
            of ``alpha_max``.
        alpha_rel_scale (float): Reliability-aware alpha slope as a fraction
            of ``alpha_max``.
        eps (float): Numerical stabilizer.
        knowledge_path (str, optional): Path to a cached knowledge ``.pth``
            produced by ``tools/s2h/build_s2h_knowledge.py``.
        num_classes (int, optional): Number of target categories. Inferred from
            ``classes`` when provided.
        stage (str): Ablation stage, one of ``ffcp``, ``ffcp_chsd`` or
            ``full``. Defaults to ``full``.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 classes: Optional[Sequence[str]] = None,
                 gamma: float = 1.0,
                 alpha_max: float = 0.5,
                 g_max: float = 0.5,
                 theta: float = 0.5,
                 T_u: float = 0.1,
                 init_alpha_bias: float = -2.5,
                 init_alpha_from_reliability: bool = False,
                 positive_only: bool = False,
                 center_delta: bool = False,
                 soft_mix: float = 0.35,
                 train_injection: bool = True,
                 reliability_power: float = 0.5,
                 alpha_rel_floor: float = 0.10,
                 alpha_rel_scale: float = 0.80,
                 eps: float = 1e-6,
                 knowledge_path: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 stage: str = 'full',
                 **kwargs) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.gamma = gamma
        self.alpha_max = alpha_max
        self.g_max = g_max
        self.theta = theta
        self.T_u = T_u
        self.eps = eps
        self.init_alpha_bias = init_alpha_bias
        self.init_alpha_from_reliability = bool(init_alpha_from_reliability)
        self.positive_only = bool(positive_only)
        self.center_delta = bool(center_delta)
        self.soft_mix = float(soft_mix)
        self.train_injection = bool(train_injection)
        # Reliability-aware alpha initialization already encodes support
        # confidence. A square-root gate avoids multiplying that confidence
        # by a second full reliability factor while remaining bounded.
        self.reliability_power = float(reliability_power)
        self.alpha_rel_floor = float(alpha_rel_floor)
        self.alpha_rel_scale = float(alpha_rel_scale)
        stage = str(stage).lower()
        if stage == 'chsd':
            stage = 'ffcp_chsd'
        if stage not in ('ffcp', 'ffcp_chsd', 'full'):
            raise ValueError(
                f"unknown S2H stage {stage!r}; expected ffcp, ffcp_chsd or full")
        self.stage = stage

        if classes is not None:
            classes = list(classes)
            num_classes = len(classes)
        self.classes: Optional[List[str]] = list(classes) if classes else None
        self.num_classes = int(num_classes) if num_classes else 0

        # routing parameters stay plain (non-trainable) buffers until real
        # knowledge is injected. This keeps a disabled / knowledge-free model
        # exactly parameter-identical to vanilla Grounding DINO, which also
        # avoids DDP "unused parameter" errors in baseline runs.
        self._init_buffers(self.num_classes)

        if knowledge_path is not None:
            self.load_knowledge(knowledge_path)

    # -- construction -----------------------------------------------------
    def _init_buffers(self, num_classes: int) -> None:
        num_classes = max(int(num_classes), 1)
        self.register_buffer('hard', torch.zeros(num_classes, self.embed_dims))
        self.register_buffer('soft', torch.zeros(num_classes, self.embed_dims))
        self.register_buffer('kappa', torch.zeros(num_classes))
        self.register_buffer('reliability', torch.zeros(num_classes))
        # flattened class->token index map (variable length per class)
        self.register_buffer(
            '_token_flat', torch.zeros(0, dtype=torch.long))
        self.register_buffer(
            '_token_offset', torch.zeros(num_classes + 1, dtype=torch.long))
        self.register_buffer(
            'raw_alpha', torch.full((num_classes, ), self.init_alpha_bias))
        self.register_buffer('raw_g_cls', torch.zeros(1))
        self.ready = False

    def set_knowledge(self,
                      hard: torch.Tensor,
                      soft: torch.Tensor,
                      kappa: torch.Tensor,
                      reliability: torch.Tensor,
                      classes: Sequence[str],
                      token_ids: Sequence[Sequence[int]],
                      ) -> None:
        """Inject the offline knowledge into the bank.

        Args:
            hard (Tensor): Hard identity anchors, shape ``[C, D]``.
            soft (Tensor): Soft appearance prototypes, shape ``[C, D]``.
            kappa (Tensor): Context dependence ``kappa_ctx``, shape ``[C]``.
            reliability (Tensor): Prototype reliability ``r_c``, shape ``[C]``.
            classes (Sequence[str]): Category names in the dataset order.
            token_ids (Sequence[Sequence[int]]): For each category, the
                Grounding DINO text-token indices that belong to it.
        """
        classes = list(classes)
        hard = torch.as_tensor(hard)
        soft = torch.as_tensor(soft)
        num_classes = hard.shape[0]
        assert len(classes) == num_classes

        # (re)create parameters with the correct size, preserving old values
        old_alpha = None
        if isinstance(self.raw_alpha, nn.Parameter) and \
                self.raw_alpha.numel() == num_classes:
            old_alpha = self.raw_alpha.detach().clone()
        old_g_cls = None
        if isinstance(self.raw_g_cls, nn.Parameter):
            old_g_cls = self.raw_g_cls.detach().clone()
        self.hard = hard.float()
        self.soft = soft.float()
        self.kappa = torch.as_tensor(kappa).float().reshape(-1)
        self.reliability = torch.as_tensor(reliability).float().reshape(-1)
        # buffers
        self.register_buffer('hard', self.hard)
        self.register_buffer('soft', self.soft)
        self.register_buffer('kappa', self.kappa)
        self.register_buffer('reliability', self.reliability)

        flat, offsets = [], [0]
        for idxs in token_ids:
            idxs = [int(i) for i in idxs]
            flat.extend(idxs)
            offsets.append(offsets[-1] + len(idxs))
        self.register_buffer(
            '_token_flat',
            torch.tensor(flat, dtype=torch.long))
        self.register_buffer(
            '_token_offset',
            torch.tensor(offsets, dtype=torch.long))

        if old_alpha is not None:
            self.raw_alpha = nn.Parameter(old_alpha)
        elif self.init_alpha_from_reliability:
            rel = self.reliability.clamp(0.0, 1.0)
            floor = max(self.alpha_rel_floor, 0.0)
            scale = max(self.alpha_rel_scale, 0.0)
            target = self.alpha_max * (floor + scale * rel)
            ratio = (target / self.alpha_max).clamp(1e-4, 1.0 - 1e-4)
            self.raw_alpha = nn.Parameter(torch.logit(ratio))
        else:
            self.raw_alpha = nn.Parameter(
                torch.full((num_classes, ), self.init_alpha_bias))
        # promote the gate to a trainable parameter only once knowledge exists
        self.raw_g_cls = nn.Parameter(
            old_g_cls if old_g_cls is not None else torch.zeros(1))
        self.classes = classes
        self.num_classes = num_classes
        self.ready = True

    def load_knowledge(self, path: str) -> None:
        """Load a cached knowledge file produced by the offline builder."""
        ckpt = torch.load(path, map_location='cpu')
        meta = ckpt.get('meta', {}) or {}
        file_stage = str(meta.get('stage', self.stage)).lower()
        if file_stage == 'chsd':
            file_stage = 'ffcp_chsd'
        if file_stage in ('ffcp', 'ffcp_chsd', 'full'):
            self.stage = file_stage
        self.set_knowledge(
            hard=ckpt['hard'],
            soft=ckpt['soft'],
            kappa=ckpt['kappa'],
            reliability=ckpt['reliability'],
            classes=ckpt['classes'],
            token_ids=ckpt['token_ids'])
        self.meta = meta

    def is_ready(self) -> bool:
        return bool(getattr(self, 'ready', False)) and self.num_classes > 0

    # -- helpers ----------------------------------------------------------
    def class_logits_from_tokens(self, token_logits: torch.Tensor) -> torch.Tensor:
        """Return class logits using Grounding DINO's score aggregation.

        Grounding DINO converts token logits to probabilities first and then
        averages the positive tokens.  Averaging raw logits here produces a
        different confidence scale for multi-token categories, which makes
        the ATAR uncertainty gate and consistency loss disagree with the
        detector's post-processing.
        """
        outs = []
        for c in range(self.num_classes):
            start = int(self._token_offset[c])
            end = int(self._token_offset[c + 1])
            if end > start:
                token_ids = self._token_flat[start:end].to(token_logits.device)
                probs = token_logits.index_select(-1, token_ids).sigmoid()
                mean_prob = probs.mean(-1).clamp(self.eps, 1.0 - self.eps)
                outs.append(torch.logit(mean_prob))
            else:
                # a category without prompt tokens must never win the softmax
                outs.append(
                    token_logits.new_full(token_logits.shape[:-1], -1e4))
        return torch.stack(outs, dim=-1)

    def broadcast_delta(self, token_logits: torch.Tensor,
                        delta: torch.Tensor) -> torch.Tensor:
        """Add the per-class correction to the corresponding text tokens."""
        out = token_logits.clone()
        for c in range(self.num_classes):
            start = int(self._token_offset[c])
            end = int(self._token_offset[c + 1])
            if end > start:
                token_ids = self._token_flat[start:end].to(out.device)
                out[..., token_ids] = out[..., token_ids] + delta[..., c:c + 1]
        return out

    def current_alpha(self) -> torch.Tensor:
        """Bounded class-wise strength ``alpha_c in [0, alpha_max]``."""
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    # -- ATAR -------------------------------------------------------------
    def atar(self, hidden_states: torch.Tensor,
             class_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Asymmetric Task-Authority Routing (Eqs. (10)-(12)).

        Args:
            hidden_states (Tensor): Decoder query embeddings of shape
                ``[..., D]`` (the matching queries).
            class_logits (Tensor): Baseline category logits ``L0`` of shape
                ``[..., C]``.

        Returns:
            dict: ``delta`` (bounded correction), ``u`` (query authority),
            ``p0`` (baseline confidence), ``agree`` (Hard-Soft agreement) and
            the intermediate ``class_logits``/``delta`` pair for the loss.
        """
        soft = l2_normalize(self.soft.to(hidden_states.dtype), dim=-1)
        z = l2_normalize(hidden_states, dim=-1)

        # FFCP is deliberately isolated from CHSD: it uses only the
        # context-cleaned visual prototype and its FFCP reliability.  The
        # later stages add the hard identity anchor and/or ATAR routing.
        if self.stage == 'ffcp':
            agree = torch.ones_like(self.reliability, dtype=z.dtype)
        else:
            hard = l2_normalize(self.hard.to(hidden_states.dtype), dim=-1)
            # Eq. (10): Hard-Soft agreement re-weights prototype reliability.
            agree = 0.5 * (1.0 + (soft * hard).sum(-1))            # [C]
        r_bar = (self.reliability.to(agree.dtype) * agree).clamp(0.0, 1.0)
        r_gate = r_bar.pow(max(self.reliability_power, 0.0))

        # FFCP and FFCP+CHSD expose the prototype directly.  Only the full
        # method enables ATAR's query-uncertainty gate (Eq. 11).
        p0 = class_logits.sigmoid().max(-1).values             # [...]
        if self.stage == 'full':
            u = torch.sigmoid((self.theta - p0) / self.T_u)    # [...]
            g = torch.clamp(u.unsqueeze(-1) * r_gate, max=self.g_max)
        else:
            u = torch.ones_like(p0)
            g = r_gate.expand_as(class_logits).clamp(max=self.g_max)

        # Eq. (12): bounded cosine correction.  FFCP uses the cleaned Soft
        # prototype; CHSD/full use a conservative Hard/Soft blend so the
        # visual descriptor cannot override the text-aligned identity anchor.
        # Compare every query with every class prototype.  ``z`` is shaped
        # ``[..., D]`` while ``soft`` is ``[C, D]``; broadcasting an element-
        # wise product would incorrectly align the query axis with classes.
        if self.stage == 'ffcp':
            direction = soft
        else:
            hard = l2_normalize(self.hard.to(hidden_states.dtype), dim=-1)
            mix = min(max(self.soft_mix, 0.0), 1.0)
            direction = l2_normalize((1.0 - mix) * hard + mix * soft,
                                     dim=-1)
        d = torch.einsum('...d,cd->...c', z, direction)        # [..., C]
        alpha = self.current_alpha().to(d.dtype)
        g_cls = torch.sigmoid(self.raw_g_cls).to(d.dtype)
        delta = self.gamma * g_cls * g * alpha * d
        bound = self.gamma * self.g_max * self.alpha_max
        delta = torch.clamp(delta, -bound, bound)
        if self.center_delta and delta.shape[-1] > 1:
            delta = delta - delta.mean(dim=-1, keepdim=True)
        if self.positive_only:
            delta = delta.clamp_min(0.0)

        return dict(delta=delta, u=u, p0=p0, agree=agree,
                    class_logits=class_logits, r_bar=r_bar)

    def extra_repr(self) -> str:
        return (f'embed_dims={self.embed_dims}, '
                f'num_classes={self.num_classes}, gamma={self.gamma}, '
                f'alpha_max={self.alpha_max}, g_max={self.g_max}, '
                f'theta={self.theta}, T_u={self.T_u}, stage={self.stage}')


__all__ = [
    'robust_mean', 'l2_normalize', 'orthonormalize', 'svd_context_basis',
    'context_leakage_energy', 'project_l2_ball', 'project_out',
    'pool_features_in_boxes', 'get_attribute_candidates',
    'STRUCTURAL_ATTRIBUTE_TEMPLATES', 'CLASS_ALIASES', 'S2HKnowledgeBank'
]

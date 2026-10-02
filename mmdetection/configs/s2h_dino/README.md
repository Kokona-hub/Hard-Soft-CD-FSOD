# S2H-CD-FSOD on the Domain-RAG Grounding DINO recipe

Implementation of **"Trust What You Transfer: Knowledge-Enhanced Hard-Soft
Reasoning for Cross-Domain Few-Shot Object Detection"** on top of the *exact*
Grounding DINO configuration reproduced by Domain-RAG (NeurIPS 2025), so that
the two methods can be compared under identical data, augmentation, optimizer
and schedule settings.

The detector, backbone, neck, transformer encoder/decoder and the
**localization branch are byte-for-byte the same** as Domain-RAG's. The only
addition is a bounded, knowledge-enhanced correction applied to the
classification logits.

---

## 1. Method → code map

| Paper component | Code |
|---|---|
| FFCP — foreground-frozen context probing (Eqs. (3)-(5)) | `tools/s2h/support_interventions.py`, `build_s2h_knowledge.py::compute_context_subspace` |
| CHSD — Hard identity anchor (Eqs. (6)-(7)) | `build_s2h_knowledge.py::build_class_knowledge` + `mmdet/models/utils/s2h_knowledge.py::STRUCTURAL_ATTRIBUTE_TEMPLATES` |
| CHSD — Soft prototype + shot-aware shrinkage (Eqs. (8)-(9)) | `build_s2h_knowledge.py::build_class_knowledge` |
| ATAR — agreement / authority routing (Eqs. (10)-(12)) | `mmdet/models/utils/s2h_knowledge.py::S2HKnowledgeBank.atar` |
| Bounded logit correction + token broadcast | `mmdet/models/dense_heads/s2h_grounding_dino_head.py` |
| Residual + consistency loss (Eq. (13)) | `S2HGroundingDINOHead.loss` (`res_weight`, `con_weight`) |
| Detector | `mmdet/models/detectors/s2h_grounding_dino.py` |

Everything is registered into the mmdet registry, so the configs below can be
launched with the standard `tools/dist_train.sh`.

---

## 2. Prerequisites

1. Install mmGroundingDINO following Domain-RAG's `README_mmlab.md`
   (`pip install -r requirements.txt`, then install mmdet from this folder).
2. Prepare the datasets exactly as Domain-RAG does (see `datasets/structure.md`
   at the repository root). The generated configs expect
   `mmdetection/data/<dataset>/annotations/{n}_shot.json` and `test.json`.

No extra dependency is required — the S2H modules only use `torch`,
`mmengine` and `PIL`/`numpy`.

---

## 3. Quick smoke test (no dataset / no checkpoint)

Verifies the operators, the zero-at-init / bounded ATAR correction and the
support interventions:

```bash
cd mmdetection
python tools/s2h/selfcheck.py
```

---

## 4. Generate the configs

```bash
cd mmdetection
python tools/s2h/gen_s2h_configs.py \
    --datasets ArTaxOr clipart1k DIOR FISH NEU-DET UODD --shots 1 5 10
```

Each `configs/s2h_dino/s2h_grounding_dino_swin-b_<DS>_<shot>shot.py`:

* copies **verbatim** the dataset block of the corresponding Domain-RAG config
  (`configs/grounding_dino/CDFSOD_detection_few-shot_<DS>_<shot>shot.py`),
  including the `CachedMosaic` / `YOLOXHSVRandomAug` / `CachedMixUp`
  pipeline and the `(800, 1333)` test resize;
* inherits the Grounding DINO Swin-B fine-tuning recipe
  (`configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py`,
  `load_from=groundingdino_swinb_cogcoor_mmdet-55949c9c.pth`);
* swaps in `S2HGroundingDINO` + `S2HGroundingDINOHead`;
* sets the paper's epoch budget (30 epochs, or 5 for Clipart1k / DeepFish).

---

## 5. Build the cached Hard-Soft knowledge

FFCP and CHSD are support-side only and are computed once per support split,
then cached (the paper: *"FFCP knowledge is computed once for each support
split and then cached"*).

```bash
python tools/s2h/build_s2h_knowledge.py \
    --config configs/s2h_dino/s2h_grounding_dino_swin-b_ArTaxOr_1shot.py \
    --out work_dirs/s2h_knowledge/ArTaxOr_1shot.pth
```

The script

1. loads the detector (default: the checkpoint from the config `load_from`);
2. renders, for every support image and category, the four foreground-frozen
   views (same-domain / class-mismatch / neutralized / reconstruction);
3. pools visual-encoder descriptors inside the dilated GT boxes;
4. estimates `U_ctx`, `kappa_ctx` (FFCP) and builds `h_c`, `s_c`, `r_c` (CHSD);
5. writes `hard / soft / kappa / reliability / classes / token_ids` to `--out`.

Useful flags: `--ffcp-rank`, `--hard-radius`, `--theta-h`, `--dilate`,
`--shrink-xi`, `--lambda-b/--lambda-cf/--lambda-kappa`, `--max-samples`.

For pipeline integration testing without a dataset use `--dry-run`, which
emits random knowledge with a correct class→token map.

---

## 6. Train / evaluate

One command per dataset/shot (builds knowledge if requested, then trains into a
dedicated work dir, mirroring Domain-RAG's `auto_modify_swin_t_config.py`):

```bash
python tools/s2h/run_s2h_all.py \
    --datasets ArTaxOr NEU-DET --shots 1 5 10 \
    --gpus 4 --build-knowledge
```

Or manually, exactly like Domain-RAG:

```bash
bash tools/dist_train.sh \
    configs/s2h_dino/s2h_grounding_dino_swin-b_ArTaxOr_1shot.py 4 \
    --work-dir cat_work_dir/s2h/ArTaxOr_1shot
```

Evaluation (COCO `classwise` mAP, identical to Domain-RAG's `CocoMetric`):

```bash
bash tools/dist_test.sh \
    configs/s2h_dino/s2h_grounding_dino_swin-b_ArTaxOr_1shot.py \
    cat_work_dir/s2h/ArTaxOr_1shot/latest.pth 4
```

---

## 7. Baselines / ablations for the comparison table

| Variant | How to run |
|---|---|
| Fine-tuned GroundingDINO (baseline) | `model.bbox_head.s2h_cfg.enabled=False` |
| FFCP | context-cleaned Soft prototype only; no CHSD Hard anchor or ATAR signal |
| FFCP + CHSD | FFCP plus Hard/Soft decomposition and agreement weighting; no ATAR signal |
| FFCP + CHSD + ATAR | complete method, including query uncertainty routing and auxiliary regularizers |
| Random knowledge (control) | build knowledge with `--dry-run` |
| w/o background retrieval / w/o generation | not applicable — those are Domain-RAG stages; this method replaces them |
| Cumulative FFCP → +CHSD → +ATAR | use `tools/s2h/run_ablation.sh`; it isolates every stage and seed |

The operational definitions above are explicit implementation choices for this
ablation: FFCP computes the context-cleaned visual prototype, CHSD adds the
structural Hard anchor and Hard/Soft agreement, and ATAR adds only the query
uncertainty gate and auxiliary regularizers. Because the correction is initialized to ~0 (`alpha_c ≈ 0`),
the first iterations remain close to the baseline while the stage-specific
parameters learn.

---

## 8. Hyper-parameters

| Symbol | Meaning | Default | Where |
|---|---|---|---|
| `gamma` | global correction scale | 1.0 | `s2h_cfg` |
| `alpha_max` | class-wise strength ceiling `α_max` | 0.5 | `s2h_cfg` |
| `g_max` | authority ceiling `g_max` | 0.5 | `s2h_cfg` |
| `theta` | query uncertainty threshold `θ` | 0.5 | `s2h_cfg` |
| `T_u` | routing temperature `T_u` | 0.1 | `s2h_cfg` |
| `res_weight` | residual penalty `λ_res` | 0.1 | `s2h_cfg` |
| `con_weight` | consistency weight `λ_con` | 0.1 | `s2h_cfg` |
| `stage` | ablation selector: `ffcp`, `ffcp_chsd`, `full` | `full` | `s2h_cfg` |
| FFCP rank `r` | context subspace size | 4 | `build_s2h_knowledge.py` |
| `ρ_H` | Hard update radius | 1.0 | `build_s2h_knowledge.py` |
| `ξ_H` | attribute acceptance threshold | 0.3 | `build_s2h_knowledge.py` |
| `T_q` | attribute softmax temperature | 0.5 | `build_s2h_knowledge.py` |

Grounding DINO side (unchanged from Domain-RAG): Swin-B + BERT-base, AdamW
`lr=1e-4`, `weight_decay=1e-4`, backbone `lr_mult=0.1`, batch size 4/GPU,
grad clip `max_norm=0.1`, `MultiStepLR(milestones=[11], gamma=0.1)`,
`max_epochs=30` (5 for Clipart1k / DeepFish).

---

## 9. Design choices / assumptions

The paper is a submission whose appendix is not included in the code drop, so a
few quantities are implemented with documented, reasonable choices:

1. **Support descriptors** are average-pooled visual-encoder features inside
   the dilated GT box (scale-selected FPN level), i.e. the *visual encoder*
   features that FFCP probes. Swapping the pooling for ROI-Align is a one-line
   change in `mmdet/models/utils/s2h_knowledge.py::pool_features_in_boxes`.
2. **Attribute candidates** come from the curated structural templates in
   `STRUCTURAL_ATTRIBUTE_TEMPLATES` (parts / shape / texture / boundary only —
   never viewpoint, scale or background terms), with a generic fallback.
3. **Prototype reliability** `r_c` is set to
   `clip((1 - kappa_c) * |V_c| / |A_c|, 0, 1)`, i.e. the context dependence
   combined with the fraction of accepted identity attributes.
4. **`alpha_c`** is parameterized as `alpha_max * sigmoid(raw_alpha)` with
   `raw_alpha` initialized to `-8`. This makes the correction numerically zero
   at initialization while keeping a non-vanishing gradient — a plain
   `clamp`/`ReLU` at 0 would have zero gradient and could never learn.
5. **Drift** uses Eq. (4) as written; the compositing is a pure pixel composite
   (no VAE), so the reconstruction control is a down-/up-sample round trip of
   the original context.
6. The correction is applied to the **last decoder layer** and only to the
   matching (non-denoising) queries, keeping the DINO denoising loss intact.

---

## 10. Full pipeline on a server (copy-paste)

A single script wraps every stage:

```bash
cd <repo>/mmdetection

bash tools/s2h/pipeline.sh env          # versions + GPU count
bash tools/s2h/pipeline.sh download     # Swin-B checkpoint -> checkpoints/
bash tools/s2h/pipeline.sh check        # verify data/<DS>/ layout
bash tools/s2h/pipeline.sh selfcheck    # math / integration self-test
bash tools/s2h/pipeline.sh configs      # generate configs/s2h_dino/*.py
bash tools/s2h/pipeline.sh knowledge    # FFCP + CHSD -> cached knowledge
bash tools/s2h/pipeline.sh train        # fine-tune all dataset/shot pairs
bash tools/s2h/pipeline.sh test         # COCO classwise mAP
bash tools/s2h/pipeline.sh baseline     # GroundingDINO baseline (S2H off)
bash tools/s2h/pipeline.sh baseline-test # baseline COCO classwise mAP
```

For the requested six-dataset, three-shot, four-stage and three-seed study,
run one command per dataset. Each command trains and evaluates all 36
combinations for that dataset:

```bash
export CKPT=/mnt/sdc/hzh/Domain-RAG-main/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
export PYTHONNOUSERSITE=1
cd /mnt/sdc/hzh/Domain-RAG-main/mmdetection

bash tools/s2h/run_ablation.sh ArTaxOr
bash tools/s2h/run_ablation.sh clipart1k
bash tools/s2h/run_ablation.sh DIOR
bash tools/s2h/run_ablation.sh FISH
bash tools/s2h/run_ablation.sh NEU-DET
bash tools/s2h/run_ablation.sh UODD
```

Defaults are `SHOTS="1 5 10"`, `STAGES="baseline ffcp ffcp_chsd full"`,
`SEEDS="3407 3408 3409"`, and `GPUS=4`. When `GPUS=1`, the launcher
automatically sets `train_dataloader.batch_size=1`; override with
`BATCH_SIZE` if the selected GPU has enough memory. Use `SKIP_BUILD=1` to reuse existing
knowledge or `FORCE=1` to rebuild and retrain. The `full` path name means
`FFCP+CHSD+ATAR`.

Outputs are isolated as
`work_dirs/s2h_knowledge/ablation/<dataset>/<stage>/seed<seed>/<shot>shot.pth`
and
`cat_work_dir/ablation/<dataset>/<stage>/seed<seed>/<shot>shot/`.
Validation-best checkpoints are saved every epoch and evaluated instead of
assuming that the final epoch is best.

After the runs finish, create the paper table with:

```bash
python tools/s2h/summarize_ablation.py \
  --root cat_work_dir/ablation \
  --out work_dirs/s2h_ablation_summary.csv
```

The CSV contains one row per seed and one `mean_std` row per dataset/stage/shot;
unreadable or unfinished runs are listed explicitly.

Or just `bash tools/s2h/pipeline.sh all` (check → selfcheck → configs →
knowledge → train).

Useful overrides (environment variables):

```bash
# subset of experiments, 8 GPUs, local checkpoint
DATASETS="ArTaxOr NEU-DET" SHOTS="1 5" GPUS=8 \
CKPT=$PWD/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth \
  bash tools/s2h/pipeline.sh all

# torch >= 2.5 (python -m torch.distributed.launch is deprecated)
LAUNCHER=torchrun bash tools/s2h/pipeline.sh train

# force a knowledge rebuild / serial knowledge jobs
FORCE=1 PARALLEL_KNOWLEDGE=0 bash tools/s2h/pipeline.sh knowledge

# offline cluster: use an uploaded HuggingFace BERT directory
export S2H_BERT_DIR=/path/to/bert-base-uncased
export TRANSFORMERS_OFFLINE=1
```

Logs are written to `work_dirs/s2h_logs/`, checkpoints to
`cat_work_dir/s2h/<DS>_<shot>shot/latest.pth`.

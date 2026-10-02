# S2H-CD-FSOD: Hard-Soft Reasoning for Cross-Domain Few-Shot Detection

This directory contains the implementation of **S2H-CD-FSOD** (Hard-Soft
reasoning) on top of the Grounding DINO Swin-B recipe used by Domain-RAG.
The goal is to improve cross-domain few-shot object detection without changing
the detector backbone, neck, transformer, localization branch, or the
Domain-RAG data protocol.

The method has three cumulative components:

1. **FFCP (Foreground-Frozen Context Probing)** estimates and removes
   context-dependent visual drift from support descriptors.
2. **CHSD (Context-Constrained Hard-Soft Decomposition)** builds a Hard
   identity anchor and a Soft appearance prototype, with a reliability score.
3. **ATAR (Asymmetric Task-Authority Routing)** applies a bounded,
   reliability-aware correction to Grounding DINO classification logits.

Only the classification logits are corrected. Bounding-box regression and the
localization branch are unchanged.

## Repository Layout

```text
mmdetection/
├── mmdet/models/detectors/s2h_grounding_dino.py
├── mmdet/models/dense_heads/s2h_grounding_dino_head.py
├── mmdet/models/utils/s2h_knowledge.py
├── tools/s2h/build_s2h_knowledge.py
├── tools/s2h/gen_s2h_configs.py
├── tools/s2h/run_ablation.sh
├── tools/s2h/analysis/run_sweep.sh
├── tools/s2h/analysis/summarize_sweep.py
└── tools/s2h/fig4/
    ├── run_clipart1k_fig4.sh
    └── summarize_clipart1k_fig4.py
```

The `fig4/` directory is isolated from the main mAP and ablation launchers.
It writes to date-scoped `fig4` directories and does not overwrite the main
experiment outputs.

## Environment

The reference training environment is:

```text
Python 3.10
PyTorch 2.0.0 + CUDA 11.7
MMCV 2.0.0
MMEngine 0.10.7
MMDetection 3.3.0
```

The original Domain-RAG Grounding DINO checkpoint and the local BERT encoder
are required for offline execution. Set these variables on the training
server:

```bash
export PYTHONNOUSERSITE=1
export S2H_BERT_DIR=/mnt/sdc/hzh/model_cache/bert-base-uncased
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CKPT=/mnt/sdc/hzh/Domain-RAG-main/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
```

`PYTHONNOUSERSITE=1` is important when the machine has another user-level
PyTorch installation. Otherwise MMCV extensions can be loaded against an
incompatible CUDA/PyTorch version.

## Data Protocol

The default configs intentionally reuse the Domain-RAG `data/` directory and
annotation protocol:

```text
data/<dataset>/
├── annotations/1_shot.json
├── annotations/5_shot.json
├── annotations/10_shot.json
├── annotations/test.json
├── train/
└── test/
```

Supported datasets are `ArTaxOr`, `clipart1k`, `DIOR`, `FISH` (DeepFish),
`NEU-DET`, and `UODD`.

Do not mix `data/` with `data_paper/` when reproducing the Domain-RAG baseline.
The support split, query split, initialization checkpoint, batch size, seed,
augmentation pipeline, and final epoch policy must be identical between the
baseline and S2H.

## Generate Configurations

From the `mmdetection` directory:

```bash
python tools/s2h/gen_s2h_configs.py \
  --datasets ArTaxOr clipart1k DIOR FISH NEU-DET UODD \
  --shots 1 5 10 \
  --out-dir configs/s2h_dino
```

This creates 18 configs by copying the Domain-RAG dataset blocks and adding
the S2H detector head. The generated configs use 30 epochs for ArTaxOr, DIOR,
NEU-DET, and UODD, and 5 epochs for clipart1k and FISH.

The current corrected inference-oriented defaults are:

```python
init_alpha_from_reliability=True
positive_only=True
center_delta=True
soft_mix=0.35
reliability_power=0.5
alpha_rel_floor=0.10
alpha_rel_scale=0.80
train_injection=False
```

`train_injection=False` keeps the fine-tuning distribution equal to the
Domain-RAG baseline and enables the Hard-Soft correction during validation and
test. The Figure 4 `lambda_con` panel explicitly overrides this setting,
because that panel measures a training-time consistency-loss factor.

## Build Support Knowledge

FFCP and CHSD are computed once for each support split and cached as a `.pth`
file. A typical command is:

```bash
python tools/s2h/build_s2h_knowledge.py \
  --config configs/s2h_dino/s2h_grounding_dino_swin-b_NEU-DET_1shot.py \
  --checkpoint "$CKPT" \
  --out work_dirs/s2h_knowledge/NEU-DET_1shot.pth \
  --device cuda:0 \
  --seed 3407 \
  --stage full
```

Important knowledge-builder controls include:

```text
--ffcp-rank       context subspace rank (default: 4)
--hard-radius     Hard update radius (default: 1.0)
--dilate          foreground box dilation (default: 1.2)
--shrink-xi       Soft prototype shrinkage (default: 5.0)
--max-samples     support-image limit; 0 means all support images
```

Use `--dry-run` only for integration tests. Random knowledge is not a valid
method result.

## Main Ablation and mAP Experiments

`run_ablation.sh` supports the four cumulative stages:

| Stage | Meaning |
|---|---|
| `baseline` | Vanilla Domain-RAG Grounding DINO; S2H disabled |
| `ffcp` | FFCP Soft prototype only |
| `ffcp_chsd` | FFCP plus CHSD Hard/Soft decomposition |
| `full` | FFCP + CHSD + ATAR |

Example: one complete dataset matrix with one seed:

```bash
CUDA_VISIBLE_DEVICES=3 \
GPUS=1 BATCH_SIZE=2 DEVICE=cuda:0 PORT=29517 LAUNCHER=torchrun \
CONFIG_DIR=configs/s2h_dino \
CHECKPOINT_POLICY=final \
WORK_ROOT=cat_work_dir/20261002_hardsoft_v2 \
KNOWLEDGE_ROOT=work_dirs/s2h_knowledge/20261002_hardsoft_v2 \
LOG_ROOT=work_dirs/ablation_logs/20261002_hardsoft_v2 \
SHOTS="1 5 10" \
STAGES="baseline ffcp ffcp_chsd full" \
SEEDS="3407" \
bash tools/s2h/run_ablation.sh NEU-DET
```

Replace `NEU-DET` with `ArTaxOr`, `clipart1k`, `DIOR`, `FISH`, or `UODD` for
the other domains. Run datasets sequentially when using one GPU.

For paper tables, use `CHECKPOINT_POLICY=final`. The query set is also used as
the validation set in this protocol, so selecting the highest query mAP is
test-set selection and should only be used for diagnostics:

```bash
CHECKPOINT_POLICY=best ...
```

After runs finish, summarize the fixed-final results with:

```bash
python tools/s2h/summarize_ablation.py \
  --root cat_work_dir/20261002_hardsoft_v2 \
  --datasets ArTaxOr clipart1k DIOR FISH NEU-DET UODD \
  --stages baseline ffcp ffcp_chsd full \
  --shots 1 5 10 \
  --seeds 3407 \
  --selection final \
  --out work_dirs/s2h_ablation_summary_20261002_seed3407_final.csv
```

## Figure 4: Hyper-Parameter Sensitivity

Figure 4 is a separate one-factor-at-a-time experiment on **Clipart1k
5-shot**. The support split, initialization checkpoint, training budget, and
seeds are fixed. The measured quantities are:

| Quantity | Values | Default |
|---|---|---|
| FFCP context rank | `1, 2, 4, 8, 16` | `4` |
| Hard update radius | `0.25, 0.5, 1, 2, 4` | `1` |
| Authority ceiling `g_max` | `0.1, 0.25, 0.5, 0.75, 1.0` | `0.5` |
| Uncertainty threshold `theta` | `0.3, 0.4, 0.5, 0.6, 0.7` | `0.5` |
| Consistency weight `lambda_con` | `0, 0.05, 0.1, 0.2` | `0.1` |

Run the complete matrix:

```bash
bash tools/s2h/fig4/run_clipart1k_fig4.sh
```

Inspect the commands without launching jobs:

```bash
DRY_RUN=1 bash tools/s2h/fig4/run_clipart1k_fig4.sh
```

Summarize and draw the five-panel figure:

```bash
python tools/s2h/fig4/summarize_clipart1k_fig4.py \
  --tag 20261002 \
  --seeds 3407 3408 3409
```

Outputs are written under:

```text
cat_work_dir/fig4/20261002/
work_dirs/s2h_knowledge/fig4/20261002/
work_dirs/s2h_analysis/fig4/20261002/
```

The generated files are `*.runs.csv`, `*.summary.csv`, `*.pdf`, `*.png`, and
`*.json`. The Figure 4 launcher refuses to write into an `ablation` path.

## Reproducibility Checklist

Before comparing two methods, record all of the following:

- dataset and shot count;
- support annotation file and split protocol;
- query annotation file;
- seed(s);
- local Grounding DINO checkpoint;
- BERT directory and framework versions;
- batch size and GPU count;
- S2H knowledge-builder arguments;
- final versus best checkpoint policy;
- output root and git commit/file hashes.

Do not report a value from a partially completed run. For the main table,
report paired seeds and mean/std across seeds. Per-class AP should be retained
for diagnosing negative gains, especially when the aggregate mAP is unchanged.

## Sanity Checks

The lightweight operator self-check does not require a dataset:

```bash
python tools/s2h/selfcheck.py
python tools/s2h/analysis/summarize_sweep.py --self-check
```

Python syntax checks:

```bash
python -m py_compile \
  mmdet/models/utils/s2h_knowledge.py \
  mmdet/models/dense_heads/s2h_grounding_dino_head.py \
  tools/s2h/gen_s2h_configs.py \
  tools/s2h/analysis/summarize_sweep.py \
  tools/s2h/fig4/summarize_clipart1k_fig4.py
```

## Method-to-Code Map

| Method component | Implementation |
|---|---|
| FFCP context probing | `tools/s2h/support_interventions.py`, `build_s2h_knowledge.py` |
| CHSD Hard/Soft prototypes | `tools/s2h/build_s2h_knowledge.py` |
| Reliability-aware routing | `mmdet/models/utils/s2h_knowledge.py` |
| Token-level logit correction | `mmdet/models/dense_heads/s2h_grounding_dino_head.py` |
| S2H detector wrapper | `mmdet/models/detectors/s2h_grounding_dino.py` |
| Main ablations | `tools/s2h/run_ablation.sh` |
| Figure 4 sweep | `tools/s2h/fig4/run_clipart1k_fig4.sh` |

## Citation

If you use this implementation, please cite the accompanying S2H-CD-FSOD
paper when it becomes public, and cite Domain-RAG for the baseline protocol:

```bibtex
@article{li2025domain,
  title={Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection},
  author={Li, Yu and Qiu, Xingyu and Fu, Yuqian and others},
  journal={arXiv preprint arXiv:2506.05872},
  year={2025}
}
```

This repository contains research code. Results should be reproduced with the
same data split, checkpoint, seed, and checkpoint-selection policy before they
are used in a paper or benchmark table.

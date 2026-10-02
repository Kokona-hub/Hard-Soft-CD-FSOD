#!/usr/bin/env bash
# Figure 4 only: Clipart1k 5-shot one-factor sensitivity experiment.
# All artifacts are date-scoped under fig4 roots and do not touch main mAP
# or ablation outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${MMDET}"

FIG4_TAG="${FIG4_TAG:-20261002}"
export DATASET="clipart1k"
export SHOT="5"
export SEEDS="${SEEDS:-3407 3408 3409}"
export PARAMS="${PARAMS:-g_max theta con_weight ffcp_rank hard_radius}"
export SWEEP_ROOT="${FIG4_ROOT:-cat_work_dir/fig4/${FIG4_TAG}}"
export KNOW_SWEEP="${FIG4_KNOW:-work_dirs/s2h_knowledge/fig4/${FIG4_TAG}}"
export ABL_ROOT="${ABL_ROOT:-cat_work_dir/20261002_hardsoft_v2}"
export ABL_KNOW="${ABL_KNOW:-work_dirs/s2h_knowledge/20261002_hardsoft_v2}"
export CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-final}"
export CONFIG_DIR="${CONFIG_DIR:-configs/s2h_dino}"
export GPUS="${GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export DEVICE="${DEVICE:-cuda:0}"
export CKPT="${CKPT:-${MMDET}/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth}"

echo "[fig4] tag=${FIG4_TAG} dataset=${DATASET} shot=${SHOT} seeds=${SEEDS}"
echo "[fig4] sweep_root=${SWEEP_ROOT} knowledge_root=${KNOW_SWEEP}"
echo "[fig4] read-only ablation_root=${ABL_ROOT} ablation_knowledge=${ABL_KNOW}"

bash tools/s2h/analysis/run_sweep.sh

echo "[fig4] completed run matrix"
echo "[fig4] summary: python tools/s2h/fig4/summarize_clipart1k_fig4.py --tag ${FIG4_TAG}"

#!/usr/bin/env bash
# Figure 4: two-factor sensitivity over the three Clipart1k shot budgets.
# This wrapper is independent from the main ablation launcher. It reuses the
# 20260930 checkpoints/knowledge read-only and writes only to FIG4_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${MMDET}"

export DATASET="clipart1k"
export SHOTS="${SHOTS:-1 5 10}"
export SEEDS="${SEEDS:-3407}"
export PARAMS="${PARAMS:-g_max theta}"
export FIG4_TAG="${FIG4_TAG:-20260930_clipart1k_shots}"
export SWEEP_ROOT="${FIG4_ROOT:-cat_work_dir/fig4/${FIG4_TAG}}"
export KNOW_SWEEP="${FIG4_KNOW:-work_dirs/s2h_knowledge/fig4/${FIG4_TAG}}"
export ABL_ROOT="${ABL_ROOT:-cat_work_dir/20260930_hardsoft}"
export ABL_KNOW="${ABL_KNOW:-work_dirs/s2h_knowledge/20260930_hardsoft}"
export CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-final}"
export CONFIG_DIR="${CONFIG_DIR:-configs/s2h_dino}"
export GPUS="${GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export DEVICE="${DEVICE:-cuda:0}"
export PORT="${PORT:-29517}"
export LAUNCHER="${LAUNCHER:-torchrun}"
export DRY_RUN="${DRY_RUN:-0}"
export FORCE="${FORCE:-0}"
export CKPT="${CKPT:-${MMDET}/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth}"
# Generated configs and newly rebuilt knowledge use the contrastive protocol.
# The historical 20260930 experiment remains available explicitly through
# FIG4_PROTOCOL=legacy_0930.
export FIG4_PROTOCOL="${FIG4_PROTOCOL:-contrastive}"
export SAFE_TEST_DATALOADER="${SAFE_TEST_DATALOADER:-1}"
case "${FIG4_PROTOCOL}" in
  legacy_0930)
    # The source checkpoints/knowledge are from the 20260930 legacy run.
    export EXTRA_CFG_OPTIONS="model.bbox_head.s2h_cfg.correction_mode=legacy model.bbox_head.s2h_cfg.positive_only=True"
    ;;
  contrastive)
    export EXTRA_CFG_OPTIONS=""
    ;;
  *)
    echo "FIG4_PROTOCOL must be legacy_0930 or contrastive" >&2
    exit 2
    ;;
esac
if [ "${SAFE_TEST_DATALOADER}" = "1" ]; then
  export EXTRA_TEST_CFG_OPTIONS="test_dataloader.num_workers=0 test_dataloader.persistent_workers=False"
else
  export EXTRA_TEST_CFG_OPTIONS=""
fi

mkdir -p "${SWEEP_ROOT}" "${KNOW_SWEEP}"
echo "[fig4-shots] tag=${FIG4_TAG} dataset=${DATASET} shots=${SHOTS} seeds=${SEEDS}"
echo "[fig4-shots] params=${PARAMS}"
echo "[fig4-shots] source checkpoints=${ABL_ROOT} knowledge=${ABL_KNOW}"
echo "[fig4-shots] output sweep=${SWEEP_ROOT} knowledge=${KNOW_SWEEP}"
echo "[fig4-shots] protocol=${FIG4_PROTOCOL}"
echo "[fig4-shots] safe test dataloader=${SAFE_TEST_DATALOADER}"

for SHOT in ${SHOTS}; do
  echo "[fig4-shots] running ${SHOT}-shot"
  SHOT="${SHOT}" \
  DATASET="${DATASET}" \
  PARAMS="${PARAMS}" \
  SEEDS="${SEEDS}" \
  SWEEP_ROOT="${SWEEP_ROOT}" \
  KNOW_SWEEP="${KNOW_SWEEP}" \
  ABL_ROOT="${ABL_ROOT}" \
  ABL_KNOW="${ABL_KNOW}" \
  CONFIG_DIR="${CONFIG_DIR}" \
  GPUS="${GPUS}" BATCH_SIZE="${BATCH_SIZE}" DEVICE="${DEVICE}" \
  PORT="${PORT}" LAUNCHER="${LAUNCHER}" DRY_RUN="${DRY_RUN}" \
  FORCE="${FORCE}" CHECKPOINT_POLICY="${CHECKPOINT_POLICY}" CKPT="${CKPT}" \
  EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS}" \
  EXTRA_TEST_CFG_OPTIONS="${EXTRA_TEST_CFG_OPTIONS}" \
  bash tools/s2h/analysis/run_sweep.sh
done

echo "[fig4-shots] completed. Summarize and draw with:"
echo "  python tools/s2h/fig4/summarize_clipart1k_shots_fig4.py \\\\"
echo "    --tag ${FIG4_TAG} --root ${SWEEP_ROOT} --params ${PARAMS} \\\\"
echo "    --shots ${SHOTS} --seeds ${SEEDS}"

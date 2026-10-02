#!/usr/bin/env bash
# Run the original Domain-RAG GroundingDINO Swin-B baseline.
#
# This entry point intentionally uses the original CDFSOD configs and data/
# support protocol. It does not load the S2H detector or data_paper splits.
#
# Usage (from mmdetection):
#   bash tools/s2h/run_domainrag_baseline.sh NEU-DET
#
# Optional environment variables:
#   SHOTS="1 5 10"  GPUS=4  BATCH_SIZE=4  SEED=3407
#   CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=29518  CKPT=/path/to/*.pth
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash tools/s2h/run_domainrag_baseline.sh <dataset>" >&2
  echo "datasets: ArTaxOr clipart1k DIOR FISH NEU-DET UODD" >&2
  exit 2
fi

DATASET="$1"
case " ${DATASETS:-ArTaxOr clipart1k DIOR FISH NEU-DET UODD} " in
  *" ${DATASET} "*) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 2 ;;
esac

MMDET="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${MMDET}"

SHOTS="${SHOTS:-1 5 10}"
GPUS="${GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PORT="${PORT:-29518}"
SEED="${SEED:-3407}"
WORK_ROOT="${WORK_ROOT:-cat_work_dir/domainrag_baseline}"
CKPT="${CKPT:-${MMDET}/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth}"
PYTHON_BIN="${PYTHON:-python}"
export PORT

if [[ ! -s "${CKPT}" ]]; then
  echo "missing local GroundingDINO Swin-B checkpoint: ${CKPT}" >&2
  exit 1
fi

# The original Domain-RAG launcher temporarily replaces the first base of the
# Swin-T config with the dataset config, then trains Swin-B on top of it. Keep
# that inheritance order here; sibling bases would make MMEngine reject the
# duplicate dataset keys.
for SHOT in ${SHOTS}; do
  DATA_CFG="configs/grounding_dino/CDFSOD_detection_few-shot_${DATASET}_${SHOT}shot.py"
  [[ -f "${DATA_CFG}" ]] || { echo "missing config: ${DATA_CFG}" >&2; exit 1; }
  [[ -f "data/${DATASET}/annotations/${SHOT}_shot.json" ]] || {
    echo "missing Domain-RAG support split: data/${DATASET}/annotations/${SHOT}_shot.json" >&2
    exit 1
  }
  [[ -f "data/${DATASET}/annotations/test.json" ]] || {
    echo "missing Domain-RAG query split: data/${DATASET}/annotations/test.json" >&2
    exit 1
  }
  WORK="${WORK_ROOT}/${DATASET}_${SHOT}shot"
  SWIN_T_CFG="${WORK}/domainrag_grounding_dino_swin-t.py"
  CFG="${WORK}/domainrag_grounding_dino_swin-b.py"
  mkdir -p "${WORK}"
  PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - "${SWIN_T_CFG}" "${DATA_CFG}" <<'PY'
from pathlib import Path
import sys

out_path, data_cfg = sys.argv[1:]
source = Path('configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py')
text = source.read_text()
config_root = Path('configs').resolve()
replacements = {
    "'../_base_/datasets/CDFSOD_detection_few-shot.py'": repr(str(Path(data_cfg).resolve()).replace('\\', '/')),
    "'../_base_/schedules/schedule_1x.py'": repr(str((config_root / '_base_/schedules/schedule_1x.py').resolve()).replace('\\', '/')),
    "'../_base_/default_runtime.py'": repr(str((config_root / '_base_/default_runtime.py').resolve()).replace('\\', '/')),
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f'cannot locate base {old} in {source}')
    text = text.replace(old, new, 1)
Path(out_path).write_text(text)
PY
  cat > "${CFG}" <<EOF
_base_ = [
    '${MMDET}/${SWIN_T_CFG}',
]

load_from = '${CKPT}'
model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
)
EOF

  echo "[domainrag-baseline][train] ${DATASET} ${SHOT}-shot gpus=${GPUS} batch=${BATCH_SIZE}"
  PYTHONNOUSERSITE=1 PYTHON_BIN="${PYTHON_BIN}" \
    bash tools/dist_train.sh "${CFG}" "${GPUS}" \
    --work-dir "${WORK}" \
    --cfg-options \
      "load_from=${CKPT}" \
      "randomness.seed=${SEED}" \
      "train_dataloader.batch_size=${BATCH_SIZE}"

  # Domain-RAG reports the final/latest checkpoint, not the best query-set
  # checkpoint. The query JSON is used for both val and test in these configs.
  [[ -s "${WORK}/latest.pth" ]] || { echo "missing ${WORK}/latest.pth" >&2; exit 1; }
  echo "[domainrag-baseline][test] ${DATASET} ${SHOT}-shot checkpoint=${WORK}/latest.pth"
  PYTHONNOUSERSITE=1 PYTHON_BIN="${PYTHON_BIN}" \
    bash tools/dist_test.sh "${CFG}" "${WORK}/latest.pth" "${GPUS}" \
    --work-dir "${WORK}"
done

echo "completed Domain-RAG GroundingDINO baseline: ${DATASET}; shots=${SHOTS}"

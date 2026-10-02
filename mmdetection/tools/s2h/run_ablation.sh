#!/usr/bin/env bash
# Reproducible S2H-CD-FSOD four-stage ablation for one dataset.
#
# Usage (from mmdetection):
#   bash tools/s2h/run_ablation.sh ArTaxOr
#
# The command runs every requested shot, stage and seed, and evaluates each
# completed run.  The support split is fixed by the generated annotation
# files; SEEDS controls detector training and the stochastic FFCP probes.
#
# Stage semantics:
#   baseline  : vanilla Grounding DINO, S2H disabled
#   ffcp      : FFCP context-cleaned Soft prototype only
#   ffcp_chsd : FFCP + CHSD Hard/Soft knowledge, no ATAR uncertainty gate
#   full      : FFCP + CHSD + ATAR (the complete method)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MMDET}"

# BERT weights for the Grounding DINO language model.  An exported but *empty*
# S2H_BERT_DIR (e.g. `S2H_BERT_DIR="$S2H_BERT_DIR"` with an unset variable) is
# forwarded verbatim and makes the config call
# `AutoTokenizer.from_pretrained('')`, failing with
# `HFValidationError ... max length is 96: ''`.  Never forward such a value.
_S2H_BERT_DEFAULT="/mnt/sdc/hzh/model_cache/bert-base-uncased"
if [ ! -d "${S2H_BERT_DIR:-}" ]; then
    if [ -d "${_S2H_BERT_DEFAULT}" ]; then
        export S2H_BERT_DIR="${_S2H_BERT_DEFAULT}"
    else
        unset S2H_BERT_DIR
    fi
fi
echo "[ablation] S2H_BERT_DIR=${S2H_BERT_DIR:-<unset>}"

if [[ $# -lt 1 || $# -gt 1 ]]; then
  echo "usage: bash tools/s2h/run_ablation.sh <dataset>" >&2
  echo "datasets: ArTaxOr clipart1k DIOR FISH NEU-DET UODD" >&2
  exit 2
fi

DATASET="$1"
SHOTS="${SHOTS:-1 5 10}"
STAGES="${STAGES:-baseline ffcp ffcp_chsd full}"
SEEDS="${SEEDS:-3407 3408 3409}"
GPUS="${GPUS:-4}"
PORT="${PORT:-29500}"
if [[ -z "${BATCH_SIZE:-}" && "${GPUS}" == "1" ]]; then
  BATCH_SIZE=1
else
  BATCH_SIZE="${BATCH_SIZE:-4}"
fi
if [[ -z "${PYTHON:-}" && -n "${CONDA_PREFIX:-}" && \
      -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi
CONFIG_DIR="${CONFIG_DIR:-configs/s2h_dino}"
KNOWLEDGE_ROOT="${KNOWLEDGE_ROOT:-work_dirs/s2h_knowledge/ablation}"
WORK_ROOT="${WORK_ROOT:-cat_work_dir/ablation}"
LOG_ROOT="${LOG_ROOT:-work_dirs/ablation_logs}"
CKPT="${CKPT:-}"
FORCE="${FORCE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
REQUIRE_LOCAL_CKPT="${REQUIRE_LOCAL_CKPT:-1}"

# Reuse the checkpoint at the conventional location when CKPT was not supplied
# (an unset or empty `CKPT="$CKPT"` would otherwise abort the whole matrix).
_S2H_CKPT_DEFAULT="${MMDET}/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
if [ -z "${CKPT}" ] && [ -s "${_S2H_CKPT_DEFAULT}" ]; then
  CKPT="${_S2H_CKPT_DEFAULT}"
  echo "[ablation] CKPT -> ${CKPT}"
fi

# An explicitly exported empty S2H_BERT_DIR overrides the config default and
# is interpreted by HuggingFace as an invalid repository id.  Fail early with
# an actionable message when offline execution requires a local model.
if [[ "${TRANSFORMERS_OFFLINE:-0}" == "1" || "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  if [[ -z "${S2H_BERT_DIR:-}" || ! -d "${S2H_BERT_DIR}" ]]; then
    echo "S2H_BERT_DIR must point to a local bert-base-uncased directory in offline mode." >&2
    echo "Example: export S2H_BERT_DIR=/mnt/sdc/hzh/model_cache/bert-base-uncased" >&2
    exit 1
  fi
fi

# The generated configs inherit an online OpenMMLab ``load_from`` URL.  On
# the offline server, fail before launching distributed workers if the local
# checkpoint was not supplied; otherwise every rank falls back to downloading
# into the user's torch cache and produces a noisy ChildFailedError.
if [[ "${REQUIRE_LOCAL_CKPT}" == "1" && ( -z "${CKPT}" || ! -s "${CKPT}" ) ]]; then
  echo "CKPT must point to a readable local Grounding DINO checkpoint." >&2
  echo "Example: CKPT=/mnt/sdc/hzh/Domain-RAG-main/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" >&2
  exit 1
fi
SKIP_TEST="${SKIP_TEST:-0}"
# `best` selects the highest recorded bbox mAP (legacy behavior).  `final`
# selects the configured final epoch, matching the Domain-RAG latest-checkpoint
# policy when val and test share the query set.
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-final}"
case "${CHECKPOINT_POLICY}" in
  best|final) ;;
  *) echo "CHECKPOINT_POLICY must be best or final" >&2; exit 2 ;;
esac
final_epoch_for() {
  case "${DATASET}" in
    clipart1k|FISH) echo 5 ;;
    *) echo 30 ;;
  esac
}

case " ${DATASETS:-ArTaxOr clipart1k DIOR FISH NEU-DET UODD} " in
  *" ${DATASET} "*) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 2 ;;
esac

CONFIG="${CONFIG_DIR}/s2h_grounding_dino_swin-b_${DATASET}_1shot.py"
if [[ ! -f "${CONFIG}" ]]; then
  echo "missing generated config: ${CONFIG}" >&2
  echo "run: ${PYTHON} tools/s2h/gen_s2h_configs.py --datasets ${DATASET} --shots 1 5 10" >&2
  exit 1
fi
mkdir -p "${KNOWLEDGE_ROOT}" "${WORK_ROOT}" "${LOG_ROOT}"

echo "[runtime] python: ${PYTHON}"
echo "[runtime] gpus: ${GPUS}; batch_size: ${BATCH_SIZE}"
PYTHONNOUSERSITE=1 "${PYTHON}" -c \
  "import sys, torch; print('[runtime] executable:', sys.executable); print('[runtime] torch:', torch.__version__, torch.__file__); print('[runtime] cuda:', torch.version.cuda)"

config_for() {
  echo "${CONFIG_DIR}/s2h_grounding_dino_swin-b_${DATASET}_$1shot.py"
}
knowledge_for() {
  echo "${KNOWLEDGE_ROOT}/${DATASET}/$1/seed$2/${3}shot.pth"
}
work_for() {
  echo "${WORK_ROOT}/${DATASET}/$1/seed$2/${3}shot"
}
log_for() {
  echo "${LOG_ROOT}/${DATASET}/$1/seed$2/${3}shot.log"
}

for SHOT_CHECK in ${SHOTS}; do
  CFG_CHECK="$(config_for "${SHOT_CHECK}")"
  if [[ ! -f "${CFG_CHECK}" ]]; then
    echo "missing generated config: ${CFG_CHECK}" >&2
    exit 1
  fi
done

train_one() {
  local cfg="$1" work="$2"; shift 2
  local -a opts=("randomness.seed=$SEED")
  if [[ -n "${CKPT}" ]]; then
    opts+=("load_from=${CKPT}")
  fi
  opts+=("train_dataloader.batch_size=${BATCH_SIZE}")
  opts+=("$@")
  mkdir -p "${work}"
  if [[ "${LAUNCHER:-dist}" == "torchrun" ]]; then
    PYTHONNOUSERSITE=1 PYTHONPATH="${MMDET}:${PYTHONPATH:-}" \
      "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${GPUS}" --master_port="${PORT}" \
      tools/train.py "${cfg}" --launcher pytorch --work-dir "${work}" \
      --cfg-options "${opts[@]}"
  else
    PYTHON_BIN="${PYTHON}" bash tools/dist_train.sh "${cfg}" "${GPUS}" --work-dir "${work}" \
      --cfg-options "${opts[@]}"
  fi
}

test_one() {
  local cfg="$1" ckpt="$2" work="$3"; shift 3
  if [[ "${LAUNCHER:-dist}" == "torchrun" ]]; then
    PYTHONNOUSERSITE=1 PYTHONPATH="${MMDET}:${PYTHONPATH:-}" \
      "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${GPUS}" --master_port="${PORT}" \
      tools/test.py "${cfg}" "${ckpt}" --launcher pytorch --work-dir "${work}" \
      --cfg-options "$@"
  else
    PYTHON_BIN="${PYTHON}" bash tools/dist_test.sh "${cfg}" "${ckpt}" "${GPUS}" --work-dir "${work}" \
      --cfg-options "$@"
  fi
}

for STAGE in ${STAGES}; do
  case "${STAGE}" in
    baseline|ffcp|ffcp_chsd|full) ;;
    *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
  esac
  for SEED in ${SEEDS}; do
    for SHOT in ${SHOTS}; do
      CFG="$(config_for "${SHOT}")"
      [[ -f "${CFG}" ]] || { echo "missing config ${CFG}" >&2; exit 1; }
      WORK="$(work_for "${STAGE}" "${SEED}" "${SHOT}")"
      LOG="$(log_for "${STAGE}" "${SEED}" "${SHOT}")"
      DONE="${WORK}/.train_complete"
      mkdir -p "$(dirname "${LOG}")"

      if [[ "${STAGE}" == "baseline" ]]; then
        KNOWLEDGE=""
        EXTRA_TRAIN=("model.bbox_head.s2h_cfg.enabled=False")
        EXTRA_TEST=("model.bbox_head.s2h_cfg.enabled=False")
      else
        KNOWLEDGE="$(knowledge_for "${STAGE}" "${SEED}" "${SHOT}")"
        mkdir -p "$(dirname "${KNOWLEDGE}")"
        if [[ "${SKIP_BUILD}" != "1" && ( "${FORCE}" == "1" || ! -s "${KNOWLEDGE}" ) ]]; then
          if [[ -z "${CKPT}" || ! -s "${CKPT}" ]]; then
            echo "CKPT must point to a readable local Grounding DINO checkpoint" >&2
            exit 1
          fi
          echo "[knowledge] ${DATASET} ${STAGE} seed=${SEED} shot=${SHOT}" | tee -a "${LOG}"
          "${PYTHON}" tools/s2h/build_s2h_knowledge.py \
            --config "${CFG}" --checkpoint "${CKPT}" \
            --out "${KNOWLEDGE}" --device "${DEVICE:-cuda:0}" \
            --seed "${SEED}" --stage "${STAGE}" 2>&1 | tee -a "${LOG}"
        fi
        [[ -s "${KNOWLEDGE}" ]] || { echo "missing knowledge ${KNOWLEDGE}" >&2; exit 1; }
        EXTRA_TRAIN=(
          "model.bbox_head.s2h_cfg.enabled=True"
          "model.bbox_head.s2h_cfg.stage=${STAGE}"
          "model.bbox_head.s2h_cfg.knowledge_path=${KNOWLEDGE}"
        )
        EXTRA_TEST=("${EXTRA_TRAIN[@]}")
      fi

      if [[ "${FORCE}" == "1" || ! -s "${WORK}/latest.pth" || ! -f "${DONE}" ]]; then
        echo "[train] ${DATASET} ${STAGE} seed=${SEED} shot=${SHOT}" | tee -a "${LOG}"
        train_one "${CFG}" "${WORK}" "${EXTRA_TRAIN[@]}" 2>&1 | tee -a "${LOG}"
        touch "${DONE}"
      else
        echo "[train] skip existing ${WORK}/latest.pth" | tee -a "${LOG}"
      fi

      if [[ "${SKIP_TEST}" != "1" ]]; then
        if [[ "${CHECKPOINT_POLICY}" == "final" ]]; then
          BEST="$(${PYTHON} tools/s2h/select_best_checkpoint.py \
            --work-dir "${WORK}" --epoch "$(final_epoch_for)" \
            2>/dev/null || true)"
        else
          BEST="$(${PYTHON} tools/s2h/select_best_checkpoint.py \
            --work-dir "${WORK}" 2>/dev/null || true)"
        fi
        [[ -s "${BEST}" ]] || { echo "missing checkpoint ${BEST}" >&2; exit 1; }
        echo "[test] ${DATASET} ${STAGE} seed=${SEED} shot=${SHOT} ckpt=${BEST}" | tee -a "${LOG}"
        test_one "${CFG}" "${BEST}" "${WORK}" "${EXTRA_TEST[@]}" 2>&1 | tee -a "${LOG}"
      fi
    done
  done
done

echo "completed ${DATASET}: stages=${STAGES}; shots=${SHOTS}; seeds=${SEEDS}"

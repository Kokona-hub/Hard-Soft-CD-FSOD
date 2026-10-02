#!/usr/bin/env bash
# ============================================================================
# S2H-CD-FSOD end-to-end pipeline (Linux server + CUDA).
#
#   "Trust What You Transfer: Knowledge-Enhanced Hard-Soft Reasoning for
#    Cross-Domain Few-Shot Object Detection"
#   built on top of Domain-RAG's Grounding DINO reproduction recipe.
#
# Usage
# -----
#   bash tools/s2h/pipeline.sh env         # show versions / GPU count
#   bash tools/s2h/pipeline.sh download    # fetch the Swin-B checkpoint
#   bash tools/s2h/pipeline.sh check       # verify dataset layout
#   bash tools/s2h/pipeline.sh selfcheck   # module sanity checks (no data)
#   bash tools/s2h/pipeline.sh configs     # (re)generate the 18 configs
#   bash tools/s2h/pipeline.sh knowledge   # FFCP + CHSD -> cached knowledge
#   bash tools/s2h/pipeline.sh train       # fine-tune every dataset/shot
#   bash tools/s2h/pipeline.sh test        # evaluate every dataset/shot
#   bash tools/s2h/pipeline.sh baseline    # GroundingDINO baseline (S2H off)
#   bash tools/s2h/pipeline.sh baseline-test # evaluate every baseline run
#   bash tools/s2h/run_ablation.sh ArTaxOr # all 4 stages x 3 shots x 3 seeds
#   bash tools/s2h/pipeline.sh all         # check -> selfcheck -> configs
#                                          #   -> knowledge -> train
#
# Everything is overridable through environment variables, e.g.
#   DATASETS="ArTaxOr NEU-DET" SHOTS="1" GPUS=8 LAUNCHER=torchrun \
#       bash tools/s2h/pipeline.sh all
#
# Notes
# -----
# * LAUNCHER=dist (default) uses Domain-RAG's tools/dist_train.sh.
#   LAUNCHER=torchrun uses `torchrun` directly -- recommended on torch>=2.5,
#   where `python -m torch.distributed.launch` is deprecated.
# * Run this script from anywhere; it cd's into <repo>/mmdetection.
# ============================================================================
set -euo pipefail

# ------------------------------- configuration ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MMDET}"

# BERT weights for the Grounding DINO language model (see run_ablation.sh):
# an exported-but-empty value would otherwise reach
# `AutoTokenizer.from_pretrained('')` and fail.
_S2H_BERT_DEFAULT="/mnt/sdc/hzh/model_cache/bert-base-uncased"
if [ ! -d "${S2H_BERT_DIR:-}" ]; then
    if [ -d "${_S2H_BERT_DEFAULT}" ]; then
        export S2H_BERT_DIR="${_S2H_BERT_DEFAULT}"
    else
        unset S2H_BERT_DIR
    fi
fi
echo "[pipeline] S2H_BERT_DIR=${S2H_BERT_DIR:-<unset>}"

PY="${PY:-python}"
DATASETS="${DATASETS:-ArTaxOr clipart1k DIOR FISH NEU-DET UODD}"
SHOTS="${SHOTS:-1 5 10}"
GPUS="${GPUS:-4}"
PORT="${PORT:-29500}"
LAUNCHER="${LAUNCHER:-dist}"
DATA_ROOT="${DATA_ROOT:-data}"

KNOWLEDGE_DIR="${KNOWLEDGE_DIR:-work_dirs/s2h_knowledge}"
WORK_DIR="${WORK_DIR:-cat_work_dir/s2h}"
LOG_DIR="${LOG_DIR:-work_dirs/s2h_logs}"
CONFIG_DIR="${CONFIG_DIR:-configs/s2h_dino}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

# empty -> let the config `load_from` URL download the Swin-B checkpoint
CKPT="${CKPT:-}"
CKPT_URL="https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"

# ... but on the offline server prefer the copy already in checkpoints/
_S2H_CKPT_DEFAULT="${MMDET}/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
if [ -z "${CKPT}" ] && [ -s "${_S2H_CKPT_DEFAULT}" ]; then
    CKPT="${_S2H_CKPT_DEFAULT}"
    echo "[pipeline] CKPT -> ${CKPT}"
fi

FORCE="${FORCE:-0}"                              # FORCE=1 rebuild knowledge
PARALLEL_KNOWLEDGE="${PARALLEL_KNOWLEDGE:-1}"    # 1 GPU per (dataset, shot)

mkdir -p "${KNOWLEDGE_DIR}" "${WORK_DIR}" "${LOG_DIR}"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

config_path()    { echo "${CONFIG_DIR}/s2h_grounding_dino_swin-b_$1_$2shot.py"; }
knowledge_path() { echo "${KNOWLEDGE_DIR}/$1_$2shot.pth"; }
work_path()      { echo "${WORK_DIR}/$1_$2shot"; }

# ---------------------------------------------------------------- launchers --
launch_train() {   # $1=config  $2=work_dir  rest -> forwarded to train.py
  local cfg="$1" wd="$2"; shift 2
  if [ "${LAUNCHER}" = "torchrun" ]; then
    PYTHONNOUSERSITE=1 PYTHONPATH="${MMDET}:${PYTHONPATH:-}" \
      "${PY}" -m torch.distributed.run \
      --nproc_per_node="${GPUS}" --master_port="${PORT}" \
      tools/train.py "${cfg}" --launcher pytorch --work-dir "${wd}" "$@"
  else
    bash tools/dist_train.sh "${cfg}" "${GPUS}" --work-dir "${wd}" "$@"
  fi
}

launch_test() {    # $1=config  $2=checkpoint  rest -> forwarded to test.py
  local cfg="$1" ck="$2"; shift 2
  if [ "${LAUNCHER}" = "torchrun" ]; then
    PYTHONNOUSERSITE=1 PYTHONPATH="${MMDET}:${PYTHONPATH:-}" \
      "${PY}" -m torch.distributed.run \
      --nproc_per_node="${GPUS}" --master_port="${PORT}" \
      tools/test.py "${cfg}" "${ck}" --launcher pytorch "$@"
  else
    bash tools/dist_test.sh "${cfg}" "${ck}" "${GPUS}" "$@"
  fi
}

ckpt_args() { if [ -n "${CKPT}" ]; then printf -- '--checkpoint %s' "${CKPT}"; fi; }

# ----------------------------------------------------------------------------
# 0. environment
# ----------------------------------------------------------------------------
stage_env() {
  log "environment"
  echo "repo    : ${MMDET}"
  echo "python  : $(${PY} -c 'import sys; print(sys.executable)')"
  ${PY} - <<'PYEOF'
import torch, mmcv, mmengine, mmdet
print('torch   :', torch.__version__, '| cuda:', torch.version.cuda,
      '| gpus:', torch.cuda.device_count())
print('mmcv    :', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmdet   :', mmdet.__version__)
PYEOF
}

# ----------------------------------------------------------------------------
# 0b. checkpoints
# ----------------------------------------------------------------------------
stage_download() {
  log "downloading the Grounding DINO Swin-B checkpoint"
  mkdir -p "${CKPT_DIR}"
  local out="${CKPT_DIR}/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
  if [ ! -f "${out}" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget -c "${CKPT_URL}" -O "${out}"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -C - "${CKPT_URL}" -o "${out}"
    else
      echo "neither wget nor curl found; download manually:"
      echo "  ${CKPT_URL}"
      return 1
    fi
  fi
  echo "local checkpoint: ${MMDET}/${out}"
  echo "reuse it with:    CKPT=${MMDET}/${out}"
}

# ----------------------------------------------------------------------------
# 1. dataset layout check
# ----------------------------------------------------------------------------
stage_check() {
  log "verifying dataset layout under '${DATA_ROOT}'"
  local ok=1
  for ds in ${DATASETS}; do
    for shot in ${SHOTS}; do
      for f in "${DATA_ROOT}/${ds}/annotations/${shot}_shot.json" \
               "${DATA_ROOT}/${ds}/annotations/test.json" \
               "${DATA_ROOT}/${ds}/train" \
               "${DATA_ROOT}/${ds}/test"; do
        if [ -e "${f}" ]; then
          printf '  [ok]   %s\n' "${f}"
        else
          printf '  [MISS] %s\n' "${f}"; ok=0
        fi
      done
    done
  done
  if [ "${ok}" -eq 0 ]; then
    cat <<'MSG'

Dataset layout is incomplete. Required per dataset:
  data/<DS>/annotations/{1,5,10}_shot.json   few-shot support split
  data/<DS>/annotations/test.json            query set
  data/<DS>/train/                           support images
  data/<DS>/test/                            query images

See ./datasets/structure.md and mmdetection/README_mmlab.md.
MSG
    return 1
  fi
  log "dataset layout OK"
}

# ----------------------------------------------------------------------------
# 2. module self-check
# ----------------------------------------------------------------------------
stage_selfcheck() {
  log "S2H self-check"
  ${PY} tools/s2h/selfcheck.py
}

# ----------------------------------------------------------------------------
# 3. generate configs
# ----------------------------------------------------------------------------
stage_configs() {
  log "generating configs into ${CONFIG_DIR}"
  ${PY} tools/s2h/gen_s2h_configs.py \
      --datasets ${DATASETS} --shots ${SHOTS} --out-dir "${CONFIG_DIR}"
}

# ----------------------------------------------------------------------------
# 4. build the cached Hard-Soft knowledge (FFCP + CHSD)
# ----------------------------------------------------------------------------
stage_knowledge() {
  log "building support knowledge (FFCP + CHSD)"
  local i=0
  for ds in ${DATASETS}; do
    for shot in ${SHOTS}; do
      local out gpu cmd
      out="$(knowledge_path "${ds}" "${shot}")"
      if [ -f "${out}" ] && [ "${FORCE}" != "1" ]; then
        echo "  [skip] ${out} already exists (FORCE=1 to rebuild)"
        i=$((i + 1)); continue
      fi
      gpu=$((i % GPUS))
      cmd="${PY} tools/s2h/build_s2h_knowledge.py \
        --config $(config_path "${ds}" "${shot}") \
        --out ${out} --device cuda:${gpu} $(ckpt_args)"
      echo "  [gpu ${gpu}] ${ds} ${shot}-shot"
      if [ "${PARALLEL_KNOWLEDGE}" = "1" ]; then
        ( eval "${cmd}" >"${LOG_DIR}/knowledge_${ds}_${shot}shot.log" 2>&1 ) &
      else
        eval "${cmd}" 2>&1 | tee "${LOG_DIR}/knowledge_${ds}_${shot}shot.log"
      fi
      i=$((i + 1))
    done
  done
  if [ "${PARALLEL_KNOWLEDGE}" = "1" ]; then
    log "waiting for knowledge jobs to finish"
    wait
  fi
  ls -lh "${KNOWLEDGE_DIR}" || true
}

# ----------------------------------------------------------------------------
# 5. fine-tune
# ----------------------------------------------------------------------------
stage_train() {
  log "fine-tuning (S2H enabled)"
  for ds in ${DATASETS}; do
    for shot in ${SHOTS}; do
      local cfg wd kn
      cfg="$(config_path "${ds}" "${shot}")"
      wd="$(work_path "${ds}" "${shot}")"
      kn="$(knowledge_path "${ds}" "${shot}")"
      if [ ! -f "${kn}" ]; then
        echo "  !! missing knowledge ${kn}; run the 'knowledge' stage first"
        exit 1
      fi
      mkdir -p "${wd}"
      echo "  -> ${ds} ${shot}-shot  (work-dir ${wd})"
      local train_args=()
      if [ -n "${CKPT}" ]; then
        train_args+=(--cfg-options "load_from=${CKPT}")
      fi
      launch_train "${cfg}" "${wd}" "${train_args[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${ds}_${shot}shot.log"
    done
  done
}

# ----------------------------------------------------------------------------
# 6. evaluate
# ----------------------------------------------------------------------------
stage_test() {
  log "evaluating"
  for ds in ${DATASETS}; do
    for shot in ${SHOTS}; do
      local cfg wd ck
      cfg="$(config_path "${ds}" "${shot}")"
      wd="$(work_path "${ds}" "${shot}")"
      ck="${wd}/latest.pth"
      if [ ! -f "${ck}" ]; then
        echo "  [skip] ${ck} not found"; continue
      fi
      echo "  -> ${ds} ${shot}-shot"
      launch_test "${cfg}" "${ck}" --work-dir "${wd}" \
        2>&1 | tee "${LOG_DIR}/test_${ds}_${shot}shot.log"
    done
  done
}

# ----------------------------------------------------------------------------
# 7. baseline (S2H correction disabled -> plain GroundingDINO recipe)
# ----------------------------------------------------------------------------
stage_baseline() {
  log "baseline runs (S2H disabled)"
  for ds in ${DATASETS}; do
    for shot in ${SHOTS}; do
      local cfg wd
      cfg="$(config_path "${ds}" "${shot}")"
      wd="${WORK_DIR}_baseline/${ds}_${shot}shot"
      mkdir -p "${wd}"
      echo "  -> ${ds} ${shot}-shot (baseline)"
      local baseline_args=(--cfg-options model.bbox_head.s2h_cfg.enabled=False)
      if [ -n "${CKPT}" ]; then
        baseline_args+=("load_from=${CKPT}")
      fi
      launch_train "${cfg}" "${wd}" "${baseline_args[@]}" \
        2>&1 | tee "${LOG_DIR}/baseline_${ds}_${shot}shot.log"
    done
  done
}

# ----------------------------------------------------------------------------
# 8. evaluate the plain GroundingDINO baseline
# ----------------------------------------------------------------------------
stage_baseline_test() {
  log "evaluating baseline runs (S2H disabled)"
  for ds in ${DATASETS}; do
    for shot in ${SHOTS}; do
      local cfg wd ck
      cfg="$(config_path "${ds}" "${shot}")"
      wd="${WORK_DIR}_baseline/${ds}_${shot}shot"
      ck="${wd}/latest.pth"
      if [ ! -f "${ck}" ]; then
        echo "  [skip] ${ck} not found"; continue
      fi
      echo "  -> ${ds} ${shot}-shot (baseline)"
      launch_test "${cfg}" "${ck}" --work-dir "${wd}" \
        --cfg-options model.bbox_head.s2h_cfg.enabled=False \
        2>&1 | tee "${LOG_DIR}/baseline_test_${ds}_${shot}shot.log"
    done
  done
}

# ----------------------------------------------------------------------------
case "${1:-all}" in
  env)        stage_env ;;
  download)   stage_download ;;
  check)      stage_check ;;
  selfcheck)  stage_selfcheck ;;
  configs)    stage_configs ;;
  knowledge)  stage_knowledge ;;
  train)      stage_train ;;
  test)       stage_test ;;
  baseline)   stage_baseline ;;
  baseline-test) stage_baseline_test ;;
  all)
    stage_env
    stage_check
    stage_selfcheck
    stage_configs
    stage_knowledge
    stage_train
    ;;
  *)
    echo "unknown stage: $1"
    echo "available: env | download | check | selfcheck | configs |" \
         "knowledge | train | test | baseline | baseline-test | all"
    exit 1
    ;;
esac

log "done: ${1:-all}"

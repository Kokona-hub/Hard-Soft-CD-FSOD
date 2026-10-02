#!/usr/bin/env bash
# Copyright (c) S2H-CD-FSOD. All rights reserved.
#
# One-at-a-time hyper-parameter sweep for Figure 4 (5-shot Clipart1k, 3 seeds).
#
# Zero-touch guarantee
# -------------------
# * Every artifact is written under ${SWEEP_ROOT} / ${KNOW_SWEEP}
#   (defaults: cat_work_dir/sweep and work_dirs/s2h_knowledge/sweep).
# * The script aborts if either root resolves into an `ablation` directory, so
#   the runs behind Table 2 / Fig. 5 can never be overwritten.
# * tools/train.py and tools/test.py are used exactly as they are, with
#   `--cfg-options` overrides only.  No repository file is modified.
#
# Protocol per panel (see the paper's Section 4.5)
# ------------------------------------------------
#   g_max, theta        inference-time gate  -> rebuild nothing, retrain nothing
#   con_weight          training-time        -> retrain with different weight
#   ffcp_rank,          knowledge-build time -> rebuild the knowledge file and
#   hard_radius                                retrain with it
#
# Usage
# -----
#   bash tools/s2h/analysis/run_sweep.sh                 # the whole matrix
#   PARAMS="g_max theta" bash tools/s2h/analysis/run_sweep.sh
#   DRY_RUN=1 bash tools/s2h/analysis/run_sweep.sh       # print the plan only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MMDET}"

# BERT weights (an exported-but-empty S2H_BERT_DIR breaks AutoTokenizer)
_S2H_BERT_DEFAULT="/mnt/sdc/hzh/model_cache/bert-base-uncased"
if [ ! -d "${S2H_BERT_DIR:-}" ] && [ -d "${_S2H_BERT_DEFAULT}" ]; then
  export S2H_BERT_DIR="${_S2H_BERT_DEFAULT}"
fi
[ -n "${S2H_BERT_DIR:-}" ] || unset S2H_BERT_DIR

DATASET="${DATASET:-clipart1k}"
SHOT="${SHOT:-5}"
SEEDS="${SEEDS:-3407 3408 3409}"
PARAMS="${PARAMS:-g_max theta con_weight ffcp_rank hard_radius}"
SWEEP_ROOT="${SWEEP_ROOT:-cat_work_dir/sweep}"
KNOW_SWEEP="${KNOW_SWEEP:-work_dirs/s2h_knowledge/sweep}"
CONFIG_DIR="${CONFIG_DIR:-configs/s2h_dino}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
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
PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

# read-only inputs: the ablation run of the same dataset/shot/seed
ABL_ROOT="${ABL_ROOT:-cat_work_dir/ablation}"
ABL_KNOW="${ABL_KNOW:-work_dirs/s2h_knowledge/ablation}"
CKPT="${CKPT:-${MMDET}/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth}"

PYTHON="${PYTHON:-python}"
CFG="${CONFIG_DIR}/s2h_grounding_dino_swin-b_${DATASET}_${SHOT}shot.py"

guard_root() {
  case "$1" in
    *ablation*) echo "refusing to write into an ablation path: $1" >&2; exit 2 ;;
  esac
}
guard_root "${SWEEP_ROOT}"
guard_root "${KNOW_SWEEP}"

select_checkpoint() {
  local work_dir="$1"
  if [ "${CHECKPOINT_POLICY}" = "final" ]; then
    "${PYTHON}" tools/s2h/select_best_checkpoint.py --work-dir "${work_dir}" \
      --epoch "$(final_epoch_for)"
  else
    "${PYTHON}" tools/s2h/select_best_checkpoint.py --work-dir "${work_dir}"
  fi
}

values_for() {
  case "$1" in
    g_max)       echo "0.1 0.25 0.5 0.75 1.0" ;;
    theta)       echo "0.3 0.4 0.5 0.6 0.7" ;;
    con_weight)  echo "0.0 0.05 0.1 0.2" ;;
    ffcp_rank)   echo "1 2 4 8 16" ;;
    hard_radius) echo "0.25 0.5 1.0 2.0 4.0" ;;
    *) echo "unknown parameter: $1" >&2; exit 2 ;;
  esac
}

kind_for() {
  case "$1" in
    g_max|theta)                 echo "test" ;;
    con_weight)                  echo "train" ;;
    ffcp_rank|hard_radius)       echo "build" ;;
  esac
}

cfg_option_for() {   # $1 = parameter, $2 = value
  case "$1" in
    g_max|theta|con_weight) echo "model.bbox_head.s2h_cfg.$1=$2" ;;
    ffcp_rank|hard_radius)  echo "" ;;      # build-time only
  esac
}

build_flag_for() {   # $1 = parameter, $2 = value
  case "$1" in
    ffcp_rank)   echo "--ffcp-rank $2" ;;
    hard_radius) echo "--hard-radius $2" ;;
    *)           echo "" ;;
  esac
}

run() {              # run <logfile> <command...>
  local log="$1"; shift
  if [ "${DRY_RUN}" = "1" ]; then
    echo "    [dry-run] $*" | tee -a "${log}"
    return 0
  fi
  mkdir -p "$(dirname "${log}")"
  echo "[cmd] $*" | tee -a "${log}"
  PYTHONNOUSERSITE=1 "${PYTHON}" "$@" 2>&1 | tee -a "${log}"
}

[ -f "${CFG}" ] || { echo "missing config ${CFG}" >&2; exit 1; }
mkdir -p "${SWEEP_ROOT}" "${KNOW_SWEEP}"

PLAN_LOG="${SWEEP_ROOT}/${DATASET}/sweep_plan.log"
echo "[sweep] dataset=${DATASET} shot=${SHOT} seeds='${SEEDS}'" | tee -a "${PLAN_LOG}"
echo "[sweep] params='${PARAMS}'" | tee -a "${PLAN_LOG}"
echo "[sweep] write roots: ${SWEEP_ROOT} , ${KNOW_SWEEP}" | tee -a "${PLAN_LOG}"

for PARAM in ${PARAMS}; do
  VALUES="$(values_for "${PARAM}")"
  KIND="$(kind_for "${PARAM}")"
  echo "[sweep] ${PARAM} (${KIND}-time): ${VALUES}" | tee -a "${PLAN_LOG}"
  for VALUE in ${VALUES}; do
    for SEED in ${SEEDS}; do
      WORK="${SWEEP_ROOT}/${DATASET}/${PARAM}/${VALUE}/seed${SEED}/${SHOT}shot"
      TEST="${WORK}_test"
      KNOW="${KNOW_SWEEP}/${DATASET}/${PARAM}/${VALUE}/seed${SEED}/${SHOT}shot.pth"
      LOG="${WORK}/sweep.log"
      DONE="${WORK}/.sweep_complete"
      mkdir -p "${WORK}"
      echo "[sweep] ${PARAM}=${VALUE} seed=${SEED} -> ${WORK}" | tee -a "${PLAN_LOG}"

      if [ "${KIND}" = "test" ]; then
        # reuse the trained `full` model and the cached ablation knowledge:
        # only the inference-time gate changes
        ABL_CKPT="$(select_checkpoint \
          "${ABL_ROOT}/${DATASET}/full/seed${SEED}/${SHOT}shot" \
          2>/dev/null || true)"
        ABL_KNOW_FILE="${ABL_KNOW}/${DATASET}/full/seed${SEED}/${SHOT}shot.pth"
        if [ -z "${ABL_CKPT}" ] || [ ! -s "${ABL_KNOW_FILE}" ]; then
          echo "[skip] ${DATASET} full seed${SEED}: missing ablation checkpoint" \
            "or knowledge (looked in ${ABL_ROOT} and ${ABL_KNOW})" | tee -a "${PLAN_LOG}"
          continue
        fi
        run "${LOG}" tools/test.py "${CFG}" "${ABL_CKPT}" \
          --work-dir "${TEST}" \
          --cfg-options "model.bbox_head.s2h_cfg.knowledge_path=${ABL_KNOW_FILE}" \
                        "$(cfg_option_for "${PARAM}" "${VALUE}")"
      else
        # A training-time sweep (con_weight) must differ in exactly one
        # quantity, so it reuses the knowledge file of the corresponding
        # ablation run (read-only); a knowledge-build sweep (ffcp_rank /
        # hard_radius) uses the file it builds below.
        if [ "${KIND}" = "build" ]; then
          TARGET_KNOW="${KNOW}"
        else
          TARGET_KNOW="${ABL_KNOW}/${DATASET}/full/seed${SEED}/${SHOT}shot.pth"
          if [ "${DRY_RUN}" != "1" ] && [ ! -s "${TARGET_KNOW}" ]; then
            echo "[skip] ${DATASET} full seed${SEED}: training-time sweep" \
              "reuses the ablation knowledge, missing ${TARGET_KNOW}" \
              | tee -a "${PLAN_LOG}"
            continue
          fi
        fi
        OPTIONS=("randomness.seed=${SEED}"
                 "train_dataloader.batch_size=${BATCH_SIZE}"
                 "model.bbox_head.s2h_cfg.enabled=True"
                 "model.bbox_head.s2h_cfg.stage=full"
                 "model.bbox_head.s2h_cfg.knowledge_path=${TARGET_KNOW}")
        # The main corrected protocol keeps Hard-Soft out of the fine-tuning
        # distribution.  lambda_con is explicitly a training-time Figure-4
        # factor, however; enable injection only for this isolated panel so
        # that its sweep is measurable without changing the main mAP runs.
        if [ "${PARAM}" = "con_weight" ]; then
          OPTIONS+=("model.bbox_head.s2h_cfg.train_injection=True")
        fi
        if [ -n "${CKPT}" ] && [ -s "${CKPT}" ]; then
          OPTIONS+=("load_from=${CKPT}")
        fi
        if [ "${KIND}" = "build" ]; then
          if [ "${FORCE}" = "1" ] || [ ! -s "${KNOW}" ]; then
            run "${LOG}" tools/s2h/build_s2h_knowledge.py \
              --config "${CFG}" --checkpoint "${CKPT}" --out "${KNOW}" \
              --device "${DEVICE}" --seed "${SEED}" --stage full \
              $(build_flag_for "${PARAM}" "${VALUE}")
          fi
          [ "${DRY_RUN}" = "1" ] || [ -s "${KNOW}" ] || {
            echo "missing knowledge ${KNOW}" >&2; exit 1; }
        fi
        # build-time parameters have no training-time counterpart, so only
        # append a `--cfg-options` entry when there is one (an empty string
        # would be passed on as a bogus argument)
        EXTRA_OPTION="$(cfg_option_for "${PARAM}" "${VALUE}")"
        if [ -n "${EXTRA_OPTION}" ]; then
          OPTIONS+=("${EXTRA_OPTION}")
        fi
        if [ "${FORCE}" = "1" ] || [ ! -f "${DONE}" ]; then
          run "${LOG}" tools/train.py "${CFG}" --work-dir "${WORK}" \
            --cfg-options "${OPTIONS[@]}"
          [ "${DRY_RUN}" = "1" ] || touch "${DONE}"
        fi
        BEST="$(select_checkpoint "${WORK}" 2>/dev/null || true)"
        if [ "${DRY_RUN}" = "1" ] || [ -s "${BEST}" ]; then
          run "${LOG}" tools/test.py "${CFG}" "${BEST}" --work-dir "${TEST}" \
            --cfg-options "${OPTIONS[@]}"
        fi
      fi
    done
  done
done

echo "[sweep] done. summarize with:"
echo "  python tools/s2h/analysis/summarize_sweep.py --root ${SWEEP_ROOT} \\"
echo "      --dataset ${DATASET} --params ${PARAMS} --seeds ${SEEDS}"

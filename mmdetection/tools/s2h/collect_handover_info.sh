#!/usr/bin/env bash
# ============================================================================
# S2H-CD-FSOD 交接信息采集脚本
#   在服务器上执行： bash tools/s2h/collect_handover_info.sh | tee handover_info.txt
#   输出即为一套完整的交接档案（数据集、权重、文本编码器、环境、评估口径）
# ============================================================================
set -u

hr() { printf '\n%s\n' "============================================================"; }

hr; echo "[0] 执行环境"; hr
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "pwd       : $(pwd)"
echo "conda env : ${CONDA_PREFIX:-<none>}"
echo "user      : $(whoami)"

hr; echo "[1] 代码仓库与数据入口"; hr
ls -ld . data checkpoints work_dirs cat_work_dir 2>/dev/null
echo
echo "-- data 软链接实际指向:"
readlink -f data 2>/dev/null || echo "  (data 不是软链接或不存在)"

hr; echo "[2] 训练用数据集布局（mmdetection/data/<DS>）"; hr
for DS in ArTaxOr clipart1k DIOR FISH NEU-DET UODD; do
  printf '%-10s ' "$DS"
  if [ -d "data/$DS" ]; then
    printf 'ann=%s ' "$(ls data/$DS/annotations/*.json 2>/dev/null | wc -l)张标注"
    printf 'train=%s ' "$(ls data/$DS/train 2>/dev/null | wc -l)项"
    printf 'test=%s\n' "$(ls data/$DS/test 2>/dev/null | wc -l)项"
  else
    echo "[MISSING]"
  fi
done

hr; echo "[3] 各 shot 支撑集规模（classes / images / annotations）"; hr
python - <<'PY'
import json, glob, os
for shot in (1, 5, 10):
    print(f'-- {shot}-shot')
    files = sorted(glob.glob(f'data/*/annotations/{shot}_shot.json'))
    if not files:
        print('   (no file)')
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f'   {f}: {e}'); continue
        ds = f.split('/')[1]
        proto = d.get('info', {}).get('s2h_split_protocol', 'official/unknown')
        print(f'   {ds:10s} classes={len(d["categories"]):3d} '
              f'imgs={len(d["images"]):5d} anns={len(d["annotations"]):6d}  [{proto}]')
    print()
PY

hr; echo "[4] 原始数据源根目录（prepare_s2h_data.py --source-root）"; hr
echo "-- 搜索包含 raw/ derived/ DIOR/coco_annotations 的目录:"
find /mnt/sdc/hzh -maxdepth 5 -type d \( -name derived -o -name coco_annotations \) 2>/dev/null
echo
echo "-- 搜索原始数据集目录:"
find /mnt/sdc/hzh -maxdepth 5 -type d \( -iname "*artaxor*" -o -iname "*deepfish*" \
     -o -iname "*uodd*" -o -iname "*clipart*" -o -iname "NEU-DET" -o -iname "DIOR" \) 2>/dev/null | head -20

hr; echo "[5] Grounding DINO 权重"; hr
for p in \
  "checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" \
  "/mnt/sdc/hzh/Domain-RAG-main/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" \
  "/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" ; do
  if [ -e "$p" ]; then
    printf '[OK]   %s\n' "$p"
    ls -lL "$p" 2>/dev/null | sed 's/^/       /'
    [ -L "$p" ] && echo "       -> $(readlink -f "$p")"
  else
    printf '[MISS] %s\n' "$p"
  fi
done

hr; echo "[6] 本地文本编码器（BERT）"; hr
echo "S2H_BERT_DIR = ${S2H_BERT_DIR:-<unset>}"
for d in "${S2H_BERT_DIR:-}" \
         "/mnt/sdc/hzh/model_cache/bert-base-uncased" \
         "$HOME/.cache/huggingface/hub/models--bert-base-uncased"; do
  [ -n "$d" ] || continue
  if [ -d "$d" ]; then
    echo "[OK]   $d"
    ls "$d" 2>/dev/null | head -8 | sed 's/^/       /'
  else
    echo "[MISS] $d"
  fi
done
echo
echo "-- 离线相关开关:"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-<unset>}  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}  HF_HOME=${HF_HOME:-<unset>}"

hr; echo "[7] 软件环境版本"; hr
python - <<'PY'
import sys
try:
    import torch
    print('python  :', sys.version.split()[0])
    print('torch   :', torch.__version__, '| cuda:', torch.version.cuda,
          '| gpus:', torch.cuda.device_count())
except Exception as e:
    print('torch   : ERROR', e)
for name in ('mmcv', 'mmengine', 'mmdet'):
    try:
        m = __import__(name)
        print(f'{name:8s}:', getattr(m, '__version__', '?'))
    except Exception as e:
        print(f'{name:8s}: ERROR', e)
PY
echo
echo "-- pip 关键包:"
python -m pip list 2>/dev/null | grep -iE "^(torch|torchvision|mmcv|mmengine|mmdet|transformers|numpy|opencv)" || true

hr; echo "[8] GPU 硬件"; hr
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null \
  || echo "nvidia-smi 不可用"
echo
echo "-- 当前占用:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null | head -20 || true

hr; echo "[9] 验证集 / 测试集口径（关键）"; hr
CFG="configs/s2h_dino/s2h_grounding_dino_swin-b_ArTaxOr_1shot.py"
if [ -f "$CFG" ]; then
  echo "-- $CFG 中与评估相关的设置:"
  grep -nE "ann_file|data_prefix|save_best|val_interval|max_keep_ckpts|max_epochs =" "$CFG" \
    | sed 's/^/   /'
else
  echo "[MISS] $CFG"
fi

hr; echo "[10] 训练/测试启动命令模板"; hr
cat <<'MSG'
# 续跑单个数据集（一站式 train + test）
CUDA_VISIBLE_DEVICES=3 GPUS=1 BATCH_SIZE=2 DEVICE=cuda:0 PORT=29503 \
SHOTS="1 5 10" STAGES="baseline ffcp ffcp_chsd full" SEEDS="3407" \
PYTHONNOUSERSITE=1 \
S2H_BERT_DIR="/mnt/sdc/hzh/model_cache/bert-base-uncased" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
bash tools/s2h/run_ablation.sh <DATASET>

# 仅训练不测试：追加 SKIP_TEST=1
# 强制重跑：   追加 FORCE=1
# 多卡：       CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 BATCH_SIZE=1
MSG

hr; echo "[11] 已有产物概览"; hr
echo "-- 知识缓存:"; ls -1 work_dirs/s2h_knowledge/ablation/*/*/seed*/*.pth 2>/dev/null | wc -l | sed 's/^/   files: /'
echo "-- 训练 run 完成数:"; find cat_work_dir/ablation -name ".train_complete" 2>/dev/null | wc -l | sed 's/^/   done: /'
echo "-- 结果汇总表:"; ls -lh work_dirs/*summary*.csv work_dirs/*retest*.tsv 2>/dev/null || echo "   (无)"

hr; echo "采集完成。请把本文件随交接材料一并提交。"; hr

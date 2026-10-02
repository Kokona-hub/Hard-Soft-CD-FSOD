#!/usr/bin/env bash
# ============================================================================
# 定位 prepare_s2h_data.py 所需的 SOURCE_ROOT（原始 COCO 标注 + 原始图片）
#
#   用法： cd /mnt/sdc/hzh/Domain-RAG-main-2/mmdetection
#          bash tools/s2h/find_source_root.sh 2>&1 | tee source_root_probe.txt
#
# 依次尝试 6 条线索，按可靠性从高到低排列。
# ============================================================================
set -u
cd "$(dirname "$0")/../.." || exit 1
echo "工作目录: $(pwd)"
echo

echo "=============================================================="
echo "[A] 从现有软链接反查（最可靠：脚本用绝对路径建链）"
echo "=============================================================="
CANDIDATES=""
for DS in ArTaxOr clipart1k DIOR FISH NEU-DET UODD; do
  printf -- '--- %s\n' "$DS"
  for sub in train test; do
    p="data/$DS/$sub"
    if [ -L "$p" ]; then
      t=$(readlink "$p")
      if [ -d "$t" ]; then st="[存在]"; else st="[悬空/源已丢失]"; fi
      printf '  %-5s -> %s   %s\n' "$sub" "$t" "$st"
      [ -n "$t" ] && CANDIDATES="$CANDIDATES $(dirname "$t")"
    elif [ -d "$p" ]; then
      printf '  %-5s  (实际目录，非软链接) 条目数=%s\n' "$sub" "$(ls "$p" 2>/dev/null | wc -l)"
    else
      printf '  %-5s  [缺失]\n' "$sub"
    fi
  done
done

echo
echo "-- 候选 SOURCE_ROOT（对 train 目标取父目录后去重）:"
for c in $(echo "$CANDIDATES" | tr ' ' '\n' | sort -u); do
  [ -n "$c" ] || continue
  hit=""
  [ -d "$c/derived" ] && hit="$hit derived"
  [ -d "$c/raw" ] && hit="$hit raw"
  [ -d "$c/DIOR" ] && hit="$hit DIOR"
  if [ -n "$hit" ]; then
    echo "  ★ $c   → 命中:$hit"
  else
    echo "    $c   → 无 derived/raw/DIOR"
  fi
done

echo
echo "=============================================================="
echo "[B] 关键标注文件逐个确认（需先确定 SOURCE_ROOT）"
echo "=============================================================="
for c in $(echo "$CANDIDATES" | tr ' ' '\n' | sort -u); do
  [ -n "$c" ] || continue
  [ -d "$c/derived" ] || [ -d "$c/DIOR" ] || continue
  echo "-- SOURCE_ROOT = $c"
  for f in \
    "$c/derived/artaxor_7class/baseline/train.json" \
    "$c/derived/artaxor_7class/baseline/val.json" \
    "$c/derived/deepfish/baseline/train.json" \
    "$c/derived/deepfish/baseline/val.json" \
    "$c/derived/uodd/baseline/train.json" \
    "$c/derived/uodd/baseline/val.json" \
    "$c/DIOR/coco_annotations/trainval.json" \
    "$c/DIOR/coco_annotations/test.json" \
    "$c/raw/NEU-DET/annotations/train.json" \
    "$c/raw/NEU-DET/annotations/val.json" ; do
    [ -f "$f" ] && echo "   [OK]   $f" || echo "   [MISS] $f"
  done
  echo "   -- 图片目录:"
  for d in "$c/ArTaxOr" "$c/DeepFish/Segmentation/images" \
           "$c/Underwater-object-detection-dataset-main/imgs" \
           "$c/DIOR/JPEGImages-trainval" "$c/raw/NEU-DET/train/images" ; do
    [ -d "$d" ] && echo "   [OK]   $d  ($(ls "$d" 2>/dev/null | wc -l) 项)" || echo "   [MISS] $d"
  done
done

echo
echo "=============================================================="
echo "[C] data/ 内部是否有嵌套保存的原始标注"
echo "=============================================================="
find data -maxdepth 4 -type d -name annotations 2>/dev/null | while read -r d; do
  echo "  $d  ($(ls "$d" 2>/dev/null | wc -l) 个文件)"
  ls "$d" 2>/dev/null | head -6 | sed 's/^/     /'
done

echo
echo "=============================================================="
echo "[D] 全盘搜索 SOURCE_ROOT 特征目录 / 文件"
echo "=============================================================="
for name in derived coco_annotations artaxor_7class deepfish uodd JPEGImages-trainval; do
  echo "-- 目录: $name"
  find /mnt/sdc -maxdepth 7 -type d -name "$name" 2>/dev/null | head -5 | sed 's/^/   /'
done
echo "-- 关键标注文件:"
find /mnt/sdc -maxdepth 8 -type f \
  \( -name "trainval.json" -o -name "*artaxor*train.json" \
     -o -name "*deepfish*train.json" -o -name "*uodd*train.json" \) 2>/dev/null \
  | head -10 | sed 's/^/   /'

echo
echo "=============================================================="
echo "[E] 备份目录 / 压缩包"
echo "=============================================================="
find /mnt/sdc/hzh -maxdepth 4 \
  \( -iname "*.tar" -o -iname "*.tar.gz" -o -iname "*.tgz" -o -iname "*.zip" -o -iname "*.7z" \) \
  2>/dev/null | head -20 | sed 's/^/  /'
echo "-- 名字含 backup/备份/原始 的目录:"
find /mnt/sdc/hzh -maxdepth 4 -type d \
  \( -iname "*backup*" -o -iname "*bak*" -o -iname "*备份*" -o -iname "*原始*" -o -iname "*original*" \) \
  2>/dev/null | head -20 | sed 's/^/  /'

echo
echo "=============================================================="
echo "[F] shell 历史 / 脚本记录中的调用"
echo "=============================================================="
grep -n "prepare_s2h_data\|source-root\|source_root" ~/.bash_history 2>/dev/null | tail -20 | sed 's/^/  /'
echo "-- 仓库内搜索结果:"
grep -rn "prepare_s2h_data" /mnt/sdc/hzh/*/mmdetection/ 2>/dev/null \
  | grep -v "Binary\|\.pyc" | head -10 | sed 's/^/  /'

echo
echo "=============================================================="
echo "[G] 上游数据来源（仓库 scp 脚本记录）"
echo "=============================================================="
echo "  scp_mmdetection_data.sh:"
echo "    TARGET_SERVER = liyu@10.176.42.44"
echo "    TARGET_DIR    = /mnt/data/liyu/mmdetection_data"
echo "  folders_to_transfer.txt 中记录的目录（含 train/annotations 嵌套）:"
sed 's/^/    /' folders_to_transfer.txt 2>/dev/null | head -25
echo
echo "  → 若本地确实丢失，可从上述上游重新拉取对应数据集。"

echo
echo "=============================================================="
echo "排查完成。请把 [A]/[B] 的输出作为 SOURCE_ROOT 结论。"
echo "=============================================================="

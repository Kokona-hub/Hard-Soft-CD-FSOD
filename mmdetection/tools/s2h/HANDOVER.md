# S2H-CD-FSOD 服务器运行交接说明

**项目**：*Trust What You Transfer: Knowledge-Enhanced Hard-Soft Reasoning for Cross-Domain Few-Shot Object Detection*（CVPR 2026 投稿）
**方法代号**：S2H-CD-FSOD（FFCP + CHSD + ATAR）
**整理日期**：2026-09-29
**信息状态**：✅ 已确认 ｜ ⚠️ 待现场核对 ｜ ❗ 需决策

> 配套采集脚本：`bash tools/s2h/collect_handover_info.sh | tee handover_info.txt`
> 在服务器上执行一次即可输出本文所有"⚠️ 待核对"项的实际值。

---

## 一、服务器与代码仓库

| 项 | 值 | 状态 |
|---|---|---|
| 主机 | `leju-Rack-Server` | ✅ |
| 登录用户 | `hhz` | ✅ |
| **工作仓库（训练用）** | `/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection` | ✅ |
| 参考仓库（含数据与权重原件） | `/mnt/sdc/hzh/Domain-RAG-main/mmdetection` | ✅ |
| Conda 环境路径 | `/mnt/sdc/conda_envs/s2h_cd_fsod` | ✅ |
| GPU 数量 | 7 张（`torch.cuda.device_count() == 7`） | ✅ |
| GPU 型号 / 驱动版本 | — | ⚠️ `nvidia-smi` 确认 |

**关键提示**：两个仓库并存，`-2` 是训练工作区，`Domain-RAG-main` 是数据与权重的原件所在。二者通过软链接关联：

```
/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection/data
    -> /mnt/sdc/hzh/Domain-RAG-main/mmdetection/data
/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
    -> /mnt/sdc/hzh/Domain-RAG-main/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
```

> ⚠️ `Domain-RAG-main` 属于历史仓库，**不要删除**；若迁移服务器，请连同它一起迁移，否则 `-2` 的数据与权重链接会全部失效。

---

## 二、数据集

### 2.1 训练用数据根目录

```
/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection/data      （软链接）
  └─ 实际位于 /mnt/sdc/hzh/Domain-RAG-main/mmdetection/data
```

每个数据集的标准布局（MMDetection CocoDataset 约定）：

```
data/<DATASET>/
├─ annotations/
│   ├─ 1_shot.json      # 1-shot 支持集
│   ├─ 5_shot.json      # 5-shot 支持集
│   ├─ 10_shot.json     # 10-shot 支持集
│   └─ test.json        # 查询集（同时充当 val 与 test，见第五节）
├─ train/               # 支持集图片
└─ test/                # 查询集图片
```

数据集清单（与配置中的 `data_root` 一一对应）：

| 数据集 | 配置中的 `data_root` | 类别数 | 训练 epoch |
|---|---|---|---|
| ArTaxOr | `data/ArTaxOr/` | 7 | 30 |
| clipart1k | `data/clipart1k/` | 20 | **5** |
| DIOR | `data/DIOR/` | 20 | 30 |
| FISH（DeepFish） | `data/FISH/` | 1 | **5** |
| NEU-DET | `data/NEU-DET/` | 6 | 30 |
| UODD | `data/UODD/` | 3 | 30 |

### 2.2 原始数据源根目录（生成 split 用）

**✅ 已定位**：`/big-disk/hzh/S2H-CD-FSOD/data`

> ⚠️ 该路径位于 **`/big-disk` 挂载点，不在 `/mnt/sdc` 下**；全盘搜索时不要遗漏这个挂载点。

定位方式（由软链接反查）：`readlink data/ArTaxOr/train` → `/big-disk/hzh/S2H-CD-FSOD/data/ArTaxOr`，
其父目录即 SOURCE_ROOT，且其下 `derived/`、`raw/`、`DIOR/` 三个子目录均存在。
经核对，6 个数据集的 `train`/`test` 软链接目标**全部存在，无悬空链接**。

```bash
python tools/s2h/prepare_s2h_data.py \
  --source-root /big-disk/hzh/S2H-CD-FSOD/data \
  --output-root data \
  --datasets ArTaxOr FISH UODD DIOR NEU-DET \
  --shots 1 5 10 --seed 20250919
```

`prepare_s2h_data.py` 内部引用的原始标注相对路径如下（据此可在服务器上反查 `<SOURCE_ROOT>`）：

| 数据集 | 原始标注 | 原始图片目录 |
|---|---|---|
| ArTaxOr | `derived/artaxor_7class/baseline/train.json` / `val.json` | `ArTaxOr` |
| FISH | `derived/deepfish/baseline/train.json` / `val.json` | `DeepFish/Segmentation/images` |
| UODD | `derived/uodd/baseline/train.json` / `val.json` | `Underwater-object-detection-dataset-main/imgs` |
| DIOR | `DIOR/coco_annotations/trainval.json` / `test.json` | `DIOR/JPEGImages-trainval`、`DIOR/JPEGImages-test` |
| NEU-DET | `raw/NEU-DET/annotations/train.json` / `val.json` | `raw/NEU-DET/train/images`、`raw/NEU-DET/val/images` |

反查命令（采集脚本第 [4] 节已包含）：

```bash
find /mnt/sdc/hzh -maxdepth 5 -type d \( -name derived -o -name coco_annotations \) 2>/dev/null
```

> 📌 **Clipart1k 例外**：其支持集使用原始划分（`data/clipart1k/annotations/*_shot.json`，文件时间为 2024-07），未经 `prepare_s2h_data.py` 重新切分。

### 2.3 ⚠️ 支持集划分协议（交接重点关注）

当前 6 个域中有 5 个使用 `prepare_s2h_data.py` 生成的 **image-level** 划分（标记为
`nested class-balanced image-shot; full labels retained`），与论文中"除 DIOR 外均为 K-instance"的描述**不一致**：

| 数据集 | 1-shot 实际标注数 | 协议要求 | 是否等价 |
|---|---|---|---|
| ArTaxOr | 7 | 7（instance） | ✅ 等价 |
| FISH | 1 | 1（instance） | ✅ 等价 |
| Clipart1k | 20 | 20（instance） | ✅（原始划分） |
| **NEU-DET** | **15** | 6（instance） | ❌ 多给 |
| **UODD** | **19** | 3（instance） | ❌ 多给 |
| DIOR | 241 | image-level | ✅ 符合声明 |

**交接影响**：NEU-DET / UODD 的绝对数值不可与论文 Table 2 的 GroundingDINO† 行直接比较；
DeepFish 1-shot 出现 70.7（基线 36.4）的异常增益，高度怀疑与此划分有关，**投稿前必须查清**。

---

## 三、Grounding DINO 权重

| 项 | 路径 | 状态 |
|---|---|---|
| 权重原件 | `/mnt/sdc/hzh/Domain-RAG-main/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth` | ✅ 已就绪（893 MB，2026-09-20） |
| 工作区引用 | `/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth`（软链接） | ✅ |
| 官方下载地址 | `https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth` | — |

- 该权重为 **Swin-B + COGCOOR** 版本，与论文 Implementation Details 中所述 `Grounding DINO†` 配置一致；
- 训练脚本通过 `load_from` 加载；`run_ablation.sh` 未显式传 `CKPT` 时会自动兜底到上述软链接路径；
- 备选下载：`bash tools/s2h/pipeline.sh download`（需联网）。

---

## 四、本地文本编码器（BERT）资源

Grounding DINO 的语言分支使用 **bert-base-uncased**，通过环境变量 `S2H_BERT_DIR` 指向本地目录，以支持全离线运行。

| 项 | 值 | 状态 |
|---|---|---|
| 本地 BERT 目录 | `/mnt/sdc/hzh/model_cache/bert-base-uncased` | ✅ **已就绪**（实测被脚本自动识别，日志打印 `[ablation] S2H_BERT_DIR=/mnt/sdc/hzh/model_cache/bert-base-uncased`） |
| 环境变量 | `S2H_BERT_DIR` | ✅ 已在启动命令中显式传入 |
| 离线开关 | `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1` | ✅ 启动命令中已设置 |

**注意事项（已在脚本中做过防御，但交接时需知）**：

1. **不要传空的 `S2H_BERT_DIR`**。形如 `S2H_BERT_DIR="$S2H_BERT_DIR"`（外层未定义）会把空串传给 Python，导致
   `AutoTokenizer.from_pretrained('')` 报 `HFValidationError`；
2. `run_ablation.sh` 的兜底顺序：若 `S2H_BERT_DIR` 无效 → 尝试 `/mnt/sdc/hzh/model_cache/bert-base-uncased` → 仍无效则 `unset`；
3. **离线模式下若两者都不可用，脚本会 fail-fast 并给出明确提示**，不会静默联网。

---

## 五、软件环境版本

| 组件 | 版本 | 状态 |
|---|---|---|
| Python | 3.10.21 | ✅ |
| **PyTorch** | **2.0.0 + CUDA 11.7** | ✅ |
| **MMCV** | **2.0.0** | ✅ |
| **MMEngine** | **0.10.7** | ✅ |
| **MMDetection** | **3.3.0** | ✅ |
| Conda 环境 | `/mnt/sdc/conda_envs/s2h_cd_fsod` | ✅ |

> ⚠️ 这是**训练侧**环境，与仓库根目录 `requirements.txt`（torch 2.7 / diffusers 0.33，属于 Domain-RAG 的**数据生成侧**）**不是同一套**，交接时勿混用。
>
> 精确版本核对请执行采集脚本第 [7] 节。

**必须遵守的环境纪律**：

```bash
export PYTHONNOUSERSITE=1     # 否则用户目录的 PyTorch 2.8/CUDA 12.8 会覆盖 conda 环境，
                              # 触发 MMCV "_ext undefined symbol"
```

> 该变量在 `dist_train.sh` / `dist_test.sh` 内部已有 `:-1` 兜底，但 `run_ablation.sh` 中**知识构建那一步**依赖外部导出，建议始终在启动命令前缀里显式写上。

---

## 六、启动命令

### 6.1 标准续跑命令（单个数据集，一站式 train + test）

```bash
cd /mnt/sdc/hzh/Domain-RAG-main-2/mmdetection

CUDA_VISIBLE_DEVICES=3 GPUS=1 BATCH_SIZE=2 DEVICE=cuda:0 PORT=29503 \
SHOTS="1 5 10" \
STAGES="baseline ffcp ffcp_chsd full" \
SEEDS="3407" \
PYTHONNOUSERSITE=1 \
S2H_BERT_DIR="/mnt/sdc/hzh/model_cache/bert-base-uncased" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
bash tools/s2h/run_ablation.sh <DATASET>
```

`<DATASET>` 取值：`ArTaxOr` `clipart1k` `DIOR` `FISH` `NEU-DET` `UODD`

### 6.2 常用开关

| 变量 | 作用 |
|---|---|
| `SKIP_TEST=1` | 只训练、不测试（与独立重测脚本配合时使用） |
| `FORCE=1` | 忽略已有结果，强制重训 / 重建知识 |
| `SKIP_BUILD=1` | 复用已缓存的知识文件，不重建 FFCP/CHSD |
| `GPUS=4` + `CUDA_VISIBLE_DEVICES=0,1,2,3` | 多卡并行（此时 `BATCH_SIZE` 为**单卡** batch） |
| `RANDOMNESS` 见 `SEEDS` | 训练随机种子，论文最终需 3407/3408/3409 三个 |

### 6.3 目录约定

| 内容 | 路径 |
|---|---|
| 知识缓存 | `work_dirs/s2h_knowledge/ablation/<DS>/<stage>/seed<seed>/<shot>shot.pth` |
| 训练工作目录 | `cat_work_dir/ablation/<DS>/<stage>/seed<seed>/<shot>shot/` |
| 脚本内层日志 | `work_dirs/ablation_logs/<DS>/<stage>/seed<seed>/<shot>shot.log` |
| 外层汇总日志 | `work_dirs/ablation_logs/<DS>_1seed.log` |

---

## 七、验证集口径（❗ 交接重点）

### 7.1 当前事实：**没有独立验证集**

配置中 `val_dataloader` 与 `test_dataloader` 指向**同一份数据**——`annotations/test.json` + `test/`：

```
val_dataloader  : ann_file='annotations/test.json', data_prefix=dict(img='test/')
test_dataloader : ann_file='annotations/test.json', data_prefix=dict(img='test/')
val_evaluator   : ann_file = data_root + 'annotations/test.json'
test_evaluator  : ann_file = data_root + 'annotations/test.json'
```

这是 CD-FSOD 协议的常规做法（目标域只有"支持集"和"查询集"两拨数据，无额外验证集）。
因此日志中的 `Epoch(val) [...]` **就是在查询集（测试集）上的评估**。

### 7.2 已确定的最终评估口径：**固定最终 epoch**

每个 epoch 仍可执行评估以记录曲线，但不再用 query/test mAP 选择
checkpoint。`run_ablation.sh` 和 `analysis/run_sweep.sh` 默认使用
`CHECKPOINT_POLICY=final`，取最终 epoch（Clipart1k/FISH 为 5，其余为
30）。这与复现的 Domain-RAG `latest.pth` 口径一致；`best` 仅保留给诊断，
不用于论文表格。

### 7.3 固定最终 epoch 的代码状态

已完成以下三处修改：

**① 配置层：关闭按最优保存**（`configs/s2h_dino/*.py` 由生成器统一生成）

```python
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best=None, max_keep_ckpts=3))
```

**② 训练脚本层：让 `run_ablation.sh` 取固定 epoch 的 checkpoint**

将 `BEST` 的选择逻辑改为按 `max_epochs` 取文件，例如固定取 `epoch_30.pth`（5-epoch 数据集为 `epoch_5.pth`），
而不是 `best_coco_bbox_mAP_epoch_*.pth`。

**③ 评估层：所有方法使用同一个固定 epoch**，论文中注明"所有方法均报告最终 epoch 结果"。

**④ 工作量影响**：旧结果仍需全部重测（训练可复用时仅重跑 test）；协议修复后的新 split 则必须重新训练。

### 7.4 交接时必须明确的一项

| 问题 | 当前状态 |
|---|---|
| 是否有独立验证集？ | ❌ 没有，`val` 与 `test` 共用 `annotations/test.json` |
| 按什么选 checkpoint？ | **固定最终 epoch**（`CHECKPOINT_POLICY=final`） |
| 是否按固定最终 epoch？ | ✅ **是**；`best` 仅用于诊断 |

---

## 八、待现场核对清单（⚠️）

在服务器上执行 `bash tools/s2h/collect_handover_info.sh | tee handover_info.txt` 后，逐项填入：

- [ ] GPU 型号、显存、驱动版本（`nvidia-smi`）
- [x] `prepare_s2h_data.py` 的 `--source-root` = `/big-disk/hzh/S2H-CD-FSOD/data`（已确认，`derived/`、`raw/`、`DIOR/` 均在）
- [ ] 各 shot 支持集的实际 `classes / images / annotations` 统计
- [ ] `mmcv / mmengine / mmdet / torch` 的精确版本输出
- [ ] `S2H_BERT_DIR` 与权重路径的软链接指向确认
- [ ] 当前 `cat_work_dir/ablation` 下 `.train_complete` 数量（进度基线）
- [x] 最终评估口径：固定最终 epoch（已确认）

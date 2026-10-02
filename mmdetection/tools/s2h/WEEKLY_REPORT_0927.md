# S2H-CD-FSOD 周报

**周期**：2026-09-20 ~ 09-27
**论文**：*Trust What You Transfer: Knowledge-Enhanced Hard-Soft Reasoning for Cross-Domain Few-Shot Object Detection*（CVPR 2026 投稿）
**代码仓库**：`/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection`

---

## 一、实验方案

### 1.1 方法

在 Domain-RAG 复现的 **Grounding DINO (Swin-B)** 配方之上，实现知识增强的 Hard-Soft 框架 **S2H-CD-FSOD**，针对"支撑集诱导的上下文泄漏"这一问题：

| 组件 | 作用 | 代码位置 |
|---|---|---|
| **FFCP** | 前景冻结的上下文探测：对支撑框做扩张保护掩码 + 三类标签保持的背景替换，估计类级上下文敏感子空间 `U_ctx` 与泄漏强度 `κ_ctx` | `tools/s2h/support_interventions.py`、`build_s2h_knowledge.py` |
| **CHSD** | 上下文约束的 Hard-Soft 分解：结构属性打分得到 Hard 身份锚点 `h_c`，在身份补空间剔除上下文后得到 Soft 外观原型 `s_c`（带 shot-aware 收缩） | `build_s2h_knowledge.py`、`mmdet/models/utils/s2h_knowledge.py` |
| **ATAR** | 非对称任务权威路由：Hard-Soft 一致性 × 查询不确定性 → 有界余弦校正，**只改分类分支、不动定位**，初始化时校正为 0 | `s2h_knowledge.py::atar`、`s2h_grounding_dino_head.py` |

### 1.2 数据集与协议

- **六个目标域**：ArTaxOr(7 类)、Clipart1k(20 类)、DIOR(20 类)、DeepFish(1 类)、NEU-DET(6 类)、UODD(3 类)
- **shot 设置**：1-shot / 5-shot / 10-shot
- **四阶段累积消融**：`baseline`（关闭 S2H，等价原版 Grounding DINO）→ `ffcp` → `ffcp_chsd` → `full`（FFCP+CHSD+ATAR）
- **随机种子**：3407 / 3408 / 3409（论文要求多样本均值±std；本轮先按 1 seed 快速推进）
- **总规模**：6 数据集 × 3 shot × 4 阶段 × 3 seed = **216 组**（单 seed 时 72 组）
- **训练预算**：Clipart1k / DeepFish 为 5 epoch，其余四个域为 30 epoch；`MultiStepLR(milestones=[11])`（5-epoch 时为 `[3]`）
- **评测**：COCO bbox mAP（classwise），按**验证最优 epoch** 选 checkpoint

### 1.3 与基线的公平性保证

S2H 配置由 `gen_s2h_configs.py` 从 Domain-RAG 的 few-shot 配置**逐字复制**数据集块（数据、增强、分辨率、evaluator 完全一致），仅替换模型头，确保与 Grounding DINO† / Domain-RAG 的对比是 apples-to-apples。

---

## 二、实验进展

### 2.1 论文主表（Table 2）当前状态

已完成 5 个域 × 3 shot（**DIOR 列仍为空**）。与两条基线对照（mAP %）：

| 域 | shot | **Ours** | GroundingDINO† | Domain-RAG | 相对 G-DINO† |
|---|---|---|---|---|---|
| ArTaxOr | 1 / 5 / 10 | **43.8 / 67.2 / 75.0** | 26.3 / 68.4 / 73.0 | 57.2 / 70.0 / 73.4 | +17.5 / −1.2 / +2.0 |
| Clipart1k | 1 / 5 / 10 | **56.4 / 59.6 / 60.6** | 55.3 / 57.6 / 58.6 | 56.1 / 59.8 / 61.1 | +1.1 / +2.0 / +2.0 |
| DeepFish | 1 / 5 / 10 | **70.7 / 74.0 / 74.5** | 36.4 / 41.6 / 38.5 | 38.0 / 43.8 / 41.3 | **+34.3 / +32.4 / +36.0** ⚠️ |
| NEU-DET | 1 / 5 / 10 | **12.0 / 28.7 / 34.3** | 9.3 / 19.7 / 25.5 | 12.1 / 24.2 / 26.3 | +2.7 / +9.0 / +8.8 |
| UODD | 1 / 5 / 10 | **23.4 / 27.0 / 29.5** | 15.9 / 25.6 / 30.3 | 20.2 / 26.8 / 31.2 | +7.5 / +1.4 / −0.8 |
| DIOR | 1 / 5 / 10 | — | 14.8 / 29.6 / 37.2 | 18.0 / 31.5 / 39.0 | 待补 |

要点：NEU-DET、UODD 在低 shot 上增益明显；Clipart1k 稳定小幅领先；**DeepFish 数值异常偏高（+30 点以上），需重点核查**；ArTaxOr 1-shot 与 5-shot 弱于 Domain-RAG。

### 2.2 服务器训练进度（1 seed，72 组）

| 数据集 | 完成度 | 说明 |
|---|---|---|
| Clipart1k | 12/12 ✅ | 全部 4 阶段 × 3 shot |
| DeepFish (FISH) | 12/12 ✅ | 同上 |
| NEU-DET | 12/12 ✅ | 同上 |
| UODD | 12/12 ✅ | 同上 |
| ArTaxOr | 3/12 | 仅 baseline 的 1/5/10-shot 完成 |
| DIOR | 1/12 | 仅 baseline 1-shot 完成 |
| **合计** | **52/72** | 缺口 20 组 |

中断原因：`ArTaxOr` 与 `DIOR` 在独立会话中运行，在 baseline 阶段末尾的 **test 环节因 CUDA 报错退出**，导致后续三个阶段和其余 shot 未启动。

### 2.3 辅助实验完成情况

- **受控上下文研究**（Table 1）：Clipart1k + NEU-DET，4 种支撑条件 × 3 seed，含 CSS-cls / CSS-box 与 bootstrap 置信区间 —— 已完成
- **原型空间可视化**（t-SNE）：155 个 query 的逐一对齐比较，full 模型匹配数 49 vs baseline 32（McNemar *p* = 0.019）—— 已完成
- **超参敏感性**：FFCP rank、`ρ_H`、`g_max`、`θ`、`λ_con` 五项扫描 —— 已完成
- **累积消融**：覆盖已完成的 4 个域（Clipart1k / DeepFish / NEU-DET / UODD）—— 已完成，DIOR、ArTaxOr 待补
- **定性分析**：Clipart1k / DeepFish / NEU-DET —— 已完成

### 2.4 本周修复的工程问题

| # | 问题 | 影响 | 处置 |
|---|---|---|---|
| 1 | `save_best='auto'` 在 `CocoMetric(classwise=True)` 下按字典序选中 `coco/Araneae_precision`，生成 `best_coco_Araneae_precision_*.pth` | 所有 test 结果基于**错误的"最优模型"** | 改为 `save_best='coco/bbox_mAP'`，并重新生成全部配置 |
| 2 | `dist_test.sh`(torchrun) 在 spawn 子进程时 `init_dist → torch.cuda.set_device` 报 `CUDA unknown error` | test 阶段中断 | `test_one` 改为**单进程 `tools/test.py`**（不调用 `init_dist`） |
| 3 | mmengine 只写 `last_checkpoint` 而未生成 `latest.pth` | 续跑判断误判 → **已完成 run 被重训** | 续跑条件改为检查 `epoch_*.pth`/`latest.pth`，ckpt 兜底链 `latest → last_checkpoint → 最大 epoch` |
| 4 | `set -euo pipefail` 使单个 test 失败即终止整个数据集脚本 | 一个错误导致整批中断 | test 增加 3 次重试 + 失败仅告警不中断 |

---

## 三、存在的问题

### 3.1 数据协议层面（最高优先级）

1. **支撑集划分与论文声明不一致**。当前 6 个域中有 5 个使用 `prepare_s2h_data.py` 生成的 **image-level** split（`ArTaxOr / DIOR / FISH / NEU-DET / UODD`，生成于 09-19），仅 Clipart1k 使用原始划分。实测 1-shot 支撑标注数：
   - ArTaxOr 7、FISH 1、Clipart1k 20 —— 与 instance-level 等价 ✅
   - **NEU-DET 15（应为 6）**、**UODD 19（应为 3）** —— 支撑标注被多给 2.5~6.3 倍 ⚠️
   - DIOR 241 —— 与论文声明的 image-level 例外一致 ✅

   论文写的是"DIOR 用 image-level，其余用 K-instance"，但脚本对所有域都套了 image-level，**与论文表述矛盾**，且使 NEU-DET / UODD 的结果无法与 Table 2 的 GroundingDINO† 行直接比较。

2. **DeepFish 数值异常**（1-shot 70.7 vs Domain-RAG 38.0，+33 点）。该域只有 1 个类别，支撑集划分对结果影响被放大，**高度怀疑由 split 差异造成，而非方法增益**。这是最容易被审稿人质疑的一处。

3. **UODD 配置文件被改动过**：`MixUp` → `CachedMixUp`、`RandomCrop(prob=0.2)` → `RandomChoice([RandomCrop, []], prob=[0.2,0.8])` 且丢失 `allow_negative_crop=False`，与 Domain-RAG 原版不再等价。

4. **有效 batch size 不一致**：本轮单卡 `BATCH_SIZE=2`（有效 2），Domain-RAG 为 4 卡 × 4 = 16，且 lr 固定 1e-4 不随 batch 缩放，优化动态差异明显。

5. **用 test 集同时做 val**：`val_evaluator.ann_file` 指向 `test.json`，`save_best` 相当于"用测试集选模型"，结果偏乐观（Domain-RAG 用 `latest.pth`）。

### 3.2 实验完整性

1. **DIOR 完全缺失**（12 组），论文主表整列为空；
2. **ArTaxOr 缺 9 组**（`ffcp / ffcp_chsd / full` × 3 shot）；
3. **仅 1 seed**，无法给出论文要求的 mean ± std 与显著性；
4. **历史 run 的 test 结果需刷新**：此前 test 使用了按单类 precision 选出的错误 ckpt。

### 3.3 效率

DIOR 的 query 集约 1.1 万张图，而 `val_interval=1` 使每个 epoch 都要全量评估一次，单卡单个 run 预计约 10 小时，11 个 run 合计超过 4 天，是本轮最大的时间瓶颈。

---

## 四、下一步计划

### 4.1 补齐实验（本周内）

1. 用修补后的 `run_ablation.sh` 续跑 **ArTaxOr 剩余 9 组**（单卡，串行）；
2. 补齐 **DIOR 12 组**，建议 3~4 卡并行（`GPUS=3~4`）以压缩验证开销；
3. 补跑过程中脚本会以正确 ckpt 自动刷新历史 run 的 test 结果，无需额外操作。

### 4.2 协议对齐（优先级最高）

1. **决策 split 方案**（三选一）：
   - A. 从备份/数据包恢复官方 few-shot split，六域全部重跑；
   - B. 六域统一使用 `prepare_s2h_data.py` 的 image-level split（含 Clipart1k），并在论文中明确说明协议；
   - C. 修改 `prepare_s2h_data.py` 增加 `--protocol instance`，按论文声明重新切分。
   **原则：六个域必须同协议，且 baseline 与 Ours 必须使用同一份 split。**
2. 修回 `CDFSOD_detection_few-shot_UODD_*.py`，与 Domain-RAG 原版对齐；
3. 对比实验统一 batch size（建议 4 卡 × batch 4）。

### 4.3 结果汇总与论文更新

1. 汇总 72 组（后续 216 组）结果为 CSV，输出每域 / shot / stage 的 mean ± std；
2. 填充 Table 2 的 **DIOR 列与 Avg. 列**，并复核 ArTaxOr 数值；
3. 更新累积消融图（当前仅 4 个域）；
4. 在实验设置中补充**支撑集划分协议的明确说明**，避免与 Domain-RAG 的对比被误读。

### 4.4 风险提示

- DeepFish 的异常增益必须在正式投稿前查清；若确认为 split 造成，应重跑或改用统一协议，否则该列数据不可用；
- 若时间不允许 3-seed 全量，至少对**主表 6 个域 × 3 shot** 补足 3 seed，消融部分可保持 1 seed 并注明。

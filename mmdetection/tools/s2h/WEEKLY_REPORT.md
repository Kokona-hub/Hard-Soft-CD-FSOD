# S2H-CD-FSOD 周报

## 一、实验方案

本周将实验固定为六个跨域数据集：ArTaxOr、Clipart1k、DIOR、DeepFish（代码目录名为
`FISH`）、NEU-DET 和 UODD。每个数据集执行 1-shot、5-shot、10-shot 三种支持集规模，
并比较以下四个递进阶段：

| 阶段 | 实际实现 |
| --- | --- |
| baseline | 关闭 S2H，使用原始 Grounding DINO 微调 |
| FFCP | 使用 FFCP 得到的去上下文 Soft 原型；不使用 CHSD Hard 锚点和 ATAR 信号 |
| FFCP+CHSD | 在 FFCP 基础上加入结构属性 Hard 锚点与 Hard/Soft 一致性；关闭 ATAR 信号 |
| FFCP+CHSD+ATAR | 完整方法，加入查询不确定性路由及残差/一致性辅助项 |

数据划分固定使用已生成的支持标注（生成随机种子 `20250919`），训练和支持侧干预使用
三个随机种子 `3407/3408/3409`。因此总规模为
`6 × 3 × 4 × 3 = 216` 个训练/评估组合。每个组合保存独立的知识文件、训练目录、日志
和验证最优 checkpoint，避免不同阶段或 seed 相互覆盖。

主要指标为 COCO bbox `mAP`，同时记录 `mAP50`、`mAP75` 和每类 AP；模型选择以每个运行的
验证集最优 epoch 为准，不直接把 `latest.pth` 当作最优结果。

## 二、本周实验进展

1. 已完成 Python 3.10.21、PyTorch 2.0.0 + CUDA 11.7、MMCV 2.0.0、MMEngine 0.10.7、
   MMDetection 3.3.0 和本地 BERT 的离线环境验证。
2. 已完成六个数据集的 COCO 格式转换、1/5/10-shot 支持集和测试集布局检查。
3. 已完成 FFCP/CHSD/ATAR 的数值自检；服务器环境中 `tools/s2h/selfcheck.py` 已通过全部检查。
4. 已修复 token 非连续聚合、token 广播、ATAR einsum、知识构建器循环加载知识文件等问题。
5. 已把消融阶段作为显式 `stage` 写入知识文件和模型配置，新增
   `tools/s2h/run_ablation.sh`，支持按一个数据集一次性运行 3 个 shot × 4 个 stage × 3 个
   seed，并自动评估验证最优 checkpoint。
6. Clipart1k 1-shot 已完成一次 5 epoch 预实验：完整 S2H 在 epoch 1 的 mAP 为 0.574，
   baseline 为 0.571；当前差异仅为单 seed 的预备结果，不能作为显著提升结论。

## 三、当前问题与风险

1. 目前只有 Clipart1k 1-shot 的单次预实验，216 组正式结果尚未全部完成，暂时不能报告
   跨 seed 的均值、标准差或统计显著性。
2. 预实验中 baseline 和完整方法均在早期 epoch 达到峰值，存在明显 few-shot 过拟合风险；
   必须按验证最优 epoch 汇总，同时保留最终 epoch 作为补充记录。
3. 服务器依赖必须保持 `PYTHONNOUSERSITE=1`，否则用户目录中的 PyTorch 2.8/CUDA 12.8
   可能覆盖 Conda 环境，导致 MMCV `_ext` undefined symbol。离线运行还需要保持本地 BERT
   和本地 Grounding DINO checkpoint 路径有效。
4. ArTaxOr、FISH、UODD、DIOR、NEU-DET 使用项目脚本生成的固定支持划分，Clipart1k 使用
   原始划分；论文和周报中需要明确这一点，避免误认为所有数据集都使用同一种官方 split。
5. 全量 216 组实验耗时和显存开销较大，需要先跑单数据集全流程作为 smoke test，再分批
   提交其余数据集，并记录失败后可续跑的 seed/stage/shot。

## 四、下周工作安排

1. 在服务器上重新执行 self-check、配置生成和数据集构建检查，抽取一个数据集完成
   `1/5/10-shot × 4-stage × 3-seed` 的 36 组试跑。
2. 确认每个运行都生成 `best_coco_bbox_mAP_epoch_*.pth`，并验证测试命令实际使用的是
   验证最优 checkpoint。
3. 分批完成剩余五个数据集的正式训练和 COCO classwise 评估，实时检查日志、GPU 利用率、
   NaN/空检测和数据路径错误。
4. 编写结果汇总脚本，输出每个数据集/shot/stage 的 `mean ± std`，并计算相对 baseline 的
   mAP、mAP50、mAP75 增量。
5. 做失败重跑和可复现性核对：固定支持 split，核对三个 seed，保存配置、知识文件、日志和
   最优 epoch；对异常结果回查类别 AP 和预测可视化。
6. 根据四阶段曲线判断增益主要来自 FFCP、CHSD 还是 ATAR，再决定是否进行 `gamma`、
   `alpha_max`、`g_max` 和训练 epoch 的小范围敏感性实验。


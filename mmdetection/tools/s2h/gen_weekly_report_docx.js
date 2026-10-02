const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, Header, Footer
} = require('docx');

const FONT = 'Microsoft YaHei';
const B = { style: BorderStyle.SINGLE, size: 1, color: 'BFBFBF' };
const BORDERS = { top: B, bottom: B, left: B, right: B };
const MARGINS = { top: 60, bottom: 60, left: 110, right: 110 };
const CONTENT_W = 9026;

function cell(text, width, o = {}) {
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    margins: MARGINS,
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: o.align || AlignmentType.LEFT,
      spacing: { before: 20, after: 20 },
      children: [new TextRun({ text: String(text), bold: !!o.bold, size: o.size || 19, color: o.color })]
    })]
  });
}

function table(widths, dataRows) {
  const rows = dataRows.map((cells, ri) => new TableRow({
    tableHeader: ri === 0,
    children: cells.map((c, ci) => cell(c.t, widths[ci], c.o || {}))
  }));
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows
  });
}

const HEAD = { bold: true, fill: 'DCE6F1' };

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function p(text, o = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60, line: 300 },
    children: [new TextRun({ text, size: 21, ...o })]
  });
}
function bullet(text, bold) {
  return new Paragraph({
    numbering: { reference: 'b', level: 0 },
    spacing: { before: 30, after: 30 },
    children: [new TextRun({ text, size: 21, bold: !!bold })]
  });
}
function spacer(h = 120) {
  return new Paragraph({ spacing: { before: h, after: 0 }, children: [new TextRun('')] });
}

const children = [];

// ---------- 标题 ----------
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 80 },
  children: [new TextRun({ text: 'S2H-CD-FSOD 周报', bold: true, size: 36, font: FONT, color: '1F3864' })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 240 },
  children: [new TextRun({ text: '周期：2026-09-20 ~ 09-27', size: 20, font: FONT, color: '595959' })]
}));

children.push(table([1500, 7526], [
  [{ t: '论文', o: HEAD }, { t: 'Trust What You Transfer: Knowledge-Enhanced Hard-Soft Reasoning for Cross-Domain Few-Shot Object Detection（CVPR 2026 投稿）' }],
  [{ t: '方法', o: HEAD }, { t: 'S2H-CD-FSOD = FFCP + CHSD + ATAR，建立在 Domain-RAG 复现的 Grounding DINO (Swin-B) 配方之上' }],
  [{ t: '代码仓库', o: HEAD }, { t: '/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection' }],
]));

// ---------- 一 ----------
children.push(h1('一、实验方案'));

children.push(h2('1.1 方法'));
children.push(p('在 Domain-RAG 复现的 Grounding DINO (Swin-B) 配方之上，实现知识增强的 Hard-Soft 框架 S2H-CD-FSOD，针对“支撑集诱导的上下文泄漏”这一问题：'));
children.push(table([1100, 5600, 2326], [
  [{ t: '组件', o: HEAD }, { t: '作用', o: HEAD }, { t: '代码位置', o: HEAD }],
  [{ t: 'FFCP' }, { t: '前景冻结的上下文探测：对支撑框做扩张保护掩码 + 三类标签保持的背景替换，估计类级上下文敏感子空间 U_ctx 与泄漏强度 κ_ctx' }, { t: 'support_interventions.py、build_s2h_knowledge.py' }],
  [{ t: 'CHSD' }, { t: '上下文约束的 Hard-Soft 分解：结构属性打分得到 Hard 身份锚点 h_c，在身份补空间剔除上下文后得到 Soft 外观原型 s_c（带 shot-aware 收缩）' }, { t: 'build_s2h_knowledge.py、s2h_knowledge.py' }],
  [{ t: 'ATAR' }, { t: '非对称任务权威路由：Hard-Soft 一致性 × 查询不确定性 → 有界余弦校正，只改分类分支、不动定位，初始化时校正为 0' }, { t: 's2h_knowledge.py::atar、s2h_grounding_dino_head.py' }],
]));

children.push(h2('1.2 数据集与协议'));
children.push(bullet('六个目标域：ArTaxOr(7 类)、Clipart1k(20 类)、DIOR(20 类)、DeepFish(1 类)、NEU-DET(6 类)、UODD(3 类)'));
children.push(bullet('shot 设置：1-shot / 5-shot / 10-shot'));
children.push(bullet('四阶段累积消融：baseline（关闭 S2H，等价原版 Grounding DINO）→ ffcp → ffcp_chsd → full（FFCP+CHSD+ATAR）'));
children.push(bullet('随机种子：3407 / 3408 / 3409（论文要求多样本均值±std；本轮先按 1 seed 快速推进）'));
children.push(bullet('总规模：6 数据集 × 3 shot × 4 阶段 × 3 seed = 216 组（单 seed 时 72 组）'));
children.push(bullet('训练预算：Clipart1k / DeepFish 为 5 epoch，其余四个域为 30 epoch；MultiStepLR(milestones=[11])，5-epoch 时为 [3]'));
children.push(bullet('评测：COCO bbox mAP（classwise），按验证最优 epoch 选 checkpoint'));

children.push(h2('1.3 与基线的公平性保证'));
children.push(p('S2H 配置由 gen_s2h_configs.py 从 Domain-RAG 的 few-shot 配置逐字复制数据集块（数据、增强、分辨率、evaluator 完全一致），仅替换模型头，确保与 Grounding DINO† / Domain-RAG 的对比是 apples-to-apples。'));

// ---------- 二 ----------
children.push(h1('二、实验进展'));

children.push(h2('2.1 论文主表（Table 2）当前状态'));
children.push(p('已完成 5 个域 × 3 shot，DIOR 列仍为空。与两条基线对照（mAP %）：'));

const W6 = [1100, 1000, 1400, 1400, 1400, 2726];
children.push(table(W6, [
  [{ t: '域', o: HEAD }, { t: 'shot', o: HEAD }, { t: 'Ours', o: HEAD }, { t: 'G-DINO†', o: HEAD }, { t: 'Domain-RAG', o: HEAD }, { t: 'Ours − G-DINO†', o: HEAD }],
  [{ t: 'ArTaxOr' }, { t: '1/5/10' }, { t: '43.8 / 67.2 / 75.0', o: { bold: true } }, { t: '26.3 / 68.4 / 73.0' }, { t: '57.2 / 70.0 / 73.4' }, { t: '+17.5 / −1.2 / +2.0' }],
  [{ t: 'Clipart1k' }, { t: '1/5/10' }, { t: '56.4 / 59.6 / 60.6', o: { bold: true } }, { t: '55.3 / 57.6 / 58.6' }, { t: '56.1 / 59.8 / 61.1' }, { t: '+1.1 / +2.0 / +2.0' }],
  [{ t: 'DeepFish' }, { t: '1/5/10' }, { t: '70.7 / 74.0 / 74.5', o: { bold: true, color: 'C00000' } }, { t: '36.4 / 41.6 / 38.5' }, { t: '38.0 / 43.8 / 41.3' }, { t: '+34.3 / +32.4 / +36.0 ⚠', o: { color: 'C00000' } }],
  [{ t: 'NEU-DET' }, { t: '1/5/10' }, { t: '12.0 / 28.7 / 34.3', o: { bold: true } }, { t: '9.3 / 19.7 / 25.5' }, { t: '12.1 / 24.2 / 26.3' }, { t: '+2.7 / +9.0 / +8.8' }],
  [{ t: 'UODD' }, { t: '1/5/10' }, { t: '23.4 / 27.0 / 29.5', o: { bold: true } }, { t: '15.9 / 25.6 / 30.3' }, { t: '20.2 / 26.8 / 31.2' }, { t: '+7.5 / +1.4 / −0.8' }],
  [{ t: 'DIOR' }, { t: '1/5/10' }, { t: '—', o: { align: AlignmentType.CENTER } }, { t: '14.8 / 29.6 / 37.2' }, { t: '18.0 / 31.5 / 39.0' }, { t: '待补' }],
]));
children.push(spacer(80));
children.push(p('要点：NEU-DET、UODD 在低 shot 上增益明显；Clipart1k 稳定小幅领先；DeepFish 数值异常偏高（+30 点以上），需重点核查；ArTaxOr 1-shot 与 5-shot 弱于 Domain-RAG。'));

children.push(h2('2.2 服务器训练进度（1 seed，72 组）'));
children.push(table([2200, 1600, 5226], [
  [{ t: '数据集', o: HEAD }, { t: '完成度', o: HEAD }, { t: '说明', o: HEAD }],
  [{ t: 'Clipart1k' }, { t: '12/12 ✓', o: { color: '2E7D32' } }, { t: '全部 4 阶段 × 3 shot 完成' }],
  [{ t: 'DeepFish (FISH)' }, { t: '12/12 ✓', o: { color: '2E7D32' } }, { t: '同上' }],
  [{ t: 'NEU-DET' }, { t: '12/12 ✓', o: { color: '2E7D32' } }, { t: '同上' }],
  [{ t: 'UODD' }, { t: '12/12 ✓', o: { color: '2E7D32' } }, { t: '同上' }],
  [{ t: 'ArTaxOr' }, { t: '3/12', o: { color: 'C00000' } }, { t: '仅 baseline 的 1/5/10-shot 完成' }],
  [{ t: 'DIOR' }, { t: '1/12', o: { color: 'C00000' } }, { t: '仅 baseline 1-shot 完成' }],
  [{ t: '合计', o: { bold: true } }, { t: '52/72', o: { bold: true, color: 'C00000' } }, { t: '缺口 20 组' }],
]));
children.push(spacer(80));
children.push(p('中断原因：ArTaxOr 与 DIOR 在独立会话中运行，在 baseline 阶段末尾的 test 环节因 CUDA 报错退出，导致后续三个阶段和其余 shot 未启动。'));

children.push(h2('2.3 辅助实验完成情况'));
children.push(bullet('受控上下文研究（Table 1）：Clipart1k + NEU-DET，4 种支撑条件 × 3 seed，含 CSS-cls / CSS-box 与 bootstrap 置信区间 —— 已完成'));
children.push(bullet('原型空间可视化（t-SNE）：155 个 query 逐一对齐比较，full 模型匹配数 49 vs baseline 32（McNemar p = 0.019）—— 已完成'));
children.push(bullet('超参敏感性：FFCP rank、ρ_H、g_max、θ、λ_con 五项扫描 —— 已完成'));
children.push(bullet('累积消融：覆盖已完成的 4 个域（Clipart1k / DeepFish / NEU-DET / UODD）—— 已完成，DIOR、ArTaxOr 待补'));
children.push(bullet('定性分析：Clipart1k / DeepFish / NEU-DET —— 已完成'));

children.push(h2('2.4 本周修复的工程问题'));
children.push(table([500, 3100, 2500, 2926], [
  [{ t: '#', o: HEAD }, { t: '问题', o: HEAD }, { t: '影响', o: HEAD }, { t: '处置', o: HEAD }],
  [{ t: '1' }, { t: "save_best='auto' 在 CocoMetric(classwise=True) 下按字典序选中 coco/Araneae_precision" }, { t: '所有 test 结果基于错误的最优模型' }, { t: "改为 save_best='coco/bbox_mAP'，重新生成全部配置" }],
  [{ t: '2' }, { t: 'dist_test.sh(torchrun) 在 spawn 子进程时 init_dist → torch.cuda.set_device 报 CUDA unknown error' }, { t: 'test 阶段中断' }, { t: 'test_one 改为单进程 tools/test.py（不调用 init_dist）' }],
  [{ t: '3' }, { t: 'mmengine 只写 last_checkpoint 而未生成 latest.pth' }, { t: '续跑判断误判，已完成 run 被重训' }, { t: '续跑条件改为检查 epoch_*.pth/latest.pth，ckpt 兜底链 latest → last_checkpoint → 最大 epoch' }],
  [{ t: '4' }, { t: 'set -euo pipefail 使单个 test 失败即终止整个数据集脚本' }, { t: '一个错误导致整批中断' }, { t: 'test 增加 3 次重试 + 失败仅告警不中断' }],
]));

// ---------- 三 ----------
children.push(h1('三、存在的问题'));

children.push(h2('3.1 数据协议层面（最高优先级）'));
children.push(bullet('支撑集划分与论文声明不一致：6 个域中 5 个使用 prepare_s2h_data.py 生成的 image-level split（ArTaxOr / DIOR / FISH / NEU-DET / UODD，生成于 09-19），仅 Clipart1k 使用原始划分。1-shot 支撑标注数实测：', true));
children.push(bullet('ArTaxOr 7、FISH 1、Clipart1k 20 —— 与 instance-level 等价 ✓'));
children.push(bullet('NEU-DET 15（应为 6）、UODD 19（应为 3）—— 支撑标注被多给 2.5~6.3 倍 ⚠'));
children.push(bullet('DIOR 241 —— 与论文声明的 image-level 例外一致 ✓'));
children.push(p('论文写的是“DIOR 用 image-level，其余用 K-instance”，但脚本对所有域都套了 image-level，与论文表述矛盾，且使 NEU-DET / UODD 的结果无法与 Table 2 的 GroundingDINO† 行直接比较。'));
children.push(bullet('DeepFish 数值异常（1-shot 70.7 vs Domain-RAG 38.0，+33 点）。该域只有 1 个类别，支撑集划分对结果影响被放大，高度怀疑由 split 差异造成，而非方法增益，是最容易被审稿人质疑的一处。', true));
children.push(bullet('UODD 配置文件被改动过：MixUp → CachedMixUp、RandomCrop(prob=0.2) → RandomChoice([RandomCrop, []], prob=[0.2,0.8]) 且丢失 allow_negative_crop=False，与 Domain-RAG 原版不再等价。'));
children.push(bullet('有效 batch size 不一致：本轮单卡 BATCH_SIZE=2（有效 2），Domain-RAG 为 4 卡 × 4 = 16，且 lr 固定 1e-4 不随 batch 缩放，优化动态差异明显。'));
children.push(bullet('用 test 集同时做 val：val_evaluator.ann_file 指向 test.json，save_best 相当于“用测试集选模型”，结果偏乐观（Domain-RAG 用 latest.pth）。'));

children.push(h2('3.2 实验完整性'));
children.push(bullet('DIOR 完全缺失（12 组），论文主表整列为空'));
children.push(bullet('ArTaxOr 缺 9 组（ffcp / ffcp_chsd / full × 3 shot）'));
children.push(bullet('仅 1 seed，无法给出论文要求的 mean ± std 与显著性'));
children.push(bullet('历史 run 的 test 结果需刷新：此前 test 使用了按单类 precision 选出的错误 ckpt'));

children.push(h2('3.3 效率'));
children.push(p('DIOR 的 query 集约 1.1 万张图，而 val_interval=1 使每个 epoch 都要全量评估一次，单卡单个 run 预计约 10 小时，11 个 run 合计超过 4 天，是本轮最大的时间瓶颈。'));

// ---------- 四 ----------
children.push(h1('四、下一步计划'));

children.push(h2('4.1 补齐实验（本周内）'));
children.push(bullet('用修补后的 run_ablation.sh 续跑 ArTaxOr 剩余 9 组（单卡，串行）'));
children.push(bullet('补齐 DIOR 12 组，建议 3~4 卡并行（GPUS=3~4）以压缩验证开销'));
children.push(bullet('补跑过程中脚本会以正确 ckpt 自动刷新历史 run 的 test 结果，无需额外操作'));

children.push(h2('4.2 协议对齐（优先级最高）'));
children.push(p('决策 split 方案（三选一）：'));
children.push(bullet('A. 从备份/数据包恢复官方 few-shot split，六域全部重跑'));
children.push(bullet('B. 六域统一使用 prepare_s2h_data.py 的 image-level split（含 Clipart1k），并在论文中明确说明协议'));
children.push(bullet('C. 修改 prepare_s2h_data.py 增加 --protocol instance，按论文声明重新切分'));
children.push(p('原则：六个域必须同协议，且 baseline 与 Ours 必须使用同一份 split。', { bold: true }));
children.push(bullet('修回 CDFSOD_detection_few-shot_UODD_*.py，与 Domain-RAG 原版对齐'));
children.push(bullet('对比实验统一 batch size（建议 4 卡 × batch 4）'));

children.push(h2('4.3 结果汇总与论文更新'));
children.push(bullet('汇总 72 组（后续 216 组）结果为 CSV，输出每域 / shot / stage 的 mean ± std'));
children.push(bullet('填充 Table 2 的 DIOR 列与 Avg. 列，并复核 ArTaxOr 数值'));
children.push(bullet('更新累积消融图（当前仅 4 个域）'));
children.push(bullet('在实验设置中补充支撑集划分协议的明确说明，避免与 Domain-RAG 的对比被误读'));

children.push(h2('4.4 风险提示'));
children.push(bullet('DeepFish 的异常增益必须在正式投稿前查清；若确认为 split 造成，应重跑或改用统一协议，否则该列数据不可用'));
children.push(bullet('若时间不允许 3-seed 全量，至少对主表 6 个域 × 3 shot 补足 3 seed，消融部分可保持 1 seed 并注明'));

// ---------- 文档 ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: '1F3864' },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 0 }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: '2E75B6' },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 }
      }
    ]
  },
  numbering: {
    config: [{
      reference: 'b',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 480, hanging: 240 } } }
      }]
    }]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'S2H-CD-FSOD 周报  ·  ', size: 18, color: '808080', font: FONT }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '808080', font: FONT })
          ]
        })]
      })
    },
    children
  }]
});

const out = 'd:/Domain-RAG-main - 副本/mmdetection/tools/s2h/S2H-CD-FSOD_周报_20260927.docx';
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(out, buf);
  console.log('written:', out, '(' + buf.length + ' bytes)');
});

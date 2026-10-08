# -*- coding: utf-8 -*-
"""生成 S2H-CD-FSOD 阶段周报 PDF（版式对齐 周报20260927.pdf）。"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether, Preformatted)

FONT_DIR = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("YaHei", os.path.join(FONT_DIR, "msyh.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("YaHei-Bold", os.path.join(FONT_DIR, "msyhbd.ttc"), subfontIndex=0))
pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHei-Bold",
                              italic="YaHei", boldItalic="YaHei-Bold")
pdfmetrics.registerFont(TTFont("TNR", os.path.join(FONT_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("TNR-It", os.path.join(FONT_DIR, "timesi.ttf")))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR", italic="TNR-It", boldItalic="TNR-It")

OUT = r"D:\打饭项目前端\周报20261007.pdf"

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
GREY = colors.HexColor("#666666")
LINE = colors.HexColor("#2b2b2b")
CODEBG = colors.HexColor("#f5f6f8")
CODEBD = colors.HexColor("#d0d4da")

st_title = ParagraphStyle("title", fontName="YaHei-Bold", fontSize=20, leading=26,
                          alignment=TA_CENTER, textColor=INK, spaceAfter=3)
st_sub = ParagraphStyle("sub", fontName="YaHei", fontSize=11, leading=16,
                        alignment=TA_CENTER, textColor=GREY, spaceAfter=10)
st_lab = ParagraphStyle("lab", fontName="YaHei-Bold", fontSize=10, leading=15, textColor=ACCENT)
st_val = ParagraphStyle("val", fontName="YaHei", fontSize=10, leading=15.5, textColor=INK, wordWrap="CJK")
st_h1 = ParagraphStyle("h1", fontName="YaHei-Bold", fontSize=14.5, leading=20,
                       textColor=ACCENT, spaceBefore=12, spaceAfter=6)
st_h2 = ParagraphStyle("h2", fontName="YaHei-Bold", fontSize=11.5, leading=17,
                       textColor=INK, spaceBefore=7, spaceAfter=3)
st_body = ParagraphStyle("body", fontName="YaHei", fontSize=10.5, leading=16.4,
                         textColor=INK, wordWrap="CJK", spaceAfter=4)
st_bullet = ParagraphStyle("bullet", parent=st_body, leftIndent=15, bulletIndent=3, spaceAfter=2.5)
st_caption = ParagraphStyle("caption", fontName="YaHei-Bold", fontSize=10, leading=15,
                            textColor=GREY, spaceBefore=6, spaceAfter=3)
st_eq = ParagraphStyle("eq", fontName="TNR-It", fontSize=11.5, leading=19,
                       alignment=TA_CENTER, textColor=INK, spaceBefore=5, spaceAfter=8)
st_code = ParagraphStyle("code", fontName="Courier", fontSize=8.8, leading=12.6, textColor=INK)
st_note = ParagraphStyle("note", fontName="YaHei", fontSize=9.5, leading=15,
                         textColor=GREY, wordWrap="CJK", spaceAfter=4)


def P(t, s=st_body):
    return Paragraph(t, s)


def H1(t):
    return Paragraph(t, st_h1)


def H2(t):
    return Paragraph(t, st_h2)


def B(t):
    return Paragraph(t, st_bullet, bulletText="•")


def EQ(t):
    return Paragraph(t, st_eq)


def CAP(t):
    return Paragraph(t, st_caption)


def codebox(text, width=150 * mm):
    t = Table([[Preformatted(text, st_code)]], colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.5, CODEBD),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def three_line(data, colWidths, aligns=None, header=True, fs=9.5):
    t = Table(data, colWidths=colWidths, hAlign="CENTER")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "YaHei"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEABOVE", (0, 0), (-1, 0), 1.1, LINE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
        ("LINEBELOW", (0, -1), (-1, -1), 1.1, LINE),
    ]
    if header:
        cmds.append(("FONTNAME", (0, 0), (-1, 0), "YaHei-Bold"))
    if aligns:
        for i, a in enumerate(aligns):
            cmds.append(("ALIGN", (i, 0), (i, -1), a))
    t.setStyle(TableStyle(cmds))
    return t


story = []

# ---------------- 标题区 ----------------
story.append(P("S2H-CD-FSOD 周报", st_title))
story.append(P("周期：2026-09-28 ~ 10-07", st_sub))

info_rows = [
    [P("论文", st_lab),
     P("Trust What You Transfer: Knowledge-Enhanced Hard-Soft Reasoning for Cross-Domain "
       "Few-Shot Object Detection（CVPR 2026 投稿）", st_val)],
    [P("方法", st_lab),
     P("S2H-CD-FSOD = FFCP + CHSD + ATAR，建立在 Domain-RAG 复现的 Grounding DINO (Swin-B) "
       "配方之上", st_val)],
    [P("代码仓库", st_lab),
     P("/mnt/sdc/hzh/Domain-RAG-main-2/mmdetection", st_val)],
]
info = Table(info_rows, colWidths=[24 * mm, 146 * mm], hAlign="CENTER")
info.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
    ("BOX", (0, 0), (-1, -1), 0.6, CODEBD),
    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde2e8")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(info)

# ---------------- 一、实验目标与方案 ----------------
story.append(H1("一、实验目标与方案"))

story.append(H2("1.1 本阶段目标"))
story.append(P("本阶段目标是验证 Hard-Soft 方法是否能够在 Grounding DINO 基础上稳定提升 "
               "few-shot detection 的 mAP，并确认："))
story.append(B("baseline → FFCP → FFCP+CHSD → full 是否能够逐阶段提升；"))
story.append(B("ATAR 的不确定性门控是否有效；"))
story.append(B("Hard/Soft 原型方向、可靠性和修正强度是否真正参与推理；"))
story.append(B("在不改变 bbox 回归分支的情况下，分类校准能否改善最终检测排序。"))

story.append(H2("1.2 数据集与消融协议"))
story.append(B("主要验证集：dataset = clipart1k，shot = 5-shot，seed = 3407，test images = 500"))
story.append(B("消融阶段：baseline → ffcp → ffcp_chsd → full"))
story.append(B("控制变量：同一 checkpoint、同一 test images、seed = 3407，"
               "gamma ∈ {0, 1, 4, 8}，soft_mix = 0.35，theta = 0.5，T_u = 0.1"))

# ---------------- 二、实验进展 ----------------
story.append(H1("二、实验进展"))

story.append(H2("2.1 配置与代码同步"))
story.append(P("当前 S2H 配置已统一为："))
story.append(codebox(
    "correction_mode     = 'contrastive'\n"
    "routing_confidence  = 'sigmoid'\n"
    "reliability_power   = 1.0\n"
    "train_injection     = False\n"
    "res_weight          = 0.0\n"
    "con_weight          = 0.0\n"
    "soft_mix            = 0.35"
))
story.append(Spacer(1, 4))
story.append(P("同时修复了一个重要问题：此前 soft_mix 只存在于配置中，而实际 atar() 中始终使用 "
               "Soft prototype，导致 Hard-Soft 混合没有真正生效。修复后："))
story.append(codebox(
    "ffcp:\n"
    "    direction = soft\n\n"
    "ffcp_chsd / full:\n"
    "    direction = normalize(\n"
    "        (1 - soft_mix) * hard + soft_mix * soft\n"
    "    )"
))
story.append(Spacer(1, 2))
story.append(P("已通过 Python 编译与运行时检查。"))

story.append(H2("2.2 诊断工具运行结果"))
story.append(P("50 张图像的诊断结果如下："))
story.append(KeepTogether([
    three_line(
        [["stage", "mean|delta|", "r_bar", "active"],
         ["ffcp", "0.0170", "0.8287", "1.0000"],
         ["ffcp_chsd", "0.0154", "0.4522", "1.0000"],
         ["full", "0.0139", "0.4514", "0.9991"]],
        colWidths=[38 * mm, 34 * mm, 30 * mm, 30 * mm]),
]))
story.append(Spacer(1, 3))
story.append(P("Hard/Soft agreement 平均约为 0.5454，mean reliability 约为 0.8289。这表明 Hard 与 "
               "Soft 原型的方向一致性较低，CHSD 阶段可靠性明显下降。"))

story.append(H2("2.3 门控尺度问题已定位"))
story.append(P("原始配置中 theta = 0.5、T_u = 0.1，但当前 Grounding DINO 的分类 logits 经过 "
               "sigmoid 后非常小：p0 median ≈ 0.0068，p0 95% ≈ 0.0272。"))
story.append(EQ("u = sigmoid( (0.5 − p<sub>0</sub>) / 0.1 ) ≈ 0.99"))
story.append(P("因此 full 阶段的 ATAR 门控几乎一直处于开启状态。将参数调整为 theta = 0.01、"
               "T_u = 0.01 后，门控数值上变得更有区分度：u ≈ 0.525，active ratio ≈ 0.726，"
               "但 mAP 从 0.593 降回 0.591。"))
story.append(P("结论：门控校准不是当前的主要性能瓶颈，最终配置暂时仍保留默认门控。"))

story.append(H2("2.4 gamma 参数已确认生效"))
story.append(KeepTogether([
    three_line(
        [["gamma", "mean|delta|", "max|delta|"],
         ["0", "0.0000", "0.0000"],
         ["1", "0.0132", "0.0563"],
         ["4", "0.0528", "0.2251"],
         ["8", "0.1056", "0.4503"]],
        colWidths=[34 * mm, 40 * mm, 40 * mm]),
]))
story.append(Spacer(1, 3))
story.append(P("修正量严格按照 gamma 成比例变化，说明：配置覆盖有效；knowledge bank 已正确加载；"
               "ATAR 实际参与推理；问题不在参数未生效。"))

story.append(H2("2.5 同一 checkpoint 的 gamma 实验"))
story.append(KeepTogether([
    three_line(
        [["gamma", "mAP", "mAP50", "mAP75", "mAP_s", "mAP_m", "mAP_l"],
         ["0", "0.591", "0.811", "0.690", "0.354", "0.551", "0.664"],
         ["1", "0.591", "0.811", "0.690", "0.354", "0.551", "0.664"],
         ["4", "0.592", "0.812", "0.690", "0.354", "0.551", "0.664"],
         ["6", "0.592", "0.813", "0.690", "0.357", "0.552", "0.664"],
         ["7", "0.593", "0.813", "0.691", "0.356", "0.551", "0.665"],
         ["8", "0.593", "0.813", "0.691", "0.356", "0.551", "0.664"],
         ["9", "0.593", "0.813", "0.691", "0.357", "0.551", "0.663"],
         ["10", "0.593", "0.813", "0.691", "0.357", "0.551", "0.663"],
         ["12", "0.593", "0.812", "0.691", "0.362", "0.551", "0.662"]],
        colWidths=[18 * mm, 20 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm], fs=9),
]))
story.append(Spacer(1, 3))
story.append(P("当前推理后校准的最高提升约为 0.591 → 0.593，说明当前 Hard-Soft 方向具有弱正作用，"
               "但收益已经出现平台期。"))

story.append(H2("2.6 soft_mix 实验"))
story.append(P("固定 gamma = 8 后："))
story.append(KeepTogether([
    three_line(
        [["soft_mix", "mAP", "mAP50", "mAP75"],
         ["0.00", "0.593", "0.813", "0.691"],
         ["0.25", "0.593", "0.813", "0.691"],
         ["0.35", "0.593", "0.813", "0.691"],
         ["0.50", "0.593", "0.812", "0.690"],
         ["0.75", "0.592", "0.812", "0.690"],
         ["1.00", "0.592", "0.812", "0.690"]],
        colWidths=[36 * mm, 32 * mm, 32 * mm, 32 * mm]),
]))
story.append(Spacer(1, 3))
story.append(P("当前最稳定的方向组合是 soft_mix = 0.25 ~ 0.35。建议继续使用 soft_mix = 0.35，"
               "因为它符合原始 Hard-Soft 方法设定，且没有证据证明 0.25 显著优于 0.35。"))

# ---------------- 三、存在的问题 ----------------
story.append(H1("三、当前存在的主要问题"))

story.append(H2("3.1 问题一：知识特征与修正对象不在同一表示空间"))
story.append(P("当前 Soft prototype 的构建流程为："))
story.append(codebox("support image\n"
                     "    |-> model.extract_feat()\n"
                     "    |-> backbone/neck feature pyramid\n"
                     "    |-> GT box pooling\n"
                     "    '-> Soft prototype", width=90 * mm))
story.append(Spacer(1, 3))
story.append(P("而运行时修正对象为："))
story.append(codebox("decoder query hidden state\n"
                     "    |-> Grounding DINO ContrastiveEmbed\n"
                     "    |-> classification logits\n"
                     "    '-> S2H correction", width=90 * mm))
story.append(Spacer(1, 4))
story.append(P("也就是说，当前 Soft prototype 来自 backbone/neck 框内池化特征，而 ATAR 中的查询特征"
               "来自 decoder hidden state。虽然二者都是 256 维，但这不代表它们属于同一个几何表示空间。"
               "当前方法实际上是在用 backbone/neck 空间中的方向修正 decoder query 空间中的分类结果，"
               "这很可能是性能受限的根本原因。"))

story.append(H2("3.2 问题二：CHSD 阶段 Hard/Soft agreement 过低"))
story.append(P("当前 Hard/Soft agreement 约为 0.54，意味着 Hard identity anchor 和 Soft appearance "
               "prototype 的夹角接近正交，CHSD 中使用二者混合方向后，可能引入错误的类别偏移。"))
story.append(KeepTogether([
    three_line(
        [["阶段", "baseline", "ffcp", "ffcp_chsd", "full"],
         ["mAP", "0.592", "0.592", "0.587", "0.591"]],
        colWidths=[30 * mm, 32 * mm, 32 * mm, 38 * mm, 32 * mm]),
]))
story.append(Spacer(1, 3))
story.append(P("这也解释了当前结果：损失主要发生在 CHSD，full 只能部分恢复。"))

story.append(H2("3.3 问题三：训练阶段没有看到 S2H 修正"))
story.append(P("当前配置 train_injection = False、res_weight = 0.0、con_weight = 0.0，因此模型训练"
               "过程实际上仍是普通 Grounding DINO，S2H 只在测试时对分类 logits 做后处理。这会导致："))
story.append(B("decoder query 没有学习适应 Soft prototype；"))
story.append(B("分类分数分布没有针对修正进行重新校准；"))
story.append(B("gamma 放大后主要改变候选排序，而不是提升真实目标置信度；"))
story.append(B("mAP 提升被限制在约 0.002。"))

story.append(H2("3.4 问题四：目前是全局修正，不是类别自适应修正"))
story.append(P("当前 gamma 是全局标量，g_cls 也是全局标量。不同类别的 AP 出现相反变化：部分类别提升，"
               "部分类别下降，全局 mAP 只小幅变化。这说明不同类别需要不同的 correction strength，"
               "单一 gamma 无法同时适配所有类别。"))

# ---------------- 四、下一步计划 ----------------
story.append(H1("四、下一步计划"))

story.append(P("下一阶段的核心任务确定为：在 decoder-query 表示空间中重新构建 Soft prototype，"
               "验证表示空间对齐是否是当前性能瓶颈。"))
story.append(P("不建议立即继续搜索 gamma，也不建议立刻扩展到六个数据集。应先在 clipart1k 5-shot "
               "seed 3407 上完成结构性验证。"))

story.append(H2("4.1 阶段 A：建立 decoder-query 特征审计"))
story.append(P("<b>目标：</b>确认 decoder-query Soft prototype 是否比当前 backbone/neck Soft "
               "prototype 更接近真实检测查询的分类空间。"))
story.append(P("<b>方法：</b>对每张 support 图像——"))
story.append(B("使用 baseline Grounding DINO，不启用 S2H；"))
story.append(B("获取最后一个 decoder layer 的 query hidden states；"))
story.append(B("获取 query 预测框；"))
story.append(B("将 query 框与 support GT 框进行 IoU 匹配；"))
story.append(B("对匹配到类别 c 的 query hidden state 做归一化；"))
story.append(B("汇总得到新的 decoder-space prototype。"))
story.append(P("定义："))
story.append(EQ("q<sub>i,c</sub> = <font face=\"TNR\">Norm</font>(H<sub>i,j*</sub>)"))
story.append(P("其中 H<sub>i,j*</sub> 是匹配到 GT 类别 c 的 decoder query；j* 是与 GT 框 IoU 最大且"
               "超过阈值的 query。新的 Soft prototype 为："))
story.append(EQ("s<sub>c</sub><super>Q</super> = <font face=\"TNR\">Norm</font>( "
                "(1 − β<sub>c</sub>)·<font face=\"TNR\">RobustMean</font>(q<sub>i,c</sub>) + "
                "β<sub>c</sub> h<sub>c</sub> )"))
story.append(P("其中 h<sub>c</sub> 是文本 Hard prototype，β<sub>c</sub> 是小样本收缩系数，"
               "RobustMean 用于降低错误 query 的影响。第一轮只验证原始 query-space Soft prototype，"
               "不同时加入 FFCP 与复杂 CHSD，避免多个因素混在一起。"))

story.append(H2("4.2 阶段 B：新增 query-space knowledge builder"))
story.append(P("计划新增 <font face=\"Courier\">tools/s2h/build_s2h_query_knowledge.py</font>，"
               "输出格式仍保持兼容："))
story.append(codebox(
    '{\n'
    '    "hard":          [C, D],\n'
    '    "soft":          [C, D],\n'
    '    "kappa":         [C],\n'
    '    "reliability":   [C],\n'
    '    "classes":       [...],\n'
    '    "token_ids":     [...],\n'
    '    "meta": {\n'
    '        "stage": "query_soft",\n'
    '        "representation": "decoder_query",\n'
    '        "source": "support_only",\n'
    '        "checkpoint": "...",\n'
    '        "iou_threshold": 0.5\n'
    '    }\n'
    '}'
))
story.append(Spacer(1, 3))
story.append(P("必须增加以下校验："))
story.append(B("knowledge 只能使用 support/training 图像；"))
story.append(B("不允许读取 test 标注；"))
story.append(B("类别顺序必须和配置一致；"))
story.append(B("query hidden dimension 必须等于 embed_dims；"))
story.append(B("knowledge 文件中记录 representation 类型；"))
story.append(B("如果是旧的 neck-space knowledge，运行时给出明确警告。"))

story.append(H2("4.3 阶段 C：同一 checkpoint 的结构对照实验"))
story.append(P("使用同一个 full checkpoint，只替换 knowledge 文件：control 为当前 neck-space Soft "
               "prototype，experiment 为 decoder-query-space Soft prototype。控制变量保持完全一致："
               "checkpoint 相同、test images 相同、gamma ∈ {1, 4, 8}、soft_mix = 0.35、theta = 0.5、"
               "T_u = 0.1、seed = 3407。"))
story.append(KeepTogether([
    three_line(
        [["knowledge", "gamma", "目的"],
         ["neck-space", "0", "关闭修正"],
         ["neck-space", "8", "当前方法基线"],
         ["query-space", "1", "检查方向是否稳定"],
         ["query-space", "4", "中等修正"],
         ["query-space", "8", "检查是否超过当前方法"]],
        colWidths=[42 * mm, 24 * mm, 72 * mm], aligns=["LEFT", "CENTER", "LEFT"]),
]))
story.append(Spacer(1, 4))
story.append(P("如果 query-space 在同一 checkpoint 上达到 mAP &gt; 0.593，并且不是只依赖某一个类别，"
               "则说明表示空间错配确实是主要问题。更理想的判据是：query-space 至少提升 "
               "0.003 ~ 0.005 mAP，同时不能只靠小目标 AP 增益而牺牲大目标 AP。"))

story.append(H2("4.4 阶段 D：在 query-space 中重新加入 FFCP"))
story.append(P("如果阶段 C 有效，再将 FFCP 从 neck-space 迁移到 query-space。对于原始图像、同域干预图像、"
               "类别错配图像和重建图像，分别提取匹配 query："))
story.append(EQ("d<sub>r,c</sub> = (q<sub>c</sub><super>fg</super> − "
                "q<sub>c</sub><super>~r</super>) − (q<sub>c</sub><super>fg</super> − "
                "q<sub>c</sub><super>rec</super>)"))
story.append(P("然后在 query-space drift 上估计："))
story.append(EQ("U<sub>ctx</sub><super>Q</super> = <font face=\"TNR\">SVD</font>(d<sub>r,c</sub>)"))
story.append(P("并执行上下文清除："))
story.append(EQ("q<sub>c</sub><super>clean</super> = (I − U<sub>ctx</sub><super>Q</super>"
                "U<sub>ctx</sub><super>Q T</super>) q<sub>c</sub><super>fg</super>"))
story.append(P("这样 FFCP 的上下文清除和最终 ATAR 修正将处于同一个 decoder-query 几何空间中。"))

story.append(H2("4.5 阶段 E：重新验证 CHSD"))
story.append(P("只有在 query-space FFCP 有正收益后，才重新加入 CHSD：Hard text anchor + "
               "query-space Soft appearance prototype。重点监控 Hard/Soft agreement、reliability、"
               "delta_mean_abs、per-class AP。"))
story.append(P("目标不是盲目让 agreement 越高，而是确认："))
story.append(B("Hard/Soft 不再出现大面积冲突；"))
story.append(B("CHSD 不再从 FFCP 阶段明显掉点；"))
story.append(B("full 至少不低于 FFCP；"))
story.append(B("最终形成合理的累计趋势。"))

story.append(H2("4.6 阶段 F：最后再开启训练注入"))
story.append(P("query-space knowledge 验证有效后，再进行训练注入实验："))
story.append(codebox(
    "train_injection = True\n"
    "res_weight      = 0.01\n"
    "con_weight      = 0.01\n"
    "gamma           = 1.0\n"
    "soft_mix        = 0.35"
))
story.append(Spacer(1, 3))
story.append(P("训练初期不直接使用 gamma = 8，原因是 gamma = 8 是测试时找到的后处理强度，"
               "不能直接假设适合训练分布。训练注入的顺序为："))
story.append(B("baseline checkpoint；"))
story.append(B("full + query-space knowledge；"))
story.append(B("gamma = 1 训练；"))
story.append(B("测试时搜索 gamma = 1, 2, 4；"))
story.append(B("再决定是否扩大训练强度。"))

# ---------------- 五、验收标准 ----------------
story.append(H1("五、下一阶段的验收标准"))
story.append(P("下一阶段不以“单次测试得到更高数字”为唯一标准，而采用以下条件。"))

story.append(H2("5.1 结构正确性"))
story.append(B("query-space knowledge 只使用 support 数据；"))
story.append(B("query prototype 与 decoder hidden state 维度和表示空间一致；"))
story.append(B("旧 neck-space knowledge 与新 query-space knowledge 可以明确区分；"))
story.append(B("同一 checkpoint 下替换 knowledge 后修正量确实发生变化。"))

story.append(H2("5.2 性能有效性"))
story.append(B("query-space 同 checkpoint mAP 至少超过当前 0.593；"))
story.append(B("不只提升单一类别；"))
story.append(B("mAP50、mAP75 不出现明显反向下降；"))
story.append(B("小目标提升不能以大目标显著下降为代价。"))

story.append(H2("5.3 消融合理性"))
story.append(P("目标趋势为 baseline ≤ ffcp ≤ ffcp_chsd ≤ full，但在单个 seed 下不强行要求严格单调。"
               "最终需要至少 3 seeds、6 datasets、1/5/10 shots 进行统计验证后，才能作为论文中的正式结果。"))

story.append(H2("5.4 当前阶段最终配置"))
story.append(KeepTogether([
    three_line(
        [["参数", "取值", "说明"],
         ["gamma", "8.0", "测试时后处理强度"],
         ["soft_mix", "0.35", "Hard/Soft 方向混合比例"],
         ["theta", "0.5", "门控阈值"],
         ["T_u", "0.1", "门控温度"],
         ["train_injection", "False", "暂不开启训练注入"]],
        colWidths=[40 * mm, 26 * mm, 72 * mm], aligns=["LEFT", "CENTER", "LEFT"]),
]))
story.append(Spacer(1, 4))
story.append(P("该配置只作为旧方法的对照配置，不应被视为最终 SOTA 方案。下一阶段的核心判断将是："
               "decoder-query 空间重建的 Soft prototype 能否把目前 0.591 ~ 0.593 的后处理收益"
               "转化为稳定、可训练、可跨数据集复现的性能提升。"))


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cccccc"))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, A4[0] - 20 * mm, 16 * mm)
    canvas.setFont("YaHei", 8.5)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0] / 2.0, 11 * mm, "S2H-CD-FSOD 周报 · %d" % doc.page)
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=17 * mm, bottomMargin=20 * mm,
                      title="S2H-CD-FSOD 阶段周报", author="S2H-CD-FSOD")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])
doc.build(story)
print("OK ->", OUT)

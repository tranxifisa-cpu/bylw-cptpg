"""Localize labels in completed vector figures without changing their data or legends."""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output_paper" / "DRCPT_PG_Full_Paper_E1_E4" / "figures"
TARGET = ROOT / "output_paper" / "应统作文模板" / "figures"
FONT = Path("C:/Windows/Fonts/simhei.ttf")

# Only explanatory labels are translated. Method names, legend entries, symbols,
# and numerical annotations keep their original wording and values.
FIGURES = {
    "e1_path_and_decision_final": (
        SOURCE / "e1_path_and_decision_final.pdf",
        {
            "Start": "起点",
            "History": "历史节点",
            "Matched endpoint": "相同终点",
            "Normalized wealth / reference": "归一化财富／参考点",
            "A  History separates the reference point": "A  路径改变参考点",
            "B  The same wealth enters different CPT domains": "B  相同财富落入不同的 CPT 区间",
            "Exact CPT objective": "精确 CPT 目标值",
            " = 0.7: 44% risky preferred": " = 0.7：优选 44% 风险资产",
            " = 0.7: 56% risky preferred": " = 0.7：优选 56% 风险资产",
            "C  Rise then fall": "C  先涨后跌",
            "D  Fall then rise": "D  先跌后涨",
        },
        ("Rise then fall: wealth", "Fall then rise: wealth", "Asymmetric", "Symmetric", "Static"),
    ),
    "e2_estimator_diagnostics": (
        SOURCE / "e2_estimator_diagnostics.pdf",
        {
            "Expected trajectory budget B": "预期轨迹预算 B",
            "Bias in the momentum coordinate": "动量因子坐标偏差",
            "A  Finite-sample bias": "A  有限样本偏差",
            "Independent complete outputs ": "独立完整估计量 ",
            "Trace covariance": "协方差迹",
            "Standard-normal quantiles": "标准正态分位数",
            "Studentized quantiles": "标准化统计量分位数",
            "C  Studentized normality": "C  标准化统计量的正态性",
            "Gradient MSE": "梯度 MSE",
            "D  Fixed-dimensional budget scaling": "D  固定维度的预算速率",
        },
        ("Proposed estimator", "CPT-PG v5 plug-in"),
    ),
    "e3_online_tracking": (
        SOURCE / "e3_online_tracking.pdf",
        {
            "Learning episode ": "学习时期 ",
            "A  Cumulative squared projected residual": "A  累积平方投影残差",
            "B  Average squared projected residual": "B  平均平方投影残差",
            "C  Cumulative DLR": "C  累积动态局部遗憾",
            "D  Average DLR": "D  平均动态局部遗憾",
        },
        ("Online update", "Frozen control"),
    ),
    "e4_robustness": (
        SOURCE / "e4_robustness.pdf",
        {
            "gain adaptation ": "收益侧适应速度 ",
            "loss adaptation ": "损失侧适应速度 ",
            "A  Reference adaptation": "A  参考点适应",
            "episode horizon ": "时期长度 ",
            "B  Episode horizon": "B  时期长度",
            "step size ": "步长 ",
            "normalization floor ": "归一化下界 ",
            "C  Optimizer scale": "C  优化器尺度",
        },
        (),
    ),
    "figure_E5_external_validity": (
        ROOT / "results" / "e5_hs300_e3sync_expcompatible_beta6_c10" / "figures" / "figure_E5_external_validity.pdf",
        {
            "Episode": "时期",
            "Normalized wealth": "归一化财富",
            "A  Normalized wealth": "A  归一化财富",
            "B  Drawdown and recovery": "B  回撤与恢复",
            "Drawdown from running peak (lower is better)": "相对历史峰值的回撤（越低越好）",
            "Maximum drawdown (lower is better)": "最大回撤（越低越好）",
            "Terminal wealth": "期末财富",
            "C  Terminal wealth, drawdown, and turnover": "C  期末财富、回撤与换手",
            "D  Turnover and transaction cost": "D  换手与交易成本",
            "Mean total turnover": "平均累计换手",
            "Mean cumulative transaction cost": "平均累计交易成本",
        },
        ("DRCPT-PG", "Symmetric CPT-PG", "Static CPT-PG", "Expected-Wealth PG", "Exponential-Utility PG"),
    ),
    "figure_E6_behavior_alignment": (
        ROOT / "results" / "e6_reference_behavior_eta041" / "model" / "figures" / "figure_E6_behavior_alignment.pdf",
        {
            "OOS error improvement vs no reference": "较无参考点模型的样本外误差改善",
            "A  Incremental decision-prediction value": "A  决策预测的增量价值",
            "Observed reduction rate": "实际减持率",
            "B  Realized behavior under each reference partition": "B  各参考点划分下的实际行为",
            "Gain minus loss reduction rate": "盈利减持率－亏损减持率",
            "C  Reference-conditioned disposition gap": "C  参考点条件下的处置效应差值",
            "Log-loss improvement vs no reference": "较无参考点模型的 log loss 改善",
            "D  Incremental value by pre-test investor history": "D  测试前投资者历史分层",
            "reduction: low": "低减持倾向",
            "reduction: medium": "中减持倾向",
            "reduction: high": "高减持倾向",
            "activity: low": "低交易活跃度",
            "activity: medium": "中交易活跃度",
            "activity: high": "高交易活跃度",
        },
        ("Static cost", "Symmetric adaptive", "Asymmetric adaptive", "Log loss", "Brier score"),
    ),
    "figure_E7_heterogeneous_agents": (
        ROOT / "results" / "e7_heterogeneous_agents" / "figures" / "figure_E7_heterogeneous_agents.pdf",
        {
            "e6_pgr": "PGR",
            "e6_plr": "PLR",
            "e6_disposition": "处置效应",
            "transnum": "交易次数",
            "stknum": "持股数",
            "sentiment_balance": "情绪倾向",
            "following_count": "关注数",
            "Active diversified investors": "活跃分散型投资者",
            "Disposition-effect investors": "处置效应型投资者",
            "Low-frequency diversified investors": "低频分散型投资者",
            "Moderate-frequency focused investors": "中频集中型投资者",
            "A  Calibrated investor profiles": "A  校准后的投资者类型",
            "B  Episode-level adoption by investor type": "B  各类投资者的时期采纳率",
            "C  Five-day satisfaction after adoption": "C  采纳后的五日满意度",
            "Average simulated adoption rate": "平均模拟采纳率",
            "D  Method-level adoption--satisfaction trade-off": "D  各方法的采纳率与满意度",
        },
        ("DRCPT-PG", "Symmetric CPT-PG", "Static CPT-PG", "Expected-Wealth PG", "Exponential-Utility PG"),
    ),
}

COMPOUNDS = {
    "e1_path_and_decision_final": (
        ("Terminal relative outcome ", "期末相对结果 X_T-r_T", "center"),
        ("Probability of +4% risky return, ", "风险资产上涨 4% 的概率 p", "center"),
    ),
    "e2_estimator_diagnostics": (
        ("B  Variance and ", "B  方差与 1/M 速率", "left"),
    ),
    "e4_robustness": (
        ("post-change DLR ratio DLR", "突变后 DLR 比值", "center"),
    ),
    "figure_E7_heterogeneous_agents": (
        ("Five-day satisfaction after adoption (×10", "采纳后的五日满意度（×10^4）", "center"),
    ),
}


def anchor(text: str, direction: tuple[float, float]) -> str:
    if text.startswith(("A  ", "B  ", "C  ", "D  ")):
        return "left"
    if text in {"Start", "History", "Matched endpoint"}:
        return "center"
    if text.endswith("investors") or text.startswith(("reduction:", "activity:")):
        return "right" if direction[0] > .95 else "center"
    if text in {"e6_pgr", "e6_plr", "e6_disposition", "transnum", "stknum",
                "sentiment_balance", "following_count"}:
        return "right"
    if text.endswith((" ", " (")):
        return "right"
    return "center"


def color_from_int(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 255) / 255 for shift in (16, 8, 0))


def redact_span(page: fitz.Page, span: dict) -> None:
    for char in span["chars"]:
        if char["c"].isspace():
            continue
        rect = fitz.Rect(char["bbox"])
        center = (rect.tl + rect.br) / 2
        page.add_redact_annot(fitz.Rect(center.x - .1, center.y - .1,
                                       center.x + .1, center.y + .1), fill=False)


def render(stem: str, source: Path, words: dict[str, str], protected: tuple[str, ...]) -> None:
    doc = fitz.open(source)
    page = doc[0]
    original = page.get_text()
    changes = []
    found = set()
    compounds_found = set()
    compounds = {old: (new, alignment) for old, new, alignment in COMPOUNDS.get(stem, ())}
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            direction = line["dir"]
            for span in line["spans"]:
                span["text"] = "".join(char["c"] for char in span["chars"])
            compound = next((key for key in compounds
                             if any(span["text"].startswith(key) for span in line["spans"])), None)
            if compound is not None:
                compounds_found.add(compound)
                spans = line["spans"]
                if stem == "e2_estimator_diagnostics":
                    # Matplotlib places the superscript in a separate PDF text line.
                    spans = [span for sibling in block["lines"] for span in sibling["spans"]]
                rect = fitz.Rect(spans[0]["bbox"])
                for span in spans:
                    rect |= fitz.Rect(span["bbox"])
                    redact_span(page, span)
                new, alignment = compounds[compound]
                changes.append((spans[0], direction, new, alignment, rect))
                continue
            if stem == "e2_estimator_diagnostics" and compounds_found and any(
                    span["text"] == " scaling" for span in line["spans"]):
                continue
            for span in line["spans"]:
                old = span["text"]
                if old not in words:
                    continue
                found.add(old)
                changes.append((span, direction, words[old], anchor(old, direction),
                                fitz.Rect(span["bbox"])))
                redact_span(page, span)
    missing = set(words) - found
    if missing:
        raise ValueError(f"{stem}: unmatched labels: {sorted(missing)}")
    if set(compounds) != compounds_found:
        raise ValueError(f"{stem}: unmatched compound labels: {set(compounds) - compounds_found}")

    page.apply_redactions(images=0, graphics=0)
    page.insert_font(fontname="SimHei", fontfile=str(FONT))
    font = fitz.Font(fontfile=str(FONT))
    for span, direction, new, alignment, old_rect in changes:
        if stem == "e2_estimator_diagnostics" and new.startswith("B  方差"):
            page.draw_rect(old_rect, color=None, fill=(1, 1, 1), overlay=True)
        dx, dy = direction
        old_length = old_rect.width if abs(dx) >= .95 else (old_rect.height if abs(dy) >= .95 else max(old_rect.width / abs(dx), old_rect.height / abs(dy)))
        size = span["size"]
        new_length = font.text_length(new, fontsize=size)
        if alignment == "left" and new_length > old_length * 1.25:
            size *= old_length * 1.25 / new_length
            new_length = font.text_length(new, fontsize=size)
        shift = {"left": 0, "center": (old_length - new_length) / 2,
                 "right": old_length - new_length}[alignment]
        x, y = span["origin"]
        point = fitz.Point(x + dx * shift, y + dy * shift)
        matrix = fitz.Matrix(dx, -dy, dy, dx, 0, 0)
        page.insert_text(point, new, fontname="SimHei", fontsize=size,
                         color=color_from_int(span["color"]), morph=(point, matrix))

    text = page.get_text()
    for old in protected:
        if original.count(old) != text.count(old):
            raise ValueError(f"{stem}: legend or method label changed: {old}")
    for old in words:
        if old in text and old not in words.values():
            raise ValueError(f"{stem}: untranslated label remains: {old}")
    TARGET.mkdir(parents=True, exist_ok=True)
    doc.subset_fonts()
    doc.save(TARGET / f"{stem}_zh.pdf", garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(TARGET / f"{stem}_zh.png")
    doc.close()


def main() -> None:
    for stem, (source, words, protected) in FIGURES.items():
        render(stem, source, words, protected)
        print(f"{stem}: {len(words)} labels localized")


if __name__ == "__main__":
    main()

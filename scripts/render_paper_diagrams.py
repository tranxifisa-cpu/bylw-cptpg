"""Render the paper's research framework and experimental evidence map."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "output_paper" / "DRCPT_PG_Full_Paper_E1_E4" / "figures"
INK = "#33434f"
ARROW = "#687d91"
BLUE = "#2a6796"
MAP_BLUE = "#1d628d"
PURPLE = "#735095"
GREEN = "#1f7d75"
ORANGE = "#af6a16"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float


def canvas(width: int, height: int):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def panel(ax, box: Box, edge: str, fill: str, radius: int = 28):
    ax.add_patch(
        FancyBboxPatch(
            (box.x, box.y), box.w, box.h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=3, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )


def arrow(ax, start, end, width: float = 2.6):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=22,
            linewidth=width, color=ARROW, zorder=3,
        )
    )


def label(ax, checks, box: Box, x: float, y: float, value: str,
          size: float, color: str = INK, bold: bool = False,
          minimum: float | None = None):
    artist = ax.text(
        x, y, value, ha="left", va="center", fontsize=size,
        fontweight="bold" if bold else "normal", color=color,
        fontfamily="DejaVu Sans", zorder=4,
    )
    checks.append((artist, box, minimum or size - 5))


def validate_text(fig, ax, checks):
    fig.canvas.draw()
    for artist, box, minimum in checks:
        while True:
            bounds = artist.get_window_extent(fig.canvas.get_renderer())
            corners = ax.transData.inverted().transform(
                ((bounds.x0, bounds.y0), (bounds.x1, bounds.y1))
            )
            left, right = sorted((corners[0, 0], corners[1, 0]))
            top, bottom = sorted((corners[0, 1], corners[1, 1]))
            if (left >= box.x + 25 and right <= box.x + box.w - 25
                    and top >= box.y + 20 and bottom <= box.y + box.h - 20):
                break
            next_size = artist.get_fontsize() - 0.5
            if next_size < minimum:
                raise ValueError(f"Text exceeds panel: {artist.get_text()}")
            artist.set_fontsize(next_size)
            fig.canvas.draw()


def export(fig, ax, checks, stem: str):
    validate_text(fig, ax, checks)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=100, facecolor="white")
    plt.close(fig)


def research_framework():
    fig, ax = canvas(3124, 1675)
    checks = []
    left = Box(140, 115, 844, 407)
    middle = Box(1139, 115, 844, 407)
    right = Box(2140, 115, 844, 407)
    statistical = Box(447, 666, 935, 364)
    online = Box(1740, 666, 935, 364)
    empirical = Box(354, 1176, 2414, 347)

    for start, end in (
        ((956, 320), (1160, 320)),
        ((1957, 320), (2160, 320)),
        ((1615, 500), (930, 680)),
        ((2540, 500), (2200, 680)),
        ((920, 1015), (1065, 1192)),
        ((2200, 1015), (2060, 1192)),
    ):
        arrow(ax, start, end, 3)

    panel(ax, left, BLUE, "#f2f3f9", 34)
    panel(ax, middle, BLUE, "#f2f3f9", 34)
    panel(ax, right, PURPLE, "#f5f5fb", 34)
    panel(ax, statistical, GREEN, "#edf8f6", 34)
    panel(ax, online, GREEN, "#edf8f6", 34)
    panel(ax, empirical, ORANGE, "#fbf4ea", 34)

    label(ax, checks, left, 210, 225, "Behavioral motivation", 38, BLUE, True, 32)
    label(ax, checks, left, 210, 310, "Adaptive and asymmetric reference", 23)
    label(ax, checks, left, 210, 360, "updating", 23)
    label(ax, checks, left, 210, 425, "Same wealth can imply different", 23)
    label(ax, checks, left, 210, 475, "gain\u2013loss domains", 23)

    label(ax, checks, middle, 1209, 225, "Dynamic-reference CPT model", 38, BLUE, True, 29)
    label(ax, checks, middle, 1209, 320, "State: wealth, reference, and portfolio", 23)
    label(ax, checks, middle, 1209, 390, "Dirichlet policy for long-only allocation", 23)

    label(ax, checks, right, 2210, 225, "Learning method", 38, PURPLE, True, 32)
    label(ax, checks, right, 2210, 310, "Unbiased frozen-target gradient", 23)
    label(ax, checks, right, 2210, 360, "estimator", 23)
    label(ax, checks, right, 2210, 425, "Normalized projected online update", 23)

    label(ax, checks, statistical, 520, 770, "Statistical guarantees", 38, GREEN, True, 32)
    label(ax, checks, statistical, 520, 850, "Exact unbiasedness, finite variance, and CLT", 23)
    label(ax, checks, statistical, 520, 915, "Fixed-dimensional minimax", 23)
    label(ax, checks, statistical, 520, 965, "trajectory-budget rate", 23)

    label(ax, checks, online, 1813, 770, "Online-tracking guarantees", 38, GREEN, True, 32)
    label(ax, checks, online, 1813, 850, "Projected residual and dynamic local regret", 23)
    label(ax, checks, online, 1813, 915, "Stationary-set tracking under an error bound", 23)

    label(ax, checks, empirical, 485, 1265, "Empirical evaluation", 42, ORANGE, True, 36)
    label(ax, checks, empirical, 485, 1350,
          "E1\u2013E4: mechanism, estimator diagnostics, online tracking, and ablation / robustness", 23)
    label(ax, checks, empirical, 485, 1415,
          "E5\u2013E7: chronological A-share replay, held-out investor behavior, and heterogeneous-agent interaction", 23)
    export(fig, ax, checks, "research_framework")


def experiment_evidence_map():
    fig, ax = canvas(2048, 1376)
    checks = []
    e1 = Box(61, 56, 635, 265)
    e2 = Box(707, 56, 635, 265)
    e3 = Box(1352, 56, 635, 265)
    e4 = Box(616, 431, 816, 260)
    e5 = Box(152, 829, 855, 244)
    e6 = Box(1040, 829, 855, 244)
    e7 = Box(597, 1113, 855, 236)

    for start, end in (
        ((670, 190), (725, 190)),
        ((1316, 190), (1370, 190)),
        ((380, 305), (690, 445)),
        ((1024, 305), (1024, 445)),
        ((1665, 305), (1361, 445)),
        ((859, 675), (630, 843)),
        ((1190, 675), (1420, 843)),
        ((584, 1056), (797, 1126)),
        ((1464, 1056), (1253, 1126)),
    ):
        arrow(ax, start, end)

    for box in (e1, e2, e3):
        panel(ax, box, MAP_BLUE, "#f3f6fb", 22)
    panel(ax, e4, PURPLE, "#f6f6fb", 22)
    for box in (e5, e6):
        panel(ax, box, GREEN, "#f1f9f7", 22)
    panel(ax, e7, ORANGE, "#fcf6ef", 22)

    label(ax, checks, e1, 121, 145, "E1  Path dependence", 26, MAP_BLUE, True, 22)
    label(ax, checks, e1, 121, 218, "Matched wealth; different histories", 18)
    label(ax, checks, e1, 121, 269, "Reference domain + portfolio ranking", 17, "#526579")

    label(ax, checks, e2, 767, 145, "E2  Finite-sample gradient estimation", 26, MAP_BLUE, True, 20)
    label(ax, checks, e2, 767, 218, "Four fixed targets; matched expected cost", 18)
    label(ax, checks, e2, 767, 269, "Bias / variance / CLT / MSE scaling", 17, "#526579")

    label(ax, checks, e3, 1412, 145, "E3  Continuous online tracking", 26, MAP_BLUE, True, 21)
    label(ax, checks, e3, 1412, 218, "Same setup; slow dynamic state path", 18)
    label(ax, checks, e3, 1412, 269, "Residual + DLR, cumulative / average", 17, "#526579")

    label(ax, checks, e4, 676, 520, "E4  Reference ablation and robustness", 27, PURPLE, True, 22)
    label(ax, checks, e4, 676, 590, "Reference dynamics / horizon / optimizer scale", 18)
    label(ax, checks, e4, 676, 640, "Where does the tracking advantage strengthen or weaken?", 17, "#526579")

    label(ax, checks, e5, 212, 916, "E5  Real-market portfolio evaluation", 27, GREEN, True, 22)
    label(ax, checks, e5, 212, 990, "Time-ordered A-share backtest", 18)
    label(ax, checks, e5, 212, 1040, "Risk-adjusted performance + trading intensity", 17, "#526579")

    label(ax, checks, e6, 1100, 916, "E6  Investor behavior prediction", 27, GREEN, True, 22)
    label(ax, checks, e6, 1100, 990, "Held-out sell / hold opportunities", 18)
    label(ax, checks, e6, 1100, 1040, "Held-out prediction + behavioral alignment", 17, "#526579")

    label(ax, checks, e7, 657, 1195, "E7  Simulated investor interaction", 27, ORANGE, True, 22)
    label(ax, checks, e7, 657, 1265, "Heterogeneous agents informed by E6", 18)
    label(ax, checks, e7, 657, 1315, "Adoption + satisfaction", 17, "#526579")
    export(fig, ax, checks, "experiment_evidence_map")


def main():
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    research_framework()
    experiment_evidence_map()


if __name__ == "__main__":
    main()

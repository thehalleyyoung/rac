"""Figure: the RAC loop, where the two objectives fork, and how they relate.

Boxes are sized from their own content (one unit = one text line), so nothing
overflows when the wording changes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

SHARED, SHARED_E = "#e9eff7", "#33507a"        # spine arrows and the recursion rail
# one colour per shared step, so the text can name a step by its colour
STEP = [
    ("#ede7f6", "#5b3fa0"),   # 1 elicit        violet
    ("#e0ecfa", "#2f5d9e"),   # 2 score         blue
    ("#e0f1f5", "#1f6f7a"),   # 3 spec          teal
    ("#fdf3dd", "#8a6a12"),   # 4 gate          amber
    ("#fae5e5", "#9e2a2b"),   # 5 select        crimson
    ("#efefef", "#5a5a5a"),   # 6 append        grey
    ("#fbe6f0", "#9c2f66"),   # 7 mine/refine   magenta
]
MAXMIN, MAXMIN_E = "#fdeae1", "#b9502f"
COVER,  COVER_E  = "#e4f1e9", "#26714b"
BAND,   BAND_E   = "#f4f4f4", "#8a8a8a"

FS = 8.6            # body font size
TITLE_UNITS = 1.55  # vertical units taken by a title
PAD = 0.55          # padding above title and below last body line


class Box:
    def __init__(self, x, w, title, body, fc, ec):
        self.x, self.w, self.title, self.fc, self.ec = x, w, title, fc, ec
        self.body = body.split("\n") if body else []
        self.h = PAD * 2 + TITLE_UNITS + len(self.body) * 1.12
        self.y = None

    @property
    def cx(self): return self.x + self.w / 2

    @property
    def cy(self): return self.y + self.h / 2

    def draw(self, ax):
        ax.add_patch(FancyBboxPatch((self.x, self.y), self.w, self.h,
                                    boxstyle="round,pad=0.10,rounding_size=0.30",
                                    facecolor=self.fc, edgecolor=self.ec, lw=1.4, zorder=2))
        ax.text(self.cx, self.y + self.h - PAD, self.title, ha="center", va="top",
                fontsize=FS + 1.1, fontweight="bold", color=self.ec, zorder=3)
        if self.body:
            ax.text(self.cx, self.y + self.h - PAD - TITLE_UNITS, "\n".join(self.body),
                    ha="center", va="top", fontsize=FS, color="#141414",
                    linespacing=1.5, zorder=3)


def arrow(ax, p, q, color, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=lw,
                                 color=color, linestyle=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


def main():
    LX, LW = 0.0, 9.4
    CX, CW = 11.4, 9.6
    RX, RW = 23.0, 9.4
    GAP = 1.5

    spine = [
        Box(CX, CW, "1 · Elicit axes",
            "Ask the generator to name the axes its\nown outputs vary along, each with\ndiscrete language-valued levels",
            SHARED, SHARED_E),
        Box(CX, CW, "2 · Score axes, pick levels",
            "Which condition moves the reachable\nslice somewhere the corpus has not been?",
            SHARED, SHARED_E),
        Box(CX, CW, "3 · Spec → K candidates",
            "One level per active axis, stated as\ncontracts, plus the ledger's avoid-list",
            SHARED, SHARED_E),
        Box(CX, CW, "4 · Judge gate",
            "Reject-only. Passing is not evidence\nof quality.", SHARED, SHARED_E),
        Box(CX, CW, "5 · Select among the survivors",
            "Which of the K candidates to keep.", SHARED, SHARED_E),
        Box(CX, CW, "6 · Append, update bounded state",
            "Per-item cost is O(1) in n: item 10,000\ncosts what item 100 cost", SHARED, SHARED_E),
        Box(CX, CW, "7 · Mine attractors, refine axes",
            "\"What do these have in common?\" → ledger.\nAn axis with nothing transverse left is\nsplit into conditional sub-axes",
            SHARED, SHARED_E),
    ]
    for b, (fc, ec) in zip(spine, STEP):
        b.fc, b.ec = fc, ec
    y = 0.0
    for b in reversed(spine):          # lay out bottom-up, then read top-down
        b.y = y
        y += b.h + GAP
    top = y - GAP

    fork_score, fork_select = spine[1], spine[4]

    mm_score = Box(LX, LW, "Score: four-factor product",
                   "spread × transversality ×\nindependence × headroom\n\n"
                   "Headroom is an entropy deficit over the\naxis's own levels. Values chosen by\n"
                   "farthest-point search in level space.", MAXMIN, MAXMIN_E)
    mm_sel = Box(LX, LW, "Select: utility, best-of-K",
                 "0.4·quality + 0.35·orthogonality\n+ 0.25·capped gap\n\n"
                 "State: running centroid, second-moment\nmatrix, EMA scales. One of K survives.",
                 MAXMIN, MAXMIN_E)
    cv_score = Box(RX, RW, "Score: one estimable scalar",
                   "expected marginal ε-ball gain of\nconditioning on the axis\n\n"
                   "It subsumes all four factors, and its\nheadroom is measure-weighted: a large\n"
                   "under-covered region beats a small one.", COVER, COVER_E)
    cv_sel = Box(RX, RW, "Select: marginal gain, keep all",
                 "uncovered reference points newly\nfalling within ε of the candidate\n\n"
                 "State: reference pool + covered mask.\nCoverage is monotone in items, so every\n"
                 "discard is a permanent loss.", COVER, COVER_E)
    for side, anchor in [(mm_score, fork_score), (mm_sel, fork_select),
                         (cv_score, fork_score), (cv_sel, fork_select)]:
        side.y = anchor.cy - side.h / 2

    cv_aim = Box(RX, RW, "Aim by retrieval (no inverse oracle)",
                 "Sample an uncovered reference region\n∝ 1/r², retrieve the nearest text we\n"
                 "already own, and use it as the exemplar.", COVER, COVER_E)
    cv_aim.y = cv_score.y - GAP - cv_aim.h

    fig_h = (top + 14.6) * 0.235
    fig, ax = plt.subplots(figsize=(15.2, fig_h))
    ax.set_xlim(-1.0, RX + RW + 1.0)
    ax.set_ylim(-13.1, top + 7.2)
    ax.axis("off")

    ax.text((CX + CW / 2), top + 6.4, "Recursive Axis Conditioning: one loop, two objectives",
            ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax.text((CX + CW / 2), top + 4.6,
            "The generator can be steered only through language, because there is no inverse "
            "embedding oracle.\nThe loop is shared; the objective enters at exactly two points.",
            ha="center", va="top", fontsize=9.6, color="#3c3c3c", linespacing=1.5)

    ax.text(LX + LW / 2, top + 2.0, "MAX-MIN · no two items alike", ha="center",
            fontsize=12, fontweight="bold", color=MAXMIN_E)
    ax.text(LX + LW / 2, top + 0.6, "packing · unbounded horizon · no reference set",
            ha="center", fontsize=8.8, color=MAXMIN_E)
    ax.text(RX + RW / 2, top + 2.0, "COVERAGE · reach as much as possible", ha="center",
            fontsize=12, fontweight="bold", color=COVER_E)
    ax.text(RX + RW / 2, top + 0.6, "covering · budget of n · scored against a reference",
            ha="center", fontsize=8.8, color=COVER_E)

    for b in spine + [mm_score, mm_sel, cv_score, cv_sel, cv_aim]:
        b.draw(ax)
    for a, b in zip(spine, spine[1:]):
        arrow(ax, (a.cx, a.y), (b.cx, b.y + b.h), SHARED_E)

    arrow(ax, (fork_score.x, fork_score.cy), (mm_score.x + mm_score.w, mm_score.cy), MAXMIN_E)
    arrow(ax, (fork_select.x, fork_select.cy), (mm_sel.x + mm_sel.w, mm_sel.cy), MAXMIN_E)
    arrow(ax, (fork_score.x + fork_score.w, fork_score.cy), (cv_score.x, cv_score.cy), COVER_E)
    arrow(ax, (fork_select.x + fork_select.w, fork_select.cy), (cv_sel.x, cv_sel.cy), COVER_E)
    arrow(ax, (cv_aim.x, cv_aim.cy), (spine[2].x + spine[2].w, spine[2].cy), COVER_E,
          ls=(0, (5, 3)))

    # recursion: step 7 back to step 2, routed down the right of the spine
    rail = CX + CW + 0.85
    ax.plot([spine[6].x + spine[6].w, rail], [spine[6].cy, spine[6].cy],
            color=SHARED_E, lw=1.6, zorder=1)
    ax.plot([rail, rail], [spine[6].cy, fork_score.cy - 1.2], color=SHARED_E, lw=1.6, zorder=1)
    arrow(ax, (rail, fork_score.cy - 1.2), (fork_score.x + fork_score.w, fork_score.cy - 1.2),
          SHARED_E)
    ax.text(rail + 0.42, (spine[6].cy + spine[5].cy) / 2,
            "recursion: the one step that raises the ceiling",
            ha="center", va="center", fontsize=8.6, style="italic", color=SHARED_E,
            rotation=90)

    # ---- relation band ------------------------------------------------
    band_y, band_h = -12.6, 11.2
    ax.add_patch(FancyBboxPatch((LX, band_y), RX + RW - LX, band_h,
                                boxstyle="round,pad=0.10,rounding_size=0.30",
                                facecolor=BAND, edgecolor=BAND_E, lw=1.3, zorder=2))
    ax.text(LX + 0.6, band_y + band_h - 0.55, "How the two relate", fontsize=10.4,
            fontweight="bold", color="#2b2b2b", va="top", zorder=3)
    cols = [
        ("Dual at maximality", SHARED_E,
         "A maximal ε-packing is already an ε-covering,\n"
         "and N_cov(ε) ≤ N_pack(ε) ≤ N_cov(ε/2). Run\n"
         "max-min to exhaustion and coverage follows."),
        ("Not dual at a finite budget", MAXMIN_E,
         "Greedy k-center wins min-gap in both domains\n"
         "and finishes last on coverage, at a sixth of\n"
         "random. Each objective's own tool degrades\n"
         "the other: orthogonalized conditioning scores\n"
         "0.294 on coverage, below plain conditioning's\n0.315."),
        ("Same binding constraint", COVER_E,
         "Both are limited by the conditional dimension\n"
         "m rather than by their optimizers, and both\n"
         "need the typicality gate. Only max-min lacks a\n"
         "reference measure, which is why its steered\n"
         "corpora drift toward regions real examples\nrarely occupy."),
    ]
    cw_ = (RX + RW - LX) / 3
    for k, (h, c, b) in enumerate(cols):
        x0 = LX + k * cw_ + 0.6
        ax.text(x0, band_y + band_h - 2.4, h, fontsize=9.4, fontweight="bold",
                color=c, va="top", zorder=3)
        ax.text(x0, band_y + band_h - 3.9, b, fontsize=8.3, color="#141414", va="top",
                linespacing=1.5, zorder=3)

    fig.savefig(HERE / "figures" / "fig15_rac_diagram.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("wrote figures/fig15_rac_diagram.png")


if __name__ == "__main__":
    main()

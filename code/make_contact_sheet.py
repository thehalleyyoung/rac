"""Two-panel contact sheet: naive renders against max-min steered renders."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
PANELS = [
    ("naive prompting", HERE / "real" / "dalle_naive" / "images_high"),
    ("max-min steering,\nliteral + latent repulsion", HERE / "real" / "dalle_steer2_maxmin" / "images"),
]
COLS, ROWS = 8, 2


def sheet(out="figures/fig13_contact_sheet.png"):
    fig, axes = plt.subplots(len(PANELS) * ROWS, COLS,
                             figsize=(COLS * 1.6, len(PANELS) * ROWS * 1.6))
    for pi, (label, d) in enumerate(PANELS):
        imgs = sorted(d.glob("*.png"))[: COLS * ROWS]
        for k in range(COLS * ROWS):
            ax = axes[pi * ROWS + k // COLS][k % COLS]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k < len(imgs):
                ax.imshow(Image.open(imgs[k]).convert("RGB").resize((256, 256)))
        # row label down the left edge of the panel's first row
        axes[pi * ROWS][0].set_ylabel(label, fontsize=9, rotation=0,
                                      ha="right", va="center", labelpad=14)
        for sp in axes[pi * ROWS][0].spines.values():
            sp.set_visible(False)
    fig.suptitle("Sixteen renders per policy, same generator, same budget, "
                 "same image model", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.06, 0, 1, 0.98])
    fig.savefig(HERE / out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    sheet()

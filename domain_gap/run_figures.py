"""
Figures + gap-vs-AP correlation for the domain-gap analysis.

Reads outputs/statistical_results.json and outputs/latent_results.json and
produces:
  1. MMD level-decomposition figure (scene/bbox/mask x real-2D/real-3D) for
     each feature space -- the headline 'background vs fragment' story.
  2. Per-class gap-vs-AP correlation (Spearman): dataset gap should predict
     the AP of the model trained purely on that dataset.
  3. Per-class gap-vs-AP scatter for the most illustrative space/level.
  4. t-SNE of instance embeddings coloured by domain (from cached embeddings).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({          # match the manuscript font (as in Figure 5)
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).parent
OUT = HERE / "outputs"
CACHE = HERE / "cache"
CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]

# paper figure names (PDF is the manuscript's figure format)
PAPER_NAMES = {
    "fig_mmd_decomposition": "fig_4_4_domain_gap_levels",
    "fig_gap_vs_ap": "fig_4_5_gap_vs_ap",
    "fig_tsne": "fig_4_6_tsne_domains",
}


def _save(fig, name):
    """Save PNG (quick view) and PDF (manuscript) under the paper's name."""
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{PAPER_NAMES.get(name, name)}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png + {PAPER_NAMES.get(name, name)}.pdf")

# per-class AP from paper Table 4
AP = {
    "2D": {"AAC": 0.669, "Ceramics": 0.944, "Mortar": 0.583, "Stones": 0.629, "Tiles": 0.912},
    "3D": {"AAC": 0.228, "Ceramics": 0.820, "Mortar": 0.414, "Stones": 0.027, "Tiles": 0.609},
}  # 3D uses Stage-1 (purely rendered) AP, matching the label-free dataset gap

SPACES = {"statistical": "Handcrafted", "dino": "DINOv2", "yolo": "YOLO-COCO"}
# Real/2D/3D dataset colors, matching the thesis results figures
DATASET_COLORS = {"real": "#E53935", "2D": "#43A047", "3D": "#1E88E5"}
LEVELS = ["scene", "bbox", "mask"]


def load():
    stat = json.loads((OUT / "statistical_results.json").read_text())
    lat = json.loads((OUT / "latent_results.json").read_text())
    return {"statistical": stat, "dino": lat["dino"], "yolo": lat["yolo"]}


def fig_decomposition(res):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, (key, title) in zip(axes, SPACES.items()):
        r2d = [res[key]["levels"][lv]["real-2D"]["mmd"] for lv in LEVELS]
        r3d = [res[key]["levels"][lv]["real-3D"]["mmd"] for lv in LEVELS]
        x = np.arange(len(LEVELS))
        ax.bar(x - 0.19, r2d, 0.36, label="real vs 2D", color=DATASET_COLORS["2D"])
        ax.bar(x + 0.19, r3d, 0.36, label="real vs 3D", color=DATASET_COLORS["3D"])
        ax.set_xticks(x); ax.set_xticklabels(["scene", "bbox", "mask"], fontsize=15)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("granularity", fontsize=16)
        if ax is axes[0]:
            ax.set_ylabel("MMD$^2$ (domain gap)", fontsize=16)
        ax.legend(fontsize=15)
    fig.tight_layout()  # no suptitle: the LaTeX caption carries the description
    _save(fig, "fig_mmd_decomposition")


def correlations(res):
    print("\n=== per-class gap vs AP (Spearman rho; expect NEGATIVE) ===")
    rows = []
    for key in res:
        for level in ["bbox", "mask"]:
            for model in ["2D", "3D"]:
                pc = res[key]["per_class"][level][f"real-{model}"]
                gaps = [pc[c]["mmd"] for c in CLASSES]
                aps = [AP[model][c] for c in CLASSES]
                rho, p = spearmanr(gaps, aps)
                rows.append((key, level, model, rho, p))
                print(f"  {SPACES[key]:12s} {level:5s} {model}: "
                      f"rho={rho:+.3f} p={p:.3f}")
    return rows


def fig_scatter(res, key="dino", level="mask"):
    """Single panel: both data routes share the same x (gap) and y (AP) axes,
    so overlaying them makes the contrast directly visible. Spearman rho/p go
    in the LaTeX caption, not on the figure."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    styles = {"2D": ("#4C78A8", "o", "composited (2D) data"),
              "3D": ("#E4572E", "s", "rendered (3D) data")}
    rhos = {}
    for model in ["2D", "3D"]:
        pc = res[key]["per_class"][level][f"real-{model}"]
        gaps = [pc[c]["mmd"] for c in CLASSES]
        aps = [AP[model][c] for c in CLASSES]
        rho, p = spearmanr(gaps, aps)
        rhos[model] = (rho, p)
        color, marker, lab = styles[model]
        ax.scatter(gaps, aps, color=color, marker=marker, s=55, label=lab,
                   zorder=3)
        for c, g, a in zip(CLASSES, gaps, aps):
            ax.annotate(c, (g, a), fontsize=7, color=color,
                        xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel(f"{SPACES[key]} {level}-level domain gap (MMD$^2$)")
    ax.set_ylabel("AP@0.5")
    ax.margins(x=0.10, y=0.10)   # room for the point labels
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "fig_gap_vs_ap")
    print(f"  rho 2D={rhos['2D'][0]:+.2f} (p={rhos['2D'][1]:.2f}) | "
          f"rho 3D={rhos['3D'][0]:+.2f} (p={rhos['3D'][1]:.2f})")
    return rhos


def fig_tsne(key="dino", level="bbox"):
    from sklearn.manifold import TSNE
    z = np.load(CACHE / "embeddings.npz", allow_pickle=True)
    emb = z[key].item()
    X, y = [], []
    for dom in ["real", "2D", "3D"]:
        e = emb[dom][level]
        X.append(e); y += [dom] * len(e)
    X = np.vstack(X)
    proj = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = DATASET_COLORS
    for dom in ["real", "2D", "3D"]:
        m = np.array(y) == dom
        ax.scatter(proj[m, 0], proj[m, 1], s=10, alpha=0.6,
                   color=colors[dom], label=dom)
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])   # t-SNE coordinates are arbitrary
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    fig.tight_layout()  # no title: the LaTeX caption carries the description
    _save(fig, "fig_tsne")


def main():
    res = load()
    fig_decomposition(res)
    correlations(res)
    # fig_scatter cut from the paper (per-class Spearman correlation removed)
    fig_tsne(key="dino", level="bbox")
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()

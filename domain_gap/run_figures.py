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
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).parent
OUT = HERE / "outputs"
CACHE = HERE / "cache"
CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]

# per-class AP from paper Table 4
AP = {
    "2D": {"AAC": 0.669, "Ceramics": 0.944, "Mortar": 0.583, "Stones": 0.629, "Tiles": 0.912},
    "3D": {"AAC": 0.228, "Ceramics": 0.820, "Mortar": 0.414, "Stones": 0.027, "Tiles": 0.609},
}  # 3D uses Stage-1 (purely rendered) AP, matching the label-free dataset gap

SPACES = {"statistical": "Handcrafted", "dino": "DINOv2", "yolo": "YOLO-COCO"}
LEVELS = ["scene", "bbox", "mask"]


def load():
    stat = json.loads((OUT / "statistical_results.json").read_text())
    lat = json.loads((OUT / "latent_results.json").read_text())
    return {"statistical": stat, "dino": lat["dino"], "yolo": lat["yolo"]}


def fig_decomposition(res):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (key, title) in zip(axes, SPACES.items()):
        r2d = [res[key]["levels"][lv]["real-2D"]["mmd"] for lv in LEVELS]
        r3d = [res[key]["levels"][lv]["real-3D"]["mmd"] for lv in LEVELS]
        x = np.arange(len(LEVELS))
        ax.bar(x - 0.19, r2d, 0.36, label="real vs 2D", color="#4C78A8")
        ax.bar(x + 0.19, r3d, 0.36, label="real vs 3D", color="#E4572E")
        ax.set_xticks(x); ax.set_xticklabels(["scene", "bbox", "mask"])
        ax.set_title(title); ax.set_xlabel("granularity")
        if ax is axes[0]:
            ax.set_ylabel("MMD$^2$ (domain gap)")
        ax.legend(fontsize=8)
    fig.suptitle("Domain gap by granularity — at the fragment (mask) level the 3D gap "
                 "exceeds the 2D gap in all three feature spaces", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mmd_decomposition.png", dpi=150)
    plt.close(fig)
    print("saved fig_mmd_decomposition.png")


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
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, model in zip(axes, ["2D", "3D"]):
        pc = res[key]["per_class"][level][f"real-{model}"]
        gaps = [pc[c]["mmd"] for c in CLASSES]
        aps = [AP[model][c] for c in CLASSES]
        rho, p = spearmanr(gaps, aps)
        ax.scatter(gaps, aps, color="#4C78A8" if model == "2D" else "#E4572E", s=60)
        for c, g, a in zip(CLASSES, gaps, aps):
            ax.annotate(c, (g, a), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel(f"{SPACES[key]} {level}-level gap (MMD$^2$)")
        ax.set_ylabel(f"{model} per-class AP@0.5")
        ax.set_title(f"real vs {model}   (Spearman ρ={rho:+.2f}, p={p:.2f})")
    fig.suptitle("Label-free per-class gap vs per-class detection AP", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gap_vs_ap.png", dpi=150)
    plt.close(fig)
    print("saved fig_gap_vs_ap.png")


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
    colors = {"real": "#333333", "2D": "#4C78A8", "3D": "#E4572E"}
    for dom in ["real", "2D", "3D"]:
        m = np.array(y) == dom
        ax.scatter(proj[m, 0], proj[m, 1], s=10, alpha=0.6,
                   color=colors[dom], label=dom)
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE of {SPACES[key]} {level} embeddings by domain")
    fig.tight_layout()
    fig.savefig(OUT / "fig_tsne.png", dpi=150)
    plt.close(fig)
    print("saved fig_tsne.png")


def main():
    res = load()
    fig_decomposition(res)
    correlations(res)
    fig_scatter(res, key="dino", level="mask")
    fig_tsne(key="dino", level="bbox")
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()

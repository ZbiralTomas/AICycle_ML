"""
Figure for Section 2.8: the three crops taken from one example instance
(scene / bbox / mask), shown side by side. The pipeline (encoders, MMD, gap)
is described in the text, so it is not drawn here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import gap_common as gc

OUT = __import__("pathlib").Path(__file__).parent / "outputs"

# ---- pick one clear real example instance ----
PAD = 0.18
inst = gc.sample_instances("real", ["test"], per_class=1, seed=0)
it = [x for x in inst if x["class"] == "Ceramics"][0]
full = gc.load_full(it["image_path"])
box, masked, fm = gc.crops_from(full, it["mask_path"], pad=PAD)
scene = gc.load_scene(it["image_path"], size=512)
mk = gc._binarize_mask(np.asarray(Image.open(it["mask_path"])))
ys, xs = np.where(mk); H, W = mk.shape
ph, pw = int((ys.max() - ys.min() + 1) * PAD), int((xs.max() - xs.min() + 1) * PAD)
sx, sy = 512 / W, 512 / H
bx0 = max(0, xs.min() - pw) * sx; bx1 = min(W, xs.max() + pw) * sx
by0 = max(0, ys.min() - ph) * sy; by1 = min(H, ys.max() + ph) * sy

fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, img, title, rect in zip(
        axes, [scene, box, masked], ["scene", "bbox", "mask"],
        [(bx0, bx1, by0, by1), None, None]):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.6)
    ax.set_title(title, fontsize=22, pad=6)
    if rect is not None:
        ax.add_patch(Rectangle((rect[0], rect[2]), rect[1] - rect[0],
                     rect[3] - rect[2], fill=False, ec="#FFD400", lw=3.0))

fig.tight_layout()
fig.savefig(OUT / "domain_gap_schema.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig_domain_gap_schema.pdf", bbox_inches="tight")
plt.close(fig)
print("saved domain_gap_schema.png + fig_domain_gap_schema.pdf   (example class:",
      it["class"], ")")

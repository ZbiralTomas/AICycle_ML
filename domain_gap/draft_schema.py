"""
Schematic of the domain-gap measurement strategy for Section 2.8.

Flow: one instance -> three crops (scene / bbox / mask) -> three neutral
encoders (rulers) -> MMD (+ energy, KID) -> gap of real vs each synthetic.
Uses a real example instance so the three crops are concrete.

Column colours: encoders = red, MMD = green, domain gap = blue.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

import gap_common as gc

OUT = __import__("pathlib").Path(__file__).parent / "outputs"

# column palette (fill, edge)
RED_F, RED_E = "#F4A7A7", "#C62828"     # encoders
GRN_F, GRN_E = "#B2DFB2", "#2E7D32"     # MMD
BLU_F, BLU_E = "#AFCDEC", "#1565C0"     # domain gap
INK = "#141414"

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

fig = plt.figure(figsize=(9, 8.5))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")


def rbox(cx, cy, w, h, text, fc, ec, fs=16, bold=False):
    ov.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.014",
                 fc=fc, ec=ec, lw=1.9, zorder=2))
    ov.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, color=INK)


def arrow(x0, x1, y, y1=None):
    ov.annotate("", xy=(x1, y1 or y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=2.3, color="#222222"), zorder=1)


def crop(x, y, w, h, arr, label, rect=None):
    a = fig.add_axes([x, y, w, h]); a.imshow(arr); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(1.6)
    a.set_title(label, fontsize=19, pad=4)
    if rect is not None:
        a.add_patch(Rectangle((rect[0], rect[2]), rect[1] - rect[0],
                    rect[3] - rect[2], fill=False, ec="#FFD400", lw=3.0))


# ---- stage 1: three crops (~2x larger, square) ----
cw, ch, cx = 0.264, 0.28, 0.06
crop(cx, 0.68, cw, ch, scene, "scene", rect=(bx0, bx1, by0, by1))
crop(cx, 0.37, cw, ch, box, "bbox")
crop(cx, 0.06, cw, ch, masked, "mask")

# ---- stage 2: three encoders (RED) ----
ex, ew = 0.48, 0.22
ov.text(ex, 0.965, "3 neutral encoders",
        ha="center", fontsize=14, style="italic")
for cy, t in [(0.72, "Handcrafted\n20-d"), (0.50, "DINOv2\n384-d"),
              (0.28, "YOLO-COCO\n512-d")]:
    rbox(ex, cy, ew, 0.15, t, RED_F, RED_E, fs=15)

# ---- stage 3: MMD (GREEN) ----
mx, mw = 0.71, 0.12
rbox(mx, 0.50, mw, 0.22, "MMD$^2$\n\n+ energy,\nKID", GRN_F, GRN_E, fs=14)

# ---- stage 4: domain gap (BLUE) ----
gx, gw = 0.90, 0.17
rbox(gx, 0.50, gw, 0.30, "domain gap\n\nreal vs 2D\nreal vs 3D\n\n(per level)",
     BLU_F, BLU_E, fs=14, bold=True)

# ---- arrows ----
arrow(cx + cw + 0.01, ex - ew / 2 - 0.005, 0.5)          # crops -> encoders
for yy in (0.72, 0.50, 0.28):                            # encoders -> MMD
    arrow(ex + ew / 2 + 0.005, mx - mw / 2 - 0.005, yy, 0.5)
arrow(mx + mw / 2 + 0.005, gx - gw / 2 - 0.005, 0.5)     # MMD -> gap

fig.savefig(OUT / "domain_gap_schema.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig_domain_gap_schema.pdf", bbox_inches="tight")
plt.close(fig)
print("saved domain_gap_schema.png + fig_domain_gap_schema.pdf   (example class:",
      it["class"], ")")

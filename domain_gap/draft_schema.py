"""
Schematic of the domain-gap measurement strategy for Section 2.8.

Flow: one instance -> three crops (scene / bbox / mask) -> three neutral
encoders (rulers) -> MMD (+ energy, KID) -> gap of real vs each synthetic.
Uses a real example instance so the three crops are concrete.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

import gap_common as gc

OUT = gc.Path if False else __import__("pathlib").Path(__file__).parent / "outputs"
BLUE, GREEN, RED = "#1E88E5", "#43A047", "#E53935"
INK, GREYBOX = "#222222", "#EEF1F4"

# ---- pick one clear real example instance ----
PAD = 0.18
inst = gc.sample_instances("real", ["test"], per_class=1, seed=0)
it = [x for x in inst if x["class"] == "Ceramics"][0]  # visually distinctive
full = gc.load_full(it["image_path"])
box, masked, fm = gc.crops_from(full, it["mask_path"], pad=PAD)
scene = gc.load_scene(it["image_path"], size=512)
mk = gc._binarize_mask(np.asarray(Image.open(it["mask_path"])))
ys, xs = np.where(mk); H, W = mk.shape
# padded box in native coords (matches crops_from), then scale to the 512 scene
ph, pw = int((ys.max() - ys.min() + 1) * PAD), int((xs.max() - xs.min() + 1) * PAD)
sx, sy = 512 / W, 512 / H
bx0 = max(0, xs.min() - pw) * sx; bx1 = min(W, xs.max() + pw) * sx
by0 = max(0, ys.min() - ph) * sy; by1 = min(H, ys.max() + ph) * sy

fig = plt.figure(figsize=(13, 4.7))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")


def rbox(cx, cy, w, h, text, fc=GREYBOX, ec=INK, fs=16, bold=False):
    ov.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.014",
                 fc=fc, ec=ec, lw=1.4, zorder=2))
    ov.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, color=INK)


def arrow(x0, x1, y, y1=None):
    ov.annotate("", xy=(x1, y1 or y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=INK), zorder=1)


def crop(x, y, w, h, arr, label, rect=None):
    a = fig.add_axes([x, y, w, h]); a.imshow(arr); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(1.3)
    a.set_title(label, fontsize=16, pad=3)
    if rect is not None:
        a.add_patch(Rectangle((rect[0], rect[2]), rect[1] - rect[0],
                    rect[3] - rect[2], fill=False, ec="#FFD400", lw=2.6))


# ---- stage 1: three crops ----
cw, ch, cx = 0.135, 0.225, 0.045
crop(cx, 0.66, cw, ch, scene, "scene", rect=(bx0, bx1, by0, by1))
crop(cx, 0.375, cw, ch, box, "bbox")
crop(cx, 0.09, cw, ch, masked, "mask")
ov.text(cx + cw / 2, 0.035, "3 crops per instance", ha="center",
        style="italic", color=INK, fontsize=13.5)

# ---- stage 2: three encoders ----
ex = 0.47
ov.text(ex, 0.93, "3 neutral encoders (fixed ``rulers'')".replace("``", "“").replace("''", "”"),
        ha="center", fontsize=14, style="italic")
rbox(ex, 0.72, 0.24, 0.13, "Handcrafted\n20-d")
rbox(ex, 0.50, 0.24, 0.13, "DINOv2\n384-d")
rbox(ex, 0.28, 0.24, 0.13, "YOLO-COCO\n512-d")

# ---- stage 3 + 4: distance and gap ----
mx, gx = 0.74, 0.91
rbox(mx, 0.50, 0.15, 0.20, "MMD$^2$\n\n+ energy,\nKID", fc="#FFF3E0", ec="#E65100")
rbox(gx, 0.50, 0.135, 0.30, "domain gap\n\nreal vs 2D\nreal vs 3D\n\n(per level)",
     fc="#E3F2FD", ec=BLUE, bold=True)

# ---- arrows between stages ----
arrow(cx + cw + 0.015, 0.335, 0.5)          # crops -> encoders (into mid)
for yy in (0.72, 0.50, 0.28):               # encoders -> MMD
    arrow(ex + 0.12 + 0.005, mx - 0.075 - 0.005, yy, 0.5)
arrow(mx + 0.075 + 0.005, gx - 0.0675 - 0.005, 0.5)   # MMD -> gap

fig.savefig(OUT / "domain_gap_schema.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig_domain_gap_schema.pdf", bbox_inches="tight")
plt.close(fig)
print("saved domain_gap_schema.png + fig_domain_gap_schema.pdf   (example class:",
      it["class"], ")")

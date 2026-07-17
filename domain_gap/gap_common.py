"""
Common indexing / sampling / image-loading for the domain-gap analysis.

Three domains (real / 2D / 3D) are compared at three granularities:
  - scene : the whole image, resized to the detector input (1024)
  - bbox  : a tight rectangle around one instance (fragment + local belt)
  - mask  : the same rectangle, but belt pixels neutralized (fragment only)

bbox and mask crops come from the SAME sampled instances and differ ONLY by
whether the background is kept, so their gap difference isolates the
background/belt contribution.

Nothing here computes a metric; it just yields numpy image arrays.
"""
import re
import random
from pathlib import Path

import numpy as np
from PIL import Image

YAML = Path("/Users/tomas/Desktop/PycharmProjects/AICycle-DS/data/yaml")
DSET = Path("/Users/tomas/Desktop/PycharmProjects/AICycle-DS/data/datasets")
CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]

DOMAINS = {  # domain -> (yaml subdir, datasets subdir)
    "real": ("real", "real"),
    "2D": ("synth2D", "synthetic2D"),
    "3D": ("synth3D", "synthetic3D"),
}

SCENE_SIZE = 1024   # detector input resolution (paper 2.6)
CROP_SIZE = 224     # instance crop size fed to encoders / features

RE_IMG_PLAIN = re.compile(r"^scene_(\d+)\.png$")
RE_IMG_TS = re.compile(r"^scene_(\d+)_(\d{8}_\d{6})\.png$")
RE_MASK_PLAIN = re.compile(r"^mask_(\d+)_([A-Za-z]+)_(\d+)\.png$")
RE_MASK_TS = re.compile(r"^mask_(\d+)_([A-Za-z]+)_(\d+)_(\d{8}_\d{6})_\d+\.png$")


def _img_key(name):
    m = RE_IMG_TS.match(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = RE_IMG_PLAIN.match(name)
    return m.group(1) if m else None


def _mask_key_cls(name):
    m = RE_MASK_TS.match(name)
    if m:
        return f"{m.group(1)}_{m.group(4)}", m.group(2)
    m = RE_MASK_PLAIN.match(name)
    if m:
        return m.group(1), m.group(2)
    return None, None


# ----------------------------- indexing --------------------------------------
def scene_index(domain, split):
    """Return {scene_key: image_path} for a domain/split (mask-side images)."""
    _, dname = DOMAINS[domain]
    d = DSET / dname / split / "images"
    if not d.exists():
        return {}
    return {_img_key(p.name): p for p in d.glob("*.png") if _img_key(p.name)}


def masks_for_scene(domain, split, key):
    """List of (class, mask_path) for one scene."""
    _, dname = DOMAINS[domain]
    mdir = DSET / dname / split / "masks"
    sid = key.split("_")[0]
    out = []
    for p in mdir.glob(f"mask_{sid}_*"):
        k, cls = _mask_key_cls(p.name)
        if k == key and cls in CLASSES:
            out.append((cls, p))
    return out


# ----------------------------- image loading ---------------------------------
def _binarize_mask(arr):
    if arr.ndim == 3:
        arr = arr[..., 3] if arr.shape[2] == 4 else arr.max(axis=2)
    return arr > 127


def load_scene(image_path, size=SCENE_SIZE):
    """Whole scene, resized square to the detector resolution."""
    im = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def load_full(image_path):
    """Full-resolution scene RGB array (cache this across instances)."""
    return np.asarray(Image.open(image_path).convert("RGB"))


def crops_from(im, mask_path, size=CROP_SIZE, pad=0.08):
    """
    Given a full-res scene array `im`, return (bbox_crop, mask_crop, frag_mask):
      bbox_crop : tight box around the instance, background KEPT
      mask_crop : same box, background pixels set to 0 (fragment only)
      frag_mask : boolean fragment mask at `size` (for masked-pixel stats)
    """
    mk = _binarize_mask(np.asarray(Image.open(mask_path)))
    if mk.shape != im.shape[:2]:
        mk = np.asarray(Image.fromarray(mk).resize(
            (im.shape[1], im.shape[0]), Image.NEAREST))
    ys, xs = np.where(mk)
    if ys.size == 0:
        return None, None, None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    py, px = int(h * pad), int(w * pad)
    y0, y1 = max(0, y0 - py), min(im.shape[0], y1 + py + 1)
    x0, x1 = max(0, x0 - px), min(im.shape[1], x1 + px + 1)
    box = im[y0:y1, x0:x1]
    m = mk[y0:y1, x0:x1]
    masked = box.copy()
    masked[~m] = 0
    to = lambda a, r=Image.BILINEAR: np.asarray(
        Image.fromarray(a).resize((size, size), r))
    fm = to(m.astype(np.uint8) * 255, Image.NEAREST) > 127
    return to(box), to(masked), fm


def load_instance(image_path, mask_path, size=CROP_SIZE, pad=0.08):
    """Convenience single-shot loader (loads the scene each call)."""
    return crops_from(load_full(image_path), mask_path, size, pad)


# ----------------------------- sampling --------------------------------------
def sample_scenes(domain, splits, n, seed=0):
    """n scene image paths pooled over the given splits."""
    rng = random.Random(seed)
    pool = []
    for sp in splits:
        pool += list(scene_index(domain, sp).items())
    rng.shuffle(pool)
    if n is not None and len(pool) > n:
        pool = pool[:n]
    return [p for _, p in pool]


def sample_instances(domain, splits, per_class, seed=0, per_scene_cap=None):
    """
    Return list of dicts {class, image_path, mask_path} with `per_class`
    instances per class, spread across scenes to limit intra-scene correlation.
    """
    rng = random.Random(seed)
    scenes = []
    for sp in splits:
        idx = scene_index(domain, sp)
        scenes += [(sp, k, p) for k, p in idx.items()]
    rng.shuffle(scenes)
    got = {c: [] for c in CLASSES}
    need = lambda: any(len(got[c]) < per_class for c in CLASSES)
    for sp, key, img in scenes:
        if not need():
            break
        by_cls = {c: [] for c in CLASSES}
        for cls, mp in masks_for_scene(domain, sp, key):
            by_cls[cls].append(mp)
        for c in CLASSES:
            if len(got[c]) >= per_class:
                continue
            rng.shuffle(by_cls[c])
            take = by_cls[c]
            if per_scene_cap:
                take = take[:per_scene_cap]
            for mp in take:
                if len(got[c]) >= per_class:
                    break
                got[c].append({"class": c, "image_path": img, "mask_path": mp})
    return [inst for c in CLASSES for inst in got[c][:per_class]]

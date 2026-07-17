"""
Data-plumbing sanity check for the domain-gap analysis.

Verifies, across all three domains (real / 2D / 3D):
  - scene image <-> YOLO bbox label pairing
  - scene image <-> per-instance mask pairing (incl. 3D timestamped names)
  - masks are full-image-sized and binary
  - per-class instance counts
Does NOT compute any metrics yet -- purely validates that we can index
every scene, its boxes, and its instance masks with correct class labels.
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

YAML = Path("/Users/tomas/Desktop/PycharmProjects/AICycle-DS/data/yaml")
DSET = Path("/Users/tomas/Desktop/PycharmProjects/AICycle-DS/data/datasets")
CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]

# domain -> (yaml subdir, datasets subdir)
DOMAINS = {
    "real": ("real", "real"),
    "2D": ("synth2D", "synthetic2D"),
    "3D": ("synth3D", "synthetic3D"),
}

# image name -> pairing key
RE_IMG_PLAIN = re.compile(r"^scene_(\d+)\.png$")               # real / 2D
RE_IMG_TS = re.compile(r"^scene_(\d+)_(\d{8}_\d{6})\.png$")    # 3D
# mask name -> (scene-key, class)
RE_MASK_PLAIN = re.compile(r"^mask_(\d+)_([A-Za-z]+)_(\d+)\.png$")
RE_MASK_TS = re.compile(r"^mask_(\d+)_([A-Za-z]+)_(\d+)_(\d{8}_\d{6})_\d+\.png$")


def img_key(name):
    m = RE_IMG_TS.match(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = RE_IMG_PLAIN.match(name)
    if m:
        return m.group(1)
    return None


def mask_key_cls(name):
    m = RE_MASK_TS.match(name)
    if m:
        return f"{m.group(1)}_{m.group(4)}", m.group(2)
    m = RE_MASK_PLAIN.match(name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def check_split(domain, split, n_scene_sample=3):
    yname, dname = DOMAINS[domain]
    img_dir = YAML / yname / "images" / split
    lbl_dir = YAML / yname / "labels" / split
    mask_img_dir = DSET / dname / split / "images"
    mask_dir = DSET / dname / split / "masks"

    if not img_dir.exists():
        return f"  [{domain}/{split}] no image dir"

    imgs = sorted([p for p in img_dir.glob("*.png")])
    out = [f"  [{domain}/{split}] images={len(imgs)}"]

    # bbox pairing
    n_box_ok, n_box_missing = 0, 0
    for p in imgs:
        lbl = lbl_dir / (p.stem + ".txt")
        if lbl.exists():
            n_box_ok += 1
        else:
            n_box_missing += 1
    out.append(f"      bbox labels: paired={n_box_ok} missing={n_box_missing}")

    # mask pairing (if mask dir exists for this split)
    if mask_dir.exists():
        imgkeys = {img_key(p.name): p for p in mask_img_dir.glob("*.png")}
        masks = list(mask_dir.glob("*.png"))
        per_class = Counter()
        per_scene = defaultdict(int)
        unpaired = 0
        bad_cls = 0
        for m in masks:
            key, cls = mask_key_cls(m.name)
            if key is None:
                unpaired += 1
                continue
            if cls not in CLASSES:
                bad_cls += 1
            per_class[cls] += 1
            per_scene[key] += 1
        keys_with_img = sum(1 for k in per_scene if k in imgkeys)
        out.append(
            f"      masks: total={len(masks)} scenes_with_masks={len(per_scene)} "
            f"scenes_matched_to_image={keys_with_img} unparsed={unpaired} bad_class={bad_cls}"
        )
        out.append(f"      per-class instances: {dict(per_class)}")

        # deep-check a few scenes: mask size vs image, binary values, crop works
        sample_keys = list(per_scene)[:n_scene_sample]
        for k in sample_keys:
            ip = imgkeys.get(k)
            if ip is None:
                out.append(f"        scene {k}: NO IMAGE MATCH")
                continue
            im = np.array(Image.open(ip).convert("RGB"))
            # find one mask for this scene
            mk = next(mm for mm in masks if mask_key_cls(mm.name)[0] == k)
            ms = np.array(Image.open(mk))
            uniq = np.unique(ms)
            same_size = ms.shape[:2] == im.shape[:2]
            out.append(
                f"        scene {k}: img={im.shape} mask={ms.shape} "
                f"same_size={same_size} mask_uniq={uniq[:6]}"
            )
    else:
        out.append("      (no mask split here)")
    return "\n".join(out)


if __name__ == "__main__":
    print("=" * 70)
    for domain in DOMAINS:
        # scene-level splits we care about
        for split in ["train", "val", "test"]:
            has_mask = (DSET / DOMAINS[domain][1] / split / "masks").exists()
            # only deep-scan splits that have masks or are small
            print(check_split(domain, split))
        print("-" * 70)

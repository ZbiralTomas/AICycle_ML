#!/usr/bin/env python3
"""
Apply relative normalization to all dataset scenes using parameters
computed from real training data (norm_params.json).

Processes all source directories:
    data/datasets/real/{train,val,test}/images/
    data/datasets/synthetic2D/{train,val}/images/
    data/datasets/synthetic3D/{train,val}/images/

By default, normalized images are saved alongside originals with a
"_normalized" suffix on the parent directory. Use --overwrite to
modify originals in-place.

Run compute_norm_params.py first to generate
data/datasets/normalization/norm_params.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

SCENE_RE = re.compile(r"^scene_(?P<sid>\d+)(?:_.*)?\.(png|jpg|jpeg)$", re.IGNORECASE)


def load_params(params_path: Path) -> dict:
    if not params_path.exists():
        raise FileNotFoundError(f"Norm params not found: {params_path}")

    if params_path.suffix.lower() == ".json":
        data = json.loads(params_path.read_text(encoding="utf-8"))
        params = {k: np.array(data[k], dtype=np.float32) for k in ("p1", "p99", "mu", "sd")}
    elif params_path.suffix.lower() == ".npz":
        d = np.load(str(params_path))
        params = {k: d[k].astype(np.float32) for k in ("p1", "p99", "mu", "sd")}
    else:
        raise ValueError(f"Unsupported params file: {params_path.suffix}")

    for k in ("p1", "p99", "mu", "sd"):
        if params[k].shape != (3,):
            raise ValueError(f"{k} must have shape (3,), got {params[k].shape}")
    return params


def normalize_image(img_rgb_float: np.ndarray, params: dict) -> np.ndarray:
    """Percentile clip + scale to [0, 1]."""
    p1, p99 = params["p1"], params["p99"]
    eps = 1e-6
    x = img_rgb_float.copy()
    for c in range(3):
        x[..., c] = np.clip(x[..., c], p1[c], p99[c])
        x[..., c] = (x[..., c] - p1[c]) / (p99[c] - p1[c] + eps)
    return x


def list_scenes(images_dir: Path) -> List[Tuple[int, Path]]:
    scenes = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        m = SCENE_RE.match(p.name)
        if m:
            scenes.append((int(m.group("sid")), p))
    scenes.sort()
    return scenes


def normalize_directory(images_dir: Path, out_dir: Path, params: dict) -> int:
    """Normalize all scene images. Returns count of processed images."""
    scenes = list_scenes(images_dir)
    if not scenes:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    for _, img_path in scenes:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = np.clip(normalize_image(rgb, params), 0.0, 1.0)
        out = cv2.cvtColor((norm * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / img_path.name), out)

    return len(scenes)


def build_output_path(images_dir: Path, datasets_root: Path) -> Path:
    """Build output path by inserting '_normalized' into the dataset directory name.

    data/datasets/real/train/images → data/datasets/real_normalized/train/images
    """
    rel = images_dir.relative_to(datasets_root)
    parts = list(rel.parts)
    parts[0] = parts[0] + "_normalized"
    return datasets_root / Path(*parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize scene images using precomputed params.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite originals in-place instead of saving to *_normalized/ dirs.")
    args = ap.parse_args()

    datasets_root = Path(__file__).resolve().parent.parent / "data" / "datasets"
    params_path = datasets_root / "normalization" / "norm_params.json"

    params = load_params(params_path)
    print(f"Loaded params from {params_path}")
    print(f"Mode: {'overwrite in-place' if args.overwrite else 'save to *_normalized/ directories'}\n")

    dirs_to_normalize = [
        datasets_root / "real" / "train" / "images",
        datasets_root / "real" / "val" / "images",
        datasets_root / "real" / "test" / "images",
        datasets_root / "synthetic2D" / "train" / "images",
        datasets_root / "synthetic2D" / "val" / "images",
        datasets_root / "synthetic3D" / "train" / "images",
        datasets_root / "synthetic3D" / "val" / "images",
    ]

    total = 0
    for src_dir in dirs_to_normalize:
        if not src_dir.exists():
            print(f"  [SKIP] {src_dir.relative_to(datasets_root)} — not found")
            continue

        out_dir = src_dir if args.overwrite else build_output_path(src_dir, datasets_root)
        n = normalize_directory(src_dir, out_dir, params)

        if n > 0:
            label = src_dir.relative_to(datasets_root)
            if args.overwrite:
                print(f"  [OK]   {label}: {n} images (overwritten)")
            else:
                print(f"  [OK]   {label} -> {out_dir.relative_to(datasets_root)}: {n} images")
        total += n

    print(f"\nDone. Normalized {total} images.")


if __name__ == "__main__":
    main()

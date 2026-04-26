#!/usr/bin/env python3
"""
Compute per-channel normalization parameters from real training data.

Collects instance pixels (using masks) from data/datasets/real/train/ and computes:
  - p1, p99:  1st and 99th percentile per channel (RGB, float in [0,1])
  - mu, sd:   mean and std after percentile scaling

Saves to data/datasets/normalization/norm_params.{json,npz}.
These parameters are used by normalize_scenes.py to normalize all datasets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

SCENE_RE = re.compile(r"^scene_(?P<sid>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
MASK_RE = re.compile(r"^mask_(?P<sid>\d+)_.*\.png$", re.IGNORECASE)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_instance_mask(mask_path: Path, thr: int = 0) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 2:
        return m > thr
    if m.shape[2] == 4:
        return m[..., 3] > thr
    return np.max(m[..., :3], axis=2) > thr


def merge_masks(mask_paths: List[Path], thr: int = 0) -> np.ndarray:
    merged = None
    for mp in mask_paths:
        m = load_instance_mask(mp, thr=thr)
        merged = m.copy() if merged is None else (merged | m)
    if merged is None:
        raise ValueError("No masks to merge.")
    return merged


def index_scenes(images_dir: Path, masks_dir: Path) -> List[Tuple[int, Path, np.ndarray]]:
    """Discover scenes, match masks, return [(scene_id, img_path, merged_mask)]."""
    scenes = {}
    for p in images_dir.iterdir():
        m = SCENE_RE.match(p.name)
        if m:
            sid = int(m.group("sid"))
            scenes[sid] = p

    if not scenes:
        raise ValueError(f"No scene files in {images_dir}")

    masks_by_sid = {}
    for p in masks_dir.iterdir():
        m = MASK_RE.match(p.name)
        if m:
            sid = int(m.group("sid"))
            masks_by_sid.setdefault(sid, []).append(p)

    items = []
    for sid in sorted(scenes):
        mp = sorted(masks_by_sid.get(sid, []))
        if not mp:
            raise ValueError(f"No masks for scene_id={sid} in {masks_dir}")
        items.append((sid, scenes[sid], merge_masks(mp)))

    return items


def compute_instance_stats(
    img_paths: List[Path],
    masks: List[np.ndarray],
    p_low: float = 1.0,
    p_high: float = 99.0,
    max_pixels: int = 20_000_000,
    seed: int = 0,
) -> dict:
    """Compute p1/p99 and post-scaling mean/std from instance pixels only."""
    rng = np.random.default_rng(seed)
    samples = [[], [], []]
    total = 0

    for img_path, mask in zip(img_paths, masks):
        img = load_rgb(img_path)
        if mask.shape[:2] != img.shape[:2]:
            raise ValueError(f"Shape mismatch for {img_path.name}: mask {mask.shape} vs img {img.shape}")

        pix = img[mask]
        if pix.size == 0:
            continue

        n = pix.shape[0]
        total += n

        if total > max_pixels:
            k = max(1, int(n * max_pixels / total))
            pix = pix[rng.choice(n, size=k, replace=False)]

        for c in range(3):
            samples[c].append(pix[:, c])

    if any(len(s) == 0 for s in samples):
        raise ValueError("No instance pixels collected. Check masks/paths.")

    vals = [np.concatenate(s) for s in samples]
    p1 = np.array([np.percentile(v, p_low) for v in vals], dtype=np.float32)
    p99 = np.array([np.percentile(v, p_high) for v in vals], dtype=np.float32)

    eps = 1e-6
    scaled = []
    for c in range(3):
        v = np.clip(vals[c], p1[c], p99[c])
        v = (v - p1[c]) / (p99[c] - p1[c] + eps)
        scaled.append(v)

    mu = np.array([np.mean(v) for v in scaled], dtype=np.float32)
    sd = np.array([np.std(v) for v in scaled], dtype=np.float32)

    return {"p1": p1, "p99": p99, "mu": mu, "sd": sd}


def save_params(params: dict, json_path: Path, npz_path: Path) -> None:
    payload = {k: params[k].astype(float).tolist() for k in ("p1", "p99", "mu", "sd")}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(str(npz_path), **{k: params[k] for k in ("p1", "p99", "mu", "sd")})


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parent.parent / "data"
    real_train = data_root / "datasets" / "real" / "train"

    items = index_scenes(real_train / "images", real_train / "masks")
    img_paths = [ip for _, ip, _ in items]
    masks = [m for _, _, m in items]

    print(f"Computing parameters from {len(items)} real train scenes...")
    params = compute_instance_stats(img_paths, masks)

    print("Computed parameters (R, G, B):")
    print(f"  p1 : {params['p1']}")
    print(f"  p99: {params['p99']}")
    print(f"  mu : {params['mu']}")
    print(f"  sd : {params['sd']}")

    out_dir = data_root / "datasets" / "normalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "norm_params.json"
    out_npz = out_dir / "norm_params.npz"
    save_params(params, out_json, out_npz)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_npz}")

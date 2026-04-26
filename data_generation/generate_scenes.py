#!/usr/bin/env python3
"""
Semi-synthetic dataset generation.
Generates train and val scenes in a single run from a pre-split fragments directory.

Expected fragments layout:
    fragments_split_root/
        train/<Class>/*.png
        val/<Class>/*.png

Output layout:
    out_dir/
        train/images/  train/masks/
        val/images/    val/masks/
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random
import cv2
import numpy as np


CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]


# ---------------------------- Configuration ----------------------------

@dataclass
class GenConfig:
    fragments_split_root: Path  # contains train/ and val/ subdirs
    backgrounds_dir: Path
    out_dir: Path

    out_w: int = 1024
    out_h: int = 1024

    # Scene counts and starting indices per split
    n_train_scenes: int = 8000
    n_val_scenes: int = 1000
    train_start_idx: int = 1
    val_start_idx: int = 1

    n_fragments_min: int = 65
    n_fragments_max: int = 90

    class_weights: Tuple[float, ...] = (1, 1, 1, 1, 1)

    # Geometry
    enable_rotation: bool = True
    angle_min: float = 0.0
    angle_max: float = 360.0

    enable_scale: bool = True
    scale_min: float = 0.9
    scale_max: float = 1.05

    # Brightness only
    enable_brightness_jitter: bool = True
    brightness_min: float = -5.0
    brightness_max: float = 5.0

    # Overlap
    max_overlap_ratio: float = 0.10
    max_tries: int = 500
    fallback_scale_decay: float = 0.95
    fallback_steps: int = 3

    seed: int = 0
    verbose: bool = True


# ---------------------------- Utilities ----------------------------

def apply_brightness(bgr: np.ndarray, beta: float) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def list_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def load_fragment(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """Load BGRA fragment and (if available) the source image (W, H) from sibling .json."""
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.shape[2] != 4:
        raise ValueError(f"Invalid fragment (expected BGRA): {path}")

    src_wh: Optional[Tuple[int, int]] = None
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        wh = meta.get("image_size_wh")
        if isinstance(wh, list) and len(wh) == 2:
            src_wh = (int(wh[0]), int(wh[1]))

    return bgra, bgra[:, :, 3] > 0, src_wh


def rotate_scale_anisotropic(
    bgra: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bgra.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    theta = np.deg2rad(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]], dtype=np.float32)
    t = np.array([cx, cy], dtype=np.float32) - A @ np.array([cx, cy], dtype=np.float32)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = A
    M[:, 2] = t

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    tc = corners @ A.T + t
    min_xy = tc.min(axis=0)
    max_xy = tc.max(axis=0)
    out_w = int(np.ceil(max_xy[0] - min_xy[0]))
    out_h = int(np.ceil(max_xy[1] - min_xy[1]))

    M2 = M.copy()
    M2[:, 2] -= min_xy

    flags_lin = cv2.INTER_LINEAR
    flags_nn = cv2.INTER_NEAREST
    border = cv2.BORDER_CONSTANT

    bgr_w = cv2.warpAffine(bgra[:, :, :3], M2, (out_w, out_h), flags=flags_lin, borderMode=border, borderValue=0)
    a_w   = cv2.warpAffine(bgra[:, :, 3],  M2, (out_w, out_h), flags=flags_nn,  borderMode=border, borderValue=0)
    m_w   = cv2.warpAffine((mask.astype(np.uint8) * 255), M2, (out_w, out_h), flags=flags_nn, borderMode=border, borderValue=0) > 0

    return np.dstack([bgr_w, a_w]), m_w


def alpha_blend(dst: np.ndarray, fg: np.ndarray, x: int, y: int) -> None:
    h, w = fg.shape[:2]
    roi = dst[y:y + h, x:x + w]
    a = fg[:, :, 3:4] / 255.0
    roi[:] = (fg[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def bbox_inter(a: Tuple, b: Tuple) -> Optional[Tuple[int, int, int, int]]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


@dataclass
class Instance:
    cls: str
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    area: int


# ---------------------------- Fragment index ----------------------------

def build_frag_index(fragments_dir: Path) -> Dict[str, List[Path]]:
    """Build per-class fragment file list from a directory containing <Class>/*.png subdirs."""
    index: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        cls_dir = fragments_dir / cls
        files = sorted(cls_dir.glob("*.png"))
        if not files:
            raise ValueError(
                f"No .png files found for class '{cls}'.\n"
                f"Checked: {cls_dir}\n"
                f"Does the folder exist and contain .png files?"
            )
        index[cls] = files
    return index


# ---------------------------- Placement ----------------------------

def place_scene(cfg: GenConfig, bg: np.ndarray, frag_index: Dict[str, List[Path]]) -> Tuple[np.ndarray, List[Instance]]:
    H, W = cfg.out_h, cfg.out_w
    canvas = bg.copy()
    occupancy = np.zeros((H, W), bool)
    instances: List[Instance] = []

    for _ in range(random.randint(cfg.n_fragments_min, cfg.n_fragments_max)):
        cls = random.choices(CLASSES, cfg.class_weights)[0]
        base_bgra, base_mask, src_wh = load_fragment(random.choice(frag_index[cls]))
        angle = random.uniform(cfg.angle_min, cfg.angle_max) if cfg.enable_rotation else 0.0

        # Per-fragment source-to-canvas scale: keep physical size consistent across resolutions.
        if src_wh is not None:
            base_scale = min(W / src_wh[0], H / src_wh[1])
        else:
            base_scale = 1.0

        placed = False
        for _ in range(cfg.fallback_steps + 1):
            sx = base_scale * random.uniform(0.85, 1.15)
            sy = base_scale * random.uniform(0.85, 1.15)
            fg, mask = rotate_scale_anisotropic(base_bgra, base_mask, angle, sx, sy)

            if cfg.enable_brightness_jitter:
                fg[:, :, :3] = apply_brightness(fg[:, :, :3], random.uniform(cfg.brightness_min, cfg.brightness_max))

            h, w = fg.shape[:2]
            area = int(mask.sum())
            if area == 0 or w > W or h > H:
                continue

            for _ in range(cfg.max_tries):
                x = random.randint(0, W - w)
                y = random.randint(0, H - h)

                if np.logical_and(mask, occupancy[y:y + h, x:x + w]).sum() / area > cfg.max_overlap_ratio:
                    continue

                ok = True
                for inst in instances:
                    inter = bbox_inter((x, y, x + w, y + h), inst.bbox)
                    if not inter:
                        continue
                    ix0, iy0, ix1, iy1 = inter
                    a_patch = mask[iy0 - y:iy1 - y, ix0 - x:ix1 - x]
                    b_patch = inst.mask[iy0 - inst.bbox[1]:iy1 - inst.bbox[1], ix0 - inst.bbox[0]:ix1 - inst.bbox[0]]
                    if b_patch.sum() and np.logical_and(a_patch, b_patch).sum() / inst.area > cfg.max_overlap_ratio:
                        ok = False
                        break
                if not ok:
                    continue

                alpha_blend(canvas, fg, x, y)

                # Update occluded area on existing instances
                for inst in instances:
                    inter = bbox_inter((x, y, x + w, y + h), inst.bbox)
                    if not inter:
                        continue
                    ix0, iy0, ix1, iy1 = inter
                    a_patch = mask[iy0 - y:iy1 - y, ix0 - x:ix1 - x]
                    b_patch = inst.mask[iy0 - inst.bbox[1]:iy1 - inst.bbox[1], ix0 - inst.bbox[0]:ix1 - inst.bbox[0]]
                    before = b_patch.sum()
                    b_patch &= ~a_patch
                    inst.area -= before - b_patch.sum()

                occupancy[y:y + h, x:x + w] |= mask
                instances.append(Instance(cls, (x, y, x + w, y + h), mask.copy(), area))
                placed = True
                break

            if placed:
                break

    return canvas, instances


# ---------------------------- I/O ----------------------------

def save_scene(split_out_dir: Path, idx: int, img: np.ndarray, insts: List[Instance], H: int, W: int) -> None:
    img_dir = split_out_dir / "images"
    mask_dir = split_out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_dir / f"scene_{idx:05d}.png"), img)

    counters = {c: 0 for c in CLASSES}
    for inst in insts:
        if inst.area <= 0:
            continue
        m = np.zeros((H, W), np.uint8)
        x0, y0, x1, y1 = inst.bbox
        m[y0:y1, x0:x1] = inst.mask.astype(np.uint8) * 255
        k = counters[inst.cls]
        counters[inst.cls] += 1
        cv2.imwrite(str(mask_dir / f"mask_{idx:05d}_{inst.cls}_{k}.png"), m)


# ---------------------------- Main ----------------------------

def generate_split(
    cfg: GenConfig,
    split: str,
    frag_index: Dict[str, List[Path]],
    bgs: List[Path],
    n_scenes: int,
    start_idx: int,
) -> None:
    split_out_dir = cfg.out_dir / split
    print(f"\n--- Generating {split} ({n_scenes} scenes, idx {start_idx}–{start_idx + n_scenes - 1}) ---")

    for i in range(n_scenes):
        idx = start_idx + i
        bg = cv2.imread(str(random.choice(bgs)))
        bg = cv2.resize(bg, (cfg.out_w, cfg.out_h))
        img, insts = place_scene(cfg, bg, frag_index)
        save_scene(split_out_dir, idx, img, insts, cfg.out_h, cfg.out_w)
        if cfg.verbose:
            print(f"  [OK] {split}/scene_{idx:05d}: instances={len(insts)}")

    print(f"--- {split} done. ---")


def main() -> None:
    cfg = GenConfig(
        fragments_split_root=Path(__file__).parent.parent / "data" / "generation_assets" / "fragments",
        backgrounds_dir=Path(__file__).parent.parent / "data" / "generation_assets" / "backgrounds",
        out_dir=Path(__file__).parent.parent / "data" / "datasets" / "synthetic2D",
        n_train_scenes=8000,
        n_val_scenes=1000,
        train_start_idx=1,
        val_start_idx=1,
        seed=0,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    bgs = list_images(cfg.backgrounds_dir)
    if not bgs:
        raise ValueError(f"No background images found in {cfg.backgrounds_dir}")

    train_frag_index = build_frag_index(cfg.fragments_split_root / "train")
    val_frag_index = build_frag_index(cfg.fragments_split_root / "val")

    generate_split(cfg, "train", train_frag_index, bgs, cfg.n_train_scenes, cfg.train_start_idx)
    generate_split(cfg, "val",   val_frag_index,   bgs, cfg.n_val_scenes,   cfg.val_start_idx)

    print("\nAll done.")


if __name__ == "__main__":
    main()

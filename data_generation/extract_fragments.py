#!/usr/bin/env python3
"""
Pipeline Step 1: Extract CDW fragments from conveyor-belt scenes into train/val.

Source layout (each split has independent scene numbering starting at 1):
    fragment_source/
        train/images/scene_00001.png ...  train/masks/mask_00001_AAC_0.png ...
        val/images/scene_00001.png ...    val/masks/mask_00001_AAC_0.png ...

Output:
    fragments/
        train/<Class>/fragment_00001_AAC_0.png ...
        val/<Class>/fragment_00001_AAC_0.png ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]


@dataclass(frozen=True)
class Config:
    fragment_source_dir: Path
    output_dir: Path

    binarize_threshold: int = 127
    morph_open_ksize: int = 0
    min_area_px: int = 50
    pad: int = 2

    erode_iterations: int = 4
    blur_ksize: int = 5

    write_metadata_json: bool = True
    verbose_print: bool = True


# ---------------------------- Extraction Utilities ----------------------------

def read_image(path: Path, flag: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[:, :, :3]
    if img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def mask_to_binary(mask_img: np.ndarray, thresh: int) -> np.ndarray:
    if mask_img.ndim == 2:
        gray = mask_img
    else:
        gray = cv2.cvtColor(mask_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return (gray > thresh).astype(np.uint8) * 255


def maybe_morph_open(bin_mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 0:
        return bin_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)


def refine_alpha_mask(bin_mask: np.ndarray, erode_iters: int, blur_ksize: int) -> np.ndarray:
    refined = bin_mask.copy()
    if erode_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.erode(refined, kernel, iterations=erode_iters)
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        refined = cv2.GaussianBlur(refined, (k, k), 0)
    return refined


def tight_bbox(bin_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def rgba_crop(scene_bgr: np.ndarray, alpha_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop_bgr = scene_bgr[y0:y1, x0:x1].copy()
    crop_alpha = alpha_mask[y0:y1, x0:x1]
    crop_bgr[crop_alpha == 0] = 0
    return np.dstack([crop_bgr, crop_alpha])


def parse_mask_filename(p: Path) -> Optional[Tuple[int, str, int]]:
    parts = p.stem.split("_")
    if len(parts) < 4 or parts[0] != "mask":
        return None
    try:
        scene_idx = int(parts[1])
        idx_within = int(parts[3])
    except ValueError:
        return None
    return scene_idx, parts[2], idx_within


def build_scene_map(images_dir: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for p in sorted(images_dir.glob("scene_*")):
        parts = p.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            mapping[int(parts[1])] = p
        except ValueError:
            continue
    return mapping


# ---------------------------- Core Pipeline ----------------------------

def extract_split(cfg: Config, split: str) -> Tuple[int, int]:
    """Extract fragments from one split directory. Returns (saved, discarded)."""
    src_images = cfg.fragment_source_dir / split / "images"
    src_masks = cfg.fragment_source_dir / split / "masks"

    if not src_images.exists():
        raise FileNotFoundError(f"Missing directory: {src_images}")
    if not src_masks.exists():
        raise FileNotFoundError(f"Missing directory: {src_masks}")

    for cls in CLASSES:
        (cfg.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    scene_map = build_scene_map(src_images)
    scene_indices = sorted(scene_map.keys())

    masks_by_scene: Dict[int, List[Tuple[Path, str, int]]] = {i: [] for i in scene_indices}
    for mp in sorted(src_masks.glob("mask_*.png")):
        parsed = parse_mask_filename(mp)
        if parsed is None:
            continue
        scene_idx, class_name, idx_within = parsed
        if scene_idx not in masks_by_scene or class_name not in CLASSES:
            continue
        masks_by_scene[scene_idx].append((mp, class_name, idx_within))

    total_saved = 0
    total_discarded = 0

    for scene_idx in scene_indices:
        scene_path = scene_map.get(scene_idx)
        if scene_path is None or not scene_path.exists():
            print(f"  [WARN] Scene {scene_idx} not found in {src_images}. Skipping.")
            continue

        scene_bgr = ensure_bgr(read_image(scene_path, cv2.IMREAD_COLOR))
        h, w = scene_bgr.shape[:2]

        triplets = sorted(masks_by_scene.get(scene_idx, []), key=lambda t: (t[1], t[2]))
        if not triplets:
            print(f"  [WARN] No masks for {scene_path.name}")
            continue

        scene_saved = 0
        for mask_path, class_name, idx_within in triplets:
            bin_mask = mask_to_binary(read_image(mask_path), cfg.binarize_threshold)
            bin_mask = maybe_morph_open(bin_mask, cfg.morph_open_ksize)

            area = int((bin_mask > 0).sum())
            if area < cfg.min_area_px:
                total_discarded += 1
                continue

            bbox0 = tight_bbox(bin_mask)
            if bbox0 is None:
                total_discarded += 1
                continue

            x0, y0, x1, y1 = bbox0
            x0, y0, x1, y1 = clamp_bbox(x0 - cfg.pad, y0 - cfg.pad, x1 + cfg.pad, y1 + cfg.pad, w, h)

            smooth_alpha = refine_alpha_mask(bin_mask, cfg.erode_iterations, cfg.blur_ksize)
            bgra = rgba_crop(scene_bgr, smooth_alpha, (x0, y0, x1, y1))

            stem = f"fragment_{scene_idx:05d}_{class_name}_{idx_within}"
            out_dir = cfg.output_dir / split / class_name

            if not cv2.imwrite(str(out_dir / f"{stem}.png"), bgra):
                raise IOError(f"Failed to write: {out_dir / f'{stem}.png'}")

            if cfg.write_metadata_json:
                meta = {
                    "source_scene": str(scene_path),
                    "source_mask": str(mask_path),
                    "scene_idx": scene_idx,
                    "split": split,
                    "class_name": class_name,
                    "idx_within_class": idx_within,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "area_px": area,
                    "image_size_wh": [w, h],
                }
                (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            total_saved += 1
            scene_saved += 1

        if cfg.verbose_print:
            print(f"  [{split}] scene_{scene_idx:05d}: {scene_saved} fragments")

    return total_saved, total_discarded


def extract_fragments(cfg: Config) -> None:
    print("--- Starting Fragment Extraction ---")

    grand_saved = 0
    grand_discarded = 0

    for split in ("train", "val"):
        print(f"\n[{split}] Processing all available scenes")
        saved, discarded = extract_split(cfg, split)
        grand_saved += saved
        grand_discarded += discarded
        print(f"  {split}: saved={saved}, discarded={discarded}")

    print(f"\nExtraction done. Total saved={grand_saved}, discarded={grand_discarded}")


if __name__ == "__main__":
    cfg = Config(
        fragment_source_dir=Path(__file__).resolve().parent.parent / "data" / "generation_assets" / "fragment_source",
        output_dir=Path(__file__).resolve().parent.parent / "data" / "generation_assets" / "fragments",
        binarize_threshold=127,
        morph_open_ksize=0,
        min_area_px=50,
        pad=2,
        erode_iterations=4,
        blur_ksize=5,
        write_metadata_json=True,
        verbose_print=True,
    )

    extract_fragments(cfg)

#!/usr/bin/env python3
"""
Convert source data (scenes + pixel masks) into YOLO-format datasets.

Reads from:   data/datasets/{real,synthetic2D,synthetic3D}/{train,val,test}/{images,masks}/
Writes to:    data/yaml/{real,synth2D,synth3D,mixed2D,mixed3D}/

Each output dataset contains:
    images/{train,val,test}/  — resized scene images (1024x1024)
    labels/{train,val,test}/  — YOLO bounding-box labels
    dataset.yaml              — ultralytics config
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
TARGET_SIZE = 1024


# ─── Mask → bbox utilities ───────────────────────────────────────────────────

def parse_mask_filename(p: Path) -> Optional[Tuple[str, str]]:
    """Parse mask_{scene_key}_{class}_{k}.png → (scene_key, class_name)."""
    parts = p.stem.split("_")
    if len(parts) < 4 or parts[0] != "mask":
        return None
    return parts[1], parts[2]


def extract_bbox(mask_path: Path, threshold: int = 127) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox (x_min, y_min, x_max, y_max) from a binary mask."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    ys, xs = np.where(img > threshold)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_to_yolo(x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int) -> str:
    """Normalized YOLO format: 'cx cy w h'."""
    cx = (x0 + x1) / 2.0 / img_w
    cy = (y0 + y1) / 2.0 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def scene_key_from_path(p: Path) -> str:
    """scene_00001.png → '00001'."""
    parts = p.stem.split("_")
    return parts[1] if len(parts) >= 2 else p.stem


def group_masks_by_scene(masks_dir: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """Group mask files by scene key → {key: [(path, class_name), ...]}."""
    groups: Dict[str, List[Tuple[Path, str]]] = {}
    for mp in sorted(masks_dir.glob("mask_*.png")):
        parsed = parse_mask_filename(mp)
        if parsed is None:
            continue
        scene_key, class_name = parsed
        if class_name not in CLASS_TO_ID:
            continue
        groups.setdefault(scene_key, []).append((mp, class_name))
    return groups


# ─── Processing ──────────────────────────────────────────────────────────────

def process_scenes(
    src_images_dir: Path,
    src_masks_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    target_size: int = TARGET_SIZE,
    name_prefix: str = "",
    multiplier: int = 1,
    max_scenes: Optional[int] = None,
    seed: int = 42,
) -> int:
    """Convert source scenes + masks → resized YOLO images + labels.

    Args:
        name_prefix: Prepended to output filenames (avoids collisions in mixed datasets).
        multiplier:  Repeat each image N times with distinct names (same labels).
        max_scenes:  If set, randomly sample this many scenes from source.
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(
        p for p in src_images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not scene_files:
        print(f"  [WARN] No images in {src_images_dir}")
        return 0

    if max_scenes is not None and max_scenes < len(scene_files):
        rng = random.Random(seed)
        scene_files = sorted(rng.sample(scene_files, max_scenes))

    mask_groups = group_masks_by_scene(src_masks_dir)
    count = 0

    for scene_path in scene_files:
        img = cv2.imread(str(scene_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (target_size, target_size))
        scene_key = scene_key_from_path(scene_path)

        lines = []
        for mask_path, class_name in mask_groups.get(scene_key, []):
            bbox = extract_bbox(mask_path)
            if bbox is None:
                continue
            lines.append(f"{CLASS_TO_ID[class_name]} {bbox_to_yolo(*bbox, w, h)}")

        label_text = "\n".join(lines)

        for rep in range(multiplier):
            stem = f"{name_prefix}{scene_path.stem}" + (f"_rep{rep}" if multiplier > 1 else "")
            cv2.imwrite(str(dst_images_dir / f"{stem}.png"), img_resized)
            (dst_labels_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
            count += 1

    return count


# ─── YAML generation ─────────────────────────────────────────────────────────

def write_yaml(yaml_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {yaml_path.parent.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )


# ─── Dataset specification ───────────────────────────────────────────────────

@dataclass
class SourceSpec:
    images_dir: Path
    masks_dir: Path
    split: str
    prefix: str = ""
    multiplier: int = 1
    max_scenes: Optional[int] = None


@dataclass
class DatasetSpec:
    name: str
    sources: List[SourceSpec] = field(default_factory=list)


def create_dataset(spec: DatasetSpec, output_root: Path) -> None:
    out_dir = output_root / spec.name
    print(f"\n=== {spec.name} ===")

    for src in spec.sources:
        if not src.images_dir.exists():
            print(f"  [SKIP] {src.images_dir} does not exist")
            continue

        n = process_scenes(
            src.images_dir, src.masks_dir,
            out_dir / "images" / src.split,
            out_dir / "labels" / src.split,
            name_prefix=src.prefix,
            multiplier=src.multiplier,
            max_scenes=src.max_scenes,
        )
        desc = src.prefix or "all"
        if src.multiplier > 1:
            desc += f" (x{src.multiplier})"
        if src.max_scenes:
            desc += f" (sampled {src.max_scenes})"
        print(f"  {src.split}: {n} images <- {desc}")

    write_yaml(out_dir / "dataset.yaml")
    print(f"  -> {out_dir / 'dataset.yaml'}")


# ─── Main ────────────────────────────────────────────────────────────────────

def _src(base: Path, split: str) -> Tuple[Path, Path]:
    """Helper: return (images_dir, masks_dir) for a source split."""
    return base / split / "images", base / split / "masks"


def main() -> None:
    data_root = Path(__file__).parent.parent / "data"
    datasets_root = data_root / "datasets"
    yaml_root = data_root / "yaml"

    real = datasets_root / "real"
    s2d = datasets_root / "synthetic2D"
    s3d = datasets_root / "synthetic3D"

    datasets = [
        # ── Real model (1-stage) ──
        DatasetSpec("real", [
            SourceSpec(*_src(real, "train"), "train"),
            SourceSpec(*_src(real, "val"),   "val"),
            SourceSpec(*_src(real, "test"),  "test"),
        ]),

        # ── Synth2D stage 1: synth train/val, real test ──
        DatasetSpec("synth2D", [
            SourceSpec(*_src(s2d, "train"),  "train"),
            SourceSpec(*_src(s2d, "val"),    "val"),
            SourceSpec(*_src(real, "test"),  "test"),
        ]),

        # ── Synth3D stage 1: synth train/val, real test ──
        DatasetSpec("synth3D", [
            SourceSpec(*_src(s3d, "train"),  "train"),
            SourceSpec(*_src(s3d, "val"),    "val"),
            SourceSpec(*_src(real, "test"),  "test"),
        ]),

        # ── Mixed2D stage 3: 80 real (16x5) + 80 synth train, real val/test ──
        DatasetSpec("mixed2D", [
            SourceSpec(*_src(real, "train"), "train", prefix="real_",    multiplier=5),
            SourceSpec(*_src(s2d, "train"),  "train", prefix="synth2d_", max_scenes=80),
            SourceSpec(*_src(real, "val"),   "val"),
            SourceSpec(*_src(real, "test"),  "test"),
        ]),

        # ── Mixed3D stage 3: 80 real (16x5) + 80 synth train, real val/test ──
        DatasetSpec("mixed3D", [
            SourceSpec(*_src(real, "train"), "train", prefix="real_",    multiplier=5),
            SourceSpec(*_src(s3d, "train"),  "train", prefix="synth3d_", max_scenes=80),
            SourceSpec(*_src(real, "val"),   "val"),
            SourceSpec(*_src(real, "test"),  "test"),
        ]),
    ]

    for spec in datasets:
        create_dataset(spec, yaml_root)

    print("\nAll datasets created.")


if __name__ == "__main__":
    main()

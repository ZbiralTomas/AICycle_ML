#!/usr/bin/env python3
"""
Training script for YOLO11 CDW fragment detection models.

Model types and their training strategies:
  real:    1-stage  — 600 epochs, patience 80, COCO pretrained weights
  synth2D: 3-stage  — pretrain on synth → frozen-backbone on real → mixed fine-tune
  synth3D: 3-stage  — pretrain on synth → frozen-backbone on real → mixed fine-tune

Stage details for synthetic models:
  Stage 1: 600 epochs, patience 80, default lr, COCO pretrained weights
  Stage 2: 600 epochs, patience 60, frozen backbone, lr0=0.001, lrf=0.001
  Stage 3: 600 epochs, patience 120, unfrozen, lr0=0.001, lrf=0.001

All stages use augmentations from training/augmentations.yaml.
Multiple model types can be trained in a single call:
    python train.py --model-type real synth2D synth3D
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_IMGSIZE = 1024
DEFAULT_BATCH = 16

STAGE_DEFAULTS = {
    "real": {
        1: {"epochs": 600, "patience": 80},
    },
    "synth": {
        1: {"epochs": 600, "patience": 80},
        2: {"epochs": 600, "patience": 60,  "lr0": 0.001, "lrf": 0.001, "freeze": 9},
        3: {"epochs": 600, "patience": 120, "lr0": 0.001, "lrf": 0.001},
    },
}

DATA_YAML_MAP = {
    "real":    {1: "real"},
    "synth2D": {1: "synth2D", 2: "real", 3: "mixed2D"},
    "synth3D": {1: "synth3D", 2: "real", 3: "mixed3D"},
}


# ─── Arg parsing ─────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train YOLO11 models for CDW fragment detection.",
    )

    ap.add_argument(
        "--model-type", nargs="+", required=True,
        choices=["real", "synth2D", "synth3D"],
        help="Training strategy/strategies to run.",
    )
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help=f"Base COCO-pretrained model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--imgsize", type=int, default=DEFAULT_IMGSIZE,
                    help=f"Image size (default: {DEFAULT_IMGSIZE}).")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"Batch size (default: {DEFAULT_BATCH}).")

    ap.add_argument("--project", default="runs", type=str,
                    help="Ultralytics project directory (default: runs).")
    ap.add_argument("--name", required=True, type=str,
                    help="Experiment name prefix.")
    ap.add_argument("--exist-ok", action="store_true",
                    help="Allow existing run directory.")

    ap.add_argument(
        "--data-root", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "yaml",
        help="Root directory containing per-strategy YAML dataset folders.",
    )
    ap.add_argument(
        "--aug-yaml", type=Path,
        default=Path(__file__).resolve().parent / "augmentations.yaml",
        help="YAML with augmentation overrides (default: training/augmentations.yaml).",
    )

    ap.add_argument("--freeze-layers", type=int, default=9,
                    help="Backbone layers to freeze in stage 2 (default: 9, keeps SPPF+C2PSA trainable).")
    ap.add_argument(
        "--override", action="append", default=None,
        help="Extra Ultralytics override as key=value (repeatable).",
    )
    ap.add_argument("--print-cuda", action="store_true",
                    help="Print CUDA availability / device info.")
    return ap


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_augmentations(aug_yaml: Path) -> dict:
    if aug_yaml and aug_yaml.exists():
        with open(aug_yaml) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_cli_overrides(override_list: list | None) -> dict:
    overrides: dict = {}
    if not override_list:
        return overrides
    for item in override_list:
        key, _, val = item.partition("=")
        try:
            parsed = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                parsed = {"true": True, "false": False}.get(val.lower(), val)
        overrides[key] = parsed
    return overrides


def get_data_yaml(data_root: Path, model_type: str, stage: int) -> Path:
    folder = DATA_YAML_MAP[model_type][stage]
    return data_root / folder / "dataset.yaml"


# ─── Training ────────────────────────────────────────────────────────────────

def train_stage(
    model_path: str | Path,
    data_yaml: Path,
    imgsize: int,
    batch: int,
    project: str,
    name: str,
    exist_ok: bool,
    stage_params: dict,
    augmentations: dict,
    cli_overrides: dict,
) -> Path:
    """Run one training stage. Returns path to best.pt."""
    model = YOLO(str(model_path))

    train_args = {
        "data": str(data_yaml),
        "imgsz": imgsize,
        "batch": batch,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        **stage_params,
        **augmentations,
        **cli_overrides,
    }

    model.train(**train_args)
    return Path(model.trainer.save_dir) / "weights" / "best.pt"


def print_stage_header(model_type: str, stage: int, description: str,
                       data_yaml: Path, stage_params: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{model_type}] Stage {stage} — {description}")
    print(f"  data:     {data_yaml}")
    for k, v in stage_params.items():
        print(f"  {k:10s} {v}")
    print(f"{'=' * 60}")


def train_real(args: argparse.Namespace, augmentations: dict, cli_overrides: dict) -> Path:
    params = STAGE_DEFAULTS["real"][1].copy()
    data_yaml = get_data_yaml(args.data_root, "real", 1)
    run_name = f"{args.name}_real"

    print_stage_header("real", 1, "training on real data", data_yaml, params)

    return train_stage(
        args.model, data_yaml, args.imgsize, args.batch,
        args.project, run_name, args.exist_ok,
        params, augmentations, cli_overrides,
    )


def train_synthetic(args: argparse.Namespace, model_type: str,
                    augmentations: dict, cli_overrides: dict) -> Path:
    base_name = f"{args.name}_{model_type}"

    # Stage 1 — pretrain on synthetic
    s1_params = STAGE_DEFAULTS["synth"][1].copy()
    data_yaml = get_data_yaml(args.data_root, model_type, 1)
    print_stage_header(model_type, 1, "synthetic pretraining", data_yaml, s1_params)

    best_s1 = train_stage(
        args.model, data_yaml, args.imgsize, args.batch,
        args.project, f"{base_name}_stage1", args.exist_ok,
        s1_params, augmentations, cli_overrides,
    )

    # Stage 2 — frozen backbone on real
    s2_params = STAGE_DEFAULTS["synth"][2].copy()
    s2_params["freeze"] = args.freeze_layers
    data_yaml = get_data_yaml(args.data_root, model_type, 2)
    print_stage_header(model_type, 2, "frozen backbone on real data", data_yaml, s2_params)

    best_s2 = train_stage(
        best_s1, data_yaml, args.imgsize, args.batch,
        args.project, f"{base_name}_stage2", args.exist_ok,
        s2_params, augmentations, cli_overrides,
    )

    # Stage 3 — unfrozen mixed fine-tune
    s3_params = STAGE_DEFAULTS["synth"][3].copy()
    data_yaml = get_data_yaml(args.data_root, model_type, 3)
    print_stage_header(model_type, 3, "mixed fine-tuning (unfrozen)", data_yaml, s3_params)

    best_s3 = train_stage(
        best_s2, data_yaml, args.imgsize, args.batch,
        args.project, f"{base_name}_stage3", args.exist_ok,
        s3_params, augmentations, cli_overrides,
    )

    print(f"\n{model_type} complete. Final weights: {best_s3}")
    return best_s3


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_argparser().parse_args()
    args.project = str(Path(args.project).resolve())

    if args.print_cuda:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")

    augmentations = load_augmentations(args.aug_yaml)
    cli_overrides = parse_cli_overrides(args.override)

    if augmentations:
        print(f"Loaded augmentations from {args.aug_yaml}")
    if cli_overrides:
        print(f"CLI overrides: {cli_overrides}")

    for model_type in args.model_type:
        if model_type == "real":
            train_real(args, augmentations, cli_overrides)
        else:
            train_synthetic(args, model_type, augmentations, cli_overrides)

    print(f"\n{'=' * 60}")
    print("All training complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

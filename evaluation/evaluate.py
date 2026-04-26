#!/usr/bin/env python3
"""
Statistical evaluation and comparison of trained YOLO models.

Computes:
  1. Per-model metrics:  mAP@0.5, per-class AP, Precision, Recall
  2. McNemar's paired test with Bonferroni correction
  3. Image-level bootstrap 95 % confidence intervals

Usage:
    python evaluation/evaluate.py \
      --models runs/exp_real/weights/best.pt \
              runs/exp_synth2D_stage3/weights/best.pt \
              runs/exp_synth3D_stage3/weights/best.pt \
      --model-names real synth2D synth3D
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import binomtest
from ultralytics import YOLO

CLASSES = ["AAC", "Ceramics", "Mortar", "Stones", "Tiles"]
N_CLASSES = len(CLASSES)


# ─── Bounding-box utilities ──────────────────────────────────────────────────

def compute_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [N,4] and [M,4] xyxy boxes → [N,M]."""
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def greedy_match(
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy-match predictions to GT (highest confidence first).
    Returns (pred_tp[N_pred], gt_matched[N_gt])."""
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    pred_tp = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)

    if n_pred == 0 or n_gt == 0:
        return pred_tp, gt_matched

    iou = compute_iou_matrix(pred_boxes, gt_boxes)

    for pi in np.argsort(-pred_conf):
        best_iou, best_gi = -1.0, -1
        for gi in range(n_gt):
            if gt_matched[gi] or gt_cls[gi] != pred_cls[pi]:
                continue
            if iou[pi, gi] > best_iou:
                best_iou = iou[pi, gi]
                best_gi = gi
        if best_iou >= iou_thr and best_gi >= 0:
            pred_tp[pi] = True
            gt_matched[best_gi] = True

    return pred_tp, gt_matched


# ─── Data loading ────────────────────────────────────────────────────────────

def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """YOLO label file → (boxes[M,4] xyxy, classes[M])."""
    boxes, classes = [], []
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    text = label_path.read_text().strip()
    if not text:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    for line in text.split("\n"):
        parts = line.split()
        c = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.append([
            (cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h,
        ])
        classes.append(c)
    return np.array(boxes), np.array(classes, dtype=int)


# Raw per-image data: (pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls)
ImageData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_inference(
    model_path: str | Path,
    images_dir: Path,
    labels_dir: Path,
    imgsize: int = 1024,
) -> List[Tuple[str, ImageData]]:
    """Run model on all test images. Returns [(image_id, data), ...]."""
    model = YOLO(str(model_path))

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    out = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        gt_boxes, gt_cls = load_gt_labels(labels_dir / f"{img_path.stem}.txt", w, h)

        preds = model.predict(str(img_path), imgsz=imgsize, conf=0.001, verbose=False)
        if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
            pred_boxes = preds[0].boxes.xyxy.cpu().numpy()
            pred_cls = preds[0].boxes.cls.cpu().numpy().astype(int)
            pred_conf = preds[0].boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))
            pred_cls = np.zeros(0, dtype=int)
            pred_conf = np.zeros(0)

        out.append((img_path.stem, (pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls)))

    return out


# ─── Pre-computation ─────────────────────────────────────────────────────────

@dataclass
class ImagePrecomp:
    """Per-image pre-computed matching results."""
    # For AP (all predictions, no conf filter)
    ap_preds_by_class: Dict[int, List[Tuple[float, bool]]] = field(default_factory=dict)
    gt_counts_by_class: Dict[int, int] = field(default_factory=dict)

    # For Precision / Recall (conf >= threshold)
    tp: int = 0
    fp: int = 0
    fn: int = 0

    # For McNemar (conf >= threshold)
    gt_matched: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


def precompute(
    image_data_list: List[ImageData],
    iou_thr: float = 0.5,
    conf_thr: float = 0.5,
) -> List[ImagePrecomp]:
    """Match predictions to GT for every image (once for AP, once for P/R)."""
    results = []

    for pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls in image_data_list:
        pc = ImagePrecomp()

        # --- AP: match ALL predictions ---
        pred_tp_all, _ = greedy_match(pred_boxes, pred_cls, pred_conf,
                                      gt_boxes, gt_cls, iou_thr)
        for c in range(N_CLASSES):
            pc.ap_preds_by_class[c] = []
            pc.gt_counts_by_class[c] = 0
        for i in range(len(pred_boxes)):
            c = pred_cls[i]
            pc.ap_preds_by_class.setdefault(c, []).append(
                (float(pred_conf[i]), bool(pred_tp_all[i]))
            )
        for c in gt_cls:
            pc.gt_counts_by_class[c] = pc.gt_counts_by_class.get(c, 0) + 1

        # --- P/R, McNemar: match predictions with conf >= threshold ---
        mask = pred_conf >= conf_thr
        pb, pcl, pco = pred_boxes[mask], pred_cls[mask], pred_conf[mask]
        pred_tp_conf, gt_matched = greedy_match(pb, pcl, pco, gt_boxes, gt_cls, iou_thr)

        pc.tp = int(pred_tp_conf.sum())
        pc.fp = int((~pred_tp_conf).sum())
        pc.fn = int((~gt_matched).sum())
        pc.gt_matched = gt_matched

        results.append(pc)

    return results


# ─── Metrics from pre-computed data ──────────────────────────────────────────

def compute_ap(class_preds: List[Tuple[float, bool]], n_gt: int) -> float:
    """AP via all-point interpolation of the PR curve."""
    if n_gt == 0 or not class_preds:
        return 0.0

    sorted_preds = sorted(class_preds, key=lambda x: -x[0])
    tp = np.cumsum([int(t) for _, t in sorted_preds])
    fp = np.cumsum([int(not t) for _, t in sorted_preds])

    recall = tp / n_gt
    precision = tp / (tp + fp)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def aggregate_metrics(precomps: List[ImagePrecomp], indices: Optional[np.ndarray] = None) -> Dict:
    """Compute mAP@0.5, per-class AP, Precision, Recall from precomputed data.
    If indices is given, use only those images (for bootstrapping)."""
    if indices is None:
        subset = precomps
    else:
        subset = [precomps[i] for i in indices]

    # Aggregate per-class AP data
    class_preds = {c: [] for c in range(N_CLASSES)}
    class_gt_n = {c: 0 for c in range(N_CLASSES)}
    total_tp, total_fp, total_fn = 0, 0, 0

    for pc in subset:
        for c in range(N_CLASSES):
            class_preds[c].extend(pc.ap_preds_by_class.get(c, []))
            class_gt_n[c] += pc.gt_counts_by_class.get(c, 0)
        total_tp += pc.tp
        total_fp += pc.fp
        total_fn += pc.fn

    per_class_ap = {}
    valid_aps = []
    for c in range(N_CLASSES):
        ap = compute_ap(class_preds[c], class_gt_n[c])
        per_class_ap[CLASSES[c]] = ap
        if class_gt_n[c] > 0:
            valid_aps.append(ap)

    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "mAP50": map50,
        "per_class_ap": per_class_ap,
        "precision": precision,
        "recall": recall,
    }


# ─── McNemar's test ──────────────────────────────────────────────────────────

def mcnemar_test(matched_a: List[np.ndarray], matched_b: List[np.ndarray]) -> Dict:
    """Exact McNemar's test on concatenated GT detection arrays."""
    a = np.concatenate(matched_a)
    b = np.concatenate(matched_b)

    b_count = int(np.sum(a & ~b))   # A detected, B missed
    c_count = int(np.sum(~a & b))   # A missed, B detected
    n = b_count + c_count

    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0}

    result = binomtest(min(b_count, c_count), n, 0.5, alternative="two-sided")
    chi2_stat = (abs(b_count - c_count) - 1) ** 2 / n

    return {
        "b": b_count,
        "c": c_count,
        "n_discordant": n,
        "chi2": round(chi2_stat, 4),
        "p_value": result.pvalue,
    }


# ─── Image-level bootstrap ───────────────────────────────────────────────────

def bootstrap_ci(
    precomps: List[ImagePrecomp],
    n_iterations: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Image-level bootstrap for mAP@0.5, Precision, Recall."""
    rng = np.random.default_rng(seed)
    n = len(precomps)
    alpha = (1 - ci) / 2

    maps, precs, recs = [], [], []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        m = aggregate_metrics(precomps, idx)
        maps.append(m["mAP50"])
        precs.append(m["precision"])
        recs.append(m["recall"])

    return {
        "mAP50": (float(np.percentile(maps, alpha * 100)),
                  float(np.percentile(maps, (1 - alpha) * 100))),
        "precision": (float(np.percentile(precs, alpha * 100)),
                      float(np.percentile(precs, (1 - alpha) * 100))),
        "recall": (float(np.percentile(recs, alpha * 100)),
                   float(np.percentile(recs, (1 - alpha) * 100))),
    }


# ─── Printing ─────────────────────────────────────────────────────────────────

def print_metrics_table(model_names: List[str], all_metrics: List[Dict]) -> None:
    header = f"{'Model':<12} {'mAP@0.5':>8} {'Prec':>8} {'Rec':>8}"
    for cls in CLASSES:
        header += f" {cls:>9}"
    print(header)
    print("-" * len(header))

    for name, m in zip(model_names, all_metrics):
        row = f"{name:<12} {m['mAP50']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}"
        for cls in CLASSES:
            row += f" {m['per_class_ap'][cls]:>9.4f}"
        print(row)


def print_bootstrap_table(model_names: List[str], all_ci: List[Dict]) -> None:
    header = f"{'Model':<12} {'mAP@0.5':>20} {'Precision':>20} {'Recall':>20}"
    print(header)
    print("-" * len(header))

    for name, ci in zip(model_names, all_ci):
        row = (f"{name:<12} "
               f"[{ci['mAP50'][0]:.4f}, {ci['mAP50'][1]:.4f}] "
               f"[{ci['precision'][0]:.4f}, {ci['precision'][1]:.4f}] "
               f"[{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")
        print(row)


def print_mcnemar_table(comparisons: List[Dict], alpha_corrected: float) -> None:
    header = f"{'Comparison':<24} {'b':>5} {'c':>5} {'chi2':>8} {'p-value':>10} {'Significant':>12}"
    print(header)
    print("-" * len(header))

    for comp in comparisons:
        sig = "Yes" if comp["p_value"] < alpha_corrected else "No"
        print(f"{comp['pair']:<24} {comp['b']:>5} {comp['c']:>5} "
              f"{comp['chi2']:>8.4f} {comp['p_value']:>10.6f} {sig:>12}")


# ─── CSV writers ──────────────────────────────────────────────────────────────

def write_metrics_csv(
    csv_path: Path,
    model_names: List[str],
    all_metrics: List[Dict],
    all_ci: List[Dict],
) -> None:
    """One row per model: aggregate metrics + bootstrap CIs + per-class AP."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "model",
        "mAP50", "mAP50_ci_lo", "mAP50_ci_hi",
        "precision", "prec_ci_lo", "prec_ci_hi",
        "recall", "rec_ci_lo", "rec_ci_hi",
    ] + [f"AP_{c}" for c in CLASSES]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for name, m, ci in zip(model_names, all_metrics, all_ci):
            row = [
                name,
                f"{m['mAP50']:.6f}",     f"{ci['mAP50'][0]:.6f}",     f"{ci['mAP50'][1]:.6f}",
                f"{m['precision']:.6f}", f"{ci['precision'][0]:.6f}", f"{ci['precision'][1]:.6f}",
                f"{m['recall']:.6f}",    f"{ci['recall'][0]:.6f}",    f"{ci['recall'][1]:.6f}",
            ] + [f"{m['per_class_ap'][c]:.6f}" for c in CLASSES]
            w.writerow(row)


def write_mcnemar_csv(csv_path: Path, comparisons: List[Dict], alpha_corrected: float) -> None:
    """One row per pairwise comparison."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["pair", "b", "c", "n_discordant", "chi2", "p_value", "alpha_corrected", "significant"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for comp in comparisons:
            w.writerow([
                comp["pair"], comp["b"], comp["c"], comp["n_discordant"],
                f"{comp['chi2']:.4f}", f"{comp['p_value']:.6f}",
                f"{alpha_corrected:.4f}",
                "yes" if comp["p_value"] < alpha_corrected else "no",
            ])


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate and statistically compare YOLO models.",
    )
    ap.add_argument("--models", nargs="+", required=True, type=Path,
                    help="Paths to model weights (best.pt).")
    ap.add_argument("--model-names", nargs="+", required=True,
                    help="Display name for each model.")
    ap.add_argument("--test-images", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "yaml" / "real" / "images" / "test",
                    help="Test images directory.")
    ap.add_argument("--test-labels", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "yaml" / "real" / "labels" / "test",
                    help="Test labels directory (YOLO format).")
    ap.add_argument("--imgsize", type=int, default=1024)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="Confidence threshold for Precision/Recall and McNemar (default: 0.5).")
    ap.add_argument("--n-bootstrap", type=int, default=10_000,
                    help="Number of bootstrap iterations (default: 10000).")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level before Bonferroni correction (default: 0.05).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Save full results to JSON.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Save per-model metrics + bootstrap CIs to CSV. "
                         "If 2+ models, also writes <csv>_mcnemar.csv with pairwise tests.")
    return ap


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_argparser().parse_args()

    if len(args.models) != len(args.model_names):
        raise ValueError("--models and --model-names must have the same length.")

    n_models = len(args.models)
    n_comparisons = n_models * (n_models - 1) // 2
    alpha_corrected = args.alpha / n_comparisons if n_comparisons > 0 else args.alpha

    # ── Inference ──
    all_image_data: Dict[str, List[ImageData]] = {}
    for model_path, name in zip(args.models, args.model_names):
        print(f"Running inference: {name} ({model_path}) ...")
        raw = run_inference(model_path, args.test_images, args.test_labels, args.imgsize)
        all_image_data[name] = [d for _, d in raw]

    # ── Pre-compute matchings ──
    all_precomp: Dict[str, List[ImagePrecomp]] = {}
    for name in args.model_names:
        all_precomp[name] = precompute(all_image_data[name], args.iou_threshold, args.conf_threshold)

    # ── Metrics ──
    all_metrics = []
    for name in args.model_names:
        m = aggregate_metrics(all_precomp[name])
        all_metrics.append(m)

    print(f"\n{'=' * 72}")
    print(f"Model Performance (IoU={args.iou_threshold}, conf={args.conf_threshold})")
    print(f"{'=' * 72}\n")
    print_metrics_table(args.model_names, all_metrics)

    # ── Bootstrap CI ──
    print(f"\n{'=' * 72}")
    print(f"Bootstrap 95% Confidence Intervals ({args.n_bootstrap} iterations)")
    print(f"{'=' * 72}\n")

    all_ci = []
    for name in args.model_names:
        print(f"  Bootstrapping {name} ...", end=" ", flush=True)
        ci = bootstrap_ci(all_precomp[name], args.n_bootstrap)
        all_ci.append(ci)
        print("done")

    print()
    print_bootstrap_table(args.model_names, all_ci)

    # ── McNemar's test ──
    if n_models >= 2:
        print(f"\n{'=' * 72}")
        print(f"McNemar's Test (Bonferroni-corrected alpha = {alpha_corrected:.4f})")
        print(f"{'=' * 72}\n")

        comparisons = []
        for i, j in combinations(range(n_models), 2):
            name_a, name_b = args.model_names[i], args.model_names[j]
            matched_a = [pc.gt_matched for pc in all_precomp[name_a]]
            matched_b = [pc.gt_matched for pc in all_precomp[name_b]]
            result = mcnemar_test(matched_a, matched_b)
            result["pair"] = f"{name_a} vs {name_b}"
            comparisons.append(result)

        print_mcnemar_table(comparisons, alpha_corrected)

    # ── Save JSON ──
    if args.output:
        output = {
            "config": {
                "iou_threshold": args.iou_threshold,
                "conf_threshold": args.conf_threshold,
                "n_bootstrap": args.n_bootstrap,
                "alpha": args.alpha,
                "alpha_corrected": alpha_corrected,
            },
            "metrics": {n: m for n, m in zip(args.model_names, all_metrics)},
            "bootstrap_ci": {n: c for n, c in zip(args.model_names, all_ci)},
        }
        if n_models >= 2:
            output["mcnemar"] = comparisons

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.output}")

    # ── Save CSV ──
    if args.csv:
        write_metrics_csv(args.csv, args.model_names, all_metrics, all_ci)
        print(f"Metrics CSV saved to {args.csv}")
        if n_models >= 2:
            mcnemar_csv = args.csv.with_name(args.csv.stem + "_mcnemar" + args.csv.suffix)
            write_mcnemar_csv(mcnemar_csv, comparisons, alpha_corrected)
            print(f"McNemar CSV saved to {mcnemar_csv}")

    print(f"\n{'=' * 72}")
    print("Evaluation complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

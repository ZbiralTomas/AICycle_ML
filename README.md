# AICycle-DS

Synthetic data generation, YOLO11 training, and statistical model comparison for construction and demolition waste (CDW) fragment detection.

The project compares three training strategies — real-only, 2D synthetic pretraining, and 3D synthetic pretraining — to evaluate the impact of synthetic data on detection performance.

## Project Structure

```
AICycle-DS/
├── data_generation/             # Stage 1: Create synthetic training data
│   ├── extract_fragments.py     #   Extract fragment PNGs from conveyor-belt scenes
│   └── generate_scenes.py       #   Render fragments onto backgrounds → synthetic scenes
│
├── data_preprocessing/          # Stage 2: Prepare data for training
│   ├── compute_norm_params.py   #   Compute normalization params from real train data
│   ├── normalize_scenes.py      #   Apply relative normalization to all datasets
│   └── create_yaml_files.py     #   Resize images, extract bboxes, create YOLO datasets
│
├── training/                    # Stage 3: Train YOLO11 models
│   ├── train.py                 #   Multi-strategy training script
│   └── augmentations.yaml       #   Ultralytics augmentation overrides
│
├── evaluation/                  # Stage 4: Compare trained models
│   └── evaluate.py              #   Metrics, McNemar's test, bootstrap CIs
│
└── data/                              # All data (not committed to git)
    ├── generation_assets/             #   Inputs for synthetic scene generation
    │   ├── fragment_source/           #     Conveyor-belt scenes (input to extract_fragments)
    │   ├── fragments/                 #     Extracted fragment PNGs (output of extract_fragments)
    │   └── backgrounds/               #     Background images for scene composition
    ├── datasets/                      #   Image-level datasets (pre-YAML staging)
    │   ├── real/                      #     Real train/val/test scenes + masks
    │   ├── real_normalized/
    │   ├── synthetic2D/               #     Generated 2D synthetic scenes + masks
    │   ├── synthetic2D_normalized/
    │   ├── synthetic3D/               #     Rendered 3D synthetic scenes + masks
    │   ├── synthetic3D_normalized/
    │   └── normalization/             #     norm_params.{json,npz}
    └── yaml/                          #   Ultralytics YAML dataset stagings
        ├── real/
        ├── synth2D/
        ├── synth3D/
        ├── mixed2D/
        └── mixed3D/
```

## Pipeline

### Prerequisites

```bash
pip install opencv-python numpy ultralytics scipy
```

Fragment classes: `AAC`, `Ceramics`, `Mortar`, `Stones`, `Tiles`

---

### Stage 1: Data Generation (2D synthetic)

**1a. Extract fragments** from conveyor-belt recordings where single-class fragments pass through the camera view. Each split (train/val) has its own source directory; scene IDs may be non-contiguous (e.g., val uses 2, 4, 6, 8) — the extractor auto-discovers all available scenes.

Place conveyor-belt scenes and masks into:
```
data/generation_assets/fragment_source/
├── train/images/scene_00001.png ...
│        /masks/mask_00001_AAC_0.png ...
└── val/images/scene_00002.png ...
         /masks/mask_00002_AAC_0.png ...
```

Run extraction:
```bash
python data_generation/extract_fragments.py
```

Config is at the bottom of the script. Key parameters:
- `erode_iterations`, `blur_ksize` — halo removal / edge feathering
- `min_area_px` — discard tiny fragments
- `pad` — padding around tight bbox before crop

The scene image is loaded with `cv2.IMREAD_COLOR` so OpenCV applies the embedded sRGB ICC profile; the mask uses `IMREAD_UNCHANGED`. A sibling `.json` is written next to each fragment with metadata including the source `image_size_wh`, which the scene generator uses to pick the right scale.

Output: `data/generation_assets/fragments/{train,val}/<Class>/fragment_*.png` (RGBA PNGs + `.json` sidecars)

**1b. Generate synthetic scenes** by rendering extracted fragments onto background images with random placement, rotation, scaling, and brightness jitter.

Place background images into `data/generation_assets/backgrounds/`.

```bash
python data_generation/generate_scenes.py
```

Config is at the bottom of the script. Key parameters:
- `out_w` / `out_h` — output canvas size (default 1024×1024)
- `n_train_scenes` / `n_val_scenes` — default 8000 / 1000
- `n_fragments_min` / `n_fragments_max` — fragments per scene (65–90)
- `max_overlap_ratio` — overlap constraint between fragments (10%)

Each fragment is scaled by `min(out_w/src_w, out_h/src_h)` (read from the sibling `.json` written at extraction time) and then jittered by ±15%, so fragment physical size stays consistent regardless of source resolution.

Output: `data/datasets/synthetic2D/{train,val}/{images,masks}/`

The 3D synthetic data (`data/datasets/synthetic3D/`) is generated externally using Blender/bpy and follows the same directory layout.

---

### Stage 2: Data Preprocessing

All three steps must run in order.

**2a. Compute normalization parameters** from real training data. Instance pixels (identified via masks) are used to compute per-channel percentile and distribution statistics.

```bash
python data_preprocessing/compute_norm_params.py
```

Reads from `data/datasets/real/train/{images,masks}/`, saves to `data/datasets/normalization/norm_params.{json,npz}`.

**2b. Normalize all scenes** using the computed parameters. Applies percentile clip and scale to match the real data distribution.

```bash
# Save to *_normalized/ directories (originals preserved)
python data_preprocessing/normalize_scenes.py

# Or overwrite originals in-place
python data_preprocessing/normalize_scenes.py --overwrite
```

Processes all images in `data/datasets/{real,synthetic2D,synthetic3D}/{train,val,test}/images/`.

**2c. Create YOLO-format datasets.** Resizes images to 1024x1024, converts pixel masks to bounding-box labels, and generates `dataset.yaml` files.

```bash
python data_preprocessing/create_yaml_files.py
```

Creates five YAML datasets under `data/yaml/`:

| Dataset | Train | Val | Test | Purpose |
|---|---|---|---|---|
| `real` | 16 real | 4 real | 10 real | Real model + synth stage 2 |
| `synth2D` | 8000 synth2D | 1000 synth2D | 10 real | Synth2D stage 1 |
| `synth3D` | synth3D | synth3D | 10 real | Synth3D stage 1 |
| `mixed2D` | 80 real (16×5) + 80 synth2D | 4 real | 10 real | Synth2D stage 3 |
| `mixed3D` | 80 real (16×5) + 80 synth3D | 4 real | 10 real | Synth3D stage 3 |

Each contains `images/{train,val,test}/`, `labels/{train,val,test}/`, and `dataset.yaml`.

---

### Stage 3: Training

Three training strategies, all using YOLO11 (COCO-pretrained weights):

**Real model** — single-stage training on real data.

**Synthetic models (2D and 3D)** — three-stage transfer learning:

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Data | Synthetic | Real | Mixed (50:50) |
| Epochs | 600 | 600 | 600 |
| Patience | 80 | 60 | 120 |
| lr0 | 0.01 (default) | 0.001 | 0.001 |
| lrf | 0.01 (default) | 0.001 | 0.001 |
| Backbone | Unfrozen | Frozen (layers 0–8) | Unfrozen |
| Init weights | COCO pretrained | Stage 1 best.pt | Stage 2 best.pt |

All stages use augmentations from `training/augmentations.yaml`.

```bash
# Train all three model types
python training/train.py --model-type real synth2D synth3D --name experiment1

# Train only real
python training/train.py --model-type real --name experiment1

# Override defaults
python training/train.py --model-type synth2D --name experiment1 \
  --model yolo11m.pt --batch 8 --override workers=4
```

Key arguments:
- `--model-type` — one or more of `real`, `synth2D`, `synth3D`
- `--model` — base model (default: `yolo11s.pt`)
- `--imgsize` — image size (default: 1024)
- `--batch` — batch size (default: 16)
- `--aug-yaml` — augmentation config (default: `training/augmentations.yaml`)
- `--freeze-layers` — backbone layers to freeze in stage 2 (default: 9)

Run names follow the pattern `{name}_{model_type}` for real and `{name}_{model_type}_stage{N}` for synthetic. Weights are saved to `runs/`.

---

### Stage 4: Evaluation

Statistical comparison of trained models on the shared real test set.

```bash
python evaluation/evaluate.py \
  --models runs/experiment1_real/weights/best.pt \
           runs/experiment1_synth2D_stage3/weights/best.pt \
           runs/experiment1_synth3D_stage3/weights/best.pt \
  --model-names real synth2D synth3D \
  --output evaluation/results.json
```

**Metrics computed:**
- **mAP@0.5** — mean Average Precision at IoU threshold 0.5 (per-class and overall)
- **Precision and Recall** — at a configurable confidence threshold (default 0.5)

**Statistical tests:**
- **McNemar's test** — paired hypothesis test on instance-level detection outcomes. For each ground-truth instance, checks whether each model detected it. Uses exact binomial test. Three pairwise comparisons with Bonferroni correction (adjusted alpha = 0.05/3 = 0.0166).
- **Image-level bootstrap** — 10,000 iterations resampling test images with replacement. Preserves intra-image correlations between fragment detections. Reports 95% confidence intervals for mAP, Precision, and Recall.

Key arguments:
- `--conf-threshold` — confidence threshold for Precision/Recall/McNemar (default: 0.5)
- `--n-bootstrap` — bootstrap iterations (default: 10000)
- `--alpha` — significance level before Bonferroni correction (default: 0.05)
- `--output` — save results to JSON

---

## Quick Reference: Full Pipeline

```bash
# 1. Data generation (2D)
python data_generation/extract_fragments.py
python data_generation/generate_scenes.py

# 2. Preprocessing
python data_preprocessing/compute_norm_params.py
python data_preprocessing/normalize_scenes.py
python data_preprocessing/create_yaml_files.py

# 3. Training
python training/train.py --model-type real synth2D synth3D --name exp1

# 4. Evaluation
python evaluation/evaluate.py \
  --models runs/exp1_real/weights/best.pt \
           runs/exp1_synth2D_stage3/weights/best.pt \
           runs/exp1_synth3D_stage3/weights/best.pt \
  --model-names real synth2D synth3D \
  --output evaluation/results.json
```

## Data Layout

Source datasets under `data/datasets/` all follow a uniform structure:
```
data/datasets/{real,synthetic2D,synthetic3D}/
├── train/
│   ├── images/scene_00001.png ...
│   └── masks/mask_00001_<Class>_<k>.png ...
├── val/
│   ├── images/
│   └── masks/
└── test/          (real only)
    ├── images/
    └── masks/
```

Scene files: `scene_{index:05d}.png`
Mask files: `mask_{index:05d}_{ClassName}_{instance}.png` (full-image binary masks, one per instance)

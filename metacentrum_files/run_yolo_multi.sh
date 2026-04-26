#!/bin/bash
#
# --- Part 1: Resources ---
#PBS -q gpu
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=16:ngpus=1:mem=80gb:scratch_local=150gb:gpu_mem=40gb
#PBS -N AICycle_yolo11s_multi
#PBS -m ae

# --- Part 2: Configuration (edit per submission) ---
EXPERIMENT_NAME="exp1"     # run name prefix; final dirs = ${EXPERIMENT_NAME}_real, _synth2D_stage{1,2,3}, _synth3D_stage{1,2,3}
BASE_MODEL="yolo11s.pt"
IMGSIZE=1024
BATCH=16

HOMEDIR="/storage/brno2/home/zbiratom"
PROJECT_DIR="$HOMEDIR/AICycle_ML"

# --- Part 3: Environment ---
module add mambaforge
mamba activate "$PROJECT_DIR/env"

# Scratch is the only writable temp area on the compute node.
export TMPDIR="$SCRATCHDIR/tmp"
export YOLO_CONFIG_DIR="$SCRATCHDIR/Ultralytics"
mkdir -p "$TMPDIR" "$YOLO_CONFIG_DIR"

# --- Part 4: Data Transfer ---
echo "Copying project to scratch..."

# Source code (small)
cp -r "$PROJECT_DIR/training"           "$SCRATCHDIR" || { echo "Error copying training/"; exit 1; }
cp    "$PROJECT_DIR/$BASE_MODEL"        "$SCRATCHDIR" 2>/dev/null  # base weights, if pre-downloaded

# Data (large)
cp -r "$PROJECT_DIR/data"               "$SCRATCHDIR" || { echo "Error copying data/"; exit 1; }

cd "$SCRATCHDIR" || { echo "Error changing directory"; exit 1; }

# --- Part 5: Execution ---
echo "Starting YOLO training: all model types, name=${EXPERIMENT_NAME}"

# train.py runs all listed model types sequentially in one process.
# Per-stage epochs / patience / lr / freeze come from STAGE_DEFAULTS in train.py.
python training/train.py \
  --model-type real synth2D synth3D \
  --name "$EXPERIMENT_NAME" \
  --model "$BASE_MODEL" \
  --imgsize "$IMGSIZE" \
  --batch "$BATCH" \
  --exist-ok \
  --override device=0

echo "Training finished."

# --- Part 6: Save Results ---
echo "Copying results back to project dir..."
mkdir -p "$PROJECT_DIR/runs"
cp -r -u runs/* "$PROJECT_DIR/runs/" || { echo "Error copying results"; exit 1; }

echo "Job done. Results saved to $PROJECT_DIR/runs"

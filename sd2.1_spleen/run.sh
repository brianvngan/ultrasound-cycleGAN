#!/usr/bin/env bash
# End-to-end MVP pipeline: preprocess -> LoRA fine-tune -> sample.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RAW_DIR="${RAW_DIR:-raw_spleen_data}"
DATA_DIR="${DATA_DIR:-data/processed}"
OUT_DIR="${OUT_DIR:-outputs/lora-spleen}"
SAMPLE_DIR="${SAMPLE_DIR:-outputs/samples}"
STEPS="${STEPS:-3000}"
BATCH="${BATCH:-16}"

python preprocess.py --input-dir "$RAW_DIR" --output-dir "$DATA_DIR"

accelerate launch --num_processes=1 train_lora.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --train-batch-size "$BATCH" \
    --max-train-steps "$STEPS"

python infer.py \
    --lora-dir "$OUT_DIR" \
    --prompt "an ultrasound image of a spleen" \
    --num-images 8 \
    --output-dir "$SAMPLE_DIR"

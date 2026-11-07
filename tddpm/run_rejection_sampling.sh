#!/bin/bash

# Rejection sampling for the "w/o spatial prior" models (train_unconditional.py, noise scale 1.0):
# 10,000 accepted samples per dataset and training run. Each run writes to its own folder,
# rejection_sampling_results/<run>/; synthetic_loader.py expects the samples as
# unconditional_rejection_<dataset>_samples-10k-<tag>.npy in synthetic_data/, where <tag> is
# 'submitted' for the run without --seed and seed42 / seed1 for the others.

set -e  # Exit on error

# Configuration
N_SAMPLES=10000
BATCH_SIZE=256
SNAPSHOT_INTERVAL=250
MAX_ITERATIONS=300  # 10,000 accepted samples took up to 128 batches of 256
REJECT_THRESHOLD=0.1
NOISE_SCALE="1.0"

for RUN in "" "-seed42" "-seed1"; do
    TAG=${RUN#-}
    OUTPUT_DIR="rejection_sampling_results/${TAG:-default}"
    mkdir -p "$OUTPUT_DIR"
    for DATASET in geolife-100 porto-100 cabspotting-100 brest-100; do
        MODEL="model_weights/unconditional-tddpm-${DATASET}-128-1${RUN}.pth"
        echo "========================================="
        echo "Processing: $DATASET (model $MODEL)"
        echo "========================================="
        if [ ! -f "$MODEL" ]; then
            echo "ERROR: Model file not found: $MODEL; skipping"
            continue
        fi
        python sample_unconditional_rejection.py \
            "$DATASET" \
            "$MODEL" \
            --n-samples "$N_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --snapshot-interval "$SNAPSHOT_INTERVAL" \
            --max-iterations "$MAX_ITERATIONS" \
            --reject-threshold "$REJECT_THRESHOLD" \
            --noise-scale "$NOISE_SCALE" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/${DATASET}_rejection_sampling.log"
    done
done

echo "========================================="
echo "All rejection sampling completed!"
echo "========================================="
echo "Results saved in: rejection_sampling_results/<run>/"
echo ""
echo "Output files for each dataset:"
echo "  - *_samples.npy     : Accepted trajectory samples"
echo "  - *_stats.pkl       : Sampling statistics"
echo "  - *_visualization.png : Comparison plots"
echo "  - snapshot_*.npy    : Intermediate snapshots"
echo "  - *_rejection_sampling.log : Execution logs"
echo "========================================="

# Rejection Sampling for Unconditional Trajectory Models

This directory contains scripts for generating synthetic trajectories using rejection sampling with unconditional diffusion models.

## Files

- **`sample_unconditional_rejection.py`**: Main rejection sampling script
- **`run_rejection_sampling.sh`**: Bash script to run rejection sampling for all four datasets and the three training runs

## Quick Start

### Run rejection sampling for all datasets (10,000 accepted samples each):

```bash
cd tddpm
./run_rejection_sampling.sh
```

This processes geolife-100, porto-100, cabspotting-100 and brest-100, for the model trained without `--seed`
and the models trained with `--seed 42` and `--seed 1` (`train_all_unconditional.sh`). Results are saved in
`rejection_sampling_results/<run>/`, with `<run>` one of `default`, `seed42` and `seed1`.

### Run for a single dataset:

```bash
python sample_unconditional_rejection.py \
    geolife-100 \
    model_weights/unconditional-tddpm-geolife-100-128-1.pth \
    --n-samples 10000 \
    --noise-scale 1.0 \
    --output-dir results/
```

## How It Works

1. **Load Target Dataset**: Loads the specified dataset and scales to [0,1] normalized space
2. **Generate Heatmap**: Creates a spatial density heatmap with dynamic resolution (21.33 pixels/km by default)
3. **Sample with Rejection**:
   - Generates trajectories from the unconditional model in batches
   - Rejects trajectories where >10% of points fall in zero-density cells (hard rejection)
   - Continues until target number of accepted samples is reached
4. **Save Results**:
   - Accepted samples as `.npy` file
   - Statistics including acceptance rate as `.pkl` file
   - Visualization comparing real vs synthetic distributions

## Command-Line Arguments

### Required:
- `dataset`: Dataset name (geolife-100, porto-100, cabspotting-100, brest-100)
- `model_weights`: Path to model checkpoint (.pth file)

### Optional:
- `--n-samples`: Target number of accepted samples (default: 10000)
- `--batch-size`: Generation batch size (default: 256)
- `--snapshot-interval`: Save intermediate results every N samples (default: 1000, 0=disable)
- `--max-iterations`: Maximum sampling iterations (default: 300)
- `--reject-threshold`: Max fraction of points outside valid regions (default: 0.1 = 10%)
- `--noise-scale`: Noise scale used during training (default: 1.0, as in `train_unconditional.py`)
- `--pixels-per-km`: Heatmap resolution (default: 21.33)
- `--seq-len`: Trajectory sequence length (default: 128)
- `--output-dir`: Output directory (default: current directory)

## Output Files

For each dataset, the following files are generated:

- `unconditional_rejection_{dataset}_samples.npy`: Final accepted trajectory samples
- `unconditional_rejection_{dataset}_stats.pkl`: Sampling statistics and acceptance rate history
- `unconditional_rejection_{dataset}_visualization.png`: 6-panel comparison plot
- `snapshot_*.npy`: Intermediate snapshots (if enabled)
- `{dataset}_rejection_sampling.log`: Execution log

`synthetic_loader.py` expects the accepted samples in `synthetic_data/` as
`unconditional_rejection_{dataset}_samples-10k-{tag}.npy`, with `{tag}` `submitted` for the run without `--seed`
and `seed42` / `seed1` for the others.

## Configuration

Edit `run_rejection_sampling.sh` to change:
- `N_SAMPLES`: Number of target samples (default: 10000)
- `BATCH_SIZE`: Batch size for generation (default: 256)
- `SNAPSHOT_INTERVAL`: Snapshot frequency (default: 250)
- `MAX_ITERATIONS`: Safety limit (default: 300; 10,000 accepted samples took up to 128 batches of 256)
- `REJECT_THRESHOLD`: Rejection threshold (default: 0.1)
- `NOISE_SCALE`: Model noise scale (default: 1.0)

## Model Weights

The script uses the unconditional model weights written by `train_unconditional.py`:
`model_weights/unconditional-tddpm-{dataset}-128-1.pth`, and `...-128-1-seed42.pth` / `...-128-1-seed1.pth`
for the seed runs.

All models use `noise_scale=1.0` and `seq_len=128`.

## Example Workflow

```bash
# 1. Run rejection sampling for all datasets
./run_rejection_sampling.sh

# 2. Check results
ls -lh rejection_sampling_results/*/

# 3. Load results in Python
import numpy as np
import pickle

# Load accepted samples
samples = np.load('rejection_sampling_results/default/unconditional_rejection_geolife-100_samples.npy')
print(f"Accepted samples shape: {samples.shape}")

# Load statistics
with open('rejection_sampling_results/default/unconditional_rejection_geolife-100_stats.pkl', 'rb') as f:
    stats = pickle.load(f)
print(f"Acceptance rate: {stats['final_acceptance_rate']:.3f}")
```

## Notes

- The rejection criterion is **hard rejection**: trajectories with >10% of points in zero-density cells are rejected
- Dynamic heatmap resolution scales with dataset size to maintain consistent spatial resolution
- Snapshots are saved periodically to prevent data loss during long runs
- The script includes a maximum iteration limit to prevent infinite loops if acceptance rate is very low

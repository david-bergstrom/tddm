# Learning to Deaggregate: Large-scale Trajectory Generation with Spatial Priors

This repository contains the code for the paper. In addition to the source code, we will provide preprocessed data, synthetic datasets and model weights through a Zenodo archive (coming soon; the link will be added here).

The archive contains three zip files:

- preprocessed.zip: preprocessed data, used for sampling and evaluation (place the *.npy files in data/preprocessed/)
- synthetic_data.zip: synthetic data for the models in the tables (place the *.npy files in synthetic_data/)
- model_weights.zip: for sampling the model (unpack the *.pth files and place them in tddpm/model_weights)

Requirements:

- docker (for preprocessing)
- docker-compose (for preprocessing)
- conda (we used miniconda3)

We trained on NVIDIA GPUs with 24 GB of VRAM or more (RTX 3090 Ti, RTX PRO 6000). Sampling TDDM (`tddpm/sample.py`) loads the conditional training data of the dataset (about 12 GB on disk) and needs about 50 GB of RAM.

The training scripts log to Weights & Biases: run `wandb login` first, or set `WANDB_MODE=offline`.

## Quickstart for setting up conda environment

```bash
conda create -n tddm python=3.10
conda activate tddm
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install tqdm cython matplotlib pandas numpy wandb einops ema_pytorch accelerate requests
cd tddpm
python setup.py build_ext --inplace
```

We used PyTorch 2.10 (CUDA 12.9) and NumPy 2.2; with PyTorch 2.13 we reproduced the Porto evaluation bit for bit.

## Quickstart for running the benchmark

1. Download preprocessed data and synthetic data from the Zenodo archive (coming soon)
2. Unzip the synthetic data, make sure that the synthetic datasets (*.npy files) are located in synthetic_data/
3. Unzip the preprocessed data, make sure the preprocessed datasets are placed in data/preprocessed/
4. Setup the environment:
    - Follow the quickstart for setting up the conda environment above
    - Enter the environment: `conda activate tddm`
5. In the environment, compute the measures (each script caches its results in a `*.pickle` file and skips entries that are already computed):
    - `python run_evaluation.py eval` (KL/JS, Density, Start/End, Pattern, OD, and the real-vs-real noise floors)
    - `python speed_kl_divergence.py` (Speed KL)
    - `python length_error_v2.py` (Length)
    - `python tstr_scaled_diff.py` (TSTR)
    - `python compute_v2_noise_floors.py` (the real-vs-real noise floors for Length, Speed KL and TSTR; run it after the three scripts above)
6. Generate the tables, see [Evaluation Pipeline](#evaluation-pipeline) for the command behind each table in the paper.

## Quickstart: Sample the model

1. Download the model weights from the Zenodo archive (coming soon; skip if already downloaded)
2. Unzip the model weights and place the *.pth files in `tddpm/model_weights/` (Skip if already done)
3. Setup the environment (Skip if already set up):
    - Follow the quickstart for setting up the conda environment above
    - Enter the environment: `conda activate tddm`
4. Compile the cython files (Skip if already compiled):
    - `cd tddpm`
    - `python setup.py build_ext --inplace`
5. Sample from the model:
    - For Porto dataset: `python sample.py porto-100 porto-100 --run-id sc7ukq34`
    - For Cabspotting dataset: `python sample.py cabspotting-100 cabspotting-100 --run-id vctqwp9s`
    - `--region-size` (default 3000, in metres) must match the region size the model was trained with; the released models use 3000.
    - Sampling also reads `data/preprocessed/<dataset>-100000-conditional-128-3000.npz` for the heatmap normalisation. It is in preprocessed.zip; to create it yourself, run `python conditional-variant-generator.py <dataset> 3000` from the repository root (about 10 hours).

## Quickstart: Full training instructions for Geolife

Since we cannot redistribute derivative work of Geolife, we instead provide full instructions for training:

1. Download Geolife
   - Get the zip-archive of Geolife from: https://www.microsoft.com/en-us/download/details.aspx?id=52367
   - Unzip the file in the `data/raw/` directory
2. Setup the environment (if you have already setup the environment, skip this step)
    - Follow the quickstart for setting up the conda environment above
    - Enter the environment: `conda activate tddm`
3. Compile the cython:
    - `cd tddpm`
    - `python setup.py build_ext --inplace`
    - `cd ..`
4. In a separate terminal, start the mapmatching service
    - `cd valhalla/geolife`
    - `docker-compose up`
    - Wait until it has downloaded and parsed the data. Look for: `INFO: Found config file. Starting valhalla service!` in the output.
5. Preprocess the data (takes ~15 minutes): 
    - `python preprocess_data.py geolife-100`
    - You may now close the docker container
6. Run the conditional training data extraction (takes ~10 hours, CPU bound):
    - `python conditional-variant-generator.py  geolife-100 3000`
7. Train the model (takes ~2 days):
    - `cd tddpm`
    - `python train_conditional.py geolife-100`
8. Sample geolife-100 (takes ~1 hour):
    - `cd tddpm` (if you skipped step 7)
    - `python sample.py geolife-100 geolife-100`

## Table of Contents

The rest of the README is structured as follows:

1. [Core Components](#core-components)
   - [Main Scripts](#main-scripts)
   - [Model Implementations](#model-implementations)
   - [Baselines](#baselines)
   - [Visualization Notebook](#visualization-notebook)
   - [Configuration and Utilities](#configuration-and-utilities)
2. [Data Sources](#data-sources)
3. [Setting up Runtime Environments](#setting-up-runtime-environments)
   - [TimeGAN](#timegan)
   - [TimeVAE](#timevae)
   - [Diffusion-TS](#diffusion-ts)
   - [COSCI-GAN](#cosci-gan)
   - [TDDM (ours)](#tddm-ours)
4. [Data Preprocessing](#data-preprocessing)
5. [TDDM](#tddm)
   - [Train TDDM](#train-tddm)
6. [Running Related Work](#running-related-work)
   - [TimeGAN](#timegan-1)
   - [TimeVAE](#timevae-1)
   - [COSCI-GAN](#cosci-gan-1)
   - [Diffusion-TS](#diffusion-ts-1)
   - [DiffTraj](#difftraj)
7. [Evaluation Pipeline](#evaluation-pipeline)
   - [Notes on the baselines](#notes-on-the-baselines)
   - [Not included](#not-included)
8. [License](#license)

## Core Components

### Main Scripts

- `run_evaluation.py`: Main evaluation script that runs all experiments and metrics
- `preprocess_data.py`: Data preprocessing pipeline for trajectory datasets
- `preproc.py`: Contains core preprocessing functions used by preprocess_data.py
- `data.py`: Core data handling utilities and dataset classes
- `synthetic_loader.py`: Utilities for loading and processing synthetic trajectories
- `generate_tables.py`: Generates result tables for the paper
- `generate_summary_table.py`: Generates summary tables of key results
- `generate_combined_generalization_table_with_brest.py`: Generates the combined intra-city / city-to-city generalization table (Table 3)
- `generate_table_generalize_25_loqo.py`: Generates the generalization-from-25% table, averaged over the four quadrant models
- `seed_mean.py`: Averages a table cell over the training runs of a model (mean ± standard error)
- `length_error_v2.py`: Length measure (distribution of per-trajectory travel distance)
- `tstr_scaled_diff.py`: TSTR measure (train on synthetic, test on real: predict the last 10 of 128 points)
- `noise_floors.py` / `floor_bolding.py`: Real-vs-real noise floors for each measure, and the † marks in the tables
- `compute_v2_noise_floors.py`: Adds the noise floors for Length, Speed KL and TSTR to their caches
- `speed_kl_divergence.py`: Speed KL measure
- `conditional-variant-generator.py`: Second step of preprocessing specific to our proposed method
- `tddpm/train_unconditional.py`: For the ablation study, trains without conditional mode
- `tddpm/train_conditional.py`: Trains TDDM in conditional mode
- `tddpm/sample.py`: Sampling script for TDDM

### Model Implementations

- `tddpm/`: Our proposed trajectory diffusion model implementation
- `eval/`: Evaluation metrics and utilities

### Baselines

We include baselines as separate directories:

- `timegan/`: TimeGAN
- `timevae/`: TimeVAE
- `cosci-gan/`: COSCI-GAN
- `diffusion-ts/`: Diffusion-TS
- `difftraj/`: DiffTraj

### Visualization Notebook

- `Compare original to synthetic samples globally.ipynb`: Comprehensive global comparison visualizations

### Configuration and Utilities

- `config.py`: Configuration settings and hyperparameters
- `common.py`: Common utility functions
- `valhalla/`: Map matching instructions and docker-compose files
- `data/`: Dataset storage and processing

## Data Sources

### Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015 (porto)

Expected path: `data/raw/porto/train.csv`

Source: https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### Geolife 

Expected path: `data/raw/Geolife Trajectories 1.3/`

Source: https://www.microsoft.com/en-us/download/details.aspx?id=52367

License: Microsoft Research License Agreement (see User Guide-1.3.pdf in the zip-file from the link above)

### Cabspotting

Expected path: `data/raw/cabspottingdata/cabspottingdata/new_*.txt`

Source: https://ieee-dataport.org/open-access/crawdad-epflmobility

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### Brest (maritime AIS)

Expected path: `data/raw/brest/ais_brest_positions_raw.csv`

Source: *Final dataset of Trajectory Synopses over AIS kinematic messages in Brest area (ver. 0.8)*, https://doi.org/10.5281/zenodo.2563256. Download `ais_brest_synopses_v0.8.zip` and extract the one file we use, unchanged:

```bash
unzip -j ais_brest_synopses_v0.8.zip ais_brest_positions_raw.csv -d data/raw/brest/
```

The file is the NARI AIS dataset (Ray et al., *Heterogeneous integrated dataset for Maritime Intelligence, Surveillance, and Reconnaissance*, Data in Brief 2019, https://doi.org/10.1016/j.dib.2019.104141) with duplicate messages removed. Filtering happens when the file is read (`read_brest_dfs` in `preproc.py`).

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Brest is not map matched: `python preprocess_data.py brest-100` skips the Valhalla step.

## Setting up Runtime Environments

### TimeGAN

TimeGAN needs TensorFlow 1.15; we used Python 3.7 with TensorFlow 1.15.5:

```bash
conda create --name timegan python=3.7
conda activate timegan
pip install tensorflow==1.15.5 protobuf==3.20.3
pip install pandas scipy scikit-learn matplotlib tqdm pyarrow wandb
```

### TimeVAE

```bash
cd timevae
virtualenv -p python3.10 env
source env/bin/activate
pip install -r requirements.txt
pip install tensorflow[and-cuda]
pip install pyarrow
pip install wandb
```

### Diffusion-TS

```bash
cd diffusion-ts
conda create --name diffusionts python=3.10
conda activate diffusionts
pip install -r requirements.txt
pip install pyyaml
pip install pyarrow
pip install wandb
pip install colored  # used by DiffTraj, which runs in this environment
```

`diffusion-ts/requirements.txt` has upstream's pins (`torch==2.0.1`), which do not support recent GPUs. We ran Diffusion-TS and DiffTraj in the TDDM environment below (Python 3.10, PyTorch 2.10 with CUDA 12.9), with `pyyaml` and `colored` added.

### COSCI-GAN

No setup necessary, the environment for Diffusion-TS works with COSCI-GAN:
```bash
conda activate diffusionts
```

### TDDM (ours)

For running our code (TDDM, in `tddpm/`) and any code not related to baselines, we use conda for better reproducibility and to avoid library conflicts:

```bash
conda create -n tddm python=3.10
conda activate tddm
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install tqdm cython matplotlib pandas numpy wandb einops ema_pytorch accelerate requests
cd tddpm
python setup.py build_ext --inplace
```

The last two commands compile the Cython version of the trajectory cropping used in preprocessing. Without it, `preproc.py` falls back to a slower pure-Python version.

## Data preprocessing

1. Preprocess data (`<dataset>` is e.g. `porto-100`; see `python preprocess_data.py --help` for the choices):
```bash
python preprocess_data.py <dataset>
```

2. Extract conditional data (for our method; the second argument is the region size in metres):
```bash
python conditional-variant-generator.py <dataset> 3000
```

## TDDM

### Train TDDM

```bash
cd tddpm
python train_conditional.py <dataset>
python train_conditional.py <dataset> --seed 42
python train_conditional.py <dataset> --seed 1
```
The paper reports the mean over these three runs. A run with `--seed N` saves `model_weights/conditional-<dataset>-100000-3000-seedN.pth`; sample it with `python sample.py <dataset> <dataset> --run-id seedN`.

The ablation without spatial prior is trained with `train_all_unconditional.sh` (three runs per dataset) and sampled with rejection by `run_rejection_sampling.sh`, see `tddpm/REJECTION_SAMPLING_README.md`.

The hyperparameters in `train_conditional.py` were selected with an Optuna search on Geolife; the search script is not included.

## Running related work

### TimeGAN

1. Activate the TimeGAN environment:
```bash
conda activate timegan
```

2. Run the model:
```bash
cd timegan
bash run_all.sh
```

The model will train on the preprocessed data and save the generated trajectories, for each dataset with seeds 42, 1 and 2 (the paper's TimeGAN rows are the mean over these three runs).

### TimeVAE

1. Activate the TimeVAE environment:
```bash
cd timevae
source env/bin/activate
```

2. Run the model:
```bash
./run_all.sh
```

The model will train and save the generated trajectories in the `timevae/` directory, three runs per dataset (without `--seed`, `--seed 42` and `--seed 1`), as in the paper.

### COSCI-GAN

1. Activate the environment (uses the same as Diffusion-TS):
```bash
conda activate diffusionts
```

2. Run the model and sample synthetic datasets:
```bash
cd cosci-gan/Code
./run_all.sh
```

The model will train and save the generated trajectories in `cosci-gan/Results/`, three runs per dataset (without `--seed`, `--seed 42` and `--seed 1`), as in the paper.

### Diffusion-TS

1. Activate the environment:
```bash
conda activate diffusionts
```

2. Run the model:
```bash
cd diffusion-ts

# For Geolife dataset
./train-geolife-100-128.sh

# For Porto dataset
./train-porto-100-128.sh

# For Cabspotting dataset
./train-cabspotting-100-128.sh

# For Brest dataset
./train-brest-100-128.sh

# Or run all datasets, three runs each (default seed, --seed 42 and --seed 1), as in the paper
./train-all.sh
```

The model will train and save the generated trajectories in the `OUTPUT` directory.

### DiffTraj

1. Activate the environment (uses the same as Diffusion-TS):
```bash
conda activate diffusionts
```

2. Run the model:
```bash
cd difftraj
for dataset in geolife-100 porto-100 cabspotting-100 brest-100; do
  python main.py --dataset $dataset --seq_len 128
  python main.py --dataset $dataset --seq_len 128 --seed 42
  python main.py --dataset $dataset --seq_len 128 --seed 1
done
```

## Evaluation Pipeline

The synthetic datasets are registered in [synthetic_loader.py](synthetic_loader.py) and the tables are defined in [config.py](config.py). Each measure script computes every (dataset, model, filter) combination that the tables need and that is not yet in its cache:

```bash
python run_evaluation.py eval
python speed_kl_divergence.py
python length_error_v2.py
python tstr_scaled_diff.py
python compute_v2_noise_floors.py
```

The tables in the paper are generated with:

| Paper table | Command |
|---|---|
| Unconditional benchmark (and its summary) | `python generate_tables.py --table 0 --seed-mean` / `python generate_summary_table.py --table 0 --seed-mean` |
| Ablation (and its summary) | `python generate_tables.py --table 1 --seed-mean` / `python generate_summary_table.py --table 1 --seed-mean` |
| Generalization from 25% of the city | `python generate_table_generalize_25_loqo.py` |
| Combined generalization (Table 3) | `python generate_combined_generalization_table_with_brest.py --seed-mean` |
| City-to-city generalization | `python generate_tables.py --table 3 --seed-mean` |
| Map dropout | `python generate_tables.py --table 4 --seed-mean --no-bold` / `python generate_summary_table.py --table 4 --seed-mean` |
| No map matching | `python generate_tables.py --table 5` / `python generate_summary_table.py --table 5` |

`--seed-mean` reports each cell as the mean ± standard error over the training runs of a model (usually the run registered under the model name plus its `-seed42` and `-seed1` runs; see `seed_mean.py`); without it, the tables show the run registered under the model name (TimeGAN has none, as its three runs are all seeded, so its row is empty). The summaries of the map dropout and no map matching tables are not in the paper.

### Notes on the baselines

- **TimeGAN:** upstream's loader reverses every window in time, which puts its newest-first stock CSV in chronological order; our windows are already chronological, so `timegan/data_loading.py` does not (`--reverse_time` restores it). The paper's TimeGAN rows (`timegan-forward`, seeds 42, 1 and 2) are trained in forward time.
- **TimeVAE:** TimeVAE trains on windows min-max scaled per timestep; `timevae/test_vae.py` applies the inverse scaling before saving, so the samples are in the same frame as the data.

### Not included

Cluster job scripts and the launchers for individual experiment runs (map dropout, the no-map-match variants, the heatmap-resolution experiment) are not included; the scripts above and their command-line flags (e.g. `--dropout` in `tddpm/sample.py`, `--no-map-match` in `preprocess_data.py`, `--res`) cover the same steps. The paper's tables average three training runs per model (`--seed-mean`): every training script takes `--seed`, and the launchers (`run_all.sh`, `train-all.sh`, `train_all_unconditional.sh`) run the three runs per dataset.

## License

Copyright 2026 David Bergström. The code in this repository is released under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The baselines in `timegan/`, `timevae/`, `cosci-gan/`, `diffusion-ts/` and `difftraj/`, and the diffusion code in `tddpm/denoising_diffusion_pytorch/`, keep their own licenses (MIT, and Apache-2.0 for TimeGAN); see the license files in those directories.

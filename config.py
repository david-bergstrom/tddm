import os

from os.path import expanduser

RAW_DATA_PATH = expanduser('data/raw/')
SYNTH_DATA_ROOT = expanduser('synthetic_data/')

os.makedirs(SYNTH_DATA_ROOT, exist_ok=True)

datasets = ['geolife-100', 'cabspotting-100', 'porto-100', 'brest-100']

models = []

table_names = [
    'Unconditional Generation Benchmark',
    'Ablation Study',
    'Generalization from 25% of the city',
    'City-to-city Generalization',
    'Robustness to Map Dropout',
    'No Map Matching',
    'Training-seed Variance',
    'Additional runs',
]


# Each element is a table, each table consist of a list of tuples
# Each tuple is a model and what should be filtered-out
tables = [
            [
                # Unconditional Generation Benchmark
                ('timegan-forward', 'none'),  # forward time; seeds 42, 1, 2 (seed_mean.RUNS)
                ('timevae', 'none'),
                ('coscigan', 'none'),
                ('diffusionts', 'none'),
                ('difftraj', 'none'),
                ('tddpm', 'none'),
            ],
            [
                # Ablation Study
                # ('diffusionts', 'none'),
                # ('difftraj', 'none'),
                ('tddpm', 'none'),
                ('tddpm-region-size-1000', 'none'),
                ('tddpm-no-conditional-same-model', 'none'),  # same model as the rejection row in every run
                ('tddpm-no-conditional-rejection-10k', 'none'),  # 10,000 accepted samples per city
            ],
            [
                # Generalization from 25% of the city: the lower-left quadrant model, as in the paper's
                # generalization figures. The paper's 25% tables (generate_table_generalize_25_loqo.py,
                # generate_combined_generalization_table_with_brest.py) list all four quadrant models themselves.
                ('tddpm', 'lower-left-quarter'),
                ('tddpm-generalize-from-25-bl-v2', 'lower-left-quarter'),
            ],
            [
                # City-to-city Generalization
                # ('diffusionts', 'none'),
                # ('difftraj', 'none'),
                # ('tddpm', 'none'),
                ('tddpm-generalize-from-geolife-100', 'none'),
                ('tddpm-generalize-from-porto-100', 'none'),
                ('tddpm-generalize-from-cabspotting-100', 'none'),
                ('tddpm-generalize-from-brest-100', 'none'),
            ],
            [
                # Robustness to Map Dropout
                ('tddpm', 'none'),
                ('tddpm-dropout-rs3k-10', 'none'),
                ('tddpm-dropout-rs3k-20', 'none'),
            ],
            [
                # No Map Matching
                ('diffusionts-no-map-match', 'none'),
                ('difftraj-no-map-match-500ep', 'none'),  # 500 epochs, like the other DiffTraj runs
                ('tddpm-no-map-match', 'none'),
            ],
            [
                # Training-seed variance: the seed-42 and seed-1 runs; with the unsuffixed entries in the
                # main benchmark, n=3 per model (TimeGAN's three runs are in seed_mean.RUNS).
                ('tddpm-seed42', 'none'),
                ('tddpm-seed1', 'none'),
                ('difftraj-seed42', 'none'),
                ('difftraj-seed1', 'none'),
                ('diffusionts-seed42', 'none'),
                ('diffusionts-seed1', 'none'),
                ('timevae-seed42', 'none'),
                ('timevae-seed1', 'none'),
                ('coscigan-seed42', 'none'),
                ('coscigan-seed1', 'none'),
                ('tddpm-dropout-rs3k-10-seed42', 'none'),
                ('tddpm-dropout-rs3k-10-seed1', 'none'),
                ('tddpm-dropout-rs3k-20-seed42', 'none'),
                ('tddpm-dropout-rs3k-20-seed1', 'none'),
                ('tddpm-generalize-from-geolife-100-seed42', 'none'),
                ('tddpm-generalize-from-geolife-100-seed1', 'none'),
                ('tddpm-generalize-from-porto-100-seed42', 'none'),
                ('tddpm-generalize-from-porto-100-seed1', 'none'),
                ('tddpm-generalize-from-cabspotting-100-seed42', 'none'),
                ('tddpm-generalize-from-cabspotting-100-seed1', 'none'),
                ('tddpm-generalize-from-brest-100-seed42', 'none'),
                ('tddpm-generalize-from-brest-100-seed1', 'none'),
            ],
            [
                # Additional runs: TimeGAN in forward time (seeds 42, 1, 2), the w/o-spatial-prior
                # seeds, rejection at 10,000 samples, and DiffTraj no-map-match at 500 epochs.
                ('timegan-forward-seed42', 'none'),
                ('timegan-forward-seed1', 'none'),
                ('timegan-forward-seed2', 'none'),
                ('tddpm-no-conditional-seed42', 'none'),
                ('tddpm-no-conditional-seed1', 'none'),
                ('tddpm-no-conditional-rejection-10k', 'none'),
                ('tddpm-no-conditional-rejection-10k-seed42', 'none'),
                ('tddpm-no-conditional-rejection-10k-seed1', 'none'),
                ('difftraj-no-map-match-500ep', 'none'),
                ('tddpm-no-conditional-same-model', 'none'),
                ('tddpm-region-size-1000-seed42', 'none'),
                ('tddpm-region-size-1000-seed1', 'none'),
            ]
        ]


# Generalization from 25% of the city, per quadrant: the full-city model and the model trained on that quadrant,
# each evaluated without the trajectories that pass through the quadrant. Used by
# generate_table_generalize_25_loqo.py and generate_combined_generalization_table_with_brest.py.
quadrant_baseline_pairs = [
    ('tddpm', 'lower-left-quarter'),
    ('tddpm', 'lower-right-quarter'),
    ('tddpm', 'upper-left-quarter'),
    ('tddpm', 'upper-right-quarter'),
]
quadrant_model_pairs = [
    ('tddpm-generalize-from-25-bl-v2', 'lower-left-quarter'),
    ('tddpm-generalize-from-25-br', 'lower-right-quarter'),
    ('tddpm-generalize-from-25-tl', 'upper-left-quarter'),
    ('tddpm-generalize-from-25-tr', 'upper-right-quarter'),
]

# Every (model, filter) cell the measure scripts compute
for table in tables + [quadrant_baseline_pairs, quadrant_model_pairs]:
    for model_context in table:
        if model_context not in models:
            models.append(model_context)

nice_names = {
    'timevae': 'TimeVAE', 
    'coscigan': 'COSCI-GAN',
    'diffusionts': 'Diffusion-TS',
    'tddpm': 'TDDM',
    'geolife-100': 'Geolife',
    'porto-100': 'Porto',
    'cabspotting-100': 'Cabspotting',
    'difftraj': 'DiffTraj',
    'tddpm-region-size-1000': '1x1 km',
    'tddpm-generalize-from-geolife-100': 'Geolife',
    'tddpm-generalize-from-porto-100': 'Porto',
    'tddpm-generalize-from-cabspotting-100': 'Cabspotting',
    'tddpm-generalize-from-brest-100': 'Brest',
    'tddpm-generalize-from-geolife-100-seed42': 'Geolife (seed 42)',
    'tddpm-generalize-from-geolife-100-seed1': 'Geolife (seed 1)',
    'tddpm-generalize-from-porto-100-seed42': 'Porto (seed 42)',
    'tddpm-generalize-from-porto-100-seed1': 'Porto (seed 1)',
    'tddpm-generalize-from-cabspotting-100-seed42': 'Cabspotting (seed 42)',
    'tddpm-generalize-from-cabspotting-100-seed1': 'Cabspotting (seed 1)',
    'tddpm-generalize-from-brest-100-seed42': 'Brest (seed 42)',
    'tddpm-generalize-from-brest-100-seed1': 'Brest (seed 1)',
    'tddpm-dropout-rs3k-10': r'10\% dropout',
    'tddpm-dropout-rs3k-20': r'20\% dropout',
    'diffusionts-no-map-match': 'Diffusion-TS',
    'difftraj-no-map-match-500ep': 'DiffTraj',
    'tddpm-no-map-match': 'TDDM',
    'tddpm-seed42': 'TDDM (seed 42)',
    'tddpm-seed1': 'TDDM (seed 1)',
    'difftraj-seed42': 'DiffTraj (seed 42)',
    'difftraj-seed1': 'DiffTraj (seed 1)',
    'diffusionts-seed42': 'Diffusion-TS (seed 42)',
    'diffusionts-seed1': 'Diffusion-TS (seed 1)',
    'timevae-seed42': 'TimeVAE (seed 42)',
    'timevae-seed1': 'TimeVAE (seed 1)',
    'coscigan-seed42': 'COSCI-GAN (seed 42)',
    'coscigan-seed1': 'COSCI-GAN (seed 1)',
    'timegan-forward': 'TimeGAN',
    'timegan-forward-seed42': 'TimeGAN (seed 42)',
    'timegan-forward-seed1': 'TimeGAN (seed 1)',
    'timegan-forward-seed2': 'TimeGAN (seed 2)',
    'tddpm-no-conditional-seed42': 'w/o spatial prior (seed 42)',
    'tddpm-no-conditional-seed1': 'w/o spatial prior (seed 1)',
    'tddpm-no-conditional-same-model': 'w/o spatial prior',
    'tddpm-region-size-1000-seed42': '1x1 km (seed 42)',
    'tddpm-region-size-1000-seed1': '1x1 km (seed 1)',
    'tddpm-no-conditional-rejection-10k': 'w/o spatial prior + rejection',
    'tddpm-no-conditional-rejection-10k-seed42': 'w/o spatial prior + rejection (seed 42)',
    'tddpm-no-conditional-rejection-10k-seed1': 'w/o spatial prior + rejection (seed 1)',
    'tddpm-dropout-rs3k-10-seed42': r'10\% dropout (seed 42)',
    'tddpm-dropout-rs3k-10-seed1': r'10\% dropout (seed 1)',
    'tddpm-dropout-rs3k-20-seed42': r'20\% dropout (seed 42)',
    'tddpm-dropout-rs3k-20-seed1': r'20\% dropout (seed 1)',
    'brest-100': 'Brest',
    'tddpm-generalize-from-25-bl-v2': r'BL (25\%)',
    'tddpm-generalize-from-25-br': r'BR (25\%)',
    'tddpm-generalize-from-25-tl': r'TL (25\%)',
    'tddpm-generalize-from-25-tr': r'TR (25\%)',
    }


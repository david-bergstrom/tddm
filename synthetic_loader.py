import numpy as np
from collections import namedtuple
import os
from config import SYNTH_DATA_ROOT

SyntheticDataset  = namedtuple('SyntheticDataset', ['model_name', 'dataset', 'seq_len', 'path'])

all_datasets = [
    SyntheticDataset('coscigan',    'cabspotting-100', 128, 'coscigan-synthetic-cabspotting-100-128-tough-feather-48.npy'),
    SyntheticDataset('coscigan',    'geolife-100',     128, 'coscigan-synthetic-geolife-100-128-lunar-energy-28.npy'),
    SyntheticDataset('coscigan',    'porto-100',       128, 'coscigan-synthetic-porto-100-128-balmy-wood-45.npy'),
    SyntheticDataset('difftraj',    'cabspotting-100', 128, 'difftraj-synthetic-cabspotting-100-128.npy'),
    SyntheticDataset('difftraj',    'geolife-100',     128, 'difftraj-synthetic-geolife-100-128.npy'),
    SyntheticDataset('difftraj',    'porto-100',       128, 'difftraj-synthetic-porto-100-128.npy'),
    # Trained for 500 epochs, like the other DiffTraj runs.
    SyntheticDataset('difftraj-no-map-match-500ep', 'cabspotting-100', 128, 'difftraj-synthetic-cabspotting-100-no-map-match-128.npy'),
    SyntheticDataset('difftraj-no-map-match-500ep', 'geolife-100',     128, 'difftraj-synthetic-geolife-100-no-map-match-128.npy'),
    SyntheticDataset('difftraj-no-map-match-500ep', 'porto-100',       128, 'difftraj-synthetic-porto-100-no-map-match-128.npy'),
    SyntheticDataset('diffusionts', 'cabspotting-100', 128, 'diffusionts-synthetic-cabspotting-100-128-experiment-peachy-surf-68.npy'),
    SyntheticDataset('diffusionts', 'geolife-100',     128, 'diffusionts-synthetic-geolife-100-128-experiment-sleek-moon-65.npy'),
    SyntheticDataset('diffusionts', 'porto-100',       128, 'diffusionts-synthetic-porto-100-128-experiment-icy-smoke-66.npy'),
    SyntheticDataset('diffusionts-no-map-match', 'cabspotting-100', 128, 'diffusionts-synthetic-cabspotting-100-no-map-match-128.npy'),
    SyntheticDataset('diffusionts-no-map-match', 'geolife-100',     128, 'diffusionts-synthetic-geolife-100-no-map-match-128.npy'),
    SyntheticDataset('diffusionts-no-map-match', 'porto-100',       128, 'diffusionts-synthetic-porto-100-no-map-match-128.npy'),
    SyntheticDataset('timevae',     'cabspotting-100', 128, 'timevae-synthetic-cabspotting-100-128-experiment-dark-resonance-57-rescaled.npy'),
    SyntheticDataset('timevae',     'geolife-100',     128, 'timevae-synthetic-geolife-100-128-experiment-helpful-water-55-rescaled.npy'),
    SyntheticDataset('timevae',     'porto-100',       128, 'timevae-synthetic-porto-100-128-experiment-northern-silence-56-rescaled.npy'),
    SyntheticDataset('tddpm',       'geolife-100',     128, 'tddpm-conditional-geolife-100-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm',       'porto-100',       128, 'tddpm-conditional-porto-100-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm',       'cabspotting-100', 128, 'tddpm-conditional-cabspotting-100-to-cabspotting-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-geolife-100',           'porto-100',             128, 'tddpm-conditional-geolife-100-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-geolife-100',           'cabspotting-100',       128, 'tddpm-conditional-geolife-100-to-cabspotting-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-porto-100',             'geolife-100',           128, 'tddpm-conditional-porto-100-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-porto-100',             'cabspotting-100',       128, 'tddpm-conditional-porto-100-to-cabspotting-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-cabspotting-100',       'porto-100',             128, 'tddpm-conditional-cabspotting-100-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-cabspotting-100',       'geolife-100',           128, 'tddpm-conditional-cabspotting-100-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-region-size-1000',                      'geolife-100',           128, 'tddpm-conditional-geolife-100-to-geolife-100-region-size-1000.npy'),
    SyntheticDataset('tddpm-region-size-1000',                      'porto-100',             128, 'tddpm-conditional-porto-100-to-porto-100-region-size-1000.npy'),
    SyntheticDataset('tddpm-region-size-1000',                      'cabspotting-100',       128, 'tddpm-conditional-cabspotting-100-to-cabspotting-100-region-size-1000.npy'),
    # Brest 1x1 km
    SyntheticDataset('tddpm-region-size-1000',                      'brest-100',             128, 'tddpm-conditional-brest-100-to-brest-100-region-size-1000.npy'),
    # Generalization from 25%: quadrant models BR (bottom-right)
    SyntheticDataset('tddpm-generalize-from-25-br', 'geolife-100', 128, 'tddpm-conditional-geolife-25-br-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-br', 'porto-100', 128, 'tddpm-conditional-porto-25-br-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-br', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-25-br-to-cabspotting-100-region-size-3000.npy'),
    # Generalization from 25%: quadrant models TL (top-left)
    SyntheticDataset('tddpm-generalize-from-25-tl', 'geolife-100', 128, 'tddpm-conditional-geolife-25-tl-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-tl', 'porto-100', 128, 'tddpm-conditional-porto-25-tl-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-tl', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-25-tl-to-cabspotting-100-region-size-3000.npy'),
    # Generalization from 25%: quadrant models TR (top-right)
    SyntheticDataset('tddpm-generalize-from-25-tr', 'geolife-100', 128, 'tddpm-conditional-geolife-25-tr-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-tr', 'porto-100', 128, 'tddpm-conditional-porto-25-tr-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-tr', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-25-tr-to-cabspotting-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-no-map-match', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-100-no-map-match-to-cabspotting-100-no-map-match-region-size-3000.npy'),
    SyntheticDataset('tddpm-no-map-match', 'geolife-100', 128, 'tddpm-conditional-geolife-100-no-map-match-to-geolife-100-no-map-match-region-size-3000.npy'),
    SyntheticDataset('tddpm-no-map-match', 'porto-100', 128, 'tddpm-conditional-porto-100-no-map-match-to-porto-100-no-map-match-region-size-3000.npy'),
    # Maritime AIS
    SyntheticDataset('tddpm', 'brest-100', 128, 'tddpm-conditional-brest-100-to-brest-100-region-size-3000.npy'),
    SyntheticDataset('timevae',     'brest-100', 128, 'timevae-synthetic-brest-100-128-experiment-None-rescaled.npy'),
    SyntheticDataset('coscigan',    'brest-100', 128, 'coscigan-synthetic-brest-100-128-None.npy'),
    SyntheticDataset('diffusionts', 'brest-100', 128, 'diffusionts-synthetic-brest-100-128-experiment.npy'),
    SyntheticDataset('difftraj',    'brest-100', 128, 'difftraj-synthetic-brest-100-128.npy'),
    SyntheticDataset('tddpm-generalize-from-geolife-100',     'brest-100', 128, 'tddpm-conditional-geolife-100-to-brest-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-porto-100',       'brest-100', 128, 'tddpm-conditional-porto-100-to-brest-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-cabspotting-100', 'brest-100', 128, 'tddpm-conditional-cabspotting-100-to-brest-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-brest-100', 'geolife-100',     128, 'tddpm-conditional-brest-100-to-geolife-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-brest-100', 'porto-100',       128, 'tddpm-conditional-brest-100-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-brest-100', 'cabspotting-100', 128, 'tddpm-conditional-brest-100-to-cabspotting-100-region-size-3000.npy'),
    # Brest quadrant models (brest-25-X → brest-100)
    SyntheticDataset('tddpm-generalize-from-25-br', 'brest-100', 128, 'tddpm-conditional-brest-25-br-to-brest-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-tl', 'brest-100', 128, 'tddpm-conditional-brest-25-tl-to-brest-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-tr', 'brest-100', 128, 'tddpm-conditional-brest-25-tr-to-brest-100-region-size-3000.npy'),
    # Seed-variance runs (2 extra training seeds per model).
    # Convention: the seed is encoded in model_name as '<base>-seed<N>'; analysis scripts
    # split it back out with r'-seed(\d+)$'. The unsuffixed base entries above are the
    # third member of each n=3 family.
    SyntheticDataset('tddpm-seed42', 'geolife-100',     128, 'tddpm-conditional-geolife-100-to-geolife-100-region-size-3000run_id-seed42.npy'),
    SyntheticDataset('tddpm-seed42', 'porto-100',       128, 'tddpm-conditional-porto-100-to-porto-100-region-size-3000run_id-seed42.npy'),
    SyntheticDataset('tddpm-seed42', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-100-to-cabspotting-100-region-size-3000run_id-seed42.npy'),
    SyntheticDataset('tddpm-seed42', 'brest-100',       128, 'tddpm-conditional-brest-100-to-brest-100-region-size-3000run_id-seed42.npy'),
    SyntheticDataset('tddpm-seed1',  'geolife-100',     128, 'tddpm-conditional-geolife-100-to-geolife-100-region-size-3000run_id-seed1.npy'),
    SyntheticDataset('tddpm-seed1',  'porto-100',       128, 'tddpm-conditional-porto-100-to-porto-100-region-size-3000run_id-seed1.npy'),
    SyntheticDataset('tddpm-seed1',  'cabspotting-100', 128, 'tddpm-conditional-cabspotting-100-to-cabspotting-100-region-size-3000run_id-seed1.npy'),
    SyntheticDataset('tddpm-seed1',  'brest-100',       128, 'tddpm-conditional-brest-100-to-brest-100-region-size-3000run_id-seed1.npy'),
    SyntheticDataset('difftraj-seed42', 'geolife-100',     128, 'difftraj-synthetic-geolife-100-128-seed42.npy'),
    SyntheticDataset('difftraj-seed42', 'porto-100',       128, 'difftraj-synthetic-porto-100-128-seed42.npy'),
    SyntheticDataset('difftraj-seed42', 'cabspotting-100', 128, 'difftraj-synthetic-cabspotting-100-128-seed42.npy'),
    SyntheticDataset('difftraj-seed42', 'brest-100',       128, 'difftraj-synthetic-brest-100-128-seed42.npy'),
    SyntheticDataset('difftraj-seed1',  'geolife-100',     128, 'difftraj-synthetic-geolife-100-128-seed1.npy'),
    SyntheticDataset('difftraj-seed1',  'porto-100',       128, 'difftraj-synthetic-porto-100-128-seed1.npy'),
    SyntheticDataset('difftraj-seed1',  'cabspotting-100', 128, 'difftraj-synthetic-cabspotting-100-128-seed1.npy'),
    SyntheticDataset('difftraj-seed1',  'brest-100',       128, 'difftraj-synthetic-brest-100-128-seed1.npy'),
    SyntheticDataset('diffusionts-seed42', 'geolife-100',     128, 'diffusionts-synthetic-geolife-100-128-seed42.npy'),
    SyntheticDataset('diffusionts-seed42', 'porto-100',       128, 'diffusionts-synthetic-porto-100-128-seed42.npy'),
    SyntheticDataset('diffusionts-seed42', 'cabspotting-100', 128, 'diffusionts-synthetic-cabspotting-100-128-seed42.npy'),
    SyntheticDataset('diffusionts-seed42', 'brest-100',       128, 'diffusionts-synthetic-brest-100-128-seed42.npy'),
    SyntheticDataset('diffusionts-seed1',  'geolife-100',     128, 'diffusionts-synthetic-geolife-100-128-seed1.npy'),
    SyntheticDataset('diffusionts-seed1',  'porto-100',       128, 'diffusionts-synthetic-porto-100-128-seed1.npy'),
    SyntheticDataset('diffusionts-seed1',  'cabspotting-100', 128, 'diffusionts-synthetic-cabspotting-100-128-seed1.npy'),
    SyntheticDataset('diffusionts-seed1',  'brest-100',       128, 'diffusionts-synthetic-brest-100-128-seed1.npy'),
    # Map dropout, sampled at region size 3000.
    SyntheticDataset('tddpm-dropout-rs3k-10', 'geolife-100',     128, 'tddpm-conditional-geolife-100-to-geolife-100-region-size-3000-dropout-10.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-10', 'porto-100',       128, 'tddpm-conditional-porto-100-to-porto-100-region-size-3000-dropout-10.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-10', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-100-to-cabspotting-100-region-size-3000-dropout-10.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-10', 'brest-100',       128, 'tddpm-conditional-brest-100-to-brest-100-region-size-3000-dropout-10.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-20', 'geolife-100',     128, 'tddpm-conditional-geolife-100-to-geolife-100-region-size-3000-dropout-20.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-20', 'porto-100',       128, 'tddpm-conditional-porto-100-to-porto-100-region-size-3000-dropout-20.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-20', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-100-to-cabspotting-100-region-size-3000-dropout-20.npy'),
    SyntheticDataset('tddpm-dropout-rs3k-20', 'brest-100',       128, 'tddpm-conditional-brest-100-to-brest-100-region-size-3000-dropout-20.npy'),
] + [
    SyntheticDataset(f'tddpm-dropout-rs3k-{pct}-seed{seed}', ds, 128,
                     f'tddpm-conditional-{ds}-to-{ds}-region-size-3000run_id-seed{seed}-dropout-{pct}.npy')
    for pct in (10, 20) for seed in (42, 1)
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
] + [
    # BL (25%): Geolife and Cabspotting use models retrained (seed 42) on the BL data built from the
    # full-city data; Porto and Brest use their original models.
    SyntheticDataset('tddpm-generalize-from-25-bl-v2', 'geolife-100',     128, 'tddpm-conditional-geolife-25-bl-to-geolife-100-region-size-3000run_id-seed42.npy'),
    SyntheticDataset('tddpm-generalize-from-25-bl-v2', 'porto-100',       128, 'tddpm-conditional-porto-25-to-porto-100-region-size-3000.npy'),
    SyntheticDataset('tddpm-generalize-from-25-bl-v2', 'cabspotting-100', 128, 'tddpm-conditional-cabspotting-25-bl-to-cabspotting-100-region-size-3000run_id-seed42.npy'),
    SyntheticDataset('tddpm-generalize-from-25-bl-v2', 'brest-100',       128, 'tddpm-conditional-brest-25-bl-to-brest-100-region-size-3000.npy'),
] + [
    # Seed runs for the remaining baselines.
    SyntheticDataset(f'{model}-seed{seed}', ds, 128, path.format(ds=ds, seed=seed))
    for model, path in (('timevae', 'timevae-synthetic-{ds}-128-experiment-None-seed{seed}-rescaled.npy'),
                        ('coscigan', 'coscigan-synthetic-{ds}-128-None-seed{seed}.npy'))
    for seed in (42, 1)
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
] + [
    # TimeGAN trained in forward time order: the three runs are seeds 42, 1 and 2 (`seed_mean.RUNS`).
    SyntheticDataset(f'timegan-forward-seed{seed}', ds, 128, f'timegan-synthetic-{ds}-128-forward-seed{seed}.npy')
    for seed in (42, 1, 2)
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
] + [
    # "w/o spatial prior" ablation, trained with seeds 42 and 1.
    SyntheticDataset(f'tddpm-no-conditional-seed{seed}', ds, 128, f'unconditional-tddpm-{ds}-128-1-seed{seed}.npy')
    for seed in (42, 1)
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
] + [
    # "w/o spatial prior", third run: samples from the same model that the rejection row's original run uses
    # (unfiltered samples saved by train_unconditional.py at the end of training), so that both rows come
    # from the same model in every run.
    SyntheticDataset('tddpm-no-conditional-same-model', ds, 128, f'unconditional-tddpm-{ds}-128-1.npy')
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100')
] + [
    SyntheticDataset('tddpm-no-conditional-same-model', 'brest-100', 128, 'unconditional-tddpm-brest-100-128.npy'),
] + [
    # "w/o spatial prior + rejection" at 10,000 accepted samples:
    # tddpm/sample_unconditional_rejection.py, --noise-scale 1.0.
    SyntheticDataset(f'tddpm-no-conditional-rejection-10k{suffix}', ds, 128,
                     f'unconditional_rejection_{ds}_samples-10k-{tag}.npy')
    for suffix, tag in (('', 'submitted'), ('-seed42', 'seed42'), ('-seed1', 'seed1'))
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
] + [
    # 1x1 km (region size 1000) ablation seeds, all four cities. The third runs are the unsuffixed
    # entries above.
    SyntheticDataset(f'tddpm-region-size-1000-seed{seed}', ds, 128,
                     f'tddpm-conditional-{ds}-to-{ds}-region-size-1000run_id-seed{seed}.npy')
    for seed in (42, 1)
    for ds in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
] + [
    # City-to-city generalization from the seed-42 and seed-1 full-city models.
    SyntheticDataset(f'tddpm-generalize-from-{src}-seed{seed}', dst, 128,
                     f'tddpm-conditional-{src}-to-{dst}-region-size-3000run_id-seed{seed}.npy')
    for seed in (42, 1)
    for src in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
    for dst in ('geolife-100', 'porto-100', 'cabspotting-100', 'brest-100')
    if src != dst
]

def find_synthetic_datasets():
    return all_datasets

def load_synthetic(dataset, silent=False):
    full_path = os.path.join(SYNTH_DATA_ROOT, dataset.path)
    a = np.load(full_path)
    if a.dtype != np.float32:
        if not silent:
            print(f"Casting {dataset} to float32, was: {a.dtype}")
        a = a.astype(np.float32)
    if len(a) > 10000:
        if not silent:
            print(f"Cropping {dataset} to length 10k, was: {len(a)}")
        rnd = np.random.default_rng(42)
        rnd.shuffle(a)
        a = a[:10000]
    return a

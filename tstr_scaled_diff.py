"""
Scaled-difference TSTR (see eval.transformer_tstr.transformer_tstr_scaled_diff) for every configured cell.
Cells: config.models; real data: run_evaluation.load_train_data; keys (dataset, seq_len, model,
filter) -> tensor of per-window MAEs on all real windows of that dataset and filter. Skips keys already in the
pickle, so it only adds keys. Seed 0.
"""
import argparse
import os
import pickle

from config import datasets, models
from eval.transformer_tstr import transformer_tstr_scaled_diff
from run_evaluation import load_train_data, filter_data, ignore_combinations
from synthetic_loader import find_synthetic_datasets, load_synthetic


def main():
    parser = argparse.ArgumentParser(description='Scaled-difference TSTR for every configured cell')
    parser.add_argument('--output', default='transformer_tstr_scaled_diff_measures.pickle')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    try:
        results = pickle.load(open(args.output, 'rb'))
    except FileNotFoundError:
        results = {}
    all_datasets = find_synthetic_datasets()
    seq_len = 128
    real_cache = {}
    for dataset_str in datasets:
        for model, filter_out in models:
            key = (dataset_str, seq_len, model, filter_out)
            if key in results or key[:3] in ignore_combinations:
                continue
            match = [s for s in all_datasets if s.model_name == model and s.dataset == dataset_str and s.seq_len == seq_len]
            if not match:
                continue
            try:
                synth = filter_data(load_synthetic(match[0], silent=True), filter_out)
            except FileNotFoundError:
                print(f'No synthetic data for {key}')
                continue
            if (dataset_str, filter_out) not in real_cache:
                real_cache.clear()
                real_cache[(dataset_str, filter_out)] = load_train_data(dataset_str, seq_len, filter_out)[0]
            losses = transformer_tstr_scaled_diff(real_cache[(dataset_str, filter_out)], synth, seed=args.seed)
            results[key] = losses
            print(f'{key}: {losses.mean().item():.6f}', flush=True)
            # Save after every cell, so an interrupted run keeps what it has
            pickle.dump(results, open(args.output + '.tmp', 'wb'))
            os.replace(args.output + '.tmp', args.output)


if __name__ == '__main__':
    main()

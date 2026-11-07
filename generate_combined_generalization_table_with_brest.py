"""
Combined generalization table (Table 3): generalization from 25% of the city (the four quadrant models,
pooled per city) and city-to-city transfer, for all four cities including Brest.
"""
import pickle
import sys
import os
import argparse
from typing import Iterable, Optional, Tuple

import numpy as np

from config import quadrant_baseline_pairs, quadrant_model_pairs


def best_and_second(values: Iterable[float], maximize: bool) -> Tuple[Optional[float], Optional[float]]:
    uniq = sorted(set(values), reverse=maximize)
    best = uniq[0] if uniq else None
    second = uniq[1] if len(uniq) > 1 else None
    return best, second


datasets = ['geolife-100', 'porto-100', 'cabspotting-100', 'brest-100']
city_columns = ['Geolife', 'Porto', 'Cabspotting', 'Brest']
source_models = {
    'Geolife': ('tddpm-generalize-from-geolife-100', 'geolife-100'),
    'Porto': ('tddpm-generalize-from-porto-100', 'porto-100'),
    'Cabspotting': ('tddpm-generalize-from-cabspotting-100', 'cabspotting-100'),
    'Brest': ('tddpm-generalize-from-brest-100', 'brest-100'),
}
seq_len = 128
decimals = 3

measure_keys = ['transformer_tstr', 'kl_synth_real', 'kl_real_synth',
                'kl_symmetric', 'js', 'speed_kl', 'density_error', 'trip_error',
                'od_error', 'length_error', 'top_n_score']

measure_nice_names = {
    'transformer_tstr': r'TSTR ($\times 10^{-3}$, ↓)',
    'kl_synth_real': r'$\mathrm{KL}(S \parallel R)$ (↓)',
    'kl_real_synth': r'$\mathrm{KL}(R \parallel S)$ (↓)',
    'kl_symmetric': r'$\mathrm{KL}_\mathrm{sym}$ (↓)',
    'js': 'JS (↓)',
    'speed_kl': r'$\mathrm{KL}_\mathrm{speed}$ (↓)',
    'density_error': 'Density (↓)',
    'trip_error': 'Start/End (↓)',
    'od_error': 'OD (↓)',
    'length_error': 'Length (↓)',
    'top_n_score': 'Pattern (↑)',
}


parser = argparse.ArgumentParser(description='Generate combined generalization table (Table 3) including Brest')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--pickle-dir', type=str, default='.')
parser.add_argument('--seed-mean', action='store_true',
                    help='City-to-city columns as the mean over the three training runs of each source model')
args = parser.parse_args()


def _load(name):
    path = os.path.join(args.pickle_dir, name)
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if args.debug:
            print(f'Loaded {path}', file=sys.stderr)
        return data
    except FileNotFoundError as e:
        if args.debug:
            print(f'Warning: {e}', file=sys.stderr)
        return {}


transformer_tstrs = {k: v * 1e3 for k, v in _load('transformer_tstr_scaled_diff_measures.pickle').items()}  # in 1e-3
kls = _load('kl_measures.pickle')
difftraj_measures = _load('difftraj_measures.pickle')
speed_kls = _load('speed_kl_measures_v2.pickle')
od_measures = _load('od_measures.pickle')
# Length (length_error_v2.py) goes in index 2 of the DiffTraj tuple; a cell without a Length value shows nan
length_v2 = _load('length_measures_v2.pickle')
difftraj_measures = {k: (v[0], v[1], length_v2.get(k, float('nan')), v[3]) for k, v in difftraj_measures.items()}
if args.seed_mean:
    # The intra-city half stays single-run: the quadrant models have no seed runs, and the 100% baseline
    # is evaluated with the quadrant filters for one run only.
    from seed_mean import average_over_seeds
    transformer_tstrs = average_over_seeds(transformer_tstrs, 2)
    kls = average_over_seeds(kls, 2)
    difftraj_measures = average_over_seeds(difftraj_measures, 2)
    od_measures = average_over_seeds(od_measures, 2)
    speed_kls = average_over_seeds(speed_kls, 1)


def _per_cell_values(model, filter_out, dataset):
    cell = {}
    key = (dataset, seq_len, model, filter_out)

    if key in transformer_tstrs:
        cell['transformer_tstr'] = transformer_tstrs[key].mean().item()
    else:
        print(f'WARNING: missing transformer_tstr key {key}', file=sys.stderr)

    if key in kls:
        kl_sr, kl_rs, kl_sym, js = kls[key]
        cell['kl_synth_real'] = kl_sr
        cell['kl_real_synth'] = kl_rs
        cell['kl_symmetric'] = kl_sym
        cell['js'] = js
    else:
        print(f'WARNING: missing kl key {key}', file=sys.stderr)

    if key in difftraj_measures:
        de, te, le, top_n_score = difftraj_measures[key]
        cell['density_error'] = de
        cell['trip_error'] = te
        cell['length_error'] = le
        cell['top_n_score'] = top_n_score
    else:
        print(f'WARNING: missing difftraj key {key}', file=sys.stderr)

    if key in od_measures:
        cell['od_error'] = od_measures[key]
    else:
        print(f'WARNING: missing od key {key}', file=sys.stderr)

    speed_kl_key = (dataset, model, filter_out)
    if speed_kl_key in speed_kls:
        _, _, kl_sym_speed, _ = speed_kls[speed_kl_key]
        cell['speed_kl'] = kl_sym_speed
    else:
        print(f'WARNING: missing speed_kl key {speed_kl_key}', file=sys.stderr)

    return cell


def compute_source_column(model, source_dataset):
    pooled = {k: [] for k in measure_keys}
    for dataset in datasets:
        if dataset == source_dataset:
            continue
        cell = _per_cell_values(model, 'none', dataset)
        for k, v in cell.items():
            pooled[k].append(v)
    out = {}
    for k in measure_keys:
        if pooled[k]:
            out[k] = (np.round(np.mean(pooled[k]), decimals=decimals),)
    return out


def pool_intra_city(model_filter_pairs):
    pooled = {k: [] for k in measure_keys}
    for model, filter_out in model_filter_pairs:
        for dataset in datasets:
            cell = _per_cell_values(model, filter_out, dataset)
            for k, v in cell.items():
                pooled[k].append(v)
    out = {}
    for k in measure_keys:
        if pooled[k]:
            out[k] = (np.round(np.mean(pooled[k]), decimals=decimals),)
    return out


def extract_intra_city_data():
    data_100 = pool_intra_city(quadrant_baseline_pairs)
    data_25 = pool_intra_city(quadrant_model_pairs)
    return {k: {'25%': data_25.get(k), '100%': data_100.get(k)} for k in measure_keys}


def extract_city_to_city_data():
    combined = {k: {} for k in measure_keys}
    for col, (model, source_dataset) in source_models.items():
        col_data = compute_source_column(model, source_dataset)
        for k in measure_keys:
            combined[k][col] = col_data.get(k)
    return combined


def parse_value(val):
    if val is None:
        return None
    return float(val[0])


def identify_best(intra, city2city):
    best = {'intra_city': {}, 'city_to_city': {}}
    second = {'intra_city': {}, 'city_to_city': {}}
    for k in measure_keys:
        maximize = '(↑)' in measure_nice_names[k]

        intra_vals = [parse_value(intra[k].get(c)) for c in ['25%', '100%']]
        intra_vals = [v for v in intra_vals if v is not None]
        if intra_vals:
            b, s = best_and_second(intra_vals, maximize)
            best['intra_city'][k] = b
            second['intra_city'][k] = s

        city_vals = [parse_value(city2city[k].get(c)) for c in city_columns]
        city_vals = [v for v in city_vals if v is not None]
        if city_vals:
            b, s = best_and_second(city_vals, maximize)
            best['city_to_city'][k] = b
            second['city_to_city'][k] = s

    return best, second


def fmt(val, is_best, is_second):
    if val is None:
        return '-'
    mean = val[0]
    s = f'{mean:.{decimals}f}'
    if is_best:
        s = r'\mathbf{' + s + '}'
    elif is_second:
        s = r'\underline{' + s + '}'
    return '$' + s + '$'


def generate_latex(intra, city2city, best, second):
    lines = []
    header = 'Measure & 25\\% & 100\\% & ' + ' & '.join(city_columns) + r'\\'
    lines.append(header)
    lines.append(r'\midrule')
    lines.append(r'\addlinespace[0.5em]')

    for idx, k in enumerate(measure_keys):
        b_i = best['intra_city'].get(k)
        s_i = second['intra_city'].get(k)
        b_c = best['city_to_city'].get(k)
        s_c = second['city_to_city'].get(k)

        cells = []
        for col in ['25%', '100%']:
            val = intra[k].get(col)
            p = parse_value(val)
            ib = p is not None and b_i is not None and abs(p - b_i) < 1e-9
            isec = p is not None and s_i is not None and abs(p - s_i) < 1e-9
            cells.append(fmt(val, ib, isec))
        for col in city_columns:
            val = city2city[k].get(col)
            p = parse_value(val)
            ib = p is not None and b_c is not None and abs(p - b_c) < 1e-9
            isec = p is not None and s_c is not None and abs(p - s_c) < 1e-9
            cells.append(fmt(val, ib, isec))

        lines.append(f'{measure_nice_names[k]} & ' + ' & '.join(cells) + r'\\')
        if idx < len(measure_keys) - 1:
            lines.append(r'\addlinespace[0.5em]')

    return '\n'.join(lines)


if __name__ == '__main__':
    intra = extract_intra_city_data()
    city2city = extract_city_to_city_data()
    best, second = identify_best(intra, city2city)
    print(generate_latex(intra, city2city, best, second))

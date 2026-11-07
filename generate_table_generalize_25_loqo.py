"""Table for `Generalization from 25% of the city`, pooled over the four quadrants of each city.

For each city, the 25% row is the mean over the four quadrant models (tddpm-generalize-from-25-{bl,br,tl,tr})
each paired with its matching {lower-left,lower-right,upper-left,upper-right}-quarter filter, and the 100%
row is the full-city model (tddpm) evaluated with the same four filters (config.quadrant_*_pairs).

The output format is that of `generate_tables.py --mode latex-transpose`.
"""
import argparse
import pickle
import sys

import numpy as np

from config import quadrant_baseline_pairs, quadrant_model_pairs
from floor_bolding import best_and_second, floor_for_cell, saturated_values


datasets = ['geolife-100', 'porto-100', 'cabspotting-100', 'brest-100']
seq_len = 128
decimals = 3

# (row_label, list of (model, filter_out)) — per-city pool over these pairs
baseline_pairs = quadrant_baseline_pairs
corner_pairs = quadrant_model_pairs
# Ordered rows for the table (top = 100%, bottom = 25%, matching the existing layout)
rows = [
    ('100\\%', baseline_pairs, datasets),       # baseline exists for all 4 cities
    ('25\\%', corner_pairs, datasets),
]

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

_MAXIMIZED = {'top_n_score'}


def _load(path):
    try:
        return pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        return {}


transformer_tstrs = {k: v * 1e3 for k, v in _load('transformer_tstr_scaled_diff_measures.pickle').items()}  # in 1e-3
kls = _load('kl_measures.pickle')
difftraj_measures = _load('difftraj_measures.pickle')
od_measures = _load('od_measures.pickle')
speed_kls = _load('speed_kl_measures_v2.pickle')
# Length (length_error_v2.py) goes in index 2 of the DiffTraj tuple; a cell without a Length value shows nan
length_v2 = _load('length_measures_v2.pickle')
difftraj_measures = {k: (v[0], v[1], length_v2.get(k, float('nan')), v[3]) for k, v in difftraj_measures.items()}

parser = argparse.ArgumentParser(description='Generalization from 25% of the city, pooled over the four quadrant models per city')
args = parser.parse_args()


def _check_required_keys():
    """Verify every (city, model, filter) cell has all metric pickle entries.

    Stops with an explicit list rather than silently averaging fewer cells.
    """
    missing = []
    for _, pairs, row_datasets in rows:
        for ds in row_datasets:
            for model, filt in pairs:
                full = (ds, seq_len, model, filt)
                speed = (ds, model, filt)
                if full not in transformer_tstrs:
                    missing.append(('transformer_tstr', full))
                if full not in kls:
                    missing.append(('kls', full))
                if full not in difftraj_measures:
                    missing.append(('difftraj', full))
                if full not in od_measures:
                    missing.append(('od', full))
                if speed not in speed_kls:
                    missing.append(('speed_kl', speed))
    return missing


def _cell_scalar(measure, ds, model, filt):
    """Pre-aggregated scalar for one (city, model, filter) cell."""
    full = (ds, seq_len, model, filt)
    speed = (ds, model, filt)
    if measure == 'transformer_tstr':
        return float(transformer_tstrs[full].mean().item())
    if measure in ('kl_synth_real', 'kl_real_synth', 'kl_symmetric', 'js'):
        idx = ('kl_synth_real', 'kl_real_synth', 'kl_symmetric', 'js').index(measure)
        return float(kls[full][idx])
    if measure in ('density_error', 'trip_error', 'length_error', 'top_n_score'):
        idx = ('density_error', 'trip_error', 'length_error', 'top_n_score').index(measure)
        return float(difftraj_measures[full][idx])
    if measure == 'od_error':
        return float(od_measures[full])
    if measure == 'speed_kl':
        return float(speed_kls[speed][2])  # symmetric speed KL
    raise KeyError(measure)


def compute_cell_values(pairs, ds):
    """Return list of per-cell scalars for each measure for one city."""
    out = {m: [] for m in measure_keys}
    for model, filt in pairs:
        for m in measure_keys:
            out[m].append(_cell_scalar(m, ds, model, filt))
    return out


def pooled_floor(measure, ds, pairs):
    """Mean of per-filter floors across the four quadrants.

    Each filter has its own real-vs-real floor; the LOQO row pools cells from
    four filters, so its floor is the mean of those four per-filter floors.
    """
    vals = []
    for _, filt in pairs:
        f = floor_for_cell(measure, ds, seq_len, filt,
                           kls, difftraj_measures, od_measures, transformer_tstrs, speed_kls)
        if f is None:
            return None
        vals.append(f)
    return float(np.mean(vals)) if vals else None


def format_value(measure, mean, std, is_best, is_second, is_saturated):
    if measure == 'transformer_tstr':
        body = f'{mean:.{decimals}f} \\pm {std:.{decimals}f}'
    else:
        body = f'{mean:.{decimals}f}'
    if is_best:
        body = r'\mathbf{' + body + '}'
    elif is_second:
        body = r'\underline{' + body + '}'
    if is_saturated:
        body = body + r'^\dagger'
    return '$' + body + '$'


def main():
    missing = _check_required_keys()
    if missing:
        print('Missing pickle entries — refusing to generate a partial table:',
              file=sys.stderr)
        for kind, key in missing:
            print(f'  [{kind}] {key}', file=sys.stderr)
        sys.exit(1)

    # Collect per-city pooled mean / std for both rows and all measures
    cell_means = {}   # (row_label, ds, measure) -> mean
    cell_stds = {}    # (row_label, ds, measure) -> std (TSTR only)
    floors = {}       # (row_label, ds, measure) -> pooled floor or None
    for row_label, pairs, row_datasets in rows:
        for ds in row_datasets:
            cells = compute_cell_values(pairs, ds)
            for m in measure_keys:
                vals = cells[m]
                cell_means[(row_label, ds, m)] = float(np.mean(vals))
                cell_stds[(row_label, ds, m)] = float(np.std(vals))
                floors[(row_label, ds, m)] = pooled_floor(m, ds, pairs)

    # Per-(dataset, measure) bold/underline across the two rows
    best_means = {}
    second_means = {}
    saturated_sets = {}
    for ds in datasets:
        for m in measure_keys:
            row_vals = []
            for row_label, _, row_datasets in rows:
                if ds in row_datasets:
                    row_vals.append(round(cell_means[(row_label, ds, m)], decimals))
            best, second = best_and_second(row_vals, maximize=(m in _MAXIMIZED))
            best_means[(ds, m)] = best
            second_means[(ds, m)] = second
            # Use the 100% row's floor for saturation detection (the floor depends on the
            # filters; both rows use the same four, so the pooled floor is the same for both).
            ref_row = '100\\%'
            saturated_sets[(ds, m)] = saturated_values(
                row_vals, floors.get((ref_row, ds, m)), m, decimals=decimals,
            )

    # Print latex-transpose, matching generate_tables.py --table 2 layout
    print()
    print()
    table_name_print = '% Generalization from 25% of the city'
    print(table_name_print)
    print('%' * len(table_name_print))

    for measure_idx, m in enumerate(measure_keys):
        for i, (row_label, _, row_datasets) in enumerate(rows):
            cells_str = []
            for ds in datasets:
                mean = cell_means[(row_label, ds, m)]
                std = cell_stds[(row_label, ds, m)]
                rounded = round(mean, decimals)
                is_best = (best_means[(ds, m)] is not None
                           and rounded == best_means[(ds, m)])
                is_second = (not is_best
                             and second_means[(ds, m)] is not None
                             and rounded == second_means[(ds, m)])
                is_saturated = rounded in saturated_sets[(ds, m)]
                cells_str.append(format_value(m, mean, std, is_best, is_second, is_saturated))

            label_field = measure_nice_names[m]
            if i == 0:
                first = r'\multirow{' + str(len(rows)) + '}*{' + label_field + '}'
            else:
                first = ' ' * (len(label_field) + 15)
            print(f"{first} & {row_label} & " + " & ".join(cells_str) + r'\\')
        if measure_idx < len(measure_keys) - 1:
            print(r'\addlinespace[0.5em]')


if __name__ == '__main__':
    main()

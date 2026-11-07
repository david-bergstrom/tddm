import pickle
import sys
from config import nice_names, tables, table_names
import numpy as np
import argparse

from floor_bolding import best_and_second, floor_for_summary, saturated_values

model_name_overrides = {
    'Generalization from 25% of the city': {
        'tddpm': r'100\%'
    }
}

datasets = ['geolife-100',
            'porto-100',
            'cabspotting-100',
            'brest-100']

seq_len = 128
decimals = 3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate summary tables')
parser.add_argument('--table', type=int, default=None,
                    help='Generate only the specified table index (0-based)')
parser.add_argument('--markdown', action='store_true',
                    help='Output markdown tables instead of LaTeX')
parser.add_argument('--debug', action='store_true',
                    help='Print debug information about averaging')
parser.add_argument('--seed-mean', action='store_true',
                    help='Report each cell as the mean ± SEM over the training runs of its model (see seed_mean.py)')
parser.add_argument('--pending-ok', action='store_true',
                    help='With --seed-mean: cells whose seed runs are not all evaluated use the runs that exist and are '
                         r'wrapped in \\pending{} (red box) instead of failing')
args = parser.parse_args()

# Read the pickle-files created by `run_evaluation.py`
try:
    # TSTR in units of 1e-3
    transformer_tstrs = {k: v * 1e3 for k, v in
                         pickle.load(open("transformer_tstr_scaled_diff_measures.pickle", "rb")).items()}
except FileNotFoundError:
    transformer_tstrs = {}

try:
    kls = pickle.load(open("kl_measures.pickle", "rb"))
except FileNotFoundError as e:
    print(e)
    kls = {}

try:
    difftraj_measures = pickle.load(open("difftraj_measures.pickle", "rb"))
except FileNotFoundError:
    difftraj_measures = {}

try:
    od_measures = pickle.load(open("od_measures.pickle", "rb"))
except FileNotFoundError:
    od_measures = {}

try:
    speed_kls = pickle.load(open("speed_kl_measures_v2.pickle", "rb"))
except FileNotFoundError:
    speed_kls = {}

# Length (length_error_v2.py) goes in index 2 of the DiffTraj tuple; a cell without a Length value shows nan
try:
    length_v2 = pickle.load(open("length_measures_v2.pickle", "rb"))
except FileNotFoundError:
    length_v2 = {}
difftraj_measures = {k: (v[0], v[1], length_v2.get(k, float('nan')), v[3]) for k, v in difftraj_measures.items()}

if args.seed_mean:
    from seed_mean import average_over_seeds, run_values, sem, PENDING
    raw_results = dict(transformer_tstr=transformer_tstrs, kl=kls, difftraj=difftraj_measures,
                       od=od_measures, speed_kl=speed_kls)
    transformer_tstrs = average_over_seeds(transformer_tstrs, 2)
    kls = average_over_seeds(kls, 2)
    difftraj_measures = average_over_seeds(difftraj_measures, 2)
    od_measures = average_over_seeds(od_measures, 2)
    speed_kls = average_over_seeds(speed_kls, 1)

measure_keys =  ['transformer_tstr', 'kl_synth_real', 'kl_real_synth',
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
    'top_n_score': 'Pattern (↑)'
}

for i, (models, table_name) in enumerate(zip(tables, table_names)):
    # Skip tables that don't match the filter
    if args.table is not None and i != args.table:
        continue

    # Per-table dataset overrides: map matching does not apply to Brest (maritime AIS)
    if table_name == 'No Map Matching':
        current_datasets = [d for d in datasets if d != 'brest-100']
    else:
        current_datasets = datasets

    res = {}

    # DEBUG: Print table info
    if args.debug and table_name == 'No Map Matching':
        print(f"\n{'='*60}")
        print(f"DEBUG: Processing table '{table_name}'")
        print(f"Models in this table: {[m for m, _ in models]}")
        print(f"Datasets to average across: {current_datasets}")
        print(f"{'='*60}")

    for model, filter_out in models:
        for dataset in current_datasets:
            key = (dataset, seq_len, model, filter_out)

            source_dataset_in_model_name = None
            if table_name == 'City-to-city Generalization':
                # Extract source dataset from model name (e.g., 'geolife-100' from 'tddpm-generalize-from-geolife-100')
                # We assume the model name format is consistent as defined in config.py
                prefix = 'tddpm-generalize-from-'
                if model.startswith(prefix):
                    source_dataset_in_model_name = model[len(prefix):]

                if source_dataset_in_model_name == dataset:
                    # This is a diagonal entry, use the standard TDDPM result
                    key = (dataset, seq_len, 'tddpm', filter_out)
    
            if key in transformer_tstrs:
                # Store per-cell trajectory mean so summary statistics treat each
                # (dataset, model, filter) cell as one observation (cell-mean
                # aggregation), matching the other measures.
                res[(dataset, model, 'transformer_tstr')] = transformer_tstrs[key].mean().item(),
            else:
                print(f"WARNING: missing transformer_tstr key {key}", file=sys.stderr)
            if key in kls:
                kl_synth_real, kl_real_synth, kl_symmetric, js = kls[key]
                res[(dataset, model, 'kl_synth_real')] = kl_synth_real,
                res[(dataset, model, 'kl_real_synth')] = kl_real_synth,
                res[(dataset, model, 'kl_symmetric')] = kl_symmetric,
                res[(dataset, model, 'js')] = js,
            else:
                print(f"WARNING: missing kl key {key}", file=sys.stderr)
            if key in difftraj_measures:
                de, te, le, top_n_score = difftraj_measures[key]
                res[(dataset, model, 'density_error')] = de,
                res[(dataset, model, 'trip_error')] = te,
                res[(dataset, model, 'length_error')] = le,
                res[(dataset, model, 'top_n_score')] = top_n_score,
            else:
                print(f"WARNING: missing difftraj key {key}", file=sys.stderr)
            if key in od_measures:
                res[(dataset, model, 'od_error')] = od_measures[key],
            else:
                print(f"WARNING: missing od key {key}", file=sys.stderr)

            # Speed KL uses 3-tuple keys (no seq_len)
            speed_kl_key = (key[0], key[2], key[3])  # (dataset, model, filter_out)
            if speed_kl_key in speed_kls:
                kl_sr, kl_rs, kl_sym, js = speed_kls[speed_kl_key]
                res[(dataset, model, 'speed_kl')] = kl_sym,
            else:
                print(f"WARNING: missing speed_kl key {speed_kl_key}", file=sys.stderr)

    filters = [filter_out for _, filter_out in models]
    filters = list(set(filters))
    assert len(filters) == 1

    summary_res = {}
    for model, filter_out in models:
        for measure_key in measure_keys:
            observations = []
            for dataset in current_datasets:
                key = (dataset, model, measure_key)

                # For city-to-city generalization, we skip the model trained on the target dataset
                prefix = 'tddpm-generalize-from-'
                if model.startswith(prefix):
                    source_dataset_in_model_name = model[len(prefix):]
                    if source_dataset_in_model_name == dataset:
                        # This is a diagonal entry, skip this entry
                        continue

                if key in res:
                    observations.append(res[key])

            if len(observations) == 0:
                continue

            # DEBUG: Print what we're averaging for "No Map Matching" table
            if args.debug and table_name == 'No Map Matching' and len(observations) > 0:
                print(f"\nDEBUG [{table_name}] Model: {model}, Measure: {measure_key}")
                print(f"  Number of observations (cities): {len(observations)}")
                print(f"  Datasets included: {current_datasets}")
                print(f"  Observations: {observations}")

            mean = np.mean(observations)
            rounded = np.round(mean, decimals=decimals)
            summary_res[(model, measure_key)] = rounded,
            if args.seed_mean:
                # Mean ± SEM over the training runs, each run averaged over the cities first
                # City-to-city models leave out their source city, as the plain mean above does
                run_datasets = [d for d in current_datasets if model != f'tddpm-generalize-from-{d}']
                runs = run_values(raw_results, measure_key, run_datasets, model, filter_out,
                                  pending_ok=args.pending_ok)
                if runs is not None:
                    summary_res[(model, measure_key)] = ((np.round(np.mean(runs), decimals=decimals), sem(runs))
                                                         if len(runs) > 1 else (np.round(np.mean(runs), decimals=decimals),))
            if args.debug and table_name == 'No Map Matching':
                print(f"  Computed mean: {rounded}")
    
    best_means = {}
    second_best_means = {}
    saturated_means = {}

    for measure in measure_keys:
        measure_values = []
        for model, filter_out in models:
            key = (model, measure)
            if key in summary_res:
                mean = round(float(summary_res[key][0]), decimals)
                measure_values.append(mean)

        if measure_values:
            maximize = (measure == 'top_n_score')
            best, second = best_and_second(measure_values, maximize)
            best_means[measure] = best
            second_best_means[measure] = second
        else:
            best_means[measure] = None
            second_best_means[measure] = None

        floor = floor_for_summary(
            measure, list(current_datasets), seq_len, filters[0],
            kls, difftraj_measures, od_measures, transformer_tstrs, speed_kls,
        )
        saturated_means[measure] = saturated_values(
            measure_values, floor, measure, decimals=decimals,
        )

    print()
    if i > 0:
        print()
        print()

    model_names = []
    for model, _ in models:
        if table_name in model_name_overrides and model in model_name_overrides[table_name]:
            model_names.append(model_name_overrides[table_name][model])
        else:
            model_names.append(nice_names[model])

    if args.markdown:
        # Markdown format
        print(f'## {table_name}')
        print()
        print('| Metric | ' + ' | '.join(model_names) + ' |')
        print('|' + '---|' * (len(model_names) + 1))
    else:
        # LaTeX format
        table_name_print = f'% {table_name}'
        print(table_name_print)
        print('%' * len(table_name_print))
        print(f'Measure & ' + ' & '.join(model_names) + r'\\')
        print(r'\midrule')
        print()

    for measure_idx, measure in enumerate(measure_keys):
        measure_strs = []
        best_mean = best_means[measure]
        second_best_mean = second_best_means[measure]
        saturated = saturated_means[measure]

        for model, filter_out in models:
            key = (model, measure)

            if key in summary_res:
                r = summary_res[key]
                if len(r) == 2:
                    mean, std = r
                    if args.markdown:
                        measure_str = f'{mean:.{decimals}f} ± {std:.{decimals}f}'
                    else:
                        measure_str = f'{mean:.{decimals}f}' + r' \pm '  + f'{std:.{decimals}f}'
                else:
                    mean = r[0]
                    measure_str = f'{mean:.{decimals}f}'

                rounded = round(float(mean), decimals)
                if not args.markdown:
                    if rounded == best_mean:
                        measure_str = r'\mathbf{' + measure_str + '}'
                    elif rounded == second_best_mean:
                        measure_str = r'\underline{' + measure_str + '}'
                    if rounded in saturated:
                        measure_str = measure_str + r'^\dagger'
                    measure_str = '$' + measure_str + '$'
                    if args.seed_mean and args.pending_ok and (tuple(current_datasets), model, filter_out, measure) in PENDING:
                        measure_str = r'\pending{' + measure_str + '}'
                else:
                    if rounded == best_mean:
                        measure_str = f'**{measure_str}**'
                    elif rounded == second_best_mean:
                        measure_str = f'_{measure_str}_'
                    if rounded in saturated:
                        measure_str = measure_str + '†'
            else:
                measure_str = '-'
            measure_strs.append(measure_str)

        if args.markdown:
            # Clean up measure name for markdown (remove LaTeX and arrows)
            measure_name = measure_nice_names[measure]
            measure_name = measure_name.replace(r'$\mathrm{KL}(S \parallel R)$', r'KL(S\|\|R)')
            measure_name = measure_name.replace(r'$\mathrm{KL}(R \parallel S)$', r'KL(R\|\|S)')
            measure_name = measure_name.replace(r'$\mathrm{KL}_\mathrm{sym}$', 'KL_sym')
            measure_name = measure_name.replace(r'$\mathrm{KL}_\mathrm{speed}$', 'KL_speed')
            measure_name = measure_name.replace('(↓)', '↓')
            measure_name = measure_name.replace('(↑)', '↑')
            row = f"| {measure_name} | " + " | ".join(measure_strs) + " |"
        else:
            row = f"{measure_nice_names[measure]} & " + " & ".join(measure_strs) + r'\\'

        print(row)
        if not args.markdown and measure_idx < len(measure_keys) - 1:
            print(r'\addlinespace[0.5em]')

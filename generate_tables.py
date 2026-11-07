import pickle
import argparse
from config import nice_names, tables, table_names
import numpy as np

from floor_bolding import best_and_second, floor_for_cell, saturated_values

# Model name overrides for specific tables
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate evaluation tables')
parser.add_argument('--table', type=int, default=None,
                    help='Generate only the specified table index (0-based)')
parser.add_argument('--mode', type=str, default='latex-transpose',
                    choices=['markdown', 'latex', 'latex-transpose'],
                    help='Output format')
parser.add_argument('--no-bold', action='store_true',
                    help='Do not mark best/second-best (dagger marks are kept)')
parser.add_argument('--seed-mean', action='store_true',
                    help='Report each cell as the mean ± SEM over the training runs of its model (see seed_mean.py)')
parser.add_argument('--pending-ok', action='store_true',
                    help='With --seed-mean: cells whose seed runs are not all evaluated use the runs that exist and are '
                         r'wrapped in \\pending{} (red box) instead of failing')
args = parser.parse_args()

mode = args.mode

# Read the pickle-files created by `run_evaluation.py`
try:
    kls = pickle.load(open("kl_measures.pickle", "rb"))
except FileNotFoundError:
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
    # TSTR in units of 1e-3
    transformer_tstrs = {k: v * 1e3 for k, v in
                         pickle.load(open("transformer_tstr_scaled_diff_measures.pickle", "rb")).items()}
except FileNotFoundError:
    transformer_tstrs = {}

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
    'top_n_score': 'Pattern (↑)',
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
    
            if key in kls:
                kl_synth_real, kl_real_synth, kl_symmetric, js = kls[key]
                res[(dataset, model, 'kl_synth_real')] = kl_synth_real,
                res[(dataset, model, 'kl_real_synth')] = kl_real_synth,
                res[(dataset, model, 'kl_symmetric')] = kl_symmetric,
                res[(dataset, model, 'js')] = js,
            if key in difftraj_measures:
                de, te, le, top_n_score = difftraj_measures[key]
                res[(dataset, model, 'density_error')] = de,
                res[(dataset, model, 'trip_error')] = te,
                res[(dataset, model, 'length_error')] = le,
                res[(dataset, model, 'top_n_score')] = top_n_score,
            if key in od_measures:
                res[(dataset, model, 'od_error')] = od_measures[key],
            if key in transformer_tstrs:
                res[(dataset, model, 'transformer_tstr')] = transformer_tstrs[key].mean().item(),

            # Speed KL uses 3-tuple keys (no seq_len)
            # At this point, 'key' may have been modified for city-to-city diagonal entries
            # Extract the model from the modified key to handle diagonal entries correctly
            speed_kl_key = (key[0], key[2], key[3])  # (dataset, model, filter_out)
            if speed_kl_key in speed_kls:
                kl_sr, kl_rs, kl_sym, js = speed_kls[speed_kl_key]
                res[(dataset, model, 'speed_kl')] = kl_sym,

    if args.seed_mean:
        # Mean ± SEM over the training runs, per city
        for model, filter_out in models:
            for dataset in current_datasets:
                # City-to-city diagonal cells use the in-city TDDM runs, as in the key mapping above
                run_model = model
                if table_name == 'City-to-city Generalization' and model == f'tddpm-generalize-from-{dataset}':
                    run_model = 'tddpm'
                for measure in measure_keys:
                    runs = run_values(raw_results, measure, [dataset], run_model, filter_out, pending_ok=args.pending_ok)
                    if runs is not None and (dataset, model, measure) in res:
                        res[(dataset, model, measure)] = ((float(np.mean(runs)), sem(runs)) if len(runs) > 1
                                                          else (float(np.mean(runs)),))

    best_means = {}
    second_best_means = {}
    saturated_means = {}

    filters = [filter_out for _, filter_out in models]
    filters = list(set(filters))
    assert len(filters) == 1
    filter = filters[0]

    for dataset in current_datasets:
        for measure in measure_keys:
            measure_values = []
            for model, filter_out in models:
                key = (dataset, model, measure)
                if key in res:
                    mean = res[key][0]

                    mean = round(float(mean), decimals)
                    measure_values.append(mean)

            if measure_values:
                maximize = (measure == 'top_n_score')
                best, second = best_and_second(measure_values, maximize)
                best_means[(dataset, measure)] = best
                second_best_means[(dataset, measure)] = second
            else:
                best_means[(dataset, measure)] = None
                second_best_means[(dataset, measure)] = None

            floor = floor_for_cell(
                measure, dataset, seq_len, filter,
                kls, difftraj_measures, od_measures, transformer_tstrs, speed_kls,
            )
            saturated_means[(dataset, measure)] = saturated_values(
                measure_values, floor, measure, decimals=decimals,
            )

    if mode == 'markdown':
        # Print table header with section title
        print()
        if i > 0:
            print()
            print()
        print(f'## {table_name}')
        print()

        # Get model names for this table
        model_names = []
        for model, _ in models:
            if table_name in model_name_overrides and model in model_name_overrides[table_name]:
                model_names.append(model_name_overrides[table_name][model])
            else:
                model_names.append(nice_names[model])

        # Print per-dataset tables in transposed format
        for dataset in current_datasets:
            print(f'### {nice_names[dataset]}')
            print()
            print('| Metric | ' + ' | '.join(model_names) + ' |')
            print('|' + '---|' * (len(model_names) + 1))

            for measure in measure_keys:
                measure_strs = []
                best_mean = best_means[(dataset, measure)]
                second_best_mean = second_best_means[(dataset, measure)]
                saturated = saturated_means[(dataset, measure)]

                for model, filter_out in models:
                    key = (dataset, model, measure)

                    if key in res:
                        r = res[key]
                        if len(r) == 2:
                            mean, sem_ = r
                            measure_str = f'{mean:.{decimals}f} ± {sem_:.{decimals}f}'
                        else:
                            mean, = r
                            measure_str = f'{mean:.{decimals}f}'

                        rounded_mean = round(float(mean), decimals)
                        if rounded_mean == best_mean:
                            measure_str = f'**{measure_str}**'
                        elif rounded_mean == second_best_mean:
                            measure_str = f'_{measure_str}_'
                        if rounded_mean in saturated:
                            measure_str = measure_str + '†'
                    else:
                        measure_str = '-'
                    measure_strs.append(measure_str)

                # Clean up measure name for markdown (remove LaTeX and escape ||)
                measure_name = measure_nice_names[measure]
                measure_name = measure_name.replace(r'$\mathrm{KL}(S \parallel R)$', r'KL(S\|\|R)')
                measure_name = measure_name.replace(r'$\mathrm{KL}(R \parallel S)$', r'KL(R\|\|S)')
                measure_name = measure_name.replace(r'$\mathrm{KL}_\mathrm{sym}$', 'KL_sym')
                measure_name = measure_name.replace(r'$\mathrm{KL}_\mathrm{speed}$', 'KL_speed')
                measure_name = measure_name.replace('(↓)', '↓')
                measure_name = measure_name.replace('(↑)', '↑')
                row = f"| {measure_name} | " + " | ".join(measure_strs) + " |"
                print(row)
            print()
    elif mode == 'latex':
        # print('Dataset & Model & TimeFID & TSTR & KL(synth, real) & KL(real, synth) & KL symmetric & JS')

        for dataset in current_datasets:
            for i, (model, filter_out) in enumerate(models):
                measure_strs = []
                for measure in measure_keys:
                    key = (dataset, model, measure)
                    best_mean = best_means[(dataset, measure)]
                    second_best_mean = second_best_means[(dataset, measure)]
                    saturated = saturated_means[(dataset, measure)]

                    if key in res:
                        r = res[key]
                        if len(r) == 2:
                            mean, std = r
                            measure_str = f'{mean:.{decimals}f} \\pm {std:.{decimals}f}'
                        elif len(r) == 1:
                            mean, = r
                            measure_str = f'{mean:.{decimals}f}'

                        rounded_mean = round(float(mean), decimals)
                        if args.no_bold:
                            pass
                        elif rounded_mean == best_mean:
                            measure_str = r'\mathbf{' + measure_str + '}'
                        elif rounded_mean == second_best_mean:
                            measure_str = r'\underline{' + measure_str + '}'
                        if rounded_mean in saturated:
                            measure_str = measure_str + r'^\dagger'
                        measure_str = '$' + measure_str + '$'
                    else:
                        measure_str = '-'
                    measure_strs.append(measure_str)
    
                if i == 0:
                    dataset_str = r'\multirow{' + str(len(models)) + '}*{' + nice_names[dataset] + '}'
                else:
                    dataset_str = ' ' * (len(nice_names[dataset]) + 15)
                
                # Use model name override if available
                if table_name in model_name_overrides and model in model_name_overrides[table_name]:
                    model_name = model_name_overrides[table_name][model]
                else:
                    model_name = nice_names[model]
                
                row = f"{dataset_str} & {model_name} & " + " & ".join(measure_strs) + r'\\'
                print(row)
            print(r'\addlinespace[0.5em]')
    elif mode == 'latex-transpose':
        print()
        if i > 0:
            print()
            print()
        table_name_print = f'% {table_name}'
        print(table_name_print)
        print('%' * len(table_name_print))
        for measure_idx, measure in enumerate(measure_keys):
            for i, (model, filter_out) in enumerate(models):
                measure_strs = []
                for dataset in current_datasets:
                    key = (dataset, model, measure)
                    best_mean = best_means[(dataset, measure)]
                    second_best_mean = second_best_means[(dataset, measure)]
                    saturated = saturated_means[(dataset, measure)]

                    if key in res:
                        r = res[key]
                        if len(r) == 2:
                            mean, sem_ = r
                            measure_str = f'{mean:.{decimals}f} \\pm {sem_:.{decimals}f}'
                        else:
                            mean, = r
                            measure_str = f'{mean:.{decimals}f}'

                        rounded_mean = round(float(mean), decimals)
                        if args.no_bold:
                            pass
                        elif rounded_mean == best_mean:
                            measure_str = r'\mathbf{' + measure_str + '}'
                        elif rounded_mean == second_best_mean:
                            measure_str = r'\underline{' + measure_str + '}'
                        if rounded_mean in saturated:
                            measure_str = measure_str + r'^\dagger'
                        measure_str = '$' + measure_str + '$'
                        if args.seed_mean and args.pending_ok and ((dataset,), model, filter_out, measure) in PENDING:
                            measure_str = r'\pending{' + measure_str + '}'
                    else:
                        measure_str = '-'
                    measure_strs.append(measure_str)
    
                if i == 0:
                    dataset_str = r'\multirow{' + str(len(models)) + '}*{' + measure_nice_names[measure] + '}'
                else:
                    dataset_str = ' ' * (len(measure_nice_names[measure]) + 15)
    
                # Use model name override if available
                if table_name in model_name_overrides and model in model_name_overrides[table_name]:
                    model_name = model_name_overrides[table_name][model]
                else:
                    model_name = nice_names[model]
                
                row = f"{dataset_str} & {model_name} & " + " & ".join(measure_strs) + r'\\'
                print(row)
            if measure_idx < len(measure_keys) - 1:
                print(r'\addlinespace[0.5em]')



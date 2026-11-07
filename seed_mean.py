"""Replace each model's cached result with the mean over its training runs.

The paper's tables report means over three training runs: the run registered
under the plain model name and the seed runs registered as '<model>-seed42' and
'<model>-seed1'. Models in RUNS list their runs explicitly: 'timegan-forward'
(seeds 42, 1, 2), and 'tddpm-no-conditional-same-model', whose seed runs are the
'tddpm-no-conditional' seeds. The table generators read one cached result per
(dataset, model); `average_over_seeds` rewrites a results dict so that the plain
model name holds the mean over the three runs wherever all three exist. Cells
without seed runs are left unchanged.
"""
import numpy as np
import torch

SEEDS = ('42', '1')
# Models whose runs don't follow '<model>', '<model>-seed42', '<model>-seed1': the run names, explicitly.
# The mean is stored under the model name; runs[0] is the key that triggers it.
RUNS = {
    'timegan-forward': ('timegan-forward-seed42', 'timegan-forward-seed1', 'timegan-forward-seed2'),
    'tddpm-no-conditional-same-model': ('tddpm-no-conditional-same-model', 'tddpm-no-conditional-seed42',
                                        'tddpm-no-conditional-seed1'),
}


def _runs(model):
    if model in RUNS:
        return list(RUNS[model])
    return [model] + [f'{model}-seed{s}' for s in SEEDS]


def _mean(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        # TSTR stores per-trajectory losses; the tables use their mean.
        return torch.tensor(float(np.mean([v.mean().item() for v in values])))
    if isinstance(first, (tuple, list)):
        return tuple(float(np.mean([v[i] for v in values])) for i in range(len(first)))
    return float(np.mean(values))


def average_over_seeds(results, model_index):
    """`model_index` is the position of the model name in the keys: 2 for
    (dataset, seq_len, model, filter_out), 1 for the speed-KL keys
    (dataset, model, filter_out)."""
    out = dict(results)
    for key, value in results.items():
        if not isinstance(key, tuple) or len(key) <= model_index:
            continue
        model = key[model_index]
        if not isinstance(model, str) or '-seed' in model:
            continue
        seed_keys = [key[:model_index] + (f'{model}-seed{s}',) + key[model_index + 1:] for s in SEEDS]
        present = [k in results for k in seed_keys]
        if not any(present):
            continue  # single-run model (e.g. ablations)
        assert all(present), f'{key}: only some seed runs are evaluated: ' + \
            ', '.join(k[model_index] for k, p in zip(seed_keys, present) if not p) + ' missing'
        out[key] = _mean([value] + [results[k] for k in seed_keys])
    for model, runs in RUNS.items():
        for key in results:
            if not isinstance(key, tuple) or len(key) <= model_index or key[model_index] != runs[0]:
                continue
            run_keys = [key[:model_index] + (run,) + key[model_index + 1:] for run in runs]
            missing = [k[model_index] for k in run_keys if k not in results]
            assert not missing, f'{key}: only some seed runs are evaluated: ' + ', '.join(missing) + ' missing'
            out[key[:model_index] + (model,) + key[model_index + 1:]] = _mean([results[k] for k in run_keys])
    return out


_KL = ('kl_synth_real', 'kl_real_synth', 'kl_symmetric', 'js')
_DIFFTRAJ = ('density_error', 'trip_error', 'length_error', 'top_n_score')


def _scalar(raw, measure, dataset, model, filter_out, seq_len=128):
    key = (dataset, seq_len, model, filter_out)
    if measure == 'transformer_tstr':
        return float(raw['transformer_tstr'][key].mean().item())
    if measure in _KL:
        return float(raw['kl'][key][_KL.index(measure)])
    if measure in _DIFFTRAJ:
        return float(raw['difftraj'][key][_DIFFTRAJ.index(measure)])
    if measure == 'od_error':
        return float(raw['od'][key])
    if measure == 'speed_kl':
        return float(raw['speed_kl'][(dataset, model, filter_out)][2])
    raise KeyError(measure)


# Cells whose runs are incomplete, as (datasets, model, filter_out, measure); filled by run_values(pending_ok=True)
PENDING = set()


def run_values(raw, measure, datasets, model, filter_out, pending_ok=False):
    """Per-run values of one measure, each averaged over `datasets` (one value per
    training run), or None for a single-run model. `raw` holds the results dicts
    before `average_over_seeds`, keyed 'transformer_tstr', 'kl', 'difftraj', 'od',
    'speed_kl'. With pending_ok, a model that has seed runs elsewhere but not (all)
    here returns the runs that exist and the cell is added to PENDING, instead of
    being treated as single-run or raising KeyError."""
    runs = _runs(model)
    probe = (datasets[0], 128, runs[1], filter_out)
    if probe not in raw['kl']:
        if not (pending_ok and any(k[2] == runs[1] and k[3] == filter_out for k in raw['kl'])):
            return None
    if not pending_ok:
        return [float(np.mean([_scalar(raw, measure, d, r, filter_out) for d in datasets])) for r in runs]
    values = []
    for r in runs:
        try:
            values.append(float(np.mean([_scalar(raw, measure, d, r, filter_out) for d in datasets])))
        except KeyError:
            PENDING.add((tuple(datasets), model, filter_out, measure))
    return values


def sem(values):
    """Standard error of the mean over the training runs (sample std / sqrt(n))."""
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))

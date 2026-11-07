"""Real-vs-real noise floor lookup and saturation tagging for evaluation tables.

Each metric has a real-vs-real noise floor (see noise_floors.py): comparing
two disjoint halves of the real data gives a positive divergence due to
finite-sample noise. A model whose synthetic-vs-real value is at or below
that floor is statistically indistinguishable from a perfect generator on
that metric -- it has saturated the measure.

Bolding and underlining keep their classic "best / second-best" meaning;
saturation is rendered separately as a dagger marker on the affected
cells, so the two pieces of information do not compete in the same visual
channel.
"""
from typing import Iterable, Optional, Set, Tuple

from noise_floors import NOISE_FLOOR_KEY


_MAXIMIZED = {'top_n_score'}

_KL_FIELDS = ('kl_synth_real', 'kl_real_synth', 'kl_symmetric', 'js')
_DIFFTRAJ_FIELDS = ('density_error', 'trip_error', 'length_error', 'top_n_score')


def floor_for_cell(measure, dataset, seq_len, filter_out,
                   kls, difftraj, od, tstr, speed_kls) -> Optional[float]:
    """Return the floor for one (dataset, measure, filter) cell, or None
    if no floor was computed for this metric, dataset and filter.
    Speed-KL floors are keyed (dataset, NOISE_FLOOR_KEY, filter), like the
    speed-KL results."""
    floor_key = (dataset, seq_len, NOISE_FLOOR_KEY, filter_out)
    if measure == 'speed_kl':
        v = speed_kls.get((dataset, NOISE_FLOOR_KEY, filter_out))
        return float(v[2]) if v is not None else None  # symmetric speed KL
    if measure == 'transformer_tstr':
        v = tstr.get(floor_key)
        return float(v.mean()) if v is not None else None
    if measure in _KL_FIELDS:
        v = kls.get(floor_key)
        if v is None:
            return None
        return float(v[_KL_FIELDS.index(measure)])
    if measure in _DIFFTRAJ_FIELDS:
        v = difftraj.get(floor_key)
        if v is None:
            return None
        return float(v[_DIFFTRAJ_FIELDS.index(measure)])
    if measure == 'od_error':
        v = od.get(floor_key)
        return float(v) if v is not None else None
    return None


def floor_for_summary(measure, datasets, seq_len, filter_out,
                      kls, difftraj, od, tstr, speed_kls) -> Optional[float]:
    """Floor for a summary row that averages across `datasets`. Computed as
    the mean of per-dataset floors; returns None if any dataset is missing
    a floor for this metric."""
    vals = []
    for ds in datasets:
        f = floor_for_cell(measure, ds, seq_len, filter_out, kls, difftraj, od, tstr, speed_kls)
        if f is None:
            return None
        vals.append(f)
    if not vals:
        return None
    return sum(vals) / len(vals)


def best_and_second(values: Iterable[float], maximize: bool) -> Tuple[Optional[float], Optional[float]]:
    """Best and second-best distinct values. (Used for bold / underline.)"""
    uniq = sorted(set(values), reverse=maximize)
    best = uniq[0] if uniq else None
    second = uniq[1] if len(uniq) > 1 else None
    return best, second


def saturated_values(values: Iterable[float], floor: Optional[float],
                     measure: str, decimals: int = 3) -> Set[float]:
    """Subset of `values` (rounded to `decimals`) that are at-or-below the
    noise floor (at-or-above for maximised metrics). Empty if no floor."""
    if floor is None:
        return set()
    f = round(float(floor), decimals)
    maximize = measure in _MAXIMIZED
    out = set()
    for v in values:
        if v is None:
            continue
        rv = round(float(v), decimals)
        if (maximize and rv >= f) or (not maximize and rv <= f):
            out.add(rv)
    return out

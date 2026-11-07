"""
Real-vs-real noise floors for Length (length_error_v2.py), Speed KL and the scaled-difference TSTR, for
every dataset and filter. Writes them under the NOISE_FLOOR_KEY model name into the pickles of those
measures, with the key formats the table generators look up:
  length_measures_v2.pickle                    (dataset, seq_len, KEY, filter) -> length_v2
  speed_kl_measures_v2.pickle                  (dataset, KEY, filter) -> (kl_sr, kl_rs, kl_sym, js)
  transformer_tstr_scaled_diff_measures.pickle (dataset, seq_len, KEY, filter) -> per-window MAEs
Only adds keys; existing keys are skipped. Run from the repository root, after speed_kl_divergence.py,
length_error_v2.py and tstr_scaled_diff.py (it adds to their pickles).
"""
import argparse
import os
import pickle
import time

from noise_floors import NOISE_FLOOR_KEY, compute_v2_floors, compute_tstr_scaled_diff_floor
from run_evaluation import load_train_data

DATASETS = ['geolife-100', 'porto-100', 'cabspotting-100', 'brest-100']
FILTERS = ['none', 'lower-left-quarter', 'lower-right-quarter', 'upper-left-quarter', 'upper-right-quarter']
SEQ_LEN = 128


def _save(obj, path):
    pickle.dump(obj, open(path + '.tmp', 'wb'))
    os.replace(path + '.tmp', path)


def main():
    parser = argparse.ArgumentParser(description='Noise floors for Length, Speed KL and TSTR')
    parser.add_argument('--measures', default='length,speed,tstr', help='Comma-separated subset of length,speed,tstr')
    args = parser.parse_args()
    todo = set(args.measures.split(','))
    length = pickle.load(open('length_measures_v2.pickle', 'rb'))
    speed = pickle.load(open('speed_kl_measures_v2.pickle', 'rb'))
    tstr = pickle.load(open('transformer_tstr_scaled_diff_measures.pickle', 'rb'))
    for ds in DATASETS:
        for filt in FILTERS:
            lkey, skey = (ds, SEQ_LEN, NOISE_FLOOR_KEY, filt), (ds, NOISE_FLOOR_KEY, filt)
            need_cpu = ('length' in todo and lkey not in length) or ('speed' in todo and skey not in speed)
            need_tstr = 'tstr' in todo and lkey not in tstr
            if not (need_cpu or need_tstr):
                continue
            train_data, extent = load_train_data(ds, SEQ_LEN, filt)
            if need_cpu:
                length_v2, speed_v2 = compute_v2_floors(train_data, extent, SEQ_LEN)
                if 'length' in todo and lkey not in length:
                    length[lkey] = length_v2
                    _save(length, 'length_measures_v2.pickle')
                if 'speed' in todo and skey not in speed:
                    speed[skey] = speed_v2
                    _save(speed, 'speed_kl_measures_v2.pickle')
                print(f'{ds} {filt}: Length {length_v2:.4f}, Speed KL {speed_v2[2]:.4f}', flush=True)
            if need_tstr:
                t0 = time.time()
                losses = compute_tstr_scaled_diff_floor(train_data, SEQ_LEN)
                tstr[lkey] = losses
                _save(tstr, 'transformer_tstr_scaled_diff_measures.pickle')
                print(f'{ds} {filt}: TSTR {losses.mean().item() * 1e3:.4f}e-3 ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()

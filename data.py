import numpy as np


def rolling_window(data, seq_len):
    """
    Input: (N, T, D) where N is the observed sequences, T is the sequence length and D is the number of features
    Output: 
    """
    rolled_data = []
    for sequence in data:
        for i in range(0, len(sequence) - seq_len + 1):
            rolled_data.append(sequence[i:i+seq_len])

    rolled_data = np.stack(rolled_data, axis=0)
    return rolled_data


def minmaxscale(data, min_v=0, max_v=1, return_bounds=False):
    if isinstance(data, list):
        """
        (N, T, D), where T might constant or not
        """
        d_min = np.min([np.min(a, axis=0) for a in data], axis=0)
        d_max = np.max([np.max(a, axis=0) for a in data], axis=0)

        scaled = []

        for arr in data:
            scaled.append(((arr - d_min) / (d_max - d_min)) * (max_v - min_v) + min_v)

        return (scaled, d_min, d_max) if return_bounds else scaled
    elif isinstance(data, np.ndarray) and len(data.shape) == 2:
        """
        (N, T, D), where T might be constant
        """
        data = np.expand_dims(data, 0)
        d_min = np.min(data, axis=(0, 1), keepdims=True)
        d_max = np.max(data, axis=(0, 1), keepdims=True)
        scaled = ((data - d_min) / (d_max - d_min)) * (max_v - min_v) + min_v
        return (scaled, d_min, d_max) if return_bounds else scaled
    elif isinstance(data, np.ndarray) and len(data.shape) == 3:
        """
        (N, T, D), where T might be constant
        """
        d_min = np.min(data, axis=(0, 1), keepdims=True)
        d_max = np.max(data, axis=(0, 1), keepdims=True)
        scaled = ((data - d_min) / (d_max - d_min)) * (max_v - min_v) + min_v
        return (scaled, d_min, d_max) if return_bounds else scaled
    else:
        raise NotImplementedError("Could not figure out how to min-max scale the data")


def load_dataset_from_str(dataset_str, prefix):
    if dataset_str == 'geolife-25':
        data = load_geolife_25(prefix)
    elif dataset_str == 'geolife-100':
        data = load_geolife_100(prefix)
    elif dataset_str == 'porto-25':
        data = load_porto_25(prefix)
    elif dataset_str == 'porto-100':
        data = load_porto_100(prefix)
    elif dataset_str == 'cabspotting-25':
        data = load_cabspotting_25(prefix)
    elif dataset_str == 'cabspotting-100':
        data = load_cabspotting_100(prefix)
    elif dataset_str == 'geolife-100-no-map-match':
        data = load_geolife_100_no_map_match(prefix)
    elif dataset_str == 'porto-100-no-map-match':
        data = load_porto_100_no_map_match(prefix)
    elif dataset_str == 'cabspotting-100-no-map-match':
        data = load_cabspotting_100_no_map_match(prefix)
    # Corner variants for 4-corner generalization experiment
    elif dataset_str == 'geolife-25-bl':    data = load_geolife_25_bl(prefix)
    elif dataset_str == 'geolife-25-br':    data = load_geolife_25_br(prefix)
    elif dataset_str == 'geolife-25-tl':    data = load_geolife_25_tl(prefix)
    elif dataset_str == 'geolife-25-tr':    data = load_geolife_25_tr(prefix)
    elif dataset_str == 'porto-25-bl':      data = load_porto_25_bl(prefix)
    elif dataset_str == 'porto-25-br':      data = load_porto_25_br(prefix)
    elif dataset_str == 'porto-25-tl':      data = load_porto_25_tl(prefix)
    elif dataset_str == 'porto-25-tr':      data = load_porto_25_tr(prefix)
    elif dataset_str == 'cabspotting-25-bl': data = load_cabspotting_25_bl(prefix)
    elif dataset_str == 'cabspotting-25-br': data = load_cabspotting_25_br(prefix)
    elif dataset_str == 'cabspotting-25-tl': data = load_cabspotting_25_tl(prefix)
    elif dataset_str == 'cabspotting-25-tr': data = load_cabspotting_25_tr(prefix)
    elif dataset_str == 'brest-100':         data = load_brest_100(prefix)
    elif dataset_str == 'brest-25-bl':       data = _load_25_corner(prefix, 'brest', 'bl')
    elif dataset_str == 'brest-25-br':       data = _load_25_corner(prefix, 'brest', 'br')
    elif dataset_str == 'brest-25-tl':       data = _load_25_corner(prefix, 'brest', 'tl')
    elif dataset_str == 'brest-25-tr':       data = _load_25_corner(prefix, 'brest', 'tr')
    else:
        raise NotImplementedError("Dataset: " + str(dataset_str) + " is not a supported dataset.")

    return data

def load_geolife_25(prefix):
    return np.load(prefix + 'preprocessed/geolife-25-100000.npy').astype(np.float32)

def load_geolife_100(prefix):
    return np.load(prefix + 'preprocessed/geolife-100-100000.npy').astype(np.float32)

def load_porto_25(prefix):
    return np.load(prefix + 'preprocessed/porto-25-100000.npy').astype(np.float32)

def load_porto_100(prefix):
    return np.load(prefix + 'preprocessed/porto-100-100000.npy').astype(np.float32)

def load_cabspotting_25(prefix):
    return np.load(prefix + 'preprocessed/cabspotting-25-100000.npy').astype(np.float32)

def load_cabspotting_100(prefix):
    return np.load(prefix + 'preprocessed/cabspotting-100-100000.npy').astype(np.float32)

def load_geolife_100_no_map_match(prefix):
    return np.load(prefix + 'preprocessed/geolife-100-no-map-match-100000.npy').astype(np.float32)

def load_porto_100_no_map_match(prefix):
    return np.load(prefix + 'preprocessed/porto-100-no-map-match-100000.npy').astype(np.float32)

def load_cabspotting_100_no_map_match(prefix):
    return np.load(prefix + 'preprocessed/cabspotting-100-no-map-match-100000.npy').astype(np.float32)

def load_brest_100(prefix):
    return np.load(prefix + 'preprocessed/brest-100-100000.npy').astype(np.float32)


def _load_25_corner(prefix, dataset, corner):
    return np.load(prefix + f'preprocessed/{dataset}-25-{corner}-100000.npy').astype(np.float32)


# Corner variant loaders for 4-corner generalization experiment
def load_geolife_25_bl(prefix): return _load_25_corner(prefix, 'geolife', 'bl')
def load_geolife_25_br(prefix): return _load_25_corner(prefix, 'geolife', 'br')
def load_geolife_25_tl(prefix): return _load_25_corner(prefix, 'geolife', 'tl')
def load_geolife_25_tr(prefix): return _load_25_corner(prefix, 'geolife', 'tr')
def load_porto_25_bl(prefix): return _load_25_corner(prefix, 'porto', 'bl')
def load_porto_25_br(prefix): return _load_25_corner(prefix, 'porto', 'br')
def load_porto_25_tl(prefix): return _load_25_corner(prefix, 'porto', 'tl')
def load_porto_25_tr(prefix): return _load_25_corner(prefix, 'porto', 'tr')
def load_cabspotting_25_bl(prefix): return _load_25_corner(prefix, 'cabspotting', 'bl')
def load_cabspotting_25_br(prefix): return _load_25_corner(prefix, 'cabspotting', 'br')
def load_cabspotting_25_tl(prefix): return _load_25_corner(prefix, 'cabspotting', 'tl')
def load_cabspotting_25_tr(prefix): return _load_25_corner(prefix, 'cabspotting', 'tr')

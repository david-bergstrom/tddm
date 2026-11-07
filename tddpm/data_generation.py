import numpy as np


def region_occupancy(trajectories, res=64):
    all_concatenated = np.concatenate(trajectories)
    H, xedges, yedges = np.histogram2d(all_concatenated[:, 1], all_concatenated[:, 0], bins=(res, res), range=((0, 1), (0, 1)), density=True)
    H = H.T

    return H, xedges, yedges


def crop_signal_with_rotation(signal, region, min_len=24):
    origin = region[0:2]
    rotation = region[2]
    size = region[3:5]
    
    signal = signal - origin
    signal = signal / size
    rotation = rotation * np.pi / 180

    rotated_signal = np.zeros_like(signal)
    rotated_signal[:, 0] = np.cos(rotation) * signal[:, 0] + np.sin(-rotation) * signal[:, 1]
    rotated_signal[:, 1] = np.sin(rotation) * signal[:, 0] + np.cos(-rotation) * signal[:, 1]
    signal = rotated_signal
    
    parts = []
    last_outside_region = 0
    
    for i in range(len(signal)):
        lat = signal[i, 0]
        lon = signal[i, 1]
        if lat < 0 or lat > 1 \
            or lon < 0 or lon > 1:
            
            part = signal[last_outside_region+1:i]
            if len(part) >= min_len:
                parts.append(part)
            
            last_outside_region = i
        
    if last_outside_region != i:
        part = signal[last_outside_region+1:i]
        if len(part) >= min_len:
            parts.append(part)
                    
    return parts


def extract_signals(dataset, region, min_len=24):
    all_parts = []

    for traj in dataset:
        parts = crop_signal_with_rotation(traj, region)
        all_parts.extend(parts)
            
    return all_parts

def random_region(size, bounds, max_rotation=360):
    min_lat, max_lat, min_lon, max_lon = bounds
    
    region = np.zeros(5, dtype=np.float32)
    
    region[0] = np.random.uniform(min_lat, max_lat, 1)
    region[1] = np.random.uniform(min_lon, max_lon, 1)
    if max_rotation != 0:
        region[2] = np.random.uniform(0, max_rotation, 1)
    else:
        region[2] = 0
    region[3:5] = size
    
    return region


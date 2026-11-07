
from multiprocessing import Pool
try:
    from tddpm.crop import crop_signal_with_rotation as crop_signal_with_rotation_optim
except ImportError:
    from tddpm.data_generation import crop_signal_with_rotation as crop_signal_with_rotation_optim
from tddpm.data_generation import random_region
import math
from glob import glob
import os

import numpy as np
import pandas as pd
from config import RAW_DATA_PATH
import torch
import torch.nn.functional as F


LIMITS = {
    # Full city regions
    'geolife-100':  (39.9, 40.0, 116.3, 116.4),
    'porto-100':    (41.1, 41.2, -8.7, -8.6),
    'cabspotting-100': (37.70, 37.83, -122.52, -122.35),  # SF + Treasure Island + Golden Gate

    # Geolife 25% corners (midpoints: lat=39.95, lon=116.35)
    'geolife-25':    (39.9, 39.95, 116.3, 116.35),     # bottom-left (default)
    'geolife-25-bl': (39.9, 39.95, 116.3, 116.35),
    'geolife-25-br': (39.9, 39.95, 116.35, 116.4),
    'geolife-25-tl': (39.95, 40.0, 116.3, 116.35),
    'geolife-25-tr': (39.95, 40.0, 116.35, 116.4),

    # Porto 25% corners (midpoints: lat=41.15, lon=-8.65)
    'porto-25':    (41.1, 41.15, -8.7, -8.65),          # bottom-left (default)
    'porto-25-bl': (41.1, 41.15, -8.7, -8.65),
    'porto-25-br': (41.1, 41.15, -8.65, -8.6),
    'porto-25-tl': (41.15, 41.2, -8.7, -8.65),
    'porto-25-tr': (41.15, 41.2, -8.65, -8.6),

    # Cabspotting 25% corners (midpoints: lat=37.765, lon=-122.435)
    'cabspotting-25':    (37.70, 37.765, -122.52, -122.435),  # bottom-left (default)
    'cabspotting-25-bl': (37.70, 37.765, -122.52, -122.435),
    'cabspotting-25-br': (37.70, 37.765, -122.435, -122.35),
    'cabspotting-25-tl': (37.765, 37.83, -122.52, -122.435),
    'cabspotting-25-tr': (37.765, 37.83, -122.435, -122.35),

    # Brest maritime AIS (Port of Brest + Ushant TSS approach)
    'brest-100': (47.9, 48.5, -5.2, -4.3),

    # Brest 25% corners (midpoints: lat=48.2, lon=-4.75)
    'brest-25':    (47.9, 48.2, -5.2, -4.75),     # bottom-left (default)
    'brest-25-bl': (47.9, 48.2, -5.2, -4.75),
    'brest-25-br': (47.9, 48.2, -4.75, -4.3),
    'brest-25-tl': (48.2, 48.5, -5.2, -4.75),
    'brest-25-tr': (48.2, 48.5, -4.75, -4.3),
}


def get_limit(dataset_str):
    if dataset_str in LIMITS:
        return LIMITS[dataset_str]
    else:
        raise ValueError(f"Unknown dataset: {dataset_str}")


def haversine_distance(data):
    lon = np.radians(data.longitude)
    lat = np.radians(data.latitude)
    dlon = lon.diff()
    dlat = lat.diff()
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat) * np.cos(lat + dlat) * np.sin(dlon/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


def find_cuts(df, limits, max_dt, kmph_range, min_seq_len):
    min_kmph, max_kmph = kmph_range
    dfs = []

    dts = (df['date_time']-df['date_time'].shift())
    dts = dts / np.timedelta64(1, 's')
    
    min_lat = limits[0]
    max_lat = limits[1]
    min_long = limits[2]
    max_long = limits[3]
    
    distance_traveled = haversine_distance(df)
    
    mps = distance_traveled * 1000 / dts
    kmph = 3.6 * mps

    last_cut = 0

    for j in range(1, len(df)):
        latitude = df.latitude[j]
        longitude = df.longitude[j]
        dt = dts[j]

        if dt == 0 or not (min_long < longitude < max_long \
                and min_lat < latitude < max_lat \
                and dt < max_dt \
                and min_kmph < kmph[j] < max_kmph):

            cut = df[last_cut:j]
            if len(cut) >= min_seq_len:
                dfs.append(cut)

            last_cut = j + 1

    cut = df[last_cut:]
    if len(cut) >= min_seq_len:
        dfs.append(cut)

    return dfs


def to_numpy_with_time_elapsed(df):
    elapsed = (df['date_time'] - df['date_time'].iloc[0]) / np.timedelta64(1, 's')

    export = df[['latitude', 'longitude']].copy()
    export['time_elapsed'] = elapsed

    return export[['time_elapsed', 'latitude', 'longitude']].to_numpy(dtype=np.float32)


def interpolate_subtraj(traj_np, target_dt):
    """
    traj_np: Nx3, where [:, 0] is time, [:, 1] latitude and [:, 2] is longitude
    target_dt: number of seconds between observations in final trajectory

    returns: Nx2 where [:, 0] is latitude and [:, 0] latitude
    """

    time = traj_np[:, 0]
    start_t = time[0]
    stop_t = time[-1]
    target_time = np.arange(start_t, stop_t, target_dt)

    latitude = traj_np[:, 1]
    longitude = traj_np[:, 2]

    latitude_interp = np.interp(x=target_time, xp=time, fp=latitude)
    longitude_interp = np.interp(x=target_time, xp=time, fp=longitude)
    
    return np.stack([latitude_interp, longitude_interp]).T  # Nx2


def _read_porto_row(polyline, timestamp):
    trajectory = pd.DataFrame(eval(polyline), columns=['longitude', 'latitude'])
    trajectory['date_time'] = pd.date_range(start=timestamp, freq='15s', periods=len(trajectory))
    return trajectory


def read_porto_dfs():
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'porto/train.csv'))
    df['TIMESTAMP'] = pd.to_datetime(df.TIMESTAMP, unit='s')
    # Remove all trajectories with missing data
    df = df[~df.MISSING_DATA]
    df = df.reset_index(drop='true')
    traj_dfs = []

    with Pool(20) as pool:
        traj_dfs = pool.starmap(_read_porto_row, zip(df.POLYLINE, df.TIMESTAMP))
    
    return traj_dfs


def read_cabspotting_dfs():
    paths = glob(os.path.join(RAW_DATA_PATH, 'cabspottingdata/cabspottingdata/new_*.txt'))
    raw_dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path, sep=' ', names=['latitude', 'longitude', 'occupancy', 'epoch'])
        df['date_time'] = pd.to_datetime(df['epoch'],unit='s')
        df = df.drop(columns=['epoch', 'occupancy'])
        df = df.sort_values(by='date_time')
        df = df.reset_index(drop=True)
        raw_dfs.append(df)
    return raw_dfs

def read_plt_file_pandas(index_plt_files_tuple):
    fieldnames = ['latitude', 'longitude', '0', 'altitude', 'date_fractional', 'date', 'time']
    relevant_columns = ['latitude', 'longitude', 'altitude', 'date', 'time']

    i, plt_files = index_plt_files_tuple
    dfs = []
    for j, plt_file in enumerate(plt_files):
        df = pd.read_csv(plt_file, skiprows=6, names=fieldnames, usecols=relevant_columns, parse_dates=[['date', 'time']])
        # df['user_id'] = i
        # df['trajectory_id'] = j
        # Convert altitude from feet to meters
        df['altitude'] = df.altitude * 0.3048
        dfs.append(df)

    return dfs

def read_geolife_dfs():
    plt_files = []
    path = os.path.join(RAW_DATA_PATH, 'Geolife Trajectories 1.3/Data/')
    users = glob(os.path.join(path, "*"))

    for i, d in enumerate(users):
        user_plt_files = glob(os.path.join(d, "Trajectory", "*.plt"))
        plt_files.append(user_plt_files)

    with Pool(processes=20) as pool:
        dfs = pool.map(read_plt_file_pandas, enumerate(plt_files))

    dfs_flattened = []
    for user_dfs in dfs:
        dfs_flattened.extend(user_dfs)

    return dfs_flattened


def read_brest_dfs():
    """Read Brest/NARI AIS dataset and return list of per-vessel DataFrames.

    The raw CSV has no header. Columns (from NARI readme):
      0: timestamp (Unix epoch in milliseconds)
      1: MMSI (vessel identifier)
      2: longitude
      3: latitude
      4: speed (SOG in knots)
      5: heading (511 = not available)
      6: annotation (critical point bitmap)
      7: turn (rate of turn; -127 = not available)
      8: course (COG in degrees)
      9-26: in-situ statistics (all zeros in raw data)

    Pre-filtering applied:
      - SOG > 1 knot (exclude stationary vessels)
      - SOG < 50 knots (exclude erroneous readings)
      - Remove duplicate MMSI+timestamp rows (multiple ground stations)
    """
    # Try repo-local path first, then global RAW_DATA_PATH
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'brest', 'ais_brest_positions_raw.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(RAW_DATA_PATH, 'brest', 'ais_brest_positions_raw.csv')
    col_names = ['timestamp_ms', 'mmsi', 'longitude', 'latitude', 'speed',
                 'heading', 'annotation', 'turn', 'course']
    df = pd.read_csv(csv_path, header=None, names=col_names, usecols=range(9))

    # Pre-filter by speed
    df = df[(df.speed > 1) & (df.speed < 50)]

    # Remove duplicate messages (same vessel + timestamp from multiple ground stations)
    df = df.drop_duplicates(subset=['timestamp_ms', 'mmsi'])

    # Convert timestamp
    df['date_time'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df = df.sort_values(['mmsi', 'date_time'])
    df = df.reset_index(drop=True)

    # Group by vessel and return one DataFrame per MMSI
    dfs = []
    for _, group in df.groupby('mmsi'):
        vessel_df = group[['date_time', 'latitude', 'longitude']].reset_index(drop=True)
        if len(vessel_df) >= 10:
            dfs.append(vessel_df)

    return dfs


def region_heatmap(dataset, region, res=64):
    all_obs = dataset.reshape(-1, 2).copy()
    all_obs -= region[:2]
    all_obs /= region[3:]

    rotation = region[2] * np.pi / 180
    cosrotation = np.cos(rotation)
    sinrotation = np.sin(rotation)
    cosnrotation = np.cos(-rotation)
    sinnrotation = np.sin(-rotation)

    rotated_obs = np.zeros_like(all_obs)
    rotated_obs[:, 0] = cosnrotation * all_obs[:, 0] + sinnrotation * all_obs[:, 1]
    rotated_obs[:, 1] = sinrotation * all_obs[:, 0] + cosnrotation * all_obs[:, 1]
    H, _, _ = np.histogram2d(rotated_obs[:, 1],
                            rotated_obs[:, 0], 
                            bins=(res, res), 
                            range=((0, 1), (0, 1)),
                            density=True)
    return H.T


def transform_trajectory(traj, region):
    origin_lat = region[0]
    origin_lon = region[1]
    rotation = region[2] * np.pi / 180
    height = region[3]
    width = region[4]

    transposed_signal = traj - [origin_lat, origin_lon]
    scaled_signal = transposed_signal / [height, width]
    
    rotated_signal = np.empty_like(traj)
    rotated_signal[:, 0] = np.cos(rotation) * scaled_signal[:, 0] + np.sin(-rotation) * scaled_signal[:, 1]
    rotated_signal[:, 1] = np.sin(rotation) * scaled_signal[:, 0] + np.cos(-rotation) * scaled_signal[:, 1]
    return rotated_signal


def find_partial_region(traj, dataset, size, bounds, fraction_inside, heatmapper=None, random_tries=10000, max_rotation=360, res=64):
    # min_lat, max_lat, min_lon, max_lon = bounds
    seq_len = len(traj)
    min_seq_len = int(math.ceil(seq_len*fraction_inside))
    # min_x = traj[:, 0].min(axis=0)
    # max_x = traj[:, 0].max(axis=0)
    # min_y = traj[:, 1].min(axis=0)
    # max_y = traj[:, 1].max(axis=0)
    # trajectory_bounds = [min_x - size[0], max_x + size[0], min_y - size[1], max_y + size[1]]
    # Ensure trajectory is float32 for Cython function
    traj = traj.astype(np.float32)
    for _ in range(random_tries):
        region = random_region(size=size, bounds=bounds, max_rotation=max_rotation)
        subtrajs = crop_signal_with_rotation_optim(traj, region, min_len=min_seq_len)
        if len(subtrajs) > 0:
            traj_transformed = transform_trajectory(traj, region)
            if heatmapper is not None:
                heatmap = heatmapper.get_heatmap(region, res=res)
            else:
                heatmap = region_heatmap(dataset, region, res=res)
            return True, region, traj_transformed, heatmap

    return False, None, None, None


class InterpolatingHeatmapper:
    """
    A faster alternative to Heatmapper that uses interpolation from a pre-computed global heatmap.
    Provides the same interface as Heatmapper for drop-in replacement.
    
    Uses PyTorch for all operations, supporting both CPU and GPU execution.
    """
    def __init__(self, dataset, meters_per_cell=1, sigma=15, device='cuda', keep_on_gpu=False):
        """
        Parameters:
        -----------
        dataset : np.ndarray
            Array of trajectories with shape (num_trajs, seq_len, 2)
        meters_per_cell : float
            Resolution of the global heatmap in meters per cell
        sigma : float
            Gaussian blur sigma for smoothing the global heatmap
        device : str
            'cuda' for GPU or 'cpu' for CPU execution
        keep_on_gpu : bool
            If True, keep blurred heatmap on GPU to eliminate CPU-GPU transfers
            Trades VRAM for faster heatmap generation (10-20% speedup)
        """

        self.device = device
        self.keep_on_gpu = keep_on_gpu
        self.meters_per_cell = meters_per_cell
        self.sigma = sigma

        # Compute dataset bounds
        min_y, min_x = dataset.min(axis=(0, 1))
        max_y, max_x = dataset.max(axis=(0, 1))

        self.global_bounds = (min_y, max_y, min_x, max_x)

        # Build high-resolution global heatmap
        width = max_x - min_x
        height = max_y - min_y

        cells_width = int(width / meters_per_cell) + 1
        cells_height = int(height / meters_per_cell) + 1

        boundries_y = np.linspace(min_y, max_y, cells_height)
        boundries_x = np.linspace(min_x, max_x, cells_width)

        all_observations = dataset.reshape(-1, 2)

        highres_heatmap, _, _ = np.histogram2d(
            all_observations[:, 1],
            all_observations[:, 0],
            bins=(boundries_x, boundries_y),
            range=((min_x, max_x), (min_y, max_y)),
            density=True
        )
        highres_heatmap = highres_heatmap.T

        # Convert to torch tensor — compute in FP32 to preserve precision
        # (density values can be ~1e-8 which underflows in FP16)
        heatmap_torch = torch.from_numpy(highres_heatmap).float().to(device)

        # Optionally apply Gaussian blur
        if sigma > 0:
            kernel_size = int(4 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()

            gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
            gauss_2d = gauss_2d / gauss_2d.sum()
            kernel = gauss_2d[None, None, :, :]

            heatmap_padded = heatmap_torch.unsqueeze(0).unsqueeze(0)
            padding = kernel_size // 2
            final_heatmap = F.conv2d(heatmap_padded, kernel, padding=padding).squeeze()
        else:
            final_heatmap = heatmap_torch

        # Store heatmap — use FP16 only on GPU to save VRAM,
        # keep FP32 on CPU where memory is less constrained
        storage_dtype = torch.float16 if device != 'cpu' else torch.float32
        if keep_on_gpu:
            self.blurred_heatmap = final_heatmap.to(storage_dtype)
        else:
            self.blurred_heatmap = final_heatmap.cpu()  # already FP32
    
    def get_heatmap(self, region, res=64):
        """
        Get a single heatmap by delegating to get_heatmaps.

        Returns a numpy array with shape (res, res).
        """
        # Create rectangular resolution tensor for square heatmap
        resolutions_hw = torch.tensor([[res, res]], device=self.device)
        result = self.get_heatmaps_adaptive([region], resolutions_hw)[0]
        if isinstance(result, torch.Tensor):
            return result.numpy()
        return result
    
    def get_heatmaps_adaptive(self, regions, resolutions_hw):
        """
        Get heatmaps for multiple regions with rectangular resolutions.
        
        CRITICAL: Region specification format and rotation behavior
        
        Parameters:
        -----------
        regions : list of lists/tuples
            List of region specifications, each as [origin_y, origin_x, rotation_degrees, height, width]
            where:
            - origin_y, origin_x: BOTTOM-LEFT CORNER of the region in global meters
            - rotation_degrees: Rotation angle in degrees (counter-clockwise)
            - height, width: Full dimensions of the region in global meters
            
        resolutions_hw : torch.Tensor
            Shape (num_regions, 2) where each row is [height_res, width_res]
        
        Returns:
        --------
        list of torch.Tensor
            List of heatmaps, where heatmaps[i] has shape (resolutions_hw[i, 0], resolutions_hw[i, 1])
        
        Region Specification and Rotation Process:
        ----------------------------------------
        
        The region specification uses the BOTTOM-LEFT CORNER as the origin, not the center.
        This is crucial for proper alignment with trajectory rotation.
        
        ASCII Diagram of Region Specification:
        
        Global Coordinate System (meters):
        
        Y (North)
        ↑
        │  ┌─────────────────┐ ← max_y
        │  │                 │
        │  │                 │
        │  │                 │
        │  │                 │
        │  │                 │
        │  │                 │
        │  │                 │
        │  └─────────────────┘ ← origin_y (bottom-left corner)
        │    ↑               ↑
        │    origin_x        max_x
        └────────────────────────────→ X (East)
        
        When rotation is applied:
        
        1. The region is defined by [origin_y, origin_x, rotation_deg, height, width]
        2. A local coordinate system is created with origin at (origin_y, origin_x)
        3. The heatmap grid is created in this local system: [0, 1] x [0, 1]
        4. Each grid point is rotated by -rotation_deg (inverse rotation)
        5. The rotated grid points are scaled by [height, width] and translated by [origin_y, origin_x]
        6. This creates a rotated heatmap that appears rotated when viewed in the local frame
        
        Rotation Mathematics:
        --------------------
        
        For a point (local_y, local_x) in the local [0,1] x [0,1] coordinate system:
        
        1. Apply inverse rotation: (rot_y, rot_x) = R(-θ) * (local_y, local_x)
        2. Scale by region dimensions: (scaled_y, scaled_x) = (rot_y * height, rot_x * width)
        3. Translate to global position: (world_y, world_x) = (scaled_y + origin_y, scaled_x + origin_x)
        
        Where R(-θ) is the rotation matrix for angle -θ:
        R(-θ) = [[cos(-θ), -sin(-θ)],
                 [sin(-θ),  cos(-θ)]]
        
        This inverse rotation ensures that when the heatmap is displayed in the local coordinate
        system, it appears rotated by the forward angle θ.
        """
        import torch

        min_y, max_y, min_x, max_x = self.global_bounds
        results = []

        # Move heatmap to GPU if not already there, and convert to FP32 for computation
        if self.keep_on_gpu:
            # Already on GPU, just cast to float32
            heatmap_gpu = self.blurred_heatmap.float()
        else:
            # Transfer from CPU to GPU and cast
            heatmap_gpu = self.blurred_heatmap.to(self.device).float()

        # Extract each region at the global heatmap's native resolution,
        # then average-pool down to the target resolution. This correctly
        # integrates density over each output cell, matching what Heatmapper
        # does when it bins observations per cell.
        hmap_H, hmap_W = heatmap_gpu.shape
        heatmap_batch = heatmap_gpu.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        for i, region in enumerate(regions):
            region_y, region_x, rotation_deg, height, width = region
            res_h = resolutions_hw[i, 0].item()
            res_w = resolutions_hw[i, 1].item()

            # Determine native extraction resolution from the global heatmap
            native_h = max(int(height / self.meters_per_cell), res_h)
            native_w = max(int(width / self.meters_per_cell), res_w)

            # Convert rotation to radians
            rotation = rotation_deg * np.pi / 180
            cos_angle = np.cos(rotation)
            sin_angle = np.sin(rotation)

            # Create grid at native resolution
            grid_y = torch.linspace(0.5/native_h, 1 - 0.5/native_h, native_h, device=self.device)
            grid_x = torch.linspace(0.5/native_w, 1 - 0.5/native_w, native_w, device=self.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

            # Apply inverse rotation
            normalized_y = cos_angle * grid_y + sin_angle * grid_x
            normalized_x = -sin_angle * grid_y + cos_angle * grid_x

            # Scale and translate to world coordinates
            world_y = normalized_y * height + region_y
            world_x = normalized_x * width + region_x

            # Convert to normalized grid coordinates for grid_sample [-1, 1]
            pixel_y = (world_y - min_y) / (max_y - min_y) * hmap_H - 0.5
            pixel_x = (world_x - min_x) / (max_x - min_x) * hmap_W - 0.5
            norm_x = 2.0 * pixel_x / (hmap_W - 1) - 1.0
            norm_y = 2.0 * pixel_y / (hmap_H - 1) - 1.0

            grid = torch.stack([norm_x, norm_y], dim=-1)

            # Extract at native resolution
            interpolated = F.grid_sample(
                heatmap_batch,
                grid.unsqueeze(0),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            # Average-pool from native resolution to target resolution
            native_result = interpolated  # [1, 1, native_h, native_w]
            if native_h != res_h or native_w != res_w:
                result = F.adaptive_avg_pool2d(native_result, (res_h, res_w))
            else:
                result = native_result
            result = result.squeeze(0).squeeze(0)  # [res_h, res_w]

            # Normalize: density should sum to res_h * res_w (matching Heatmapper convention)
            result_sum = result.sum().item()
            if result_sum > 0:
                result = (res_h * res_w) * result / result_sum
            else:
                result = torch.ones_like(result) / (res_h * res_w)

            results.append(result)

        # Clean up GPU memory
        del heatmap_gpu
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return results
    


class Heatmapper:
    def __init__(self, dataset, cells=100):
        self.cells = cells

        min_y, min_x = dataset.min(axis=(0, 1))
        max_y, max_x = dataset.max(axis=(0, 1))

        self.boundries_y = np.linspace(min_y, max_y, cells + 1)
        self.boundries_x = np.linspace(min_x, max_x, cells + 1)

        side_y = (max_y - min_y) / (cells)
        side_x = (max_x - min_x) / (cells)

        self.tree = {}

        for traj in dataset:
            for obs in traj:
                # Subtract minimum and then divide by cell size
                a = int((obs[0] - min_y) / side_y)
                b = int((obs[1] - min_x) / side_x)
                
                # Clamp to valid range to handle edge cases
                a = min(max(a, 0), cells - 1)
                b = min(max(b, 0), cells - 1)
                key = a, b
                if key not in self.tree:
                    self.tree[key] = [obs]
                else:
                    self.tree[key].append(obs)

        for key in self.tree.keys():
            self.tree[key] = np.stack(self.tree[key], axis=0)

    def _get_overlapping_cells(self, region, debug=False, padding=0.02):
        region_y, region_x = region[0], region[1]
        angle = region[2] * np.pi / 180
        height, width = region[3], region[4]
        
        # First get the bounding box to limit candidates
        corners_transformed = np.array([
            [0, 0], [0, 1], [1, 1], [1, 0]
        ], dtype=np.float32)
        
        cosrotation = np.cos(angle)
        sinrotation = np.sin(angle)
        
        corners_rotated = np.zeros_like(corners_transformed)
        corners_rotated[:, 0] = cosrotation * corners_transformed[:, 0] + sinrotation * corners_transformed[:, 1]
        corners_rotated[:, 1] = -sinrotation * corners_transformed[:, 0] + cosrotation * corners_transformed[:, 1]
        
        corners_scaled = corners_rotated * np.array([height, width])
        corners_world = corners_scaled + np.array([region_y, region_x])
        
        # Find bounding box
        min_y, max_y = corners_world[:, 0].min(), corners_world[:, 0].max()
        min_x, max_x = corners_world[:, 1].min(), corners_world[:, 1].max()
        
        # Get candidate cells from bounding box
        cell_min_y = max(0, np.searchsorted(self.boundries_y, min_y, side='right') - 1)
        cell_max_y = min(len(self.boundries_y) - 2, np.searchsorted(self.boundries_y, max_y, side='left'))
        cell_min_x = max(0, np.searchsorted(self.boundries_x, min_x, side='right') - 1)
        cell_max_x = min(len(self.boundries_x) - 2, np.searchsorted(self.boundries_x, max_x, side='left'))

        # Only test cells within the bounding box
        y_indices = np.arange(cell_min_y, cell_max_y + 1)
        x_indices = np.arange(cell_min_x, cell_max_x + 1)

        cell_y_grid, cell_x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')

        flat_y_indices = cell_y_grid.flatten()
        flat_x_indices = cell_x_grid.flatten()
        
        # Now the rest of your code is correct
        cell_corners_y = np.stack([
            self.boundries_y[flat_y_indices],      # bottom-left
            self.boundries_y[flat_y_indices],      # bottom-right  
            self.boundries_y[flat_y_indices + 1],  # top-right
            self.boundries_y[flat_y_indices + 1]   # top-left
        ], axis=1)
        
        cell_corners_x = np.stack([
            self.boundries_x[flat_x_indices],      # bottom-left
            self.boundries_x[flat_x_indices + 1],  # bottom-right
            self.boundries_x[flat_x_indices + 1],  # top-right
            self.boundries_x[flat_x_indices]       # top-left
        ], axis=1)

        dy = cell_corners_y - region_y
        dx = cell_corners_x - region_x

        norm_y = dy / height
        norm_x = dx / width
        
        cos_neg_angle = np.cos(-angle)
        sin_neg_angle = np.sin(-angle)
        
        local_y = cos_neg_angle * norm_y + sin_neg_angle * norm_x
        local_x = -sin_neg_angle * norm_y + cos_neg_angle * norm_x
        
        inside_corners = ((local_y >= -padding) & (local_y <= 1 + padding) & 
                        (local_x >= -padding) & (local_x <= 1 + padding))
        
        overlapping_mask = np.any(inside_corners, axis=1)

        valid_y_indices = flat_y_indices[overlapping_mask]
        valid_x_indices = flat_x_indices[overlapping_mask]

        return list(zip(valid_y_indices, valid_x_indices))

    def get_heatmap(self, region, res=64):
        # Get overlapping cells
        overlapping_cells = self._get_overlapping_cells(region)
        
        # Collect observations from overlapping cells only
        filtered_obs = []
        for cell_key in overlapping_cells:
            if cell_key in self.tree:
                filtered_obs.append(self.tree[cell_key])
        
        if not filtered_obs:
            return np.full((res, res), np.nan)

        filtered_obs = np.concatenate(filtered_obs, axis=0)
        
        # Your existing transformation code
        filtered_obs -= region[:2]
        filtered_obs /= region[3:]
        rotation = region[2] * np.pi / 180
        cosrotation = np.cos(rotation)
        sinrotation = np.sin(rotation)
        cosnrotation = np.cos(-rotation)
        sinnrotation = np.sin(-rotation)
        rotated_obs = np.zeros_like(filtered_obs)
        rotated_obs[:, 0] = cosnrotation * filtered_obs[:, 0] + sinnrotation * filtered_obs[:, 1]
        rotated_obs[:, 1] = sinrotation * filtered_obs[:, 0] + cosnrotation * filtered_obs[:, 1]
        
        H, _, _ = np.histogram2d(rotated_obs[:, 1],
                                rotated_obs[:, 0], 
                                bins=(res, res), 
                                range=((0, 1), (0, 1)),
                                density=True)
        return H.T

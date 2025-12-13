#!/usr/bin/env python3
"""
2D Top-Down Gaussian Process Disturbance Field Visualization

- Loads buffer poses and cause location
- Fits a 3D Gaussian Process centered at the cause location
- Creates a clean 2D top-down view of the disturbance field
- Shows trajectory, cause location, and GP field contours clearly
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

BUFFER_DIR = "/home/navin/ros2_ws/src/buffers/run_20250828_093052_931_2e959586/buffer2"
NOMINAL_PATH = "/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json"


def load_buffer_xyz_drift(buffer_dir: str):
    """Load buffer poses and cause location."""
    buffer_dir = Path(buffer_dir)
    poses_path = buffer_dir / "poses.npy"
    meta_path = buffer_dir / "metadata.json"
    cause_loc_path = buffer_dir / "cause_location.json"

    assert poses_path.exists(), f"poses.npy missing in {buffer_dir}"

    poses = np.load(poses_path)
    xyz = poses[:, 1:4]  # x, y, z coordinates

    cause = None
    cause_loc = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cause = meta.get("cause")
        cause_loc = meta.get("cause_location")
    if cause_loc is None and cause_loc_path.exists():
        with open(cause_loc_path, "r") as f:
            d = json.load(f)
        cause = d.get("cause", cause)
        cause_loc = d.get("location_3d")

    cause_xyz = None
    if cause_loc is not None:
        cause_xyz = np.array(cause_loc[:3], dtype=float)

    return xyz, cause, cause_xyz


def load_nominal_xyz(nominal_path: str):
    """Load nominal trajectory."""
    p = Path(nominal_path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        data = json.load(f)
    pts = data.get("points") if isinstance(data, dict) else data
    if isinstance(pts, list) and len(pts) > 0:
        if isinstance(pts[0], dict):
            xyz_list = []
            for item in pts:
                pos = item.get("position") if isinstance(item, dict) else None
                if pos and all(k in pos for k in ("x", "y", "z")):
                    xyz_list.append([float(pos["x"]), float(pos["y"]), float(pos["z"])])
            if xyz_list:
                return np.array(xyz_list, dtype=float)
        else:
            arr = np.array(pts, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
    return None


def compute_trajectory_drift_vectors(actual_xyz: np.ndarray, nominal_xyz: np.ndarray):
    """Compute drift vectors and magnitudes from nominal trajectory."""
    if nominal_xyz is None or len(nominal_xyz) == 0:
        return None, None
    
    drift_vectors = []
    drift_magnitudes = []
    
    for actual_point in actual_xyz:
        diffs = nominal_xyz - actual_point
        dists = np.linalg.norm(diffs, axis=1)
        closest_idx = int(np.argmin(dists))
        drift_vec = actual_point - nominal_xyz[closest_idx]
        drift_mag = float(np.linalg.norm(drift_vec))
        drift_vectors.append(drift_vec)
        drift_magnitudes.append(drift_mag)
    
    return np.array(drift_vectors), np.array(drift_magnitudes)


def compute_disturbance_at_nominal_points(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, cause_xyz: np.ndarray):
    """Compute disturbance field at nominal trajectory points.
    
    For each nominal point, find the corresponding actual point and compute the disturbance.
    This gives us the disturbance field at the intended positions, not the deviated ones.
    """
    disturbances = []
    nominal_points_used = []
    
    # First, find the temporal correspondence between nominal and actual trajectories
    # by matching the overall trajectory shape rather than individual points
    
    # Find the nominal segment that best matches the actual trajectory
    # Use the actual trajectory bounds to select relevant nominal points
    actual_bounds = {
        'x': (actual_xyz[:, 0].min(), actual_xyz[:, 0].max()),
        'y': (actual_xyz[:, 1].min(), actual_xyz[:, 1].max()),
        'z': (actual_xyz[:, 2].min(), actual_xyz[:, 2].max())
    }
    
    # Add some padding to include nominal points slightly outside actual bounds
    pad = 0.3
    mask = (
        (nominal_xyz[:, 0] >= actual_bounds['x'][0] - pad) & 
        (nominal_xyz[:, 0] <= actual_bounds['x'][1] + pad) &
        (nominal_xyz[:, 1] >= actual_bounds['y'][0] - pad) & 
        (nominal_xyz[:, 1] <= actual_bounds['y'][1] + pad) &
        (nominal_xyz[:, 2] >= actual_bounds['z'][0] - pad) & 
        (nominal_xyz[:, 2] <= actual_bounds['z'][1] + pad)
    )
    
    relevant_nominal = nominal_xyz[mask]
    
    if len(relevant_nominal) == 0:
        print("Warning: No nominal points found in actual trajectory region")
        return np.array([]), np.array([])
    
    # For each relevant nominal point, find the closest actual point
    for nominal_point in relevant_nominal:
        distances = np.linalg.norm(actual_xyz - nominal_point, axis=1)
        closest_idx = np.argmin(distances)
        closest_actual = actual_xyz[closest_idx]
        
        # Compute disturbance as the deviation from nominal
        disturbance = np.linalg.norm(closest_actual - nominal_point)
        
        # Only include if the nominal point is reasonably close to the actual trajectory
        # (to avoid spurious matches at trajectory ends)
        if distances[closest_idx] < 0.3:  # Reduced threshold for better matching
            disturbances.append(disturbance)
            nominal_points_used.append(nominal_point)
    
    return np.array(nominal_points_used), np.array(disturbances)


def create_cause_centric_features_nominal(nominal_points: np.ndarray, cause_xyz: np.ndarray):
    """Create features relative to cause location for nominal points."""
    if cause_xyz is None:
        # If no cause location, use nominal trajectory centroid
        cause_xyz = nominal_points.mean(axis=0)
        print(f"Warning: No cause location found, using nominal trajectory centroid: {cause_xyz}")
    
    # Features: [distance_to_cause, x_rel, y_rel, z_rel]
    distances = np.linalg.norm(nominal_points - cause_xyz, axis=1)
    relative_positions = nominal_points - cause_xyz
    
    features = np.column_stack([
        distances,
        relative_positions[:, 0],  # x relative to cause
        relative_positions[:, 1],  # y relative to cause
        relative_positions[:, 2]   # z relative to cause
    ])
    
    return features, cause_xyz


def fit_cause_centric_gp(features: np.ndarray, drift_magnitudes: np.ndarray):
    """Fit a Gaussian Process to model disturbance as function of position relative to cause."""
    
    # Define kernel: RBF on distance + RBF on relative positions
    kernel_distance = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    kernel_position = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    kernel = kernel_distance + kernel_position
    
    # Fit GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=10,
        random_state=42
    )
    
    gp.fit(features, drift_magnitudes)
    
    print(f"Fitted GP kernel: {gp.kernel_}")
    print(f"GP score: {gp.score(features, drift_magnitudes):.4f}")
    
    return gp


def create_2d_prediction_grid(xyz: np.ndarray, cause_xyz: np.ndarray, resolution=0.05):
    """Create a 2D grid for top-down prediction."""
    pad = 0.5
    xmin, xmax = xyz[:, 0].min() - pad, xyz[:, 0].max() + pad
    ymin, ymax = xyz[:, 1].min() - pad, xyz[:, 1].max() + pad
    
    # Include cause location in grid bounds
    if cause_xyz is not None:
        xmin = min(xmin, cause_xyz[0] - pad)
        xmax = max(xmax, cause_xyz[0] + pad)
        ymin = min(ymin, cause_xyz[1] - pad)
        ymax = max(ymax, cause_xyz[1] + pad)
    
    xs = np.arange(xmin, xmax + resolution, resolution)
    ys = np.arange(ymin, ymax + resolution, resolution)
    
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
    
    # Use the average Z height of the trajectory for the 2D projection
    avg_z = xyz[:, 2].mean()
    Zg = np.full_like(Xg, avg_z)
    
    grid_points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    
    return Xg, Yg, Zg, grid_points


def predict_gp_field_2d(gp, grid_points: np.ndarray, cause_xyz: np.ndarray):
    """Predict disturbance field on 2D grid using fitted GP."""
    # Create features for grid points
    distances = np.linalg.norm(grid_points - cause_xyz, axis=1)
    relative_positions = grid_points - cause_xyz
    
    grid_features = np.column_stack([
        distances,
        relative_positions[:, 0],
        relative_positions[:, 1],
        relative_positions[:, 2]
    ])
    
    # Predict mean and uncertainty
    mean_pred, std_pred = gp.predict(grid_features, return_std=True)
    
    return mean_pred, std_pred


def plot_2d_topdown_gp_field(Xg, Yg, disturbance_field, xyz, disturbance_magnitudes, 
                            cause_xyz, nominal_seg=None, cause=None, nominal_points_used=None):
    """Plot 2D top-down view of GP disturbance field."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    field_reshaped = disturbance_field.reshape(Xg.shape)
    
    # Plot 1: GP field with trajectory overlay
    contour = ax1.contourf(Xg, Yg, field_reshaped, levels=20, cmap='viridis', alpha=0.8)
    ax1.contour(Xg, Yg, field_reshaped, levels=10, colors='white', alpha=0.6, linewidths=0.5)
    
    # Plot actual trajectory
    ax1.plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=3, label='Actual trajectory')
    
    # Plot nominal trajectory if available
    if nominal_seg is not None:
        ax1.plot(nominal_seg[:, 0], nominal_seg[:, 1], '--', color='orange', 
                linewidth=2, label='Nominal trajectory')
    
    # Plot cause location
    if cause_xyz is not None:
        ax1.scatter(cause_xyz[0], cause_xyz[1], marker='*', s=300, color='red', 
                   label=f'Cause: {cause or "Unknown"}', edgecolors='white', linewidth=2)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('GP Disturbance Field (Top-Down View)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_aspect('equal')
    
    # Add colorbar for GP field
    cbar1 = plt.colorbar(contour, ax=ax1, shrink=0.8)
    cbar1.set_label('Disturbance magnitude', fontsize=12)
    
    # Plot 2: Nominal points with disturbance magnitude
    ax2.plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=2, alpha=0.5, label='Actual trajectory')
    
    if nominal_points_used is not None:
        scatter2 = ax2.scatter(nominal_points_used[:, 0], nominal_points_used[:, 1], 
                              c=disturbance_magnitudes, cmap='plasma', s=50, alpha=0.8, 
                              edgecolors='white', linewidth=1)
        ax2.plot(nominal_points_used[:, 0], nominal_points_used[:, 1], '--', color='orange', 
                linewidth=2, label='Nominal trajectory (sampled)')
    else:
        # Fallback to actual trajectory
        scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], c=disturbance_magnitudes, 
                              cmap='plasma', s=50, alpha=0.8, edgecolors='white', linewidth=1)
    
    if cause_xyz is not None:
        ax2.scatter(cause_xyz[0], cause_xyz[1], marker='*', s=300, color='red', 
                   label=f'Cause: {cause or "Unknown"}', edgecolors='white', linewidth=2)
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Nominal Points with Disturbance Magnitude', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_aspect('equal')
    
    # Add colorbar for disturbance
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Disturbance magnitude', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_gp_field_slices(Xg, Yg, disturbance_field, xyz, disturbance_magnitudes, 
                        cause_xyz, cause=None, nominal_points_used=None):
    """Plot GP field at different Z heights."""
    field_reshaped = disturbance_field.reshape(Xg.shape)
    
    # Get Z range from trajectory
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    z_range = z_max - z_min
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot at different Z heights
    z_heights = [z_min, z_min + z_range/3, z_min + 2*z_range/3, z_max]
    titles = ['Bottom', 'Lower Middle', 'Upper Middle', 'Top']
    
    for i, (z_height, title) in enumerate(zip(z_heights, titles)):
        ax = axes[i]
        
        # Create a slice at this Z height
        # For simplicity, we'll use the same field but adjust the visualization
        contour = ax.contourf(Xg, Yg, field_reshaped, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(Xg, Yg, field_reshaped, levels=10, colors='white', alpha=0.6, linewidths=0.5)
        
        # Plot trajectory points near this Z height
        z_tolerance = z_range / 6
        mask = np.abs(xyz[:, 2] - z_height) < z_tolerance
        if np.sum(mask) > 0:
            ax.scatter(xyz[mask, 0], xyz[mask, 1], c='blue', s=30, alpha=0.8, 
                      edgecolors='white', linewidth=1, label='Actual trajectory')
        
        # Plot nominal points near this Z height if available
        if nominal_points_used is not None:
            nominal_mask = np.abs(nominal_points_used[:, 2] - z_height) < z_tolerance
            if np.sum(nominal_mask) > 0:
                ax.scatter(nominal_points_used[nominal_mask, 0], nominal_points_used[nominal_mask, 1], 
                          c='orange', s=30, alpha=0.8, edgecolors='white', linewidth=1, label='Nominal trajectory')
        
        # Plot cause location
        if cause_xyz is not None:
            ax.scatter(cause_xyz[0], cause_xyz[1], marker='*', s=200, color='red', 
                      label=f'Cause: {cause or "Unknown"}', edgecolors='white', linewidth=2)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'GP Field at Z = {z_height:.2f}m ({title})', fontsize=12)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = 'xy'):
    """Clip nominal trajectory to the segment between the closest points to the actual
    trajectory's start and end, using a chosen coordinate plane ('xy' or 'xz').
    """
    if nominal_xyz is None or len(nominal_xyz) == 0 or actual_xyz is None or len(actual_xyz) == 0:
        return nominal_xyz
    plane = plane.lower()
    if plane not in ('xy', 'xz'):
        plane = 'xy'
    # Select components based on plane
    if plane == 'xy':
        nom_proj = nominal_xyz[:, [0, 1]]
        act_start = actual_xyz[0, [0, 1]]
        act_end = actual_xyz[-1, [0, 1]]
    else:  # 'xz'
        nom_proj = nominal_xyz[:, [0, 2]]
        act_start = actual_xyz[0, [0, 2]]
        act_end = actual_xyz[-1, [0, 2]]

    # Find nearest nominal indices to start and end in the projection
    d_start = np.linalg.norm(nom_proj - act_start[None, :], axis=1)
    d_end = np.linalg.norm(nom_proj - act_end[None, :], axis=1)
    i_start = int(np.argmin(d_start))
    i_end = int(np.argmin(d_end))

    # Ensure correct ordering
    lo, hi = (i_start, i_end) if i_start <= i_end else (i_end, i_start)
    # Clip with a small safety margin inside the range to avoid edge mismatches
    lo = max(0, lo)
    hi = min(len(nominal_xyz) - 1, hi)
    if hi <= lo:
        return nominal_xyz
    return nominal_xyz[lo:hi + 1]


def main():
    print("Loading buffer data...")
    xyz, cause, cause_xyz = load_buffer_xyz_drift(BUFFER_DIR)
    nominal_xyz = load_nominal_xyz(NOMINAL_PATH)
    
    print(f"Loaded {len(xyz)} trajectory points")
    print(f"Cause: {cause}")
    print(f"Cause location: {cause_xyz}")
    
    # Clip nominal trajectory to actual start/end in XY plane to avoid spurious edge points
    clipped_nominal_xyz = None
    if nominal_xyz is not None:
        clipped_nominal_xyz = clip_nominal_to_actual_segment(nominal_xyz, xyz, plane='xy')
        if clipped_nominal_xyz is not None and len(clipped_nominal_xyz) > 0:
            print(f"Clipped nominal from {len(nominal_xyz)} to {len(clipped_nominal_xyz)} points using XY plane")
        else:
            clipped_nominal_xyz = nominal_xyz
    
    # Compute disturbances at nominal trajectory points (using clipped nominal)
    if clipped_nominal_xyz is not None:
        nominal_points_used, disturbance_magnitudes = compute_disturbance_at_nominal_points(clipped_nominal_xyz, xyz, cause_xyz)
        if nominal_points_used.size == 0:
            print("Warning: No nominal points were used for disturbance computation.")
            # Fallback to original method
            drift_vectors, drift_magnitudes = compute_trajectory_drift_vectors(xyz, clipped_nominal_xyz)
            if drift_vectors is None:
                print("Warning: Could not compute drift from nominal trajectory; using zeros")
                disturbance_magnitudes = np.zeros(len(clipped_nominal_xyz))
            else:
                disturbance_magnitudes = drift_magnitudes
            nominal_points_used = clipped_nominal_xyz
    else:
        print("Warning: No nominal trajectory available; using zeros")
        disturbance_magnitudes = np.zeros(len(xyz))
        nominal_points_used = xyz
    
    print(f"Disturbance magnitudes - min: {disturbance_magnitudes.min():.4f}, max: {disturbance_magnitudes.max():.4f}")
    print(f"Using {len(nominal_points_used)} nominal points for GP training")
    
    # Create cause-centric features for nominal points
    print("Creating cause-centric features for nominal points...")
    features, final_cause_xyz = create_cause_centric_features_nominal(nominal_points_used, cause_xyz)
    
    # Fit Gaussian Process
    print("Fitting Gaussian Process...")
    gp = fit_cause_centric_gp(features, disturbance_magnitudes)
    
    # Create 2D prediction grid
    print("Creating 2D prediction grid...")
    Xg, Yg, Zg, grid_points = create_2d_prediction_grid(xyz, final_cause_xyz, resolution=0.05)
    
    # Predict disturbance field
    print("Predicting disturbance field...")
    mean_pred, std_pred = predict_gp_field_2d(gp, grid_points, final_cause_xyz)
    
    # Nominal segment for visualization (use clipped)
    nominal_seg = clipped_nominal_xyz
    
    # Plot 2D top-down view
    print("Creating 2D top-down visualization...")
    fig_main = plot_2d_topdown_gp_field(Xg, Yg, mean_pred, xyz, disturbance_magnitudes, 
                                       final_cause_xyz, nominal_seg, cause, nominal_points_used)
    plt.suptitle("2D Top-Down Gaussian Process Disturbance Field Analysis (Corrected)", fontsize=16)
    plt.show()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 
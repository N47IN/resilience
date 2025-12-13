#!/usr/bin/env python3
"""
3D Gaussian Process Disturbance Field Visualization (with vertical component)

- Loads buffer poses and cause location
- Clips nominal trajectory to actual segment (same as 2D script)
- Computes disturbance (including Z/vertical deviations)
- Fits a 3D Gaussian Process centered at the cause location
- Predicts on a 3D grid and visualizes:
  * 2D points (XY) colored by disturbance
  * 3D volume with translucent isosurfaces (preferred) or colored point cloud (fallback)
  * Overlays 3D trajectory and cause location
- Enhanced contrast via percentile normalization; optional PyVista renderer for smoother 3D
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional: 3D isosurfaces via marching cubes
try:
    from skimage import measure as _sk_measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# Optional: High-performance 3D rendering
try:
    import pyvista as _pv
    from pyvista import themes as _pv_themes
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

# NEW: Additional sklearn kernels and CV utilities for kernel learning
try:
    from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, WhiteKernel
    from sklearn.model_selection import KFold
    _HAS_SKLEARN_ADV = True
except Exception:
    _HAS_SKLEARN_ADV = False

BUFFER_DIR = "/home/navin/ros2_ws/src/buffers/run_20250828_093052_931_2e959586/buffer2"
NOMINAL_PATH = "/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json"

# Visualization settings
USE_PYVISTA = True  # try to use PyVista if available
MAX_POINTS_POINTCLOUD = 50000
POINT_CLOUD_ALPHA = 0.28
PERCENTILE_RANGE_2D = (5, 95)
PERCENTILE_RANGE_3D = (5, 97)


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
    """Compute disturbance field at nominal trajectory points in 3D."""
    disturbances = []
    nominal_points_used = []

    actual_bounds = {
        'x': (actual_xyz[:, 0].min(), actual_xyz[:, 0].max()),
        'y': (actual_xyz[:, 1].min(), actual_xyz[:, 1].max()),
        'z': (actual_xyz[:, 2].min(), actual_xyz[:, 2].max())
    }

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

    for nominal_point in relevant_nominal:
        distances = np.linalg.norm(actual_xyz - nominal_point, axis=1)
        closest_idx = np.argmin(distances)
        closest_actual = actual_xyz[closest_idx]
        disturbance = np.linalg.norm(closest_actual - nominal_point)  # full 3D deviation
        if distances[closest_idx] < 0.3:
            disturbances.append(disturbance)
            nominal_points_used.append(nominal_point)

    return np.array(nominal_points_used), np.array(disturbances)


def create_cause_centric_features_nominal(nominal_points: np.ndarray, cause_xyz: np.ndarray):
    """Create features relative to cause location for nominal points."""
    if cause_xyz is None:
        cause_xyz = nominal_points.mean(axis=0)
        print(f"Warning: No cause location found, using nominal trajectory centroid: {cause_xyz}")

    distances = np.linalg.norm(nominal_points - cause_xyz, axis=1)
    relative_positions = nominal_points - cause_xyz

    features = np.column_stack([
        distances,
        relative_positions[:, 0],
        relative_positions[:, 1],
        relative_positions[:, 2]
    ])

    return features, cause_xyz


def fit_cause_centric_gp(features: np.ndarray, drift_magnitudes: np.ndarray):
    """Fit a Gaussian Process to model disturbance as function of position relative to cause."""
    kernel_distance = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    kernel_position = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    kernel = kernel_distance + kernel_position

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=8,
        random_state=42
    )

    gp.fit(features, drift_magnitudes)

    print(f"Fitted GP kernel: {gp.kernel_}")
    print(f"GP score: {gp.score(features, drift_magnitudes):.4f}")

    return gp


# NEW: Kernel learning with ARD and candidate search
def _build_candidate_kernels(input_dim: int):
    """Return a list of candidate kernels to try. Uses ARD where possible."""
    if not _HAS_SKLEARN_ADV:
        return []
    ard_len = np.ones(input_dim)
    candidates = []
    # RBF (ARD)
    candidates.append(ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=ard_len, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)))
    # Matern family (ARD)
    for nu in (0.5, 1.5, 2.5):
        candidates.append(ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=ard_len, length_scale_bounds=(1e-2, 1e2), nu=nu) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)))
    # RationalQuadratic
    candidates.append(ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)))
    # Sum of RBF (distance vs xyz) like original but ARD
    candidates.append(
        (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))) +
        (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=ard_len, length_scale_bounds=(1e-2, 1e2))) +
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )
    return candidates


def fit_gp_with_kernel_learning(
    features: np.ndarray,
    targets: np.ndarray,
    kernel_mode: str = "auto",
    n_restarts_optimizer: int = 8,
    alpha: float = 1e-6,
    cv_folds: int = 0,
    normalize_y: bool = False,
):
    """Fit a GP with either a specified kernel or automatic kernel selection.

    kernel_mode:
      - 'auto': try several kernels and pick best by log-marginal likelihood or CV score
      - 'rbf', 'matern12', 'matern32', 'matern52', 'rq': force specific family
      - 'original': use the legacy kernel structure
    """
    kernel_mode = (kernel_mode or "auto").lower()
    input_dim = features.shape[1]

    def _legacy_kernel():
        kd = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kp = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-2, 1e2))
        return kd + kp

    def _named_kernel(name: str):
        if name == 'rbf':
            return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-2, 1e2)) + (WhiteKernel(1e-3, (1e-6, 1e1)) if _HAS_SKLEARN_ADV else 0)
        if name == 'matern12':
            return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-2, 1e2), nu=0.5) + (WhiteKernel(1e-3, (1e-6, 1e1)) if _HAS_SKLEARN_ADV else 0)
        if name == 'matern32':
            return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-2, 1e2), nu=1.5) + (WhiteKernel(1e-3, (1e-6, 1e1)) if _HAS_SKLEARN_ADV else 0)
        if name == 'matern52':
            return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + (WhiteKernel(1e-3, (1e-6, 1e1)) if _HAS_SKLEARN_ADV else 0)
        if name == 'rq':
            return ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0) + (WhiteKernel(1e-3, (1e-6, 1e1)) if _HAS_SKLEARN_ADV else 0)
        if name == 'original':
            return _legacy_kernel()
        return _legacy_kernel()

    if kernel_mode != 'auto':
        kernel = _named_kernel(kernel_mode)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, random_state=42)
        gp.fit(features, targets)
        print(f"Fitted GP kernel ({kernel_mode}): {gp.kernel_}")
        print(f"GP score: {gp.score(features, targets):.4f}")
        return gp

    # Auto selection
    candidates = _build_candidate_kernels(input_dim)
    if not candidates:
        print("Advanced kernels unavailable; falling back to legacy kernel")
        return fit_cause_centric_gp(features, targets)

    best_gp = None
    best_score = -np.inf
    best_desc = None

    if cv_folds and cv_folds >= 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for ker in candidates:
            try:
                cv_scores = []
                for train_idx, val_idx in kf.split(features):
                    Xtr, Xval = features[train_idx], features[val_idx]
                    ytr, yval = targets[train_idx], targets[val_idx]
                    gp = GaussianProcessRegressor(kernel=ker, alpha=alpha, n_restarts_optimizer=max(2, n_restarts_optimizer // 2), normalize_y=normalize_y, random_state=42)
                    gp.fit(Xtr, ytr)
                    cv_scores.append(gp.score(Xval, yval))
                mean_cv = float(np.mean(cv_scores))
                if mean_cv > best_score:
                    best_score = mean_cv
                    best_desc = f"CV R2={mean_cv:.4f}"
                    best_gp = GaussianProcessRegressor(kernel=ker, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, random_state=42)
            except Exception as e:
                print(f"Kernel {ker} failed in CV: {e}")
        if best_gp is None:
            print("All candidate kernels failed in CV; using legacy kernel")
            return fit_cause_centric_gp(features, targets)
        best_gp.fit(features, targets)
        print(f"Selected kernel (auto via CV): {best_gp.kernel_} [{best_desc}]")
        print(f"Train R2: {best_gp.score(features, targets):.4f}")
        return best_gp

    # No CV: pick by log-marginal likelihood on full data
    for ker in candidates:
        try:
            gp = GaussianProcessRegressor(kernel=ker, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, random_state=42)
            gp.fit(features, targets)
            lml = gp.log_marginal_likelihood(gp.kernel_.theta)
            if lml > best_score:
                best_score = lml
                best_desc = f"LML={lml:.3f}"
                best_gp = gp
        except Exception as e:
            print(f"Kernel {ker} failed: {e}")

    if best_gp is None:
        print("All candidate kernels failed; using legacy kernel")
        return fit_cause_centric_gp(features, targets)

    print(f"Selected kernel (auto): {best_gp.kernel_} [{best_desc}]")
    print(f"Train R2: {best_gp.score(features, targets):.4f}")
    return best_gp


def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = 'xy'):
    """Clip nominal trajectory to the segment between the closest points to the actual
    trajectory's start and end, using a chosen coordinate plane ('xy' or 'xz').
    """
    if nominal_xyz is None or len(nominal_xyz) == 0 or actual_xyz is None or len(actual_xyz) == 0:
        return nominal_xyz
    plane = plane.lower()
    if plane not in ('xy', 'xz'):
        plane = 'xy'
    if plane == 'xy':
        nom_proj = nominal_xyz[:, [0, 1]]
        act_start = actual_xyz[0, [0, 1]]
        act_end = actual_xyz[-1, [0, 1]]
    else:
        nom_proj = nominal_xyz[:, [0, 2]]
        act_start = actual_xyz[0, [0, 2]]
        act_end = actual_xyz[-1, [0, 2]]

    d_start = np.linalg.norm(nom_proj - act_start[None, :], axis=1)
    d_end = np.linalg.norm(nom_proj - act_end[None, :], axis=1)
    i_start = int(np.argmin(d_start))
    i_end = int(np.argmin(d_end))

    lo, hi = (i_start, i_end) if i_start <= i_end else (i_end, i_start)
    lo = max(0, lo)
    hi = min(len(nominal_xyz) - 1, hi)
    if hi <= lo:
        return nominal_xyz
    return nominal_xyz[lo:hi + 1]


def create_3d_prediction_grid(xyz: np.ndarray, cause_xyz: np.ndarray, resolution_xy: float = 0.06, resolution_z: float = 0.06):
    """Create a 3D grid for prediction bounded by the actual trajectory and including the cause."""
    pad = 0.5
    xmin, xmax = xyz[:, 0].min() - pad, xyz[:, 0].max() + pad
    ymin, ymax = xyz[:, 1].min() - pad, xyz[:, 1].max() + pad
    zmin, zmax = xyz[:, 2].min() - pad, xyz[:, 2].max() + pad

    if cause_xyz is not None:
        xmin = min(xmin, cause_xyz[0] - pad)
        xmax = max(xmax, cause_xyz[0] + pad)
        ymin = min(ymin, cause_xyz[1] - pad)
        ymax = max(ymax, cause_xyz[1] + pad)
        zmin = min(zmin, cause_xyz[2] - pad)
        zmax = max(zmax, cause_xyz[2] + pad)

    xs = np.arange(xmin, xmax + resolution_xy, resolution_xy)
    ys = np.arange(ymin, ymax + resolution_xy, resolution_xy)
    zs = np.arange(zmin, zmax + resolution_z, resolution_z)

    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='xy')  # shapes: (Ny, Nx, Nz)
    grid_points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    return Xg, Yg, Zg, grid_points, xs, ys, zs


def predict_gp_field_3d(gp, grid_points: np.ndarray, cause_xyz: np.ndarray):
    """Predict disturbance field on 3D grid using fitted GP."""
    distances = np.linalg.norm(grid_points - cause_xyz, axis=1)
    relative_positions = grid_points - cause_xyz

    grid_features = np.column_stack([
        distances,
        relative_positions[:, 0],
        relative_positions[:, 1],
        relative_positions[:, 2]
    ])

    mean_pred, std_pred = gp.predict(grid_features, return_std=True)
    return mean_pred, std_pred


def _normalize_percentile(values: np.ndarray, lower_pct: float, upper_pct: float):
    """Percentile-based normalization to [0,1] with clipping for better contrast."""
    lo = np.percentile(values, lower_pct)
    hi = np.percentile(values, upper_pct)
    if hi <= lo:
        hi = lo + 1e-9
    v = np.clip(values, lo, hi)
    v = (v - lo) / (hi - lo)
    return v, lo, hi


def plot_2d_points(xyz: np.ndarray, nominal_points_used: np.ndarray, disturbance_magnitudes: np.ndarray, cause_xyz, cause=None):
    """2D XY scatter of points colored by normalized disturbance magnitude, with trajectory overlay and cause."""
    norm_vals, vmin, vmax = _normalize_percentile(disturbance_magnitudes, *PERCENTILE_RANGE_2D)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=2, alpha=0.55, label='Actual trajectory')

    if nominal_points_used is not None and nominal_points_used.size > 0:
        sc = ax.scatter(nominal_points_used[:, 0], nominal_points_used[:, 1], c=norm_vals, cmap='plasma', s=42, alpha=0.95, edgecolors='white', linewidths=0.4)
        ax.plot(nominal_points_used[:, 0], nominal_points_used[:, 1], '--', color='orange', linewidth=1.3, alpha=0.8, label='Nominal (sampled)')
    else:
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=norm_vals, cmap='plasma', s=42, alpha=0.95, edgecolors='white', linewidths=0.4)

    if cause_xyz is not None:
        ax.scatter(cause_xyz[0], cause_xyz[1], marker='*', s=320, color='red', edgecolors='white', linewidths=1.2, label=f'Cause: {cause or "Unknown"}')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2D Points: Normalized Disturbance (XY)')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label('Normalized disturbance')

    fig.tight_layout()
    return fig


def plot_3d_isosurfaces(mean_field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ax, num_levels: int = 5):
    """Draw translucent isosurfaces into provided 3D axis using marching cubes (if available), without mesh edges."""
    if not _HAS_SKIMAGE:
        raise RuntimeError("skimage not available for isosurface rendering")

    Ny, Nx, Nz = len(ys), len(xs), len(zs)
    field_3d = mean_field.reshape(Ny, Nx, Nz)
    volume = np.transpose(field_3d, (2, 0, 1))

    # More levels for richer isolines
    levels = np.percentile(mean_field, [60, 72, 84, 92, 97])[:num_levels]

    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    dz = zs[1] - zs[0] if len(zs) > 1 else 1.0

    xmin, ymin, zmin = xs.min(), ys.min(), zs.min()

    colors = ['#2196F3', '#4CAF50', '#FFC107', '#FF9800', '#F44336']
    alphas = [0.2, 0.22, 0.22, 0.22, 0.22]

    for level, color, alpha in zip(levels, colors, alphas):
        try:
            verts, faces, normals, values = _sk_measure.marching_cubes(volume=volume, level=level, spacing=(dz, dy, dx))
            verts_world = np.column_stack([
                verts[:, 2] + xmin,
                verts[:, 1] + ymin,
                verts[:, 0] + zmin,
            ])
            ax.plot_trisurf(verts_world[:, 0], verts_world[:, 1], faces, verts_world[:, 2], color=color, lw=0.0, edgecolor='none', alpha=alpha)
        except Exception as e:
            print(f"Isosurface at level {level:.4f} failed: {e}")


def plot_3d_pointcloud(mean_field: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, Zg: np.ndarray, ax, max_points: int = MAX_POINTS_POINTCLOUD):
    """Colored 3D point cloud (downsampled) showing the scalar field distribution, normalized for contrast."""
    X = Xg.ravel(); Y = Yg.ravel(); Z = Zg.ravel(); V = mean_field.ravel()
    N = X.shape[0]
    if N > max_points:
        idx = np.random.RandomState(42).choice(N, size=max_points, replace=False)
        X, Y, Z, V = X[idx], Y[idx], Z[idx], V[idx]

    Vn, vmin, vmax = _normalize_percentile(V, *PERCENTILE_RANGE_3D)
    sc = ax.scatter(X, Y, Z, c=Vn, cmap='viridis', s=2, alpha=POINT_CLOUD_ALPHA)
    return sc


def plot_3d_volume_with_overlays(Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_xyz, cause=None, use_isosurfaces: bool = True):
    """Compose a single 3D view: isosurfaces (or point cloud) + trajectory + cause marker."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if use_isosurfaces and _HAS_SKIMAGE:
        plot_3d_isosurfaces(mean_field, xs, ys, zs, ax=ax, num_levels=5)
    else:
        sc = plot_3d_pointcloud(mean_field.reshape(Xg.shape), Xg, Yg, Zg, ax=ax, max_points=MAX_POINTS_POINTCLOUD)

    ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='blue', linewidth=2.0, alpha=0.9, label='Actual trajectory')

    if cause_xyz is not None:
        ax.scatter(cause_xyz[0], cause_xyz[1], cause_xyz[2], marker='*', s=300, color='red', edgecolors='white', linewidths=1.2, label=f'Cause: {cause or "Unknown"}')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D GP Volume with Trajectory and Cause (Normalized)')
    ax.legend(loc='upper right')

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zs.min(), zs.max())

    mappable = None
    for coll in ax.collections:
        if hasattr(coll, 'get_array') and coll.get_array() is not None:
            mappable = coll
            break
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=12, pad=0.1)
        cbar.set_label('Normalized disturbance')

    fig.tight_layout()
    return fig


def plot_3d_pyvista_volume(xs, ys, zs, mean_field, xyz, cause_xyz, cause=None):
    """High-performance 3D rendering using PyVista (VTK): more isolines w/o edges + trajectory + cause."""
    if not _HAS_PYVISTA:
        raise RuntimeError("PyVista not available")

    Ny, Nx, Nz = len(ys), len(xs), len(zs)
    volume = mean_field.reshape(Ny, Nx, Nz)

    theme = _pv_themes.DefaultTheme()
    theme.cmap = 'viridis'
    theme.colorbar_orientation = 'vertical'

    plotter = _pv.Plotter(window_size=(1024, 768), theme=theme)

    grid = _pv.UniformGrid()
    grid.dimensions = np.array(volume.shape) + 1
    spacing = (ys[1] - ys[0] if Ny > 1 else 1.0, xs[1] - xs[0] if Nx > 1 else 1.0, zs[1] - zs[0] if Nz > 1 else 1.0)
    origin = (ys.min(), xs.min(), zs.min())
    grid.spacing = spacing
    grid.origin = origin
    grid.cell_data['disturbance'] = volume.ravel(order='F')

    Vn, vmin, vmax = _normalize_percentile(mean_field, *PERCENTILE_RANGE_3D)
    levels = np.percentile(Vn, [60, 72, 84, 92, 97])
    raw_levels = vmin + levels * (vmax - vmin)

    try:
        contour = grid.contour(isosurfaces=list(raw_levels), scalars='disturbance')
        plotter.add_mesh(contour, opacity=0.24, cmap='viridis', show_edges=False)
    except Exception as e:
        print(f"PyVista contour failed: {e}; falling back to volume rendering")
        plotter.add_volume(grid, scalars='disturbance', opacity='sigmoid', cmap='viridis', shade=True)

    traj = _pv.Spline(xyz, len(xyz)) if xyz.shape[0] >= 2 else _pv.PolyData(xyz)
    plotter.add_mesh(traj, color='blue', line_width=3, label='Actual trajectory')

    if cause_xyz is not None:
        cause_pt = _pv.Sphere(radius=0.03, center=cause_xyz)
        plotter.add_mesh(cause_pt, color='red')

    plotter.add_axes(interactive=False)
    plotter.show_bounds(grid='front', location='outer', ticks='outside')
    plotter.add_scalar_bar(title='Disturbance', n_labels=4)
    plotter.set_background('white')

    print("Launching PyVista interactive window... Close the window to continue.")
    plotter.show()


def plot_gp_orthogonal_views(xs, ys, zs, mean_field, xyz, cause_xyz):
    """Plot orthogonal GP slices: XY (mid Z), YZ (mid X), ZX (mid Y) in one window, with trajectory and cause overlays."""
    Ny, Nx, Nz = len(ys), len(xs), len(zs)
    volume = mean_field.reshape(Ny, Nx, Nz)

    kz = Nz // 2
    kx = Nx // 2
    ky = Ny // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY slice at mid Z
    xy_slice = volume[:, :, kz]
    xy_norm, _, _ = _normalize_percentile(xy_slice, *PERCENTILE_RANGE_3D)
    im0 = axes[0].imshow(
        xy_norm,
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        origin='lower',
        aspect='equal',
        cmap='viridis'
    )
    axes[0].plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=1.5, alpha=0.8)
    if cause_xyz is not None:
        axes[0].scatter(cause_xyz[0], cause_xyz[1], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
    axes[0].set_xlim(xs.min(), xs.max())
    axes[0].set_ylim(ys.min(), ys.max())
    axes[0].set_title('GP slice XY (z = mid)')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # YZ slice at mid X (Z on x-axis, Y on y-axis)
    yz_slice = volume[:, kx, :]
    yz_norm, _, _ = _normalize_percentile(yz_slice, *PERCENTILE_RANGE_3D)
    im1 = axes[1].imshow(
        yz_norm,
        extent=(zs.min(), zs.max(), ys.min(), ys.max()),
        origin='lower',
        aspect='equal',
        cmap='viridis'
    )
    axes[1].plot(xyz[:, 2], xyz[:, 1], 'b-', linewidth=1.5, alpha=0.8)
    if cause_xyz is not None:
        axes[1].scatter(cause_xyz[2], cause_xyz[1], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
    axes[1].set_xlim(zs.min(), zs.max())
    axes[1].set_ylim(ys.min(), ys.max())
    axes[1].set_title('GP slice YZ (x = mid)')
    axes[1].set_xlabel('Z (m)')
    axes[1].set_ylabel('Y (m)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # ZX slice at mid Y (transpose so Z is vertical axis)
    zx_slice = volume[ky, :, :].T
    zx_norm, _, _ = _normalize_percentile(zx_slice, *PERCENTILE_RANGE_3D)
    im2 = axes[2].imshow(
        zx_norm,
        extent=(xs.min(), xs.max(), zs.min(), zs.max()),
        origin='lower',
        aspect='equal',
        cmap='viridis'
    )
    axes[2].plot(xyz[:, 0], xyz[:, 2], 'b-', linewidth=1.5, alpha=0.8)
    if cause_xyz is not None:
        axes[2].scatter(cause_xyz[0], cause_xyz[2], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
    axes[2].set_xlim(xs.min(), xs.max())
    axes[2].set_ylim(zs.min(), zs.max())
    axes[2].set_title('GP slice ZX (y = mid)')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Z (m)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def main():
    print("Loading buffer data...")
    xyz, cause, cause_xyz = load_buffer_xyz_drift(BUFFER_DIR)
    nominal_xyz = load_nominal_xyz(NOMINAL_PATH)

    print(f"Loaded {len(xyz)} trajectory points")
    print(f"Cause: {cause}")
    print(f"Cause location: {cause_xyz}")

    # NEW: CLI flags for kernel selection
    try:
        import argparse as _argparse
        parser = _argparse.ArgumentParser(description="3D GP disturbance field with kernel learning")
        parser.add_argument('--kernel', type=str, default='auto', help="Kernel mode: auto|rbf|matern12|matern32|matern52|rq|original")
        parser.add_argument('--cv-folds', type=int, default=0, help="If >1, use K-fold CV to select kernel (slower)")
        parser.add_argument('--alpha', type=float, default=1e-6, help="Gaussian noise added to the diagonal of the kernel matrix")
        parser.add_argument('--restarts', type=int, default=8, help="Number of optimizer restarts for kernel hyperparameters")
        parser.add_argument('--normalize-y', action='store_true', help="Normalize targets in GP")
        args, _unknown = parser.parse_known_args()
    except Exception:
        class _Args:
            kernel = 'auto'
            cv_folds = 0
            alpha = 1e-6
            restarts = 8
            normalize_y = False
        args = _Args()

    clipped_nominal_xyz = None
    if nominal_xyz is not None:
        clipped_nominal_xyz = clip_nominal_to_actual_segment(nominal_xyz, xyz, plane='xy')
        if clipped_nominal_xyz is not None and len(clipped_nominal_xyz) > 0:
            print(f"Clipped nominal from {len(nominal_xyz)} to {len(clipped_nominal_xyz)} points using XY plane")
        else:
            clipped_nominal_xyz = nominal_xyz

    if clipped_nominal_xyz is not None:
        nominal_points_used, disturbance_magnitudes = compute_disturbance_at_nominal_points(clipped_nominal_xyz, xyz, cause_xyz)
        if nominal_points_used.size == 0:
            print("Warning: No nominal points were used for disturbance computation.")
            drift_vectors, drift_magnitudes = compute_trajectory_drift_vectors(xyz, clipped_nominal_xyz)
            if drift_vectors is None:
                print("Warning: Could not compute drift from nominal trajectory; using zeros")
                disturbance_magnitudes = np.zeros(len(clipped_nominal_xyz))
                nominal_points_used = clipped_nominal_xyz
            else:
                disturbance_magnitudes = drift_magnitudes
                nominal_points_used = clipped_nominal_xyz
    else:
        print("Warning: No nominal trajectory available; using zeros")
        disturbance_magnitudes = np.zeros(len(xyz))
        nominal_points_used = xyz

    print(f"Disturbance magnitudes - min: {disturbance_magnitudes.min():.4f}, max: {disturbance_magnitudes.max():.4f}")
    print(f"Using {len(nominal_points_used)} nominal points for GP training")

    print("Creating cause-centric features for nominal points...")
    features, final_cause_xyz = create_cause_centric_features_nominal(nominal_points_used, cause_xyz)

    print("Fitting Gaussian Process (3D)...")
    try:
        gp = fit_gp_with_kernel_learning(
            features,
            disturbance_magnitudes,
            kernel_mode=args.kernel,
            n_restarts_optimizer=args.restarts,
            alpha=args.alpha,
            cv_folds=args.cv_folds,
            normalize_y=args.normalize_y,
        )
    except Exception as e:
        print(f"Kernel learning failed ({e}); falling back to legacy GP")
        gp = fit_cause_centric_gp(features, disturbance_magnitudes)

    print("Creating 3D prediction grid...")
    Xg, Yg, Zg, grid_points, xs, ys, zs = create_3d_prediction_grid(xyz, final_cause_xyz, resolution_xy=0.06, resolution_z=0.06)

    print("Predicting 3D disturbance field...")
    mean_pred, std_pred = predict_gp_field_3d(gp, grid_points, final_cause_xyz)

    print("Rendering 2D points visualization (XY)...")
    fig_2d = plot_2d_points(xyz, nominal_points_used, disturbance_magnitudes, final_cause_xyz, cause)

    print("Rendering orthogonal GP slices (XY, YZ, ZX)...")
    fig_slices = plot_gp_orthogonal_views(xs, ys, zs, mean_pred, xyz, final_cause_xyz)

    used_pyvista = False
    if USE_PYVISTA and _HAS_PYVISTA:
        try:
            print("Rendering 3D (PyVista)...")
            plot_3d_pyvista_volume(xs, ys, zs, mean_pred, xyz, final_cause_xyz, cause)
            used_pyvista = True
        except Exception as e:
            print(f"PyVista rendering failed: {e}. Falling back to Matplotlib.")

    if not used_pyvista:
        print("Rendering 3D (Matplotlib)...")
        fig_3d = plot_3d_volume_with_overlays(Xg, Yg, Zg, mean_pred, xs, ys, zs, xyz, final_cause_xyz, cause, use_isosurfaces=True)
        plt.show()
    else:
        plt.show()

    print("3D GP analysis complete!")


if __name__ == "__main__":
    main() 
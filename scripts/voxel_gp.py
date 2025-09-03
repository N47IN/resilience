#!/usr/bin/env python3
"""
3D Direct Disturbance Field Modeling with Superposed Anisotropic Kernels

- Loads buffer poses and cause location
- Clips nominal trajectory to actual segment (same as before)
- Computes disturbance (including Z/vertical deviations)
- Directly fits a superposed anisotropic kernel model to observed disturbances:
  * A * sum_j exp(-0.5 * Q_l(x - c_j)) + b
  * Where Q_l is anisotropic distance metric with shared parameters lxy, lz
  * Each cause point c_j contributes identically to the field
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import minimize

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

# Optional: Open3D for PCD loading
try:
    import open3d as _o3d
    _HAS_OPEN3D = True
except Exception:
    _HAS_OPEN3D = False

BUFFER_DIR = "/home/navin/ros2_ws/src/buffers/run_20250828_093052_931_2e959586/buffer2"
NOMINAL_PATH = "/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json"
PCD_PATH = "/home/navin/ros2_ws/src/buffers/run_20250828_093052_931_2e959586/semantic_voxels_20250828_093113.pcd"

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


# Removed: create_cause_centric_features_nominal and fit_cause_centric_gp functions
# These are no longer needed since we eliminated the global GP step


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


# Removed: predict_gp_field_3d function
# This is no longer needed since we eliminated the global GP step


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


def plot_3d_isosurfaces(mean_field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ax, num_levels: int = 8):
    """Draw translucent isosurfaces into provided 3D axis using marching cubes (if available), without mesh edges."""
    if not _HAS_SKIMAGE:
        raise RuntimeError("skimage not available for isosurface rendering")

    Ny, Nx, Nz = len(ys), len(xs), len(zs)
    field_3d = mean_field.reshape(Ny, Nx, Nz)
    volume = np.transpose(field_3d, (2, 0, 1))

    # Compute levels based on actual data range with more granularity
    field_min, field_max = mean_field.min(), mean_field.max()
    field_range = field_max - field_min
    
    if field_range < 1e-8:
        print("Warning: Field has very low variance, skipping isosurfaces")
        return
    
    # Use more percentiles for better visualization
    percentiles = np.linspace(50, 98, num_levels)
    levels = np.percentile(mean_field, percentiles)
    
    # Ensure levels are within the volume range with some padding
    levels = np.clip(levels, field_min + 0.005 * field_range, field_max - 0.005 * field_range)

    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    dz = zs[1] - zs[0] if len(zs) > 1 else 1.0

    xmin, ymin, zmin = xs.min(), ys.min(), zs.min()

    # More varied colors and alphas for better visualization
    colors = ['#2196F3', '#4CAF50', '#FFC107', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548']
    alphas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]

    print(f"Rendering {len(levels)} isosurfaces at levels: {levels}")
    
    for i, (level, color, alpha) in enumerate(zip(levels, colors, alphas)):
        try:
            verts, faces, normals, values = _sk_measure.marching_cubes(volume=volume, level=level, spacing=(dz, dy, dx))
            if len(verts) > 0:
                verts_world = np.column_stack([
                    verts[:, 2] + xmin,
                    verts[:, 1] + ymin,
                    verts[:, 0] + zmin,
                ])
                ax.plot_trisurf(verts_world[:, 0], verts_world[:, 1], faces, verts_world[:, 2], 
                              color=color, lw=0.0, edgecolor='none', alpha=alpha)
                print(f"  Isosurface {i+1}: level={level:.4f}, vertices={len(verts)}")
            else:
                print(f"  Isosurface {i+1}: level={level:.4f} - no vertices found")
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

# New: 3D visualization of superposed field with cause points (Matplotlib)

def plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces: bool = True, max_cause_points: int = 5000):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if use_isosurfaces and _HAS_SKIMAGE:
        plot_3d_isosurfaces(mean_field, xs, ys, zs, ax=ax, num_levels=5)
    else:
        sc = plot_3d_pointcloud(mean_field.reshape(Xg.shape), Xg, Yg, Zg, ax=ax, max_points=MAX_POINTS_POINTCLOUD)

    ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='blue', linewidth=2.0, alpha=0.9, label='Actual trajectory')

    # Overlay cause points (downsample for clarity)
    if cause_points is not None and cause_points.size > 0:
        pts = cause_points
        if pts.shape[0] > max_cause_points:
            idx = np.random.RandomState(42).choice(pts.shape[0], size=max_cause_points, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, c='red', alpha=0.6, edgecolors='none', label='Cause points')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Reconstructed 3D Field (Superposed) + Cause Points')
    ax.legend(loc='upper right')

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zs.min(), zs.max())

    # Colorbar if available
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

# New: PyVista version for reconstructed field with cause points

def plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_field, xyz, cause_points):
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
    # More isosurfaces for better visualization
    levels = np.percentile(Vn, [55, 65, 75, 82, 88, 93, 96, 98])
    raw_levels = vmin + levels * (vmax - vmin)

    print(f"PyVista: Rendering {len(raw_levels)} isosurfaces at levels: {raw_levels}")

    try:
        contour = grid.contour(isosurfaces=list(raw_levels), scalars='disturbance')
        plotter.add_mesh(contour, opacity=0.20, cmap='viridis', show_edges=False)
    except Exception as e:
        print(f"PyVista contour failed: {e}; falling back to volume rendering")
        plotter.add_volume(grid, scalars='disturbance', opacity='sigmoid', cmap='viridis', shade=True)

    traj = _pv.Spline(xyz, len(xyz)) if xyz.shape[0] >= 2 else _pv.PolyData(xyz)
    plotter.add_mesh(traj, color='blue', line_width=3, label='Actual trajectory')

    if cause_points is not None and cause_points.size > 0:
        cloud = _pv.PolyData(cause_points)
        plotter.add_mesh(cloud, color='red', point_size=6, render_points_as_spheres=True, opacity=0.6)

    plotter.add_axes(interactive=False)
    plotter.show_bounds(grid='front', location='outer', ticks='outside')
    plotter.add_scalar_bar(title='Disturbance', n_labels=4)
    plotter.set_background('white')

    print("Launching PyVista reconstructed window... Close the window to continue.")
    plotter.show()

# ----------------------------
# Extended-cause modeling utils
# ----------------------------

def _load_pcd_points(pcd_path: str) -> np.ndarray:
    """Load points from a PCD file. Uses Open3D when available; otherwise a minimal ASCII PCD parser.
    Returns an (N,3) numpy array. If file missing or empty, returns empty array.
    """
    p = Path(pcd_path)
    if not p.exists():
        print(f"Warning: PCD not found at {pcd_path}")
        return np.empty((0, 3), dtype=float)
    # Try Open3D first
    if _HAS_OPEN3D:
        try:
            pc = _o3d.io.read_point_cloud(str(p))
            pts = np.asarray(pc.points, dtype=float)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                return pts[:, :3]
        except Exception as e:
            print(f"Open3D failed to read PCD: {e}; attempting ASCII parse")
    # Fallback ASCII parse (simple, supports common PCD ASCII format)
    try:
        with open(p, 'r') as f:
            header = True
            fields = []
            data_started = False
            pts = []
            for line in f:
                line = line.strip()
                if header:
                    if line.startswith('FIELDS'):
                        fields = line.split()[1:]
                    if line.startswith('DATA'):
                        data_started = True
                        header = False
                    continue
                if data_started and line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
                            pts.append((x, y, z))
                        except Exception:
                            pass
        if len(pts) == 0:
            print("Warning: No points parsed from PCD")
            return np.empty((0, 3), dtype=float)
        return np.array(pts, dtype=float)
    except Exception as e:
        print(f"Failed to parse PCD: {e}")
        return np.empty((0, 3), dtype=float)


def _sum_of_anisotropic_rbf(grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
    """Compute phi(x) = sum_j exp(-0.5 * [((dx/lxy)^2 + (dy/lxy)^2 + (dz/lz)^2)] ) for all grid points.
    Returns vector of length len(grid_points).
    Computed in chunks for memory efficiency.
    """
    if centers.size == 0:
        return np.zeros(grid_points.shape[0], dtype=float)
    num_points = grid_points.shape[0]
    phi = np.zeros(num_points, dtype=float)
    chunk = 200000  # adjust for memory
    inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
    inv_lz2 = 1.0 / (lz * lz + 1e-12)
    for start in range(0, num_points, chunk):
        end = min(num_points, start + chunk)
        gp_chunk = grid_points[start:end]
        # Broadcast centers over chunk
        dx = gp_chunk[:, None, 0] - centers[None, :, 0]
        dy = gp_chunk[:, None, 1] - centers[None, :, 1]
        dz = gp_chunk[:, None, 2] - centers[None, :, 2]
        d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
        np.exp(-0.5 * d2, out=d2)
        # Sum over centers
        phi[start:end] = d2.sum(axis=1)
    return phi


# Removed: fit_extended_cause_superposition function
# This is no longer needed since we eliminated the global GP step and now fit directly to disturbances


def fit_direct_superposition_to_disturbances(nominal_points: np.ndarray, disturbance_magnitudes: np.ndarray, cause_points: np.ndarray):
    """Directly fit superposed identical anisotropic kernel to observed disturbances.
    
    Fits A * sum_j exp(-0.5 * Q_l(x - c_j)) + b to match the actual disturbance measurements
    at nominal trajectory points, where Q_l is the anisotropic distance metric.
    
    Returns dict with best params and the reconstructed disturbance field.
    """
    if cause_points.size == 0:
        print("Warning: No cause points to fit direct model; skipping.")
        return {
            'lxy': None,
            'lz': None,
            'A': 0.0,
            'b': 0.0,
            'recon': np.zeros_like(disturbance_magnitudes),
            'mse': float('inf'),
            'r2_score': 0.0,
            'mae': float('inf'),
            'rmse': float('inf')
        }

    target = disturbance_magnitudes.astype(float)
    
    # Normalize target to have better numerical properties
    target_mean = np.mean(target)
    target_std = np.std(target)
    if target_std < 1e-8:
        print("Warning: Target disturbances have very low variance, fitting may be unstable")
        target_std = 1.0
    target_norm = (target - target_mean) / target_std
    
    def objective(params):
        """Objective function: MSE between target disturbances and reconstructed field."""
        lxy, lz = params
        # Ensure positive length scales
        lxy = max(lxy, 0.01)
        lz = max(lz, 0.01)
        
        phi = _sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
        
        # Normalize phi for better conditioning
        phi_mean = np.mean(phi)
        phi_std = np.std(phi)
        if phi_std < 1e-8:
            return float('inf')  # Avoid singular cases
        phi_norm = (phi - phi_mean) / phi_std
        
        # Closed-form solution for A and b
        n = phi_norm.shape[0]
        X = np.column_stack([phi_norm, np.ones(n, dtype=float)])
        try:
            XtX = X.T @ X
            XtY = X.T @ target_norm
            params_ab = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            params_ab = np.linalg.lstsq(X, target_norm, rcond=None)[0]
        
        A_norm, b_norm = params_ab[0], params_ab[1]
        recon_norm = A_norm * phi_norm + b_norm
        mse = np.mean((recon_norm - target_norm) ** 2)
        
        # Add regularization to prefer reasonable length scales
        reg_term = 0.05 * (1.0 / (lxy + 0.05) + 1.0 / (lz + 0.05))
        return mse + reg_term
    
    # More comprehensive initial guesses covering different scales
    initial_guesses = [
        [0.02, 0.02],   # Very small scales
        [0.05, 0.05],   # Small scales
        [0.1, 0.1],     # Medium scales
        [0.2, 0.2],     # Large scales
        [0.3, 0.3],     # Very large scales
        [0.1, 0.05],    # Anisotropic (xy > z)
        [0.05, 0.1],    # Anisotropic (z > xy)
        [0.15, 0.08],   # Anisotropic (xy > z)
        [0.08, 0.15],   # Anisotropic (z > xy)
    ]
    
    # Wider bounds for better exploration
    bounds = [(0.005, 1.0), (0.005, 1.0)]
    
    print("Direct optimization: fitting superposed model to observed disturbances...")
    print(f"Target disturbances - min: {target.min():.6f}, max: {target.max():.6f}, mean: {target_mean:.6f}, std: {target_std:.6f}")
    print(f"Number of cause points: {cause_points.shape[0]}")
    print(f"Number of nominal points: {nominal_points.shape[0]}")
    
    best_result = None
    best_mse = float('inf')
    
    for i, x0 in enumerate(initial_guesses):
        print(f"  Trying initial guess {i+1}/{len(initial_guesses)}: lxy={x0[0]:.3f}, lz={x0[1]:.3f}")
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < best_mse:
                best_result = result
                best_mse = result.fun
                print(f"    New best: MSE={result.fun:.8f}, lxy={result.x[0]:.4f}, lz={result.x[1]:.4f}")
            
        except Exception as e:
            print(f"    Failed: {e}")
    
    if best_result is None:
        print("All optimization attempts failed. Falling back to grid search.")
        # More comprehensive grid search
        lxy_grid = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5], dtype=float)
        lz_grid = np.array([0.01, 0.02, 0.04, 0.06, 0.10, 0.16, 0.24, 0.35, 0.5], dtype=float)
        
        best = {'mse': float('inf')}
        
        for lxy in lxy_grid:
            for lz in lz_grid:
                phi = _sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
                n = phi.shape[0]
                X = np.column_stack([phi, np.ones(n, dtype=float)])
                try:
                    XtX = X.T @ X
                    XtY = X.T @ target
                    params = np.linalg.solve(XtX, XtY)
                except np.linalg.LinAlgError:
                    params = np.linalg.lstsq(X, target, rcond=None)[0]
                A, b = float(params[0]), float(params[1])
                recon = A * phi + b
                mse = float(np.mean((recon - target) ** 2))
                if mse < best['mse']:
                    best = {'lxy': lxy, 'lz': lz, 'A': A, 'b': b, 'recon': recon, 'mse': mse}
        
        print(f"Grid search fallback -> lxy: {best['lxy']:.4f} m, lz: {best['lz']:.4f} m, A: {best['A']:.6f}, b: {best['b']:.6f}, MSE: {best['mse']:.8f}")
        return best
    
    # Extract optimal parameters
    lxy_opt, lz_opt = best_result.x
    phi_opt = _sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy_opt, lz=lz_opt)
    
    # Final closed-form solution for A and b (using original scale)
    n = phi_opt.shape[0]
    X = np.column_stack([phi_opt, np.ones(n, dtype=float)])
    try:
        XtX = X.T @ X
        XtY = X.T @ target
        params_ab = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        params_ab = np.linalg.lstsq(X, target, rcond=None)[0]
    
    A_opt, b_opt = float(params_ab[0]), float(params_ab[1])
    recon_opt = A_opt * phi_opt + b_opt
    
    # Comprehensive error metrics
    mse_opt = float(np.mean((recon_opt - target) ** 2))
    rmse_opt = float(np.sqrt(mse_opt))
    mae_opt = float(np.mean(np.abs(recon_opt - target)))
    
    # R-squared score
    ss_res = np.sum((target - recon_opt) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Optimal parameters:")
    print(f"  lxy: {lxy_opt:.4f} m")
    print(f"  lz:  {lz_opt:.4f} m")
    print(f"  A:   {A_opt:.6f}")
    print(f"  b:   {b_opt:.6f}")
    print(f"\nError metrics:")
    print(f"  MSE:  {mse_opt:.8f}")
    print(f"  RMSE: {rmse_opt:.6f}")
    print(f"  MAE:  {mae_opt:.6f}")
    print(f"  R²:   {r2_score:.4f}")
    print(f"\nOptimization details:")
    print(f"  Iterations: {best_result.nit}")
    print(f"  Function evaluations: {best_result.nfev}")
    print(f"  Success: {best_result.success}")
    print(f"  Message: {best_result.message}")
    print(f"\nField statistics:")
    print(f"  Target - mean: {target_mean:.6f}, std: {target_std:.6f}")
    print(f"  Reconstructed - mean: {np.mean(recon_opt):.6f}, std: {np.std(recon_opt):.6f}")
    print(f"  Residual - mean: {np.mean(recon_opt - target):.6f}, std: {np.std(recon_opt - target):.6f}")
    
    return {
        'lxy': lxy_opt,
        'lz': lz_opt,
        'A': A_opt,
        'b': b_opt,
        'recon': recon_opt,
        'mse': mse_opt,
        'rmse': rmse_opt,
        'mae': mae_opt,
        'r2_score': r2_score,
        'optimization_result': best_result
    }


def predict_direct_field_3d(fit_params: dict, grid_points: np.ndarray, cause_points: np.ndarray):
    """Predict disturbance field on 3D grid using the directly fitted superposed model."""
    if fit_params is None or 'lxy' not in fit_params or fit_params['lxy'] is None:
        return np.zeros(grid_points.shape[0]), np.zeros(grid_points.shape[0])
    
    lxy = fit_params['lxy']
    lz = fit_params['lz']
    A = fit_params['A']
    b = fit_params['b']
    
    phi = _sum_of_anisotropic_rbf(grid_points, cause_points, lxy=lxy, lz=lz)
    mean_pred = A * phi + b
    
    # For simplicity, we'll use a constant uncertainty estimate
    # In a more sophisticated approach, you could model the uncertainty based on distance to training points
    std_pred = np.full(grid_points.shape[0], 0.1 * np.std(mean_pred))
    
    return mean_pred, std_pred


def main():
    print("Loading buffer data...")
    xyz, cause, cause_xyz = load_buffer_xyz_drift(BUFFER_DIR)
    nominal_xyz = load_nominal_xyz(NOMINAL_PATH)

    print(f"Loaded {len(xyz)} trajectory points")
    print(f"Cause: {cause}")
    print(f"Cause location: {cause_xyz}")

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
    print(f"Using {len(nominal_points_used)} nominal points for direct fitting")

    # ----------------------------
    # Direct fitting to disturbances (eliminating global GP)
    # ----------------------------
    print("Loading cause object point cloud (PCD)...")
    cause_points = _load_pcd_points(PCD_PATH)
    print(f"Loaded {cause_points.shape[0]} cause points from PCD")

    direct_fit = None
    if cause_points.size > 0:
        # Directly fit superposed model to observed disturbances
        print("Directly fitting superposed identical-kernel model to observed disturbances...")
        direct_fit = fit_direct_superposition_to_disturbances(nominal_points_used, disturbance_magnitudes, cause_points)
        
        # Quick diagnostic: compare fitted vs observed disturbances
        try:
            fig_direct, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Scatter plot of fitted vs observed
            axes[0,0].scatter(disturbance_magnitudes, direct_fit['recon'], alpha=0.7, s=30)
            axes[0,0].plot([disturbance_magnitudes.min(), disturbance_magnitudes.max()], 
                        [disturbance_magnitudes.min(), disturbance_magnitudes.max()], 'r--', alpha=0.8, linewidth=2)
            axes[0,0].set_xlabel('Observed disturbances')
            axes[0,0].set_ylabel('Fitted disturbances')
            axes[0,0].set_title(f'Direct Fit: Observed vs Fitted (R² = {direct_fit["r2_score"]:.4f})')
            axes[0,0].grid(True, alpha=0.3)
            
            # Residual plot
            residuals = direct_fit['recon'] - disturbance_magnitudes
            axes[0,1].scatter(direct_fit['recon'], residuals, alpha=0.7, s=30)
            axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
            axes[0,1].set_xlabel('Fitted disturbances')
            axes[0,1].set_ylabel('Residuals (fitted - observed)')
            axes[0,1].set_title(f'Residual Plot (MAE = {direct_fit["mae"]:.6f})')
            axes[0,1].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1,0].hist(residuals, bins=10, alpha=0.7, edgecolor='black')
            axes[1,0].axvline(x=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
            axes[1,0].set_xlabel('Residuals')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title(f'Residual Distribution (std = {np.std(residuals):.6f})')
            axes[1,0].grid(True, alpha=0.3)
            
            # Parameter summary
            axes[1,1].text(0.1, 0.9, f'Optimal Parameters:', fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.8, f'lxy = {direct_fit["lxy"]:.4f} m', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.7, f'lz = {direct_fit["lz"]:.4f} m', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.6, f'A = {direct_fit["A"]:.6f}', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.5, f'b = {direct_fit["b"]:.6f}', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.4, f'', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.3, f'Error Metrics:', fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.2, f'MSE = {direct_fit["mse"]:.8f}', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.1, f'RMSE = {direct_fit["rmse"]:.6f}', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.0, f'R² = {direct_fit["r2_score"]:.4f}', fontsize=11, transform=axes[1,1].transAxes)
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].set_title('Model Summary')
            axes[1,1].axis('off')
            
            fig_direct.tight_layout()
            
        except Exception as e:
            print(f"Direct fit diagnostic visualization failed: {e}")
    else:
        print("Skipping direct fitting (no PCD points).")

    print("Creating 3D prediction grid...")
    Xg, Yg, Zg, grid_points, xs, ys, zs = create_3d_prediction_grid(xyz, cause_xyz, resolution_xy=0.06, resolution_z=0.06)

    print("Predicting 3D disturbance field using direct fit...")
    if direct_fit is not None:
        mean_pred, std_pred = predict_direct_field_3d(direct_fit, grid_points, cause_points)
    else:
        # Fallback: use zeros if no fit available
        mean_pred = np.zeros(grid_points.shape[0])
        std_pred = np.zeros(grid_points.shape[0])

    print("Rendering 2D points visualization (XY)...")
    fig_2d = plot_2d_points(xyz, nominal_points_used, disturbance_magnitudes, cause_xyz, cause)

    print("Rendering orthogonal slices (XY, YZ, ZX)...")
    fig_slices = plot_gp_orthogonal_views(xs, ys, zs, mean_pred, xyz, cause_xyz)

    # 3D figures:
    # 1) Direct fit 3D view
    used_pyvista = False
    if USE_PYVISTA and _HAS_PYVISTA:
        try:
            print("Rendering 3D (PyVista) - Direct fit...")
            plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_pred, xyz, cause_points)
            used_pyvista = True
        except Exception as e:
            print(f"PyVista rendering failed (direct): {e}. Falling back to Matplotlib.")

    if not used_pyvista:
        print("Rendering 3D (Matplotlib) - Direct fit...")
        fig_3d_direct = plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_pred, xs, ys, zs, xyz, cause_points, use_isosurfaces=True)

    plt.show()

    print("Direct 3D disturbance field analysis complete!")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Disturbance Field Helper - Class-based utility for 3D disturbance modeling

This helper wraps the core functionality from `resilience/scripts/voxel_gp.py` into a
reusable class that can be imported in ROS2 nodes or other Python modules.

Primary capabilities:
- Load actual trajectory and cause metadata from a buffer directory
- Accept a nominal trajectory path or pass nominal directly
- Accept a point cloud (Nx3 numpy array) representing cause points (PCD content)
- Compute disturbances against nominal, fit a superposed anisotropic kernel model
- Predict a 3D field on a grid around the trajectory and cause
- Provide Matplotlib and optional PyVista visualizations

Inputs (main methods):
- pointcloud: numpy.ndarray of shape (N, 3)
- buffer_dir: directory containing `poses.npy` and `cause_location.json`/`metadata.json`
- nominal_path or nominal_xyz

Outputs:
- Fitted parameters (lxy, lz, A, b) and quality metrics
- Optional visualizations (2D scatter, orthogonal slices, 3D volume)

Dependency policy:
- Functions are directly adapted from `voxel_gp.py` to avoid cross-module script imports.
- All public APIs accept numpy arrays and paths; no ROS messages are required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Optional dependencies
try:
    from skimage import measure as _sk_measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import pyvista as _pv
    from pyvista import themes as _pv_themes
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

# ----------------------------
# Configuration defaults
# ----------------------------

_DEFAULTS = {
    'resolution_xy': 0.06,
    'resolution_z': 0.06,
    'pad_bounds': 0.5,
    'percentile_range_2d': (5, 95),
    'percentile_range_3d': (5, 97),
    'max_points_pointcloud': 50000,
    'point_cloud_alpha': 0.28,
}


class DisturbanceFieldHelper:
    """
    Helper for computing and visualizing 3D disturbance fields using a superposed
    anisotropic kernel model, refactored from `voxel_gp.py` for clean reuse.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update(config)
        self.cfg = cfg

    # ----------------------------
    # Data loading utilities
    # ----------------------------

    @staticmethod
    def load_buffer_xyz_drift(buffer_dir: str) -> Tuple[np.ndarray, Optional[str], Optional[np.ndarray]]:
        """Load actual trajectory xyz and cause metadata from a buffer directory.
        Returns (xyz, cause, cause_xyz).
        """
        buffer_path = Path(buffer_dir)
        poses_path = buffer_path / "poses.npy"
        meta_path = buffer_path / "metadata.json"
        cause_loc_path = buffer_path / "cause_location.json"

        if not poses_path.exists():
            raise FileNotFoundError(f"poses.npy missing in {buffer_dir}")

        poses = np.load(poses_path)
        xyz = poses[:, 1:4]

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

    @staticmethod
    def load_nominal_xyz(nominal_path: str) -> Optional[np.ndarray]:
        """Load nominal trajectory from JSON format used in assets.
        Accepts dict with `points` -> `position` (x,y,z) or list of lists.
        """
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
                        xyz_list.append([float(pos["x"]), float(pos["y"]), float(pos["z"])] )
                if xyz_list:
                    return np.array(xyz_list, dtype=float)
            else:
                arr = np.array(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3]
        return None

    # ----------------------------
    # Core computations (adapted)
    # ----------------------------

    @staticmethod
    def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = 'xy') -> np.ndarray:
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

    @staticmethod
    def compute_trajectory_drift_vectors(actual_xyz: np.ndarray, nominal_xyz: np.ndarray):
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

    @staticmethod
    def compute_disturbance_at_nominal_points(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute disturbance at nominal trajectory points near the actual trajectory bounds.
        Signature mirrors voxel_gp.py (cause_xyz unused but kept for API parity).
        """
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
            return np.array([]), np.array([])
        for nominal_point in relevant_nominal:
            distances = np.linalg.norm(actual_xyz - nominal_point, axis=1)
            closest_idx = np.argmin(distances)
            closest_actual = actual_xyz[closest_idx]
            disturbance = np.linalg.norm(closest_actual - nominal_point)
            if distances[closest_idx] < 0.3:
                disturbances.append(disturbance)
                nominal_points_used.append(nominal_point)
        return np.array(nominal_points_used), np.array(disturbances)

    @staticmethod
    def _sum_of_anisotropic_rbf(grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
        if centers.size == 0:
            return np.zeros(grid_points.shape[0], dtype=float)
        num_points = grid_points.shape[0]
        phi = np.zeros(num_points, dtype=float)
        chunk = 200000
        inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
        inv_lz2 = 1.0 / (lz * lz + 1e-12)
        for start in range(0, num_points, chunk):
            end = min(num_points, start + chunk)
            gp_chunk = grid_points[start:end]
            dx = gp_chunk[:, None, 0] - centers[None, :, 0]
            dy = gp_chunk[:, None, 1] - centers[None, :, 1]
            dz = gp_chunk[:, None, 2] - centers[None, :, 2]
            d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
            np.exp(-0.5 * d2, out=d2)
            phi[start:end] = d2.sum(axis=1)
        return phi

    def fit_direct_superposition_to_disturbances(self, nominal_points: np.ndarray, disturbance_magnitudes: np.ndarray, cause_points: np.ndarray) -> Dict[str, Any]:
        if cause_points.size == 0:
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
        target_mean = np.mean(target)
        target_std = np.std(target)
        if target_std < 1e-8:
            target_std = 1.0
        target_norm = (target - target_mean) / target_std

        def objective(params):
            lxy, lz = params
            lxy = max(lxy, 0.01)
            lz = max(lz, 0.01)
            phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
            phi_mean = np.mean(phi)
            phi_std = np.std(phi)
            if phi_std < 1e-8:
                return float('inf')
            phi_norm = (phi - phi_mean) / phi_std
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
            reg_term = 0.05 * (1.0 / (lxy + 0.05) + 1.0 / (lz + 0.05))
            return mse + reg_term

        initial_guesses = [
            [0.02, 0.02], [0.05, 0.05], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
            [0.1, 0.05], [0.05, 0.1], [0.15, 0.08], [0.08, 0.15],
        ]
        bounds = [(0.005, 1.0), (0.005, 1.0)]

        best_result = None
        best_mse = float('inf')
        for x0 in initial_guesses:
            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
                if result.success and result.fun < best_mse:
                    best_result = result
                    best_mse = result.fun
            except Exception:
                pass

        if best_result is None:
            # Grid search fallback
            lxy_grid = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5], dtype=float)
            lz_grid = np.array([0.01, 0.02, 0.04, 0.06, 0.10, 0.16, 0.24, 0.35, 0.5], dtype=float)
            best = {'mse': float('inf')}
            for lxy in lxy_grid:
                for lz in lz_grid:
                    phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
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
            return best

        lxy_opt, lz_opt = best_result.x
        phi_opt = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy_opt, lz=lz_opt)
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
        mse_opt = float(np.mean((recon_opt - target) ** 2))
        rmse_opt = float(np.sqrt(mse_opt))
        mae_opt = float(np.mean(np.abs(recon_opt - target)))
        ss_res = np.sum((target - recon_opt) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        return {
            'lxy': float(lxy_opt),
            'lz': float(lz_opt),
            'A': A_opt,
            'b': b_opt,
            'recon': recon_opt,
            'mse': mse_opt,
            'rmse': rmse_opt,
            'mae': mae_opt,
            'r2_score': r2_score,
            'optimization_result': best_result,
        }

    # ----------------------------
    # Grid and prediction
    # ----------------------------

    def create_3d_prediction_grid(self, xyz: np.ndarray, cause_xyz: Optional[np.ndarray], resolution_xy: Optional[float] = None, resolution_z: Optional[float] = None):
        pad = float(self.cfg['pad_bounds'])
        res_xy = float(resolution_xy or self.cfg['resolution_xy'])
        res_z = float(resolution_z or self.cfg['resolution_z'])
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
        xs = np.arange(xmin, xmax + res_xy, res_xy)
        ys = np.arange(ymin, ymax + res_xy, res_xy)
        zs = np.arange(zmin, zmax + res_z, res_z)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='xy')
        grid_points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
        return Xg, Yg, Zg, grid_points, xs, ys, zs

    def predict_direct_field_3d(self, fit_params: Dict[str, Any], grid_points: np.ndarray, cause_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if fit_params is None or 'lxy' not in fit_params or fit_params['lxy'] is None:
            return np.zeros(grid_points.shape[0]), np.zeros(grid_points.shape[0])
        lxy = float(fit_params['lxy'])
        lz = float(fit_params['lz'])
        A = float(fit_params['A'])
        b = float(fit_params['b'])
        phi = self._sum_of_anisotropic_rbf(grid_points, cause_points, lxy=lxy, lz=lz)
        mean_pred = A * phi + b
        std_pred = np.full(grid_points.shape[0], 0.1 * np.std(mean_pred))
        return mean_pred, std_pred

    # ----------------------------
    # Visualizations
    # ----------------------------

    def _normalize_percentile(self, values: np.ndarray, lower_pct: float, upper_pct: float):
        lo = np.percentile(values, lower_pct)
        hi = np.percentile(values, upper_pct)
        if hi <= lo:
            hi = lo + 1e-9
        v = np.clip(values, lo, hi)
        v = (v - lo) / (hi - lo)
        return v, lo, hi

    def plot_2d_points(self, xyz: np.ndarray, nominal_points_used: np.ndarray, disturbance_magnitudes: np.ndarray, cause_xyz: Optional[np.ndarray], cause: Optional[str] = None):
        pr = self.cfg['percentile_range_2d']
        norm_vals, _, _ = self._normalize_percentile(disturbance_magnitudes, *pr)
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

    def plot_gp_orthogonal_views(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, mean_field: np.ndarray, xyz: np.ndarray, cause_xyz: Optional[np.ndarray]):
        Ny, Nx, Nz = len(ys), len(xs), len(zs)
        volume = mean_field.reshape(Ny, Nx, Nz)
        kz = Nz // 2
        kx = Nx // 2
        ky = Ny // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        xy_slice = volume[:, :, kz]
        xy_norm, _, _ = self._normalize_percentile(xy_slice, *self.cfg['percentile_range_3d'])
        im0 = axes[0].imshow(xy_norm, extent=(xs.min(), xs.max(), ys.min(), ys.max()), origin='lower', aspect='equal', cmap='viridis')
        axes[0].plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=1.5, alpha=0.8)
        if cause_xyz is not None:
            axes[0].scatter(cause_xyz[0], cause_xyz[1], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
        axes[0].set_xlim(xs.min(), xs.max())
        axes[0].set_ylim(ys.min(), ys.max())
        axes[0].set_title('GP slice XY (z = mid)')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        yz_slice = volume[:, kx, :]
        yz_norm, _, _ = self._normalize_percentile(yz_slice, *self.cfg['percentile_range_3d'])
        im1 = axes[1].imshow(yz_norm, extent=(zs.min(), zs.max(), ys.min(), ys.max()), origin='lower', aspect='equal', cmap='viridis')
        axes[1].plot(xyz[:, 2], xyz[:, 1], 'b-', linewidth=1.5, alpha=0.8)
        if cause_xyz is not None:
            axes[1].scatter(cause_xyz[2], cause_xyz[1], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
        axes[1].set_xlim(zs.min(), zs.max())
        axes[1].set_ylim(ys.min(), ys.max())
        axes[1].set_title('GP slice YZ (x = mid)')
        axes[1].set_xlabel('Z (m)')
        axes[1].set_ylabel('Y (m)')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        zx_slice = volume[ky, :, :].T
        zx_norm, _, _ = self._normalize_percentile(zx_slice, *self.cfg['percentile_range_3d'])
        im2 = axes[2].imshow(zx_norm, extent=(xs.min(), xs.max(), zs.min(), zs.max()), origin='lower', aspect='equal', cmap='viridis')
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

    def plot_3d_volume_with_cause_points(self, Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces: bool = True, max_cause_points: int = 5000):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if use_isosurfaces and _HAS_SKIMAGE:
            self._plot_3d_isosurfaces(mean_field, xs, ys, zs, ax=ax, num_levels=5)
        else:
            self._plot_3d_pointcloud(mean_field.reshape(Xg.shape), Xg, Yg, Zg, ax=ax, max_points=self.cfg['max_points_pointcloud'])
        ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='blue', linewidth=2.0, alpha=0.9, label='Actual trajectory')
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
        fig.tight_layout()
        return fig

    def _plot_3d_pointcloud(self, mean_field: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, Zg: np.ndarray, ax, max_points: int = 50000):
        X = Xg.ravel(); Y = Yg.ravel(); Z = Zg.ravel(); V = mean_field.ravel()
        N = X.shape[0]
        if N > max_points:
            idx = np.random.RandomState(42).choice(N, size=max_points, replace=False)
            X, Y, Z, V = X[idx], Y[idx], Z[idx], V[idx]
        Vn, _, _ = self._normalize_percentile(V, *self.cfg['percentile_range_3d'])
        ax.scatter(X, Y, Z, c=Vn, cmap='viridis', s=2, alpha=self.cfg['point_cloud_alpha'])

    def _plot_3d_isosurfaces(self, mean_field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ax, num_levels: int = 8):
        if not _HAS_SKIMAGE:
            raise RuntimeError("skimage not available for isosurface rendering")
        Ny, Nx, Nz = len(ys), len(xs), len(zs)
        field_3d = mean_field.reshape(Ny, Nx, Nz)
        volume = np.transpose(field_3d, (2, 0, 1))
        field_min, field_max = mean_field.min(), mean_field.max()
        field_range = field_max - field_min
        if field_range < 1e-8:
            return
        percentiles = np.linspace(50, 98, num_levels)
        levels = np.percentile(mean_field, percentiles)
        levels = np.clip(levels, field_min + 0.005 * field_range, field_max - 0.005 * field_range)
        dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
        dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
        dz = zs[1] - zs[0] if len(zs) > 1 else 1.0
        xmin, ymin, zmin = xs.min(), ys.min(), zs.min()
        colors = ['#2196F3', '#4CAF50', '#FFC107', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548']
        alphas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]
        for i, (level, color, alpha) in enumerate(zip(levels, colors, alphas)):
            try:
                verts, faces, normals, values = _sk_measure.marching_cubes(volume=volume, level=level, spacing=(dz, dy, dx))
                if len(verts) > 0:
                    verts_world = np.column_stack([
                        verts[:, 2] + xmin,
                        verts[:, 1] + ymin,
                        verts[:, 0] + zmin,
                    ])
                    ax.plot_trisurf(verts_world[:, 0], verts_world[:, 1], faces, verts_world[:, 2], color=color, lw=0.0, edgecolor='none', alpha=alpha)
            except Exception:
                pass

    # ----------------------------
    # High-level pipeline
    # ----------------------------

    def fit_from_pointcloud_and_buffer(self, pointcloud_xyz: np.ndarray, buffer_dir: str, nominal_path: Optional[str] = None, nominal_xyz: Optional[np.ndarray] = None, clip_plane: str = 'xy') -> Dict[str, Any]:
        """
        End-to-end fitting using a provided cause pointcloud and buffer directory.

        Args:
            pointcloud_xyz: numpy array (N,3) of cause points
            buffer_dir: path to buffer dir containing poses and metadata
            nominal_path: optional path to nominal JSON
            nominal_xyz: alternatively, pass nominal points directly
            clip_plane: 'xy' or 'xz' for clipping nominal to actual segment
        Returns:
            dict with fitted parameters and diagnostics
        """
        actual_xyz, cause, cause_xyz = self.load_buffer_xyz_drift(buffer_dir)
        if nominal_xyz is None and nominal_path:
            nominal_xyz = self.load_nominal_xyz(nominal_path)
        clipped_nominal = None
        if nominal_xyz is not None:
            clipped_nominal = self.clip_nominal_to_actual_segment(nominal_xyz, actual_xyz, plane=clip_plane)
            if clipped_nominal is None or len(clipped_nominal) == 0:
                clipped_nominal = nominal_xyz
        if clipped_nominal is not None:
            nominal_points_used, disturbance_magnitudes = self.compute_disturbance_at_nominal_points(clipped_nominal, actual_xyz, cause_xyz)
            if nominal_points_used.size == 0:
                drift_vectors, drift_magnitudes = self.compute_trajectory_drift_vectors(actual_xyz, clipped_nominal)
                if drift_vectors is None:
                    disturbance_magnitudes = np.zeros(len(clipped_nominal))
                    nominal_points_used = clipped_nominal
                else:
                    disturbance_magnitudes = drift_magnitudes
                    nominal_points_used = clipped_nominal
        else:
            disturbance_magnitudes = np.zeros(len(actual_xyz))
            nominal_points_used = actual_xyz
        fit = self.fit_direct_superposition_to_disturbances(nominal_points_used, disturbance_magnitudes, pointcloud_xyz if pointcloud_xyz is not None else np.empty((0, 3), dtype=float))
        return {
            'fit': fit,
            'actual_xyz': actual_xyz,
            'nominal_used': nominal_points_used,
            'disturbances': disturbance_magnitudes,
            'cause': cause,
            'cause_xyz': cause_xyz,
        }

    def predict_grid_from_fit(self, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray], fit: Dict[str, Any], pointcloud_xyz: np.ndarray, resolution_xy: Optional[float] = None, resolution_z: Optional[float] = None):
        Xg, Yg, Zg, grid_points, xs, ys, zs = self.create_3d_prediction_grid(actual_xyz, cause_xyz, resolution_xy, resolution_z)
        mean_pred, std_pred = self.predict_direct_field_3d(fit, grid_points, pointcloud_xyz)
        return Xg, Yg, Zg, grid_points, xs, ys, zs, mean_pred, std_pred

    # ----------------------------
    # PyVista visualization (optional, off by default)
    # ----------------------------

    def plot_3d_pyvista_volume_with_points(self, xs, ys, zs, mean_field, xyz, cause_points):
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
        Vn, vmin, vmax = self._normalize_percentile(mean_field, *self.cfg['percentile_range_3d'])
        levels = np.percentile(Vn, [55, 65, 75, 82, 88, 93, 96, 98])
        raw_levels = vmin + levels * (vmax - vmin)
        try:
            contour = grid.contour(isosurfaces=list(raw_levels), scalars='disturbance')
            plotter.add_mesh(contour, opacity=0.20, cmap='viridis', show_edges=False)
        except Exception:
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
        plotter.show() 

# ============================================================
# Module-level wrappers to mirror voxel_gp.py function API
# ============================================================

def load_buffer_xyz_drift(buffer_dir: str):
	return DisturbanceFieldHelper.load_buffer_xyz_drift(buffer_dir)

def load_nominal_xyz(nominal_path: str):
	return DisturbanceFieldHelper.load_nominal_xyz(nominal_path)

def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = 'xy') -> np.ndarray:
	return DisturbanceFieldHelper.clip_nominal_to_actual_segment(nominal_xyz, actual_xyz, plane)

def compute_trajectory_drift_vectors(actual_xyz: np.ndarray, nominal_xyz: np.ndarray):
	return DisturbanceFieldHelper.compute_trajectory_drift_vectors(actual_xyz, nominal_xyz)

def compute_disturbance_at_nominal_points(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray] = None):
	return DisturbanceFieldHelper.compute_disturbance_at_nominal_points(nominal_xyz, actual_xyz, cause_xyz)

def create_3d_prediction_grid(xyz: np.ndarray, cause_xyz: Optional[np.ndarray], resolution_xy: Optional[float] = None, resolution_z: Optional[float] = None):
	helper = DisturbanceFieldHelper()
	return helper.create_3d_prediction_grid(xyz, cause_xyz, resolution_xy, resolution_z)

def fit_direct_superposition_to_disturbances(nominal_points: np.ndarray, disturbance_magnitudes: np.ndarray, cause_points: np.ndarray):
	helper = DisturbanceFieldHelper()
	return helper.fit_direct_superposition_to_disturbances(nominal_points, disturbance_magnitudes, cause_points)

def predict_direct_field_3d(fit_params: Dict[str, Any], grid_points: np.ndarray, cause_points: np.ndarray):
	helper = DisturbanceFieldHelper()
	return helper.predict_direct_field_3d(fit_params, grid_points, cause_points)

def plot_2d_points(xyz: np.ndarray, nominal_points_used: np.ndarray, disturbance_magnitudes: np.ndarray, cause_xyz: Optional[np.ndarray], cause: Optional[str] = None):
	helper = DisturbanceFieldHelper()
	return helper.plot_2d_points(xyz, nominal_points_used, disturbance_magnitudes, cause_xyz, cause)

def plot_gp_orthogonal_views(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, mean_field: np.ndarray, xyz: np.ndarray, cause_xyz: Optional[np.ndarray]):
	helper = DisturbanceFieldHelper()
	return helper.plot_gp_orthogonal_views(xs, ys, zs, mean_field, xyz, cause_xyz)

def plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces: bool = True, max_cause_points: int = 5000):
	helper = DisturbanceFieldHelper()
	return helper.plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces, max_cause_points)

def plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_field, xyz, cause_points):
	helper = DisturbanceFieldHelper()
	return helper.plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_field, xyz, cause_points) 
#!/usr/bin/env python3
"""
GP Fit Utility Module

Handles Gaussian Process fitting for disturbance field modeling with PCD cause points.
Provides direct fitting of superposed anisotropic kernels to observed disturbances.

FEATURES:
=========
1. Direct GP Fitting: Fits superposed anisotropic kernel model directly to observed disturbances
2. PCD Integration: Loads cause points from PCD files for extended-cause modeling
3. Comprehensive Error Metrics: MSE, RMSE, MAE, R² score reporting
4. Optimization: Multiple initial guesses with BFGS optimization and grid search fallback
5. Memory Efficient: Chunked computation for large point clouds
6. Visualization Support: Provides fitted parameters for 3D field prediction

USAGE:
======
from resilience.gp_fit_utility import GPFitUtility

# Initialize utility
gp_utility = GPFitUtility()

# Fit GP to buffer data with PCD cause points
fit_result = gp_utility.fit_direct_gp_to_buffer(
    buffer_dir="/path/to/buffer",
    pcd_path="/path/to/cause_points.pcd"
)

# Get fitted parameters
lxy = fit_result['lxy']
lz = fit_result['lz']
A = fit_result['A']
b = fit_result['b']

# Predict field at new points
new_points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
predicted_field = gp_utility.predict_field_at_points(
    fit_result, new_points, cause_points
)
"""

import numpy as np
import json
import os
from pathlib import Path
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Optional: Open3D for PCD loading
try:
    import open3d as _o3d
    _HAS_OPEN3D = True
except Exception:
    _HAS_OPEN3D = False


class GPFitUtility:
    """Gaussian Process fitting utility for disturbance field modeling."""
    
    def __init__(self):
        """Initialize GP fit utility."""
        self.optimization_bounds = [(0.005, 1.0), (0.005, 1.0)]
        self.initial_guesses = [
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
    
    def load_buffer_data(self, buffer_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Load buffer data including poses, disturbances, and cause information.
        
        Args:
            buffer_dir: Path to buffer directory
            
        Returns:
            Tuple of (nominal_points, disturbance_magnitudes, cause_xyz, cause_name)
        """
        try:
            buffer_path = Path(buffer_dir)
            
            # Load poses
            poses_path = buffer_path / "poses.npy"
            if not poses_path.exists():
                print(f"poses.npy not found in {buffer_dir}")
                return None, None, None, None
            
            poses = np.load(poses_path)
            xyz = poses[:, 1:4]  # x, y, z coordinates
            
            # Load metadata for cause information
            meta_path = buffer_path / "metadata.json"
            cause_path = buffer_path / "cause_location.json"
            
            cause_xyz = None
            cause_name = None
            
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                cause_name = meta.get("cause")
                cause_xyz = meta.get("cause_location")
            
            if cause_xyz is None and cause_path.exists():
                with open(cause_path, "r") as f:
                    d = json.load(f)
                cause_name = d.get("cause", cause_name)
                cause_xyz = d.get("location_3d")
            
            if cause_xyz is not None:
                cause_xyz = np.array(cause_xyz[:3], dtype=float)
            
            # Load nominal trajectory for disturbance computation
            nominal_path = buffer_path / "nominal_trajectory.json"
            nominal_xyz = None
            if nominal_path.exists():
                with open(nominal_path, "r") as f:
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
                            nominal_xyz = np.array(xyz_list, dtype=float)
                    else:
                        arr = np.array(pts, dtype=float)
                        if arr.ndim == 2 and arr.shape[1] >= 3:
                            nominal_xyz = arr[:, :3]
            
            # Compute disturbances
            if nominal_xyz is not None and len(nominal_xyz) > 0:
                # Clip nominal to actual trajectory bounds
                nominal_xyz = self._clip_nominal_to_actual_bounds(nominal_xyz, xyz)
                
                # Compute disturbances at nominal points
                nominal_points_used, disturbance_magnitudes = self._compute_disturbances_at_nominal_points(
                    nominal_xyz, xyz, cause_xyz)
                
                if nominal_points_used.size == 0:
                    print("Warning: No nominal points used for disturbance computation")
                    return None, None, cause_xyz, cause_name
                
                return nominal_points_used, disturbance_magnitudes, cause_xyz, cause_name
            else:
                print("Warning: No nominal trajectory available, using actual trajectory points with zero disturbances")
                # Use actual trajectory points as nominal points with zero disturbances
                return xyz, np.zeros(len(xyz)), cause_xyz, cause_name
                
        except Exception as e:
            print(f"Error loading buffer data: {e}")
            return None, None, None, None
    
    def _clip_nominal_to_actual_bounds(self, nominal_xyz: np.ndarray, actual_xyz: np.ndarray) -> np.ndarray:
        """Clip nominal trajectory to actual trajectory bounds."""
        if nominal_xyz is None or len(nominal_xyz) == 0:
            return nominal_xyz
        
        # Get actual trajectory bounds
        actual_bounds = {
            'x': (actual_xyz[:, 0].min(), actual_xyz[:, 0].max()),
            'y': (actual_xyz[:, 1].min(), actual_xyz[:, 1].max()),
            'z': (actual_xyz[:, 2].min(), actual_xyz[:, 2].max())
        }
        
        # Add padding
        pad = 0.3
        mask = (
            (nominal_xyz[:, 0] >= actual_bounds['x'][0] - pad) &
            (nominal_xyz[:, 0] <= actual_bounds['x'][1] + pad) &
            (nominal_xyz[:, 1] >= actual_bounds['y'][0] - pad) &
            (nominal_xyz[:, 1] <= actual_bounds['y'][1] + pad) &
            (nominal_xyz[:, 2] >= actual_bounds['z'][0] - pad) &
            (nominal_xyz[:, 2] <= actual_bounds['z'][1] + pad)
        )
        
        return nominal_xyz[mask]
    
    def _compute_disturbances_at_nominal_points(self, nominal_xyz: np.ndarray, actual_xyz: np.ndarray, 
                                               cause_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute disturbance field at nominal trajectory points."""
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
    
    def load_pcd_points(self, pcd_path: str) -> np.ndarray:
        """
        Load points from a PCD file.
        
        Args:
            pcd_path: Path to PCD file
            
        Returns:
            np.ndarray: (N, 3) array of points, or empty array if failed
        """
        p = Path(pcd_path)
        if not p.exists():
            print(f"PCD file not found: {pcd_path}")
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
        
        # Fallback ASCII parse
        try:
            with open(p, 'r') as f:
                header = True
                data_started = False
                pts = []
                for line in f:
                    line = line.strip()
                    if header:
                        if line.startswith('DATA'):
                            data_started = True
                            header = False
                        continue
                    if data_started and line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                z = float(parts[2])
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
    
    def _sum_of_anisotropic_rbf(self, grid_points: np.ndarray, centers: np.ndarray, 
                               lxy: float, lz: float) -> np.ndarray:
        """
        Compute phi(x) = sum_j exp(-0.5 * [((dx/lxy)^2 + (dy/lxy)^2 + (dz/lz)^2)]) for all grid points.
        
        Args:
            grid_points: (N, 3) array of points to evaluate at
            centers: (M, 3) array of center points
            lxy: Length scale for XY directions
            lz: Length scale for Z direction
            
        Returns:
            np.ndarray: Vector of length N
        """
        if centers.size == 0:
            return np.zeros(grid_points.shape[0], dtype=float)
        
        num_points = grid_points.shape[0]
        phi = np.zeros(num_points, dtype=float)
        chunk = 200000  # Process in chunks for memory efficiency
        
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
    
    def fit_direct_gp_to_buffer(self, buffer_dir: str, pcd_path: str) -> Optional[Dict]:
        """
        Fit direct GP to buffer data with PCD cause points.
        
        Args:
            buffer_dir: Path to buffer directory
            pcd_path: Path to PCD file with cause points
            
        Returns:
            Dict with fitted parameters and error metrics, or None if failed
        """
        try:
            print(f"Fitting direct GP to buffer: {buffer_dir}")
            print(f"Using PCD cause points: {pcd_path}")
            
            # Load buffer data
            nominal_points, disturbance_magnitudes, cause_xyz, cause_name = self.load_buffer_data(buffer_dir)
            
            if nominal_points is None or disturbance_magnitudes is None:
                print("Failed to load buffer data")
                return None
            
            print(f"Loaded {len(nominal_points)} nominal points with disturbances")
            print(f"Cause: {cause_name}, Location: {cause_xyz}")
            
            # Load PCD cause points
            cause_points = self.load_pcd_points(pcd_path)
            
            if cause_points.size == 0:
                print("No cause points loaded from PCD")
                return None
            
            print(f"Loaded {cause_points.shape[0]} cause points from PCD")
            
            # Fit the GP
            fit_result = self._fit_superposition_to_disturbances(
                nominal_points, disturbance_magnitudes, cause_points)
            
            if fit_result is None:
                print("GP fitting failed")
                return None
            
            # Add metadata
            fit_result['buffer_dir'] = buffer_dir
            fit_result['pcd_path'] = pcd_path
            fit_result['cause_name'] = cause_name
            fit_result['cause_xyz'] = cause_xyz.tolist() if cause_xyz is not None else None
            fit_result['num_nominal_points'] = len(nominal_points)
            fit_result['num_cause_points'] = cause_points.shape[0]
            
            print(f"✓ GP fitting completed successfully")
            print(f"  Optimal parameters: lxy={fit_result['lxy']:.4f}, lz={fit_result['lz']:.4f}")
            print(f"  Error metrics: MSE={fit_result['mse']:.8f}, R²={fit_result['r2_score']:.4f}")
            
            return fit_result
            
        except Exception as e:
            print(f"Error in direct GP fitting: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _fit_superposition_to_disturbances(self, nominal_points: np.ndarray, 
                                         disturbance_magnitudes: np.ndarray, 
                                         cause_points: np.ndarray) -> Optional[Dict]:
        """
        Fit superposed identical anisotropic kernel to observed disturbances.
        
        Args:
            nominal_points: (N, 3) array of nominal trajectory points
            disturbance_magnitudes: (N,) array of disturbance magnitudes
            cause_points: (M, 3) array of cause points from PCD
            
        Returns:
            Dict with fitted parameters and error metrics
        """
        if cause_points.size == 0:
            print("Warning: No cause points to fit direct model")
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
        
        # Normalize target for better numerical properties
        target_mean = np.mean(target)
        target_std = np.std(target)
        if target_std < 1e-8:
            print("Warning: Target disturbances have very low variance")
            target_std = 1.0
        target_norm = (target - target_mean) / target_std
        
        def objective(params):
            """Objective function: MSE between target and reconstructed field."""
            lxy, lz = params
            lxy = max(lxy, 0.01)
            lz = max(lz, 0.01)
            
            phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
            
            # Normalize phi for better conditioning
            phi_mean = np.mean(phi)
            phi_std = np.std(phi)
            if phi_std < 1e-8:
                return float('inf')
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
            
            # Add regularization
            reg_term = 0.05 * (1.0 / (lxy + 0.05) + 1.0 / (lz + 0.05))
            return mse + reg_term
        
        print("Optimizing length scales with BFGS...")
        print(f"Target disturbances - min: {target.min():.6f}, max: {target.max():.6f}, mean: {target_mean:.6f}, std: {target_std:.6f}")
        print(f"Number of cause points: {cause_points.shape[0]}")
        print(f"Number of nominal points: {nominal_points.shape[0]}")
        
        best_result = None
        best_mse = float('inf')
        
        for i, x0 in enumerate(self.initial_guesses):
            print(f"  Trying initial guess {i+1}/{len(self.initial_guesses)}: lxy={x0[0]:.3f}, lz={x0[1]:.3f}")
            
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=self.optimization_bounds,
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
            return self._grid_search_fallback(nominal_points, target, cause_points)
        
        # Extract optimal parameters
        lxy_opt, lz_opt = best_result.x
        phi_opt = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy_opt, lz=lz_opt)
        
        # Final closed-form solution for A and b
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
        
        # Compute error metrics
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
    
    def _grid_search_fallback(self, nominal_points: np.ndarray, target: np.ndarray, 
                             cause_points: np.ndarray) -> Dict:
        """Grid search fallback when optimization fails."""
        print("Performing grid search fallback...")
        
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
        
        print(f"Grid search fallback -> lxy: {best['lxy']:.4f} m, lz: {best['lz']:.4f} m, A: {best['A']:.6f}, b: {best['b']:.6f}, MSE: {best['mse']:.8f}")
        
        # Compute additional error metrics
        rmse = float(np.sqrt(best['mse']))
        mae = float(np.mean(np.abs(best['recon'] - target)))
        ss_res = np.sum((target - best['recon']) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        return {
            'lxy': best['lxy'],
            'lz': best['lz'],
            'A': best['A'],
            'b': best['b'],
            'recon': best['recon'],
            'mse': best['mse'],
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2_score,
            'optimization_result': None
        }
    
    def predict_field_at_points(self, fit_result: Dict, points: np.ndarray, 
                               cause_points: np.ndarray) -> np.ndarray:
        """
        Predict disturbance field at given points using fitted parameters.
        
        Args:
            fit_result: Result from fit_direct_gp_to_buffer
            points: (N, 3) array of points to predict at
            cause_points: (M, 3) array of cause points
            
        Returns:
            np.ndarray: Predicted disturbance field at points
        """
        if fit_result is None or 'lxy' not in fit_result or fit_result['lxy'] is None:
            return np.zeros(points.shape[0])
        
        lxy = fit_result['lxy']
        lz = fit_result['lz']
        A = fit_result['A']
        b = fit_result['b']
        
        phi = self._sum_of_anisotropic_rbf(points, cause_points, lxy=lxy, lz=lz)
        predicted_field = A * phi + b
        
        return predicted_field
    
    def save_fit_result(self, fit_result: Dict, output_path: str):
        """
        Save fit result to JSON file.
        
        Args:
            fit_result: Result from fit_direct_gp_to_buffer
            output_path: Path to save JSON file
        """
        try:
            # Make result JSON serializable
            serializable_result = {}
            for key, value in fit_result.items():
                if key == 'optimization_result' and value is not None:
                    # Convert optimization result to dict
                    opt_dict = {
                        'success': value.success,
                        'message': value.message,
                        'nit': value.nit,
                        'nfev': value.nfev,
                        'fun': float(value.fun),
                        'x': value.x.tolist() if hasattr(value.x, 'tolist') else list(value.x)
                    }
                    serializable_result[key] = opt_dict
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            print(f"✓ Saved fit result to: {output_path}")
            
        except Exception as e:
            print(f"Error saving fit result: {e}")
    
    def load_fit_result(self, input_path: str) -> Optional[Dict]:
        """
        Load fit result from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            Dict with fit result, or None if failed
        """
        try:
            with open(input_path, 'r') as f:
                result = json.load(f)
            
            # Convert arrays back to numpy
            for key in ['recon']:
                if key in result and isinstance(result[key], list):
                    result[key] = np.array(result[key])
            
            print(f"✓ Loaded fit result from: {input_path}")
            return result
            
        except Exception as e:
            print(f"Error loading fit result: {e}")
            return None 
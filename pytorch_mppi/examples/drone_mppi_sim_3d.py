#!/usr/bin/env python3
"""
MPPI 3D Navigation with Drone Model
- 3D double integrator dynamics (x, y, z, vx, vy, vz)
- Disturbance-aware cost via 3D GP field from cause.pcd
- Each point in cause.pcd is a GP source with anisotropic RBF kernel
- Bayesian uncertainty integration from sample_gp.py for conservative risk planning
- Visualization with 3D GP field
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pytorch_mppi import MPPI

# Import DisturbanceFieldHelper from sample_gp.py for GP fitting
_script_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_script_dir))
try:
    from sample_gp import DisturbanceFieldHelper
    _HAS_SAMPLE_GP = True
except ImportError:
    print("Warning: Could not import DisturbanceFieldHelper from sample_gp.py")
    print("Bayesian uncertainty will not be available. Using mean-only GP.")
    _HAS_SAMPLE_GP = False

# ============================================================================
# PCD LOADING UTILITY
# ============================================================================

def load_pcd_points(pcd_path: str) -> np.ndarray:
    """Load points from a PCD file. Uses Open3D when available; otherwise a minimal ASCII PCD parser.
    Returns an (N,3) numpy array. If file missing or empty, returns empty array.
    """
    p = Path(pcd_path)
    if not p.exists():
        print(f"Warning: PCD not found at {pcd_path}")
        return np.empty((0, 3), dtype=float)
    
    # Try Open3D first
    try:
        import open3d as o3d
        try:
            pc = o3d.io.read_point_cloud(str(p))
            pts = np.asarray(pc.points, dtype=float)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                print(f"Loaded {len(pts)} points from PCD using Open3D")
                return pts[:, :3]
        except Exception as e:
            print(f"Open3D failed to read PCD: {e}; attempting ASCII parse")
    except ImportError:
        print("Open3D not available, using ASCII parser")
    
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
        print(f"Loaded {len(pts)} points from PCD using ASCII parser")
        return np.array(pts, dtype=float)
    except Exception as e:
        print(f"Failed to parse PCD: {e}")
        return np.empty((0, 3), dtype=float)


# ============================================================================
# DISTURBANCE GP MODEL (3D) with Bayesian Uncertainty
# ============================================================================

class DisturbanceGP3D(torch.nn.Module):
    """
    3D superposed anisotropic RBF GP with Bayesian uncertainty:
        f(p) = A * sum_j exp(-0.5 * d^2(p, c_j)) + b
    where d^2 = (dx^2 + dy^2) / lxy^2 + (dz^2) / lz^2
    
    Each point in cause.pcd is a GP source with the same kernel.
    
    Supports two modes:
    1. Mean-only: forward() returns mean prediction
    2. Mean + Uncertainty: forward_with_uncertainty() returns (mean, std) using Bayesian linear regression
    """
    def __init__(self, cause_points, lxy, lz, A, b, 
                 sigma2_noise=None, nominal_points=None, disturbances=None,
                 device="cpu", dtype=torch.double):
        super().__init__()
        cause_points = torch.as_tensor(cause_points, device=device, dtype=dtype)
        if cause_points.dim() == 1:
            cause_points = cause_points.unsqueeze(0)
        if cause_points.shape[1] == 2:
            # If 2D points provided, add z=0
            z_zeros = torch.zeros(cause_points.shape[0], 1, device=device, dtype=dtype)
            cause_points = torch.cat([cause_points, z_zeros], dim=1)
        self.register_buffer("cause", cause_points)  # (N, 3)

        self.lxy = torch.tensor(float(lxy), device=device, dtype=dtype)
        self.lz = torch.tensor(float(lz), device=device, dtype=dtype)
        self.A = torch.tensor(float(A), device=device, dtype=dtype)
        self.b = torch.tensor(float(b), device=device, dtype=dtype)

        self.inv_lxy2 = 1.0 / (self.lxy * self.lxy + 1e-12)
        self.inv_lz2 = 1.0 / (self.lz * self.lz + 1e-12)
        
        # Bayesian uncertainty parameters (optional)
        self.sigma2_noise = torch.tensor(float(sigma2_noise), device=device, dtype=dtype) if sigma2_noise is not None else None
        self.nominal_points = None
        self.disturbances = None
        self.param_cov = None  # (2, 2) covariance matrix for [A, b]
        
        if nominal_points is not None and disturbances is not None and sigma2_noise is not None:
            self._setup_bayesian_uncertainty(nominal_points, disturbances, device, dtype)

    def _setup_bayesian_uncertainty(self, nominal_points, disturbances, device, dtype):
        """Pre-compute Bayesian uncertainty parameters from training data."""
        nominal_points = torch.as_tensor(nominal_points, device=device, dtype=dtype)
        disturbances = torch.as_tensor(disturbances, device=device, dtype=dtype)
        
        if nominal_points.dim() == 1:
            nominal_points = nominal_points.unsqueeze(0)
        if nominal_points.shape[1] == 2:
            z_zeros = torch.zeros(nominal_points.shape[0], 1, device=device, dtype=dtype)
            nominal_points = torch.cat([nominal_points, z_zeros], dim=1)
        
        self.nominal_points = nominal_points
        self.disturbances = disturbances
        
        # Compute phi at training points
        # Note: cause points are centered, so we need to center nominal_points too
        if hasattr(self, 'cause_location'):
            nom_centered = nominal_points - self.cause_location.unsqueeze(0)
        else:
            nom_centered = nominal_points
        
        diff = nom_centered.unsqueeze(1) - self.cause.unsqueeze(0)  # (K, N, 3)
        dx2 = diff[..., 0] ** 2
        dy2 = diff[..., 1] ** 2
        dz2 = diff[..., 2] ** 2
        d2 = (dx2 + dy2) * self.inv_lxy2 + dz2 * self.inv_lz2  # (K, N)
        phi_train = torch.exp(-0.5 * d2).sum(dim=1)  # (K,)
        
        # Build feature matrix X = [phi, 1]
        X_train = torch.stack([phi_train, torch.ones_like(phi_train)], dim=1)  # (K, 2)
        
        # Compute parameter covariance: Cov(A, b) = sigma^2 * (X^T X)^-1
        XtX = X_train.T @ X_train  # (2, 2)
        XtX[0, 0] += 1e-6  # Regularization
        XtX[1, 1] += 1e-6
        
        try:
            XtX_inv = torch.linalg.inv(XtX)
            self.param_cov = self.sigma2_noise * XtX_inv  # (2, 2)
        except Exception:
            # Fallback: diagonal covariance
            self.param_cov = torch.eye(2, device=device, dtype=dtype) * self.sigma2_noise

    def forward(self, pos):
        """
        Args:
            pos: (K, 3) or (3,) positions [x, y, z] in world coordinates
        Returns:
            (K,) mean prediction μ(p) at each position
        """
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)  # (1, 3)
        
        # Handle 2D positions by adding z=0
        if pos.shape[1] == 2:
            z_zeros = torch.zeros(pos.shape[0], 1, device=pos.device, dtype=pos.dtype)
            pos = torch.cat([pos, z_zeros], dim=1)

        # Cause points are centered (relative to origin), so we need to center evaluation points too
        # If cause_location is stored, subtract it from pos; otherwise assume already centered
        if hasattr(self, 'cause_location'):
            pos_centered = pos - self.cause_location.unsqueeze(0)
        else:
            pos_centered = pos

        # (K, 1, 3) - (1, N, 3) -> (K, N, 3)
        diff = pos_centered.unsqueeze(1) - self.cause.unsqueeze(0)
        dx2 = diff[..., 0] ** 2
        dy2 = diff[..., 1] ** 2
        dz2 = diff[..., 2] ** 2
        d2 = (dx2 + dy2) * self.inv_lxy2 + dz2 * self.inv_lz2  # (K, N)
        phi = torch.exp(-0.5 * d2).sum(dim=1)  # (K,)
        return self.A * phi + self.b
    
    def forward_with_uncertainty(self, pos):
        """
        Compute both mean and Bayesian uncertainty.
        
        Args:
            pos: (K, 3) or (3,) positions [x, y, z] in world coordinates
        Returns:
            mean: (K,) mean prediction μ(p)
            std: (K,) predictive standard deviation σ(p)
        """
        if self.param_cov is None or self.sigma2_noise is None:
            # Fallback: return mean with small constant uncertainty
            mean = self.forward(pos)
            std = torch.full_like(mean, 0.1 * torch.std(mean) if mean.numel() > 0 else 0.1)
            return mean, std
        
        # Normalize input shape (don't modify original)
        pos_input = pos.clone() if isinstance(pos, torch.Tensor) else torch.as_tensor(pos, device=self.cause.device, dtype=self.cause.dtype)
        if pos_input.dim() == 1:
            pos_input = pos_input.unsqueeze(0)  # (1, 3)
        if pos_input.shape[1] == 2:
            z_zeros = torch.zeros(pos_input.shape[0], 1, device=pos_input.device, dtype=pos_input.dtype)
            pos_input = torch.cat([pos_input, z_zeros], dim=1)
        
        # Compute mean
        if hasattr(self, 'cause_location'):
            pos_centered = pos_input - self.cause_location.unsqueeze(0)
        else:
            pos_centered = pos_input
        
        diff = pos_centered.unsqueeze(1) - self.cause.unsqueeze(0)  # (K, N, 3)
        dx2 = diff[..., 0] ** 2
        dy2 = diff[..., 1] ** 2
        dz2 = diff[..., 2] ** 2
        d2 = (dx2 + dy2) * self.inv_lxy2 + dz2 * self.inv_lz2  # (K, N)
        phi_query = torch.exp(-0.5 * d2).sum(dim=1)  # (K,)
        mean = self.A * phi_query + self.b
        
        # Build feature vector v = [phi, 1] for each query point
        v = torch.stack([phi_query, torch.ones_like(phi_query)], dim=1)  # (K, 2)
        
        # Epistemic variance: v^T * Cov * v for each query point
        # For each row v[i], compute v[i]^T @ param_cov @ v[i]
        # This is: sum_j sum_k v[i,j] * param_cov[j,k] * v[i,k]
        epistemic_var = torch.sum(v * (v @ self.param_cov), dim=1)  # (K,)
        
        # Total variance = epistemic + aleatoric
        total_variance = epistemic_var + self.sigma2_noise
        
        std = torch.sqrt(torch.clamp(total_variance, min=0.0))
        
        return mean, std


# ============================================================================
# 3D DRONE DYNAMICS
# ============================================================================

class DroneDynamics3D:
    """
    3D double integrator model (simple drone dynamics)
    State: [x, y, z, vx, vy, vz]
    Control: [ax, ay, az] (accelerations)
    """
    def __init__(self, dt=0.005,
                 disturb_gp=None,
                 device="cpu", dtype=torch.double):
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.disturb_gp = disturb_gp

        # Bounds
        self.a_min, self.a_max = -3.0, 3.0
        self.v_min, self.v_max = -5.0, 5.0

    def __call__(self, state, action):
        """
        Args:
            state: (K x 6) [x, y, z, vx, vy, vz]
            action: (K x 3) [ax, ay, az]
        Returns:
            next_state: (K x 6)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
        
        ax = torch.clamp(action[:, 0], self.a_min, self.a_max)
        ay = torch.clamp(action[:, 1], self.a_min, self.a_max)
        az = torch.clamp(action[:, 2], self.a_min, self.a_max)

        # Deterministic dynamics: GP is used only in cost (risk), not in propagation
        vx_next = torch.clamp(vx + ax * self.dt, self.v_min, self.v_max)
        vy_next = torch.clamp(vy + ay * self.dt, self.v_min, self.v_max)
        vz_next = torch.clamp(vz + az * self.dt, self.v_min, self.v_max)

        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        z_next = z + vz_next * self.dt

        return torch.stack([x_next, y_next, z_next, vx_next, vy_next, vz_next], dim=1)


# ============================================================================
# COST FUNCTION (3D)
# ============================================================================

class DroneCost3D:
    """
    Look-ahead path tracking + GP-based risk shaping + 3D box obstacles.
    Dynamics remain deterministic; GP only affects cost.
    """
    def __init__(self, goal,
                 ref_traj=None,  # (N, 3) reference path
                 disturb_gp=None,  # GP risk model (variance-like field)
                 obstacles=None,  # list of boxes: [(min_corner, max_corner), ...] where corners are (x,y,z)
                 ref_weight=1.50,       # Path tracking weight
                 goal_weight=3.0,       # Goal attraction
                 obstacle_weight=10.0,  # Obstacle avoidance weight
                 control_weight=0.02,
                 lookahead_dist=1.5,    # Look-ahead distance for path tracking
                 alpha=1.0,             # Weight on variance risk term
                 beta=2.0,              # Weight on drift term
                 gamma=0.5,             # Speed scaling of risk
                 device="cpu", dtype=torch.double):
        self.goal = torch.tensor(goal, device=device, dtype=dtype)  # (3,)
        self.goal_weight = goal_weight
        self.ref_weight = ref_weight
        self.obstacle_weight = obstacle_weight
        self.control_weight = control_weight
        self.lookahead_dist = lookahead_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.dtype = dtype
        self.ref_traj = None if ref_traj is None else torch.tensor(ref_traj, device=device, dtype=dtype)  # (N, 3)
        self.disturb_gp = disturb_gp
        self.obstacles = obstacles or []  # List of (min_corner, max_corner) tuples
        
    def __call__(self, state, action, step=None):
        """
        Args:
            state: (K x 6) [x, y, z, vx, vy, vz]
            action: (K x 3) [ax, ay, az]
        Returns:
            cost: (K,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        pos = state[:, :3]  # (K, 3)
        v = torch.norm(state[:, 3:6], dim=1)  # (K,) speed magnitude

        # 1) Look-ahead path tracking: track point ahead on trajectory
        if self.ref_traj is not None:
            # Find closest point on reference path
            diff = pos.unsqueeze(1) - self.ref_traj.unsqueeze(0)  # (K, N, 3)
            ref_dists = torch.norm(diff, dim=2)  # (K, N)
            min_dist, closest_idx = torch.min(ref_dists, dim=1)  # (K,)
            
            # Look-ahead: target point ahead on path
            N = self.ref_traj.shape[0]
            lookahead_steps = max(5, int(self.lookahead_dist / 0.1))  # ~1.5m ahead
            target_idx = torch.clamp(closest_idx + lookahead_steps, max=N-1)
            
            # Track the look-ahead point
            target_points = self.ref_traj[target_idx]  # (K, 3)
            lookahead_dist = torch.norm(pos - target_points, dim=1)
            ref_cost = (lookahead_dist ** 2) * self.ref_weight
        else:
            ref_cost = 0.0

        # 2) Goal attraction for forward progress
        goal = self.goal.unsqueeze(0)
        goal_dist = torch.norm(pos - goal, dim=1)
        goal_cost = (goal_dist ** 2) * self.goal_weight

        # 3) 3D box obstacle cost
        obstacle_cost = torch.zeros(pos.shape[0], device=self.device, dtype=self.dtype)
        min_safe_dist = torch.full((pos.shape[0],), 10.0, device=self.device, dtype=self.dtype)
        
        for box_min, box_max in self.obstacles:
            box_min_t = torch.tensor(box_min, device=self.device, dtype=self.dtype)
            box_max_t = torch.tensor(box_max, device=self.device, dtype=self.dtype)
            
            # Check if point is inside box
            inside = torch.all((pos >= box_min_t.unsqueeze(0)) & (pos <= box_max_t.unsqueeze(0)), dim=1)
            
            # Compute distance to box (0 if inside, otherwise distance to nearest face)
            # Distance to each face
            dist_to_min = pos - box_min_t.unsqueeze(0)  # (K, 3)
            dist_to_max = box_max_t.unsqueeze(0) - pos  # (K, 3)
            
            # For points outside, compute distance to nearest face
            outside_dist = torch.minimum(
                torch.minimum(dist_to_min, dist_to_max),
                torch.zeros_like(dist_to_min)
            )  # Negative inside box, positive outside
            dist_to_box = torch.norm(torch.clamp(outside_dist, min=0.0), dim=1)  # (K,)
            
            # If inside box, set very high cost
            obstacle_cost += torch.where(
                inside,
                torch.full_like(inside, 1000.0 * self.obstacle_weight, dtype=self.dtype),
                torch.zeros_like(inside, dtype=self.dtype)
            )
            
            # Track minimum safe distance for risk scaling
            min_safe_dist = torch.minimum(min_safe_dist, dist_to_box + 0.1)  # Add small buffer

        # 4) GP-based risk shaping with Bayesian uncertainty (conservative risk)
        risk_cost = 0.0
        if self.disturb_gp is not None:
            with torch.no_grad():
                # Check if GP supports uncertainty computation
                if hasattr(self.disturb_gp, 'forward_with_uncertainty') and \
                   self.disturb_gp.param_cov is not None:
                    # Use conservative risk: mean + 2*std (95% confidence upper bound)
                    mean_risk, std_risk = self.disturb_gp.forward_with_uncertainty(pos)
                    conservative_risk = mean_risk + 2.0 * std_risk  # 95% CI upper bound
                    base_risk = torch.clamp(conservative_risk, min=0.0)
                else:
                    # Fallback: use mean-only (backward compatibility)
                    mean_risk = torch.clamp(self.disturb_gp(pos), min=0.0)
                    base_risk = mean_risk

            # Speed scaling: higher speed -> higher risk
            speed_scale = 1.0 + self.gamma * torch.abs(v)
            risk_cost = base_risk * speed_scale

        # 5) Control effort
        control_cost = torch.norm(action, dim=1) * self.control_weight

        return ref_cost + goal_cost + obstacle_cost + risk_cost + control_cost
    
    def terminal_cost(self, states, actions):
        """
        Terminal cost at end of trajectory
        Args:
            states: (K, T, nx) trajectory states
            actions: (K, T, nu) actions  
        Returns:
            cost: (K,) terminal costs per trajectory
        """
        # Extract final position (x, y, z)
        final_pos = states[..., -1, :3]  # (K, 3) or (3,)
        
        # Ensure 3D: (K, 3) for broadcasting
        if final_pos.dim() == 1:
            final_pos = final_pos.unsqueeze(0)  # (1, 3)
        
        goal = self.goal.unsqueeze(0)  # (1, 3)
        dist = torch.norm(final_pos - goal, dim=-1)  # (K,)
        # Moderate terminal cost for path tracking (less critical than running cost)
        return (dist ** 2) * self.goal_weight * 20.0


# ============================================================================
# GP FITTING FROM BUFFER DATA
# ============================================================================

def fit_gp_from_buffer(buffer_dir=None, cause_pcd_path=None, nominal_path=None, 
                       objective="nll", device="cpu", dtype=torch.double):
    """
    Fit GP parameters from buffer data using DisturbanceFieldHelper.
    
    Args:
        buffer_dir: Path to buffer directory (contains poses.npy, metadata.json, etc.)
        cause_pcd_path: Path to points.pcd file (cause points)
        nominal_path: Path to nominal trajectory JSON
        objective: "mse" or "nll" for fitting
        device: PyTorch device
        dtype: PyTorch dtype
    
    Returns:
        disturb_gp: DisturbanceGP3D instance with fitted parameters and uncertainty support
        fit_info: Dictionary with fit statistics
    """
    if not _HAS_SAMPLE_GP:
        print("Warning: DisturbanceFieldHelper not available. Using default GP parameters.")
        return None, None
    
    helper = DisturbanceFieldHelper()
    
    # Default paths
    if buffer_dir is None:
        buffer_dir = "/home/navin/ros2_ws/src/buffers/run_20251221_144638_231_738a9b22/buffer1"
    if cause_pcd_path is None:
        cause_pcd_path = Path(buffer_dir) / "points.pcd"
    if nominal_path is None:
        nominal_path = "/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json"
    
    buffer_dir = Path(buffer_dir)
    cause_pcd_path = Path(cause_pcd_path)
    nominal_path = Path(nominal_path)
    
    print(f"\n=== Fitting GP from buffer data ===")
    print(f"Buffer dir: {buffer_dir}")
    print(f"PCD path: {cause_pcd_path}")
    print(f"Nominal path: {nominal_path}")
    
    # Load cause points
    cause_points_raw = load_pcd_points(str(cause_pcd_path))
    if len(cause_points_raw) == 0:
        print("Warning: No cause points loaded. Cannot fit GP.")
        return None, None
    
    # Fit GP using DisturbanceFieldHelper
    try:
        fit_result = helper.fit_from_pointcloud_and_buffer(
            pointcloud_xyz=cause_points_raw,
            buffer_dir=str(buffer_dir),
            nominal_path=str(nominal_path) if nominal_path.exists() else None,
            clip_plane="xy",
            objective=objective,
        )
        
        fit = fit_result["fit"]
        nominal_used = fit_result["nominal_used"]
        disturbances = fit_result["disturbances"]
        cause_xyz = fit_result["cause_xyz"]
        
        if fit["lxy"] is None:
            print("Warning: GP fitting failed. Using default parameters.")
            return None, None
        
        lxy = fit["lxy"]
        lz = fit["lz"]
        A = fit["A"]
        b = fit["b"]
        sigma2 = fit.get("sigma2", fit["mse"])
        
        print(f"\n✓ GP Fit Results:")
        print(f"  lxy: {lxy:.6f} m")
        print(f"  lz: {lz:.6f} m")
        print(f"  A: {A:.6f}")
        print(f"  b: {b:.6f}")
        print(f"  MSE: {fit['mse']:.6f}")
        print(f"  R²: {fit['r2_score']:.6f}")
        print(f"  σ²: {sigma2:.6f}")
        
        # Compute cause location (center of cause points)
        cause_location = cause_points_raw.mean(axis=0) if cause_xyz is None else cause_xyz
        
        # Center cause points
        cause_points_centered = cause_points_raw - cause_location
        
        # Create GP with uncertainty support
        disturb_gp = DisturbanceGP3D(
            cause_points=cause_points_centered,
            lxy=lxy, lz=lz, A=A, b=b,
            sigma2_noise=sigma2,
            nominal_points=nominal_used,
            disturbances=disturbances,
            device=device, dtype=dtype,
        )
        
        # Store cause location for evaluation offset
        disturb_gp.cause_location = torch.tensor(cause_location, device=device, dtype=dtype)
        
        fit_info = {
            "lxy": lxy, "lz": lz, "A": A, "b": b,
            "sigma2": sigma2, "mse": fit["mse"], "r2": fit["r2_score"],
            "cause_location": cause_location,
        }
        
        print(f"✓ GP created with Bayesian uncertainty support")
        return disturb_gp, fit_info
        
    except Exception as e:
        print(f"Error fitting GP from buffer: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_drone_mppi_3d(cause_pcd_path=None, buffer_dir=None, 
                           nominal_path=None, use_fitted_gp=True):
    """Run MPPI simulation for 3D drone navigation
    
    Args:
        cause_pcd_path: Path to points.pcd file. If None, uses default buffer path.
        buffer_dir: Path to buffer directory for GP fitting. If None, uses default.
        nominal_path: Path to nominal trajectory JSON. If None, uses default.
        use_fitted_gp: If True, fit GP from buffer data. If False, use default parameters.
    """
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    print(f"Using device: {device}")
    
    # Try to fit GP from buffer data if requested
    disturb_gp = None
    fit_info = None
    cause_location = None
    
    if use_fitted_gp and _HAS_SAMPLE_GP:
        disturb_gp, fit_info = fit_gp_from_buffer(
            buffer_dir=buffer_dir,
            cause_pcd_path=cause_pcd_path,
            nominal_path=nominal_path,
            objective="nll",
            device=device,
            dtype=dtype,
        )
        if fit_info is not None:
            cause_location = fit_info["cause_location"]
    
    # Fallback: use default GP parameters
    if disturb_gp is None:
        print("\n=== Using default GP parameters (no buffer fitting) ===")
        default_pcd_path = "/home/navin/ros2_ws/src/buffers/run_20251221_144638_231_738a9b22/buffer1/points.pcd"
        
        # Determine PCD path
        if cause_pcd_path is None:
            pcd_path = Path(default_pcd_path)
        else:
            pcd_path = Path(cause_pcd_path)
            if not pcd_path.exists() and not pcd_path.is_absolute():
                current_dir = Path.cwd()
                potential_paths = [
                    current_dir / cause_pcd_path,
                    current_dir.parent / cause_pcd_path,
                    Path(__file__).parent / cause_pcd_path,
                    Path(__file__).parent.parent / cause_pcd_path,
                ]
                for pp in potential_paths:
                    if pp.exists():
                        pcd_path = pp
                        break
        
        # Load cause points from PCD file
        print(f"Loading cause points from {pcd_path}")
        if not pcd_path.exists():
            print(f"Warning: PCD file not found at {pcd_path}")
            print("Will use synthetic points for demonstration.")
        cause_points_raw = load_pcd_points(str(pcd_path))
        
        if len(cause_points_raw) == 0:
            print(f"Warning: No points loaded from {pcd_path}")
            print("Generating synthetic cause points for demonstration...")
            rng = np.random.default_rng(42)
            gp_center = np.array([4.0, 0.0, 1.0])
            cause_points_raw = rng.normal(size=(200, 3)) * 0.5 + gp_center
        
        print(f"Loaded {len(cause_points_raw)} cause points")
        
        # Compute cause location (center) from PCD points
        cause_location = cause_points_raw.mean(axis=0)
        print(f"Cause location (center): [{cause_location[0]:.3f}, {cause_location[1]:.3f}, {cause_location[2]:.3f}]")
        
        # Center the PCD points
        cause_points_centered = cause_points_raw - cause_location
        
        # Default GP parameters
        lxy = 0.45
        lz = 0.45
        A, b = 0.15, 0.0
        
        # Create GP without uncertainty (backward compatibility)
        disturb_gp = DisturbanceGP3D(
            cause_points=cause_points_centered,
            lxy=lxy, lz=lz, A=2*A, b=b,
            device=device, dtype=dtype,
        )
        disturb_gp.cause_location = torch.tensor(cause_location, device=device, dtype=dtype)
    
    # Simulation parameters
    dt = 0.05  # 20 Hz control
    sim_steps = 150  # Longer simulation for 3D navigation
    
    # State: [x, y, z, vx, vy, vz], Control: [ax, ay, az]
    # Set start/goal relative to cause location
    start = np.array([cause_location[0] - 3.0, cause_location[1] - 2.0, cause_location[2], 0., 0., 0.])
    goal = np.array([cause_location[0] + 3.0, cause_location[1] + 2.0, cause_location[2] + 1.0])

    # Reference trajectory: goes through center of cause.pcd
    # Create a trajectory that passes through cause_location
    ref_len = 120
    # Split trajectory into two segments: start -> cause_location -> goal
    mid_len = ref_len // 2
    
    # First segment: start to cause_location
    seg1_x = np.linspace(start[0], cause_location[0], mid_len)
    seg1_y = np.linspace(start[1], cause_location[1], mid_len)
    seg1_z = np.linspace(start[2], cause_location[2], mid_len)
    
    # Second segment: cause_location to goal
    seg2_x = np.linspace(cause_location[0], goal[0], ref_len - mid_len)
    seg2_y = np.linspace(cause_location[1], goal[1], ref_len - mid_len)
    seg2_z = np.linspace(cause_location[2], goal[2], ref_len - mid_len)
    
    # Combine segments (avoid duplicate point at cause_location)
    ref_traj = np.stack([
        np.concatenate([seg1_x, seg2_x[1:]]),  # Skip first point of seg2 to avoid duplicate
        np.concatenate([seg1_y, seg2_y[1:]]),
        np.concatenate([seg1_z, seg2_z[1:]]),
    ], axis=1)

    # Create 3D box obstacles along the reference trajectory
    # Place boxes at intervals along the path
    num_obstacles = 4
    obstacle_boxes = []
    box_size = 0.6  # Size of each box (half-extent)
    
    # Sample points along reference trajectory for obstacle placement
    obstacle_indices = np.linspace(len(ref_traj) // 4, 3 * len(ref_traj) // 4, num_obstacles, dtype=int)
    
    for idx in obstacle_indices:
        if idx < len(ref_traj):
            # Get point on reference trajectory
            center = ref_traj[idx]
            
            # Offset box slightly to the side of trajectory (alternating sides)
            side_offset = 0.8 * (-1 if len(obstacle_boxes) % 2 == 0 else 1)
            # Compute perpendicular direction (simplified: use cross product with up vector)
            traj_dir = ref_traj[min(idx + 1, len(ref_traj) - 1)] - ref_traj[max(idx - 1, 0)]
            traj_dir = traj_dir / (np.linalg.norm(traj_dir) + 1e-6)
            up_vec = np.array([0, 0, 1])
            perp_dir = np.cross(traj_dir, up_vec)
            perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-6)
            
            box_center = center + perp_dir * side_offset
            
            # Create axis-aligned bounding box
            box_min = box_center - box_size
            box_max = box_center + box_size
            obstacle_boxes.append((box_min.tolist(), box_max.tolist()))
    
    print(f"\nCreated {len(obstacle_boxes)} box obstacles along reference trajectory")

    # Initialize dynamics (deterministic; GP only used in cost)
    dynamics = DroneDynamics3D(
        dt=dt,
        disturb_gp=disturb_gp,
        device=device, dtype=dtype,
    )
    cost_fn = DroneCost3D(
        goal=goal,
        ref_traj=ref_traj,
        disturb_gp=disturb_gp,
        obstacles=obstacle_boxes,
        ref_weight=15.0,       # Path tracking (look-ahead)
        goal_weight=3.0,       # Forward progress
        obstacle_weight=10.0,  # Obstacle avoidance
        control_weight=0.02,
        lookahead_dist=1.5,    # 1.5m look-ahead
        alpha=1.0,
        beta=2.0,
        gamma=0.5,
        device=device, dtype=dtype,
    )
    
    # MPPI Configuration (3D navigation)
    nx = 6  # [x, y, z, vx, vy, vz]
    nu = 3  # [ax, ay, az]
    num_samples = 3000 if device == "cuda" else 1500
    horizon = 20
    lambda_ = 0.7
    
    # Control bounds
    u_max = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype)
    u_min = -u_max
    
    # Control noise
    noise_sigma = torch.diag(torch.tensor([0.4, 0.4, 0.4], device=device, dtype=dtype))
    
    # Initialize MPPI
    mppi_ctrl = MPPI(
        dynamics=dynamics,
        running_cost=cost_fn,
        nx=nx,
        noise_sigma=noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        device=device,
        u_min=u_min,
        u_max=u_max,
        terminal_state_cost=cost_fn.terminal_cost,
    )
    
    # Simulation loop
    state = torch.tensor(start, device=device, dtype=dtype)
    trajectory = [state[:3].cpu().numpy()]  # store xyz only
    states_full = [state.cpu().numpy()]     # full state for cost snapshots
    
    print("\n=== Starting MPPI Simulation (3D Drone) ===")
    print(f"Goal: {goal}, Start: {start[:3]}")
    
    for step in range(sim_steps):
        action = mppi_ctrl.command(state, shift_nominal_trajectory=True)
        
        state = dynamics(state, action)
        
        if state.dim() > 1:
            state = state.squeeze(0)
        
        pos = state[:3].cpu().numpy()
        trajectory.append(pos)
        states_full.append(state.cpu().numpy())
        
        dist_to_goal = np.linalg.norm(pos - goal)
        if step % 10 == 0 or step < 5:
            print(f"Step {step:2d}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], dist_to_goal={dist_to_goal:.2f}m")
        
        if dist_to_goal < 0.5:
            print(f"✓ Goal reached at step {step}!")
            break
    
    trajectory = np.array(trajectory)
    states_full = np.array(states_full)  # (T+1, 6)
    
    # Get cause points for visualization
    if fit_info is not None:
        # Load cause points from PCD for visualization
        if cause_pcd_path is None:
            default_pcd = Path("/home/navin/ros2_ws/src/buffers/run_20251221_144638_231_738a9b22/buffer1/points.pcd")
            cause_points_viz = load_pcd_points(str(default_pcd))
        else:
            cause_points_viz = load_pcd_points(str(cause_pcd_path))
        if len(cause_points_viz) == 0:
            cause_points_viz = None
    else:
        cause_points_viz = cause_points_raw if 'cause_points_raw' in locals() else None
    
    visualize_trajectory_3d(trajectory, start[:3], goal,
                           disturb_gp=disturb_gp, ref_traj=ref_traj, 
                           cause_points=cause_points_viz, cause_location=cause_location,
                           obstacles=obstacle_boxes)
    
    return trajectory


def visualize_trajectory_3d(trajectory, start, goal, disturb_gp=None, ref_traj=None, 
                           cause_points=None, cause_location=None, obstacles=None):
    """3D visualization with GP isosurfaces using marching cubes, reference path, MPPI trajectory, and box obstacles"""
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine bounds around cause location and trajectory
    if cause_location is not None:
        center = cause_location
        extent = 4.0  # Extent around cause location
        x_min, x_max = center[0] - extent, center[0] + extent
        y_min, y_max = center[1] - extent, center[1] + extent
        z_min, z_max = center[2] - extent, center[2] + extent
    else:
        x_min, x_max = trajectory[:, 0].min() - 1, trajectory[:, 0].max() + 1
        y_min, y_max = trajectory[:, 1].min() - 1, trajectory[:, 1].max() + 1
        z_min, z_max = trajectory[:, 2].min() - 1, trajectory[:, 2].max() + 1
    
    # Expand bounds to include trajectory
    x_min = min(x_min, trajectory[:, 0].min() - 0.5, start[0], goal[0])
    x_max = max(x_max, trajectory[:, 0].max() + 0.5, start[0], goal[0])
    y_min = min(y_min, trajectory[:, 1].min() - 0.5, start[1], goal[1])
    y_max = max(y_max, trajectory[:, 1].max() + 0.5, start[1], goal[1])
    z_min = min(z_min, trajectory[:, 2].min() - 0.5, start[2], goal[2])
    z_max = max(z_max, trajectory[:, 2].max() + 0.5, start[2], goal[2])
    
    # Visualize GP field as 3D isosurfaces using marching cubes
    if disturb_gp is not None:
        print("\n=== Generating GP field for isosurface visualization ===")
        resolution = 0.15  # Finer resolution for better isosurfaces
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        z_grid = np.arange(z_min, z_max, resolution)
        X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Flatten for GP evaluation
        grid_points = np.column_stack([
            X_grid.ravel(), 
            Y_grid.ravel(), 
            Z_grid.ravel()
        ])
        
        print(f"Evaluating GP at {len(grid_points)} grid points...")
        # Evaluate GP in chunks to avoid memory issues
        chunk_size = 50000
        gp_values = np.zeros(len(grid_points))
        grid_tensor_base = torch.tensor(grid_points, device=disturb_gp.cause.device, dtype=disturb_gp.cause.dtype)
        
        with torch.no_grad():
            for i in range(0, len(grid_points), chunk_size):
                end_idx = min(i + chunk_size, len(grid_points))
                chunk_tensor = grid_tensor_base[i:end_idx]
                gp_values[i:end_idx] = disturb_gp(chunk_tensor).cpu().numpy()
                if (i // chunk_size) % 5 == 0:
                    print(f"  Processed {end_idx}/{len(grid_points)} points...")
        
        gp_field = gp_values.reshape(X_grid.shape)
        print(f"GP field range: [{gp_field.min():.4f}, {gp_field.max():.4f}]")
        
        # Normalize for visualization
        gp_min, gp_max = gp_field.min(), gp_field.max()
        gp_field_norm = (gp_field - gp_min) / (gp_max - gp_min + 1e-9)
        
        # Use marching cubes for proper 3D isosurfaces
        try:
            from skimage import measure as sk_measure
            print("\n=== Rendering 3D isosurfaces using marching cubes ===")
            
            # Reshape for skimage (needs z, y, x order)
            Ny, Nx, Nz = len(y_grid), len(x_grid), len(z_grid)
            volume = np.transpose(gp_field_norm, (2, 1, 0))  # (Nz, Nx, Ny)
            
            # Compute isosurface levels based on percentiles
            iso_levels_norm = [0.3, 0.5, 0.7, 0.85]  # Normalized levels
            colors_iso = ['#2196F3', '#4CAF50', '#FFC107', '#F44336']  # Blue, Green, Yellow, Red
            alphas_iso = [0.2, 0.25, 0.3, 0.35]
            
            dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else resolution
            dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else resolution
            dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else resolution
            
            xmin, ymin, zmin = x_grid.min(), y_grid.min(), z_grid.min()
            
            for iso_val, color, alpha in zip(iso_levels_norm, colors_iso, alphas_iso):
                try:
                    verts, faces, normals, values = sk_measure.marching_cubes(
                        volume=volume, 
                        level=iso_val, 
                        spacing=(dz, dx, dy)
                    )
                    # Transform vertices to world coordinates
                    verts_world = np.column_stack([
                        verts[:, 2] + xmin,  # x
                        verts[:, 1] + ymin,  # y
                        verts[:, 0] + zmin,  # z
                    ])
                    ax.plot_trisurf(
                        verts_world[:, 0], 
                        verts_world[:, 1], 
                        faces, 
                        verts_world[:, 2], 
                        color=color, 
                        lw=0.0, 
                        edgecolor='none', 
                        alpha=alpha,
                        shade=True
                    )
                    print(f"  Rendered isosurface at level {iso_val:.2f} ({len(verts)} vertices)")
                except Exception as e:
                    print(f"  Warning: Isosurface at level {iso_val:.2f} failed: {e}")
            
        except ImportError:
            print("Warning: skimage not available, using fallback visualization")
            # Fallback: show as colored surface slices
            z_slice = trajectory[:, 2].mean() if len(trajectory) > 0 else (z_min + z_max) / 2
            z_idx = np.argmin(np.abs(z_grid - z_slice))
            gp_slice = gp_field_norm[:, :, z_idx]
            X_slice = X_grid[:, :, z_idx]
            Y_slice = Y_grid[:, :, z_idx]
            Z_slice = np.full_like(X_slice, z_grid[z_idx])
            
            colors = plt.cm.viridis(gp_slice)
            ax.plot_surface(X_slice, Y_slice, Z_slice, 
                           facecolors=colors, alpha=0.5, 
                           antialiased=True, zorder=1, shade=False)
    
    # Plot 3D box obstacles
    if obstacles is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for box_min, box_max in obstacles:
            # Create box vertices
            x_min, y_min, z_min = box_min
            x_max, y_max, z_max = box_max
            
            # Define the 8 vertices of the box
            vertices = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ])
            
            # Define the 6 faces of the box
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
            ]
            
            # Create 3D polygon collection
            box = Poly3DCollection(faces, alpha=0.4, facecolor='red', 
                                  edgecolor='darkred', linewidths=1.5, zorder=6)
            ax.add_collection3d(box)
    
    # Plot cause location marker
    if cause_location is not None:
        ax.scatter([cause_location[0]], [cause_location[1]], [cause_location[2]],
                  c='red', s=200, marker='X', label='Cause Location', 
                  zorder=12, edgecolors='white', linewidths=1.5)
    
    # Plot cause points (small, subtle)
    if cause_points is not None and len(cause_points) > 0:
        ax.scatter(cause_points[:, 0], cause_points[:, 1], cause_points[:, 2],
                  c='purple', s=8, alpha=0.4, label='Cause Points (PCD)', zorder=5)
    
    # Reference trajectory
    if ref_traj is not None:
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 
               '--', color='orange', linewidth=2.5, label='Reference', zorder=8, alpha=0.8)
    
    # MPPI Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
           'b-', linewidth=3.0, label='MPPI Trajectory', zorder=10)
    
    # Start and goal
    ax.scatter(start[0], start[1], start[2], color='green', s=200, 
              marker='o', label='Start', zorder=11, edgecolors='white', linewidths=2.0)
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=300, 
              marker='*', label='Goal', zorder=11, edgecolors='white', linewidths=2.0)
    
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_zlabel('Z [m]', fontsize=11)
    ax.set_title('3D MPPI Navigation with GP Isosurfaces from points.pcd', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/tmp/drone_mppi_3d_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Trajectory saved to /tmp/drone_mppi_3d_trajectory.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MPPI 3D Navigation with GP from points.pcd')
    parser.add_argument('--cause-pcd', type=str, default=None,
                       help='Path to points.pcd file (default: uses buffer1/points.pcd)')
    parser.add_argument('--buffer-dir', type=str, default=None,
                       help='Path to buffer directory for GP fitting')
    parser.add_argument('--nominal-path', type=str, default=None,
                       help='Path to nominal trajectory JSON')
    parser.add_argument('--no-fit-gp', action='store_true',
                       help='Disable GP fitting from buffer (use default parameters)')
    
    args = parser.parse_args()
    
    trajectory = simulate_drone_mppi_3d(
        cause_pcd_path=args.cause_pcd,
        buffer_dir=args.buffer_dir,
        nominal_path=args.nominal_path,
        use_fitted_gp=not args.no_fit_gp,
    )
    print(f"\nSimulation complete. Final trajectory length: {len(trajectory)} steps")


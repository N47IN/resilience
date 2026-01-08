#!/usr/bin/env python3
"""
MPPI 2D Navigation with Bicycle Model
- 2D bicycle dynamics (x, y, heading, speed)
- Disturbance-aware cost via 2D GP field
- Visualization with GP colormap (top-down)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pytorch_mppi import MPPI

# ============================================================================
# DISTURBANCE GP MODEL (2D)
# ============================================================================

class DisturbanceGP(torch.nn.Module):
    """
    2D superposed anisotropic RBF GP:
        f(p) = A * sum_j exp(-0.5 * d^2(p, c_j)) + b
    where d^2 = (dx^2 + dy^2) / lxy^2
    
    Outputs variance scale σ²(p) for position-dependent disturbance covariance.
    """
    def __init__(self, cause_points, lxy, A, b, device="cpu", dtype=torch.double):
        super().__init__()
        cause_points = torch.as_tensor(cause_points, device=device, dtype=dtype)
        self.register_buffer("cause", cause_points)  # (N, 2)

        self.lxy = torch.tensor(float(lxy), device=device, dtype=dtype)
        self.A = torch.tensor(float(A), device=device, dtype=dtype)
        self.b = torch.tensor(float(b), device=device, dtype=dtype)

        self.inv_lxy2 = 1.0 / (self.lxy * self.lxy + 1e-12)

    def forward(self, pos):
        """
        Args:
            pos: (K, 2) or (2,) positions
        Returns:
            (K,) variance scale σ²(p) at each position
        """
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)  # (1, 2)

        # (K, 1, 2) - (1, N, 2) -> (K, N, 2)
        diff = pos.unsqueeze(1) - self.cause.unsqueeze(0)
        dx2 = diff[..., 0] ** 2
        dy2 = diff[..., 1] ** 2
        d2 = (dx2 + dy2) * self.inv_lxy2  # (K, N)
        phi = torch.exp(-0.5 * d2).sum(dim=1)  # (K,)
        return self.A * phi + self.b


# ============================================================================
# BICYCLE MODEL DYNAMICS (2D)
# ============================================================================

class BicycleDynamics:
    """
    Bicycle model (2D) with worst-case disturbance
    State: [x, y, theta, v]
    Control: [a, delta]
    """
    def __init__(self, dt=0.1, wheelbase=0.5,
                 disturb_gp=None, obstacles=None,
                 disturb_scale=0.15,
                 device="cpu", dtype=torch.double):
        self.dt = dt
        self.L = wheelbase
        self.device = device
        self.dtype = dtype
        self.disturb_gp = disturb_gp
        self.obstacles = obstacles or []
        self.disturb_scale = disturb_scale  # Scale factor for disturbance push

        # Bounds
        self.a_min, self.a_max = -2.0, 2.0
        self.delta_min, self.delta_max = -0.6, 0.6
        self.v_min, self.v_max = -0.5, 3.0

    def __call__(self, state, action):
        """
        Args:
            state: (K x 4) [x, y, theta, v]
            action: (K x 2) [a, delta]
        Returns:
            next_state: (K x 4)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x, y, th, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        a = torch.clamp(action[:, 0], self.a_min, self.a_max)
        delta = torch.clamp(action[:, 1], self.delta_min, self.delta_max)

        # Deterministic dynamics: GP is used only in cost (risk), not in propagation
        disturb_x = torch.zeros_like(x)
        disturb_y = torch.zeros_like(y)

        v_next = torch.clamp(v + a * self.dt, self.v_min, self.v_max)
        th_next = th + (v / self.L) * torch.tan(delta) * self.dt

        x_next = x + v_next * torch.cos(th_next) * self.dt + disturb_x
        y_next = y + v_next * torch.sin(th_next) * self.dt + disturb_y

        return torch.stack([x_next, y_next, th_next, v_next], dim=1)


# ============================================================================
# COST FUNCTION (2D)
# ============================================================================

class DroneCost:
    """
    Look-ahead path tracking + geometric obstacle cost + GP-based risk shaping.
    Dynamics remain deterministic; GP only affects cost.
    """
    def __init__(self, goal,
                 ref_traj=None,  # (N, 2) reference path
                 obstacles=None,  # list of (center[x,y], radius)
                 disturb_gp=None,  # GP risk model (variance-like field)
                 ref_weight=1.50,       # Path tracking weight
                 obstacle_weight=10.0,  # Base obstacle avoidance
                 goal_weight=3.0,       # Goal attraction
                 control_weight=0.02,
                 lookahead_dist=1.5,    # Look-ahead distance for path tracking
                 alpha=1.0,             # Weight on variance risk term
                 beta=2.0,              # Weight on drift/dist-to-obstacle term
                 gamma=0.5,             # Speed scaling of risk
                 device="cpu", dtype=torch.double):
        self.goal = torch.tensor(goal, device=device, dtype=dtype)  # (2,)
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
        self.ref_traj = None if ref_traj is None else torch.tensor(ref_traj, device=device, dtype=dtype)  # (N, 2)
        self.obstacles = obstacles or []
        self.disturb_gp = disturb_gp
        
    def __call__(self, state, action, step=None):
        """
        Args:
            state: (K x 4)
            action: (K x 2)
        Returns:
            cost: (K,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        pos = state[:, :2]  # (K, 2)
        v = state[:, 3]     # (K,)

        # 1) Look-ahead path tracking: track point ahead on trajectory
        if self.ref_traj is not None:
            # Find closest point on reference path
            diff = pos.unsqueeze(1) - self.ref_traj.unsqueeze(0)  # (K, N, 2)
            ref_dists = torch.norm(diff, dim=2)  # (K, N)
            min_dist, closest_idx = torch.min(ref_dists, dim=1)  # (K,)
            
            # Look-ahead: target point ahead on path
            N = self.ref_traj.shape[0]
            lookahead_steps = max(5, int(self.lookahead_dist / 0.1))  # ~1.5m ahead
            target_idx = torch.clamp(closest_idx + lookahead_steps, max=N-1)
            
            # Track the look-ahead point
            target_points = self.ref_traj[target_idx]  # (K, 2)
            lookahead_dist = torch.norm(pos - target_points, dim=1)
            ref_cost = (lookahead_dist ** 2) * self.ref_weight
        else:
            ref_cost = 0.0

        # 2) Goal attraction for forward progress
        goal = self.goal.unsqueeze(0)
        goal_dist = torch.norm(pos - goal, dim=1)
        goal_cost = (goal_dist ** 2) * self.goal_weight

        # 3) Base obstacle cost (geometric, no extended field)
        obstacle_cost = torch.zeros(pos.shape[0], device=self.device, dtype=self.dtype)
        min_safe_dist = torch.full((pos.shape[0],), 10.0, device=self.device, dtype=self.dtype)
        for obs_center, obs_radius in self.obstacles:
            oc = torch.tensor(obs_center, device=self.device, dtype=self.dtype)
            dist = torch.norm(pos - oc, dim=1)
            safe_dist = dist - obs_radius
            min_safe_dist = torch.minimum(min_safe_dist, safe_dist)
            # Pure collision penalty (no surrounding cost field)
            obstacle_cost += torch.where(
                safe_dist < 0.0,
                torch.full_like(safe_dist, 1000.0 * self.obstacle_weight),
                torch.zeros_like(safe_dist),
            )

        # 4) GP-based risk shaping (semantic-conditioned residual risk)
        risk_cost = 0.0
        if self.disturb_gp is not None:
            with torch.no_grad():
                sigma_sq = torch.clamp(self.disturb_gp(pos), min=0.0)  # variance-like field
            var_risk = sigma_sq
            drift_mag = sigma_sq  # if you later learn a bias, plug it here

            # Distance to nearest obstacle (avoid division by zero)
            d_near = torch.clamp(min_safe_dist, min=0.1)

            # Core risk term: alpha * var + beta * drift / distance
            base_risk = self.alpha * var_risk + self.beta * (drift_mag / d_near)

            # Speed scaling
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
        # Extract final position (x, y)
        final_pos = states[..., -1, :2]  # (K, 2) or (2,)
        
        # Ensure 2D: (K, 2) for broadcasting
        if final_pos.dim() == 1:
            final_pos = final_pos.unsqueeze(0)  # (1, 2)
        
        goal = self.goal.unsqueeze(0)  # (1, 2)
        dist = torch.norm(final_pos - goal, dim=-1)  # (K,)
        # Moderate terminal cost for path tracking (less critical than running cost)
        return (dist ** 2) * self.goal_weight * 20.0


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_drone_mppi():
    """Run MPPI simulation for 2D bicycle navigation"""
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    print(f"Using device: {device}")
    
    # Simulation parameters
    dt = 0.05  # 10 Hz control
    sim_steps = 100  # Longer simulation for path tracking
    
    # State: [x, y, theta, v], Control: [a, delta]
    start = np.array([0., 0., 0., 0.])
    goal = np.array([8., 0.])  # straight-line goal
    
    # Randomize obstacles and GP each run (simple, reproducible if you set the seed above)
    rng = np.random.default_rng()  # independent from global np.random

    # Obstacles: one near start, one near goal, one near middle (above line)
    obs_start_x = rng.uniform(1.0, 2.0)
    obs_start_y = rng.uniform(-1.0, -0.4)
    obs_goal_x = rng.uniform(6.5, 7.5)
    obs_goal_y = rng.uniform(0.4, 1.2)
    obs_mid_x = rng.uniform(3.0, 5.0)
    obs_mid_y = rng.uniform(0.8, 1.6)

    obstacles = [
        ([0, 0], 0.4),
        ([obs_goal_x,  obs_goal_y],  0.5),
        ([obs_mid_x,   obs_mid_y],   0.4),
    ]
    
    # Single GP blob roughly in the middle, slightly randomized
    gp_center_x = rng.uniform(3.0, 5.0)
    gp_center_y = rng.uniform(-0.4, 0.4)
    gp_center = np.array([gp_center_x, gp_center_y])
    cause_points = rng.normal(size=(200, 2)) * 0.5 + gp_center  # 200 points around middle
    
    lxy = 0.45  # Length scale for GP
    A, b = 0.15, 0.0  # Amplitude and bias
    
    disturb_gp = DisturbanceGP(
        cause_points=cause_points,
        lxy=lxy, A=2*A, b=b,
        device=device, dtype=dtype,
    )

    # Reference trajectory: straight line along x-axis from start to goal
    ref_len = 120
    ref_traj = np.stack([
        np.linspace(start[0], goal[0], ref_len),
        np.linspace(start[1], goal[1], ref_len),
    ], axis=1)

    # Initialize dynamics (deterministic; GP only used in cost)
    dynamics = BicycleDynamics(
        dt=dt, wheelbase=0.5,
        disturb_gp=disturb_gp,
        obstacles=obstacles,
        disturb_scale=0.10,  # Moderate disturbance strength
        device=device, dtype=dtype,
    )
    cost_fn = DroneCost(
        goal=goal,
        ref_traj=ref_traj,
        obstacles=obstacles,
        disturb_gp=disturb_gp,
        ref_weight=15.0,       # Path tracking (look-ahead)
        obstacle_weight=10.0,  # Base obstacle avoidance
        goal_weight=3.0,       # Forward progress
        control_weight=0.02,
        lookahead_dist=1.5,    # 1.5m look-ahead
        alpha=1.0,
        beta=2.0,
        gamma=0.5,
        device=device, dtype=dtype,
    )
    
    # MPPI Configuration (path tracking with obstacle avoidance)
    nx = 4  # [x, y, theta, v]
    nu = 2  # [a, delta]
    num_samples = 3000 if device == "cuda" else 1500  # More samples for complex scenario
    horizon = 20  # Longer horizon to plan around obstacles
    lambda_ = 0.7  # Moderate temp for exploration
    
    # Control bounds
    u_max = torch.tensor([2.0, 0.6], device=device, dtype=dtype)
    u_min = -u_max
    
    # Control noise (moderate for obstacle avoidance exploration)
    noise_sigma = torch.diag(torch.tensor([0.35, 0.35], device=device, dtype=dtype))
    
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
    trajectory = [state[:2].cpu().numpy()]  # store xy only
    states_full = [state.cpu().numpy()]     # full state for cost snapshots
    
    print("\n=== Starting MPPI Simulation (2D Bicycle) ===")
    print(f"Goal: {goal}, Start: {start[:2]}")
    
    for step in range(sim_steps):
        action = mppi_ctrl.command(state, shift_nominal_trajectory=True)
        
        state = dynamics(state, action)
        
        if state.dim() > 1:
            state = state.squeeze(0)
        
        pos = state[:2].cpu().numpy()
        trajectory.append(pos)
        states_full.append(state.cpu().numpy())
        
        dist_to_goal = np.linalg.norm(pos - goal)
        if step % 5 == 0 or step < 5:
            print(f"Step {step:2d}: pos=[{pos[0]:.2f}, {pos[1]:.2f}], dist_to_goal={dist_to_goal:.2f}m")
        
        if dist_to_goal < 0.4:
            print(f"✓ Goal reached at step {step}!")
            break
    
    trajectory = np.array(trajectory)
    states_full = np.array(states_full)  # (T+1, 4)
    
    visualize_trajectory(trajectory, start[:2], goal,
                         disturb_gp=disturb_gp, ref_traj=ref_traj, obstacles=obstacles)
    visualize_cost_slices(states_full, cost_fn, obstacles)
    
    return trajectory


def visualize_trajectory(trajectory, start, goal, disturb_gp=None, ref_traj=None, obstacles=None):
    """2D visualization with GP field colormap, reference path, and obstacles"""
    fig = plt.figure(figsize=(8, 7))
    
    x_min, x_max = trajectory[:, 0].min() - 1, trajectory[:, 0].max() + 1
    y_min, y_max = trajectory[:, 1].min() - 1, trajectory[:, 1].max() + 1
    x_min = min(x_min, start[0], goal[0]) - 0.5
    x_max = max(x_max, start[0], goal[0]) + 0.5
    y_min = min(y_min, start[1], goal[1]) - 0.5
    y_max = max(y_max, start[1], goal[1]) + 0.5
    if obstacles:
        for oc, r in obstacles:
            x_min = min(x_min, oc[0] - r - 0.5)
            x_max = max(x_max, oc[0] + r + 0.5)
            y_min = min(y_min, oc[1] - r - 0.5)
            y_max = max(y_max, oc[1] + r + 0.5)

    resolution = 0.1
    ax = fig.add_subplot(111)

    if disturb_gp is not None:
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        grid_tensor = torch.tensor(grid_points, device=disturb_gp.cause.device, dtype=disturb_gp.cause.dtype)
        with torch.no_grad():
            gp_values = disturb_gp(grid_tensor).cpu().numpy()
        gp_field = gp_values.reshape(X_grid.shape)
        gp_field_norm = (gp_field - gp_field.min()) / (gp_field.max() - gp_field.min() + 1e-9)

        im = ax.contourf(X_grid, Y_grid, gp_field_norm, levels=30, cmap='viridis', alpha=0.75, zorder=1)
        ax.contour(X_grid, Y_grid, gp_field_norm, levels=10, colors='black', alpha=0.15, linewidths=0.5, zorder=2)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('GP Disturbance Field (normalized)', fontsize=9)

    # Trajectory and points
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2.5, label='Trajectory', zorder=10)
    ax.scatter(start[0], start[1], color='green', s=100, marker='o', label='Start', zorder=11, edgecolors='white', linewidths=1.2)
    ax.scatter(goal[0], goal[1], color='red', s=150, marker='*', label='Goal', zorder=11, edgecolors='white', linewidths=1.2)

    # Obstacles (simple discs only)
    if obstacles:
        for oc, r in obstacles:
            circle = plt.Circle((oc[0], oc[1]), r, color='red', alpha=0.35,
                                edgecolor='darkred', linewidth=1.5, zorder=6)
            ax.add_patch(circle)

    # Reference path
    if ref_traj is not None:
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], '--', color='orange', linewidth=1.8, label='Reference', zorder=8)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Top-Down View with GP Field')
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig('/tmp/drone_mppi_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Trajectory saved to /tmp/drone_mppi_trajectory.png")
    plt.show()


def visualize_cost_slices(states_full, cost_fn, obstacles=None, grid_extent=1.5, resolution=0.1):
    """
    Visualize running cost heatmaps at a few poses along the trajectory.
    For each selected pose, we fix theta, v and sweep over (x, y), evaluating cost_fn.
    """
    # Pick a few indices: start, middle, near end
    T = states_full.shape[0]
    idxs = [0, T // 2, max(T - 2, 0)]

    fig, axes = plt.subplots(1, len(idxs), figsize=(5 * len(idxs), 4))
    if len(idxs) == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        s = states_full[idx]  # [x, y, theta, v]
        x0, y0, th0, v0 = s

        # Local grid around this pose
        x_grid = np.arange(x0 - grid_extent, x0 + grid_extent, resolution)
        y_grid = np.arange(y0 - grid_extent, y0 + grid_extent, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts = np.column_stack([X.ravel(), Y.ravel()])  # (M, 2)

        # Build state tensor with fixed theta, v
        num_pts = pts.shape[0]
        state_grid = np.zeros((num_pts, 4), dtype=np.float64)
        state_grid[:, 0:2] = pts
        state_grid[:, 2] = th0
        state_grid[:, 3] = v0

        state_t = torch.tensor(state_grid, device=cost_fn.goal.device, dtype=cost_fn.goal.dtype)
        action_zero = torch.zeros((num_pts, 2), device=state_t.device, dtype=state_t.dtype)

        with torch.no_grad():
            c = cost_fn(state_t, action_zero)  # (M,)
        C = c.cpu().numpy().reshape(X.shape)

        im = ax.contourf(X, Y, C, levels=30, cmap='magma')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot obstacles on top (simple discs)
        if obstacles:
            for oc, r in obstacles:
                circle = plt.Circle((oc[0], oc[1]), r, color='cyan', alpha=0.3, zorder=3)
                ax.add_patch(circle)

        ax.set_title(f"Cost around pose t={idx}")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/drone_mppi_cost_slices.png', dpi=150, bbox_inches='tight')
    print("\n✓ Cost slices saved to /tmp/drone_mppi_cost_slices.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    trajectory = simulate_drone_mppi()
    print(f"\nSimulation complete. Final trajectory length: {len(trajectory)} steps")


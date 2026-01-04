#!/usr/bin/env python3
"""
MPPI Control Node for Robot-Centric GP Navigation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import json
import time


# Add pytorch_mppi to path
sys.path.insert(0, '/home/navin/ros2_ws/src/resilience/pytorch_mppi/src')
try:
    from pytorch_mppi import MPPI
except ImportError:
    print("Error: pytorch_mppi not found. Make sure the path is correct.")

# ============================================================================
# GRID-BASED GP MODEL
# ============================================================================

class GridDisturbanceGP(torch.nn.Module):
    """
    GP Model that interpolates values from a 3D grid.
    Supports Mean and Uncertainty channels.
    """
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Grid state
        self.grid_tensor = None  # (1, C, D, H, W) for grid_sample
        self.min_bound = None    # (3,) [min_x, min_y, min_z]
        self.max_bound = None    # (3,) [max_x, max_y, max_z]
        self.resolution = 0.1
        self.grid_size = None    # (3,) [nx, ny, nz]
        
    def update_grid(self, mean_data, uncert_data, metadata):
        """
        Update the internal grid from raw data.
        metadata: [min_x, min_y, min_z, res, nx, ny, nz]
        """
        min_x, min_y, min_z = metadata[0:3]
        res = metadata[3]
        nx, ny, nz = int(metadata[4]), int(metadata[5]), int(metadata[6])
        
        self.resolution = res
        self.min_bound = torch.tensor([min_x, min_y, min_z], device=self.device, dtype=self.dtype)
        self.grid_size = torch.tensor([nx, ny, nz], device=self.device, dtype=self.dtype)
        # Calculate max bound (inclusive of the last voxel center)
        # Grid covers [min, min + (n-1)*res] in terms of centers? 
        # Usually grid represents volume. Let's assume min is the corner.
        # Max bound for normalization should be min + size * res
        self.max_bound = self.min_bound + self.grid_size * res
        
        # Reshape data
        # Incoming data is flattened (N,). Shape is (nx, ny, nz)
        # Note: frontier_mapping_node used indexing='ij' for meshgrid -> (Nx, Ny, Nz)
        try:
            mean_grid = torch.as_tensor(mean_data, device=self.device, dtype=self.dtype).reshape(nx, ny, nz)
            uncert_grid = torch.as_tensor(uncert_data, device=self.device, dtype=self.dtype).reshape(nx, ny, nz)
            
            # Stack: (C, D, H, W) -> Here (C, Nx, Ny, Nz)
            # grid_sample expects (N, C, D, H, W) where D, H, W are Z, Y, X usually or arbitrary spatial dims.
            # We map: x->D (0), y->H (1), z->W (2) or similar.
            # Let's align with grid_sample's expected coordinate system [-1, 1].
            self.grid_tensor = torch.stack([mean_grid, uncert_grid], dim=0).unsqueeze(0) # (1, 2, nx, ny, nz)
            
        except Exception as e:
            print(f"Error reshaping grid: {e}")

    def normalize_coords(self, pos):
        """
        Normalize world coords to [-1, 1] for grid_sample.
        pos: (N, 3) [x, y, z]
        """
        # Map [min, max] to [-1, 1]
        # range = max - min
        # norm = (pos - min) / range * 2 - 1
        
        grid_range = self.max_bound - self.min_bound
        # Avoid div by zero
        grid_range = torch.clamp(grid_range, min=1e-6)
        
        norm_pos = (pos - self.min_bound) / grid_range * 2.0 - 1.0
        
        # Important: grid_sample uses (x, y, z) order as (W, H, D) or (D, H, W)?
        # 5D input is (N, C, D, H, W). grid is (N, D, H, W, 3).
        # Coordinates should be (x, y, z).
        # In our setup: grid dims are (nx, ny, nz) -> (D, H, W) correspond to (x, y, z) if we fed them that way.
        # We reshaped as (nx, ny, nz). So index 0 is x, 1 is y, 2 is z.
        # grid_sample expects coords in range [-1, 1].
        # By default grid_sample treats the last dimension of grid as W (x), then H (y), then D (z).
        # Wait, PyTorch docs say: 5D input (N, C, D, H, W). grid (N, D_out, H_out, W_out, 3).
        # The 3 coords are (x, y, z). x is W, y is H, z is D.
        # So coords should be (z_norm, y_norm, x_norm) if we map input dims to D, H, W?
        # Let's map: Input tensor (1, C, nx, ny, nz).
        # nx is depth (D), ny is height (H), nz is width (W)? No.
        # We want to access data[ix, iy, iz].
        # If we construct tensor as [nx, ny, nz], then:
        # dim 2 (d) corresponds to x index.
        # dim 3 (h) corresponds to y index.
        # dim 4 (w) corresponds to z index.
        # grid_sample uses x (last coord) -> W (dim 4), y -> H (dim 3), z -> D (dim 2).
        # So if we want to access [ix, iy, iz] using coords (u, v, w):
        # We need u -> iz (W), v -> iy (H), w -> ix (D).
        # So we should pass coords as (z, y, x).
        
        # Check alignment:
        # Pos is (x, y, z).
        # We want to sample from tensor of shape (nx, ny, nz).
        # Tensor D=nx, H=ny, W=nz.
        # grid_sample coords (x,y,z) map to (W, H, D).
        # So Grid X (last dim) samples W (dim nz) -> Z coordinate data.
        # Grid Y samples H (dim ny) -> Y coordinate data.
        # Grid Z samples D (dim nx) -> X coordinate data.
        # So we should pass (z, y, x) as the sampling coordinates.
        
        return torch.stack([norm_pos[:, 2], norm_pos[:, 1], norm_pos[:, 0]], dim=1)

    def forward(self, pos):
        """Return mean risk."""
        return self._sample(pos, channel=0)
        
    def forward_with_uncertainty(self, pos):
        """Return (mean, std)."""
        mean = self._sample(pos, channel=0)
        std = self._sample(pos, channel=1)
        return mean, std

    def _sample(self, pos, channel=0):
        if self.grid_tensor is None:
            return torch.zeros(pos.shape[0], device=self.device, dtype=self.dtype)
            
        # Reshape pos for grid_sample: (1, 1, 1, N, 3)
        # We act as if we are sampling a line of points (or just N points)
        # grid_sample expects (N_batch, D_out, H_out, W_out, 3)
        N = pos.shape[0]
        norm_coords = self.normalize_coords(pos) # (N, 3) -> (z, y, x)
        
        # Reshape to (1, 1, 1, N, 3)
        grid_coords = norm_coords.view(1, 1, 1, N, 3)
        
        # Sample
        # Input: (1, 2, nx, ny, nz)
        sampled = F.grid_sample(self.grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        # Output: (1, 2, 1, 1, N)
        
        # Extract channel values
        values = sampled[0, channel, 0, 0, :] # (N,)
        
        return values

# ============================================================================
# DYNAMICS & COST
# ============================================================================

class DroneDynamics3D:
    def __init__(self, dt=0.05, device="cpu", dtype=torch.float32):
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.v_min, self.v_max = -3.0, 3.0
        self.a_min, self.a_max = -2.0, 2.0

    def __call__(self, state, action):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
        
        ax = torch.clamp(action[:, 0], self.a_min, self.a_max)
        ay = torch.clamp(action[:, 1], self.a_min, self.a_max)
        az = torch.clamp(action[:, 2], self.a_min, self.a_max)

        vx_next = torch.clamp(vx + ax * self.dt, self.v_min, self.v_max)
        vy_next = torch.clamp(vy + ay * self.dt, self.v_min, self.v_max)
        vz_next = torch.clamp(vz + az * self.dt, self.v_min, self.v_max)

        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        z_next = z + vz_next * self.dt

        return torch.stack([x_next, y_next, z_next, vx_next, vy_next, vz_next], dim=1)

class MPPICost3D:
    def __init__(self, target_goal, nominal_path=None, gp_model=None, obstacle_cloud=None, 
                 device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.goal = torch.as_tensor(target_goal, device=device, dtype=dtype)
        self.nominal_path = None
        if nominal_path is not None:
            self.nominal_path = torch.as_tensor(nominal_path, device=device, dtype=dtype)
            
        self.gp_model = gp_model
        
        # Weights
        self.w_goal = 2.0
        self.w_ref = 1.0
        self.w_obs = 10.0
        self.w_risk = 2.0
        self.w_ctrl = 0.01
        
        # Obstacle Grid (Simple occupancy check or distance)
        # We'll use a KDTree or simple voxel lookup for now?
        # Since we are on GPU, a grid lookup is best.
        # We can reuse the GP grid's geometry for obstacle occupancy if we map it.
        self.obstacle_grid_tensor = None # (1, 1, nx, ny, nz)
        self.gp_meta = None # To map world to obstacle grid coords
        
    def set_obstacle_grid(self, occupied_indices, grid_shape):
        """
        occupied_indices: (N_obs, 3) integer indices in grid
        grid_shape: (nx, ny, nz)
        """
        if len(occupied_indices) == 0:
            self.obstacle_grid_tensor = None
            return

        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        grid = torch.zeros((nx, ny, nz), device=self.device, dtype=self.dtype)
        
        # Fill occupancy
        # Filter indices
        inds = torch.as_tensor(occupied_indices, device=self.device, dtype=torch.long)
        # Bounds check
        mask = (inds[:,0]>=0) & (inds[:,0]<nx) & (inds[:,1]>=0) & (inds[:,1]<ny) & (inds[:,2]>=0) & (inds[:,2]<nz)
        valid_inds = inds[mask]
        
        if len(valid_inds) > 0:
            grid[valid_inds[:,0], valid_inds[:,1], valid_inds[:,2]] = 1.0
            
        # Distance transform or Gaussian blur could smear obstacles for gradient
        # For now, binary cost
        self.obstacle_grid_tensor = grid.unsqueeze(0).unsqueeze(0) # (1, 1, nx, ny, nz)

    def __call__(self, state, action, step=None):
        pos = state[:, :3]
        
        # 1. Goal
        d_goal = torch.norm(pos - self.goal, dim=1)
        c_goal = self.w_goal * (d_goal**2)
        
        # 2. Ref Path
        c_ref = 0.0
        if self.nominal_path is not None:
            # Find nearest
            # Naive: broadcast (K, 1, 3) - (1, P, 3) -> (K, P, 3)
            # This is heavy if P is large.
            # Assume nominal path is pruned to local window
            dists = torch.norm(pos.unsqueeze(1) - self.nominal_path.unsqueeze(0), dim=2)
            min_dist, _ = torch.min(dists, dim=1)
            c_ref = self.w_ref * (min_dist**2)
            
        # 3. Obstacles (Grid Lookup)
        c_obs = 0.0
        if self.gp_model is not None and self.obstacle_grid_tensor is not None:
            # Re-use GP model's normalization logic to sample obstacle grid
            # Need to expose the sampling method or duplicate logic
             # Reshape pos for grid_sample: (1, 1, 1, N, 3)
            N = pos.shape[0]
            norm_coords = self.gp_model.normalize_coords(pos) # (z, y, x)
            grid_coords = norm_coords.view(1, 1, 1, N, 3)
            
            # Sample (Nearest for obstacles to be sharp, or Bilinear for smooth)
            samp = F.grid_sample(self.obstacle_grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
            occ = samp[0, 0, 0, 0, :] # (N,)
            
            c_obs = self.w_obs * (occ > 0.5).float() * 1000.0 # High penalty
            
        # 4. GP Risk
        c_risk = 0.0
        if self.gp_model is not None:
            mean, std = self.gp_model.forward_with_uncertainty(pos)
            # Conservative risk
            risk = mean + 2.0 * std
            c_risk = self.w_risk * torch.clamp(risk, min=0.0)
            
        # 5. Control
        c_ctrl = self.w_ctrl * torch.norm(action, dim=1)
        
        return c_goal + c_ref + c_obs + c_risk + c_ctrl

# ============================================================================
# NODE
# ============================================================================

class MPPIControlNode(Node):
    def __init__(self):
        super().__init__('mppi_control_node')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"MPPI Node starting on {self.device}")
        
        # Params
        self.nominal_path_file = '/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json'
        
        # State
        self.gp_model = GridDisturbanceGP(device=self.device)
        self.robot_pose = None
        self.nominal_path_points = None
        self.latest_grid_meta = None
        self.latest_obstacles_indices = None # VDB voxels mapped to grid indices
        
        # Load Nominal Path
        self.load_nominal_path()
        
        # MPPI
        self.mppi = None
        self.init_mppi()
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/mppi_path', 10)
        
        # Subscribers
        self.create_subscription(Float32MultiArray, '/gp_grid_raw', self.grid_callback, 10)
        self.create_subscription(PoseStamped, '/robot_1/sensors/front_stereo/pose', self.pose_callback, 10)
        self.create_subscription(PointCloud2, '/vdb_frontiers', self.obstacle_callback, 10) # Using frontiers/obstacles
        # Note: using /semantic_octomap_colored_cloud or something that gives occupied voxels might be better.
        # User said "treat regular voxels (vdb mapping) as obstacles".
        # Check frontier_node for the occupied cloud topic. it's /semantic_octomap_colored_cloud
        self.create_subscription(PointCloud2, '/semantic_octomap_colored_cloud', self.obstacle_callback, 10)

    def load_nominal_path(self):
        if os.path.exists(self.nominal_path_file):
            try:
                with open(self.nominal_path_file, 'r') as f:
                    data = json.load(f)
                    # format check needed, assume list of [x,y,z]
                    if isinstance(data, list):
                        self.nominal_path_points = np.array(data, dtype=np.float32)
                        self.get_logger().info(f"Loaded {len(data)} nominal points")
            except Exception as e:
                self.get_logger().error(f"Failed to load nominal path: {e}")

    def init_mppi(self):
        # Dynamics
        dynamics = DroneDynamics3D(device=self.device)
        
        # Initial Cost (dummy goal)
        cost_fn = MPPICost3D(target_goal=[0,0,0], device=self.device)
        
        nx = 6
        nu = 3
        noise_sigma = 0.5 * torch.eye(nu, device=self.device)
        
        self.mppi = MPPI(
            dynamics=dynamics,
            running_cost=cost_fn,
            nx=nx,
            noise_sigma=noise_sigma,
            num_samples=500,
            horizon=20,
            device=self.device,
            lambda_=0.1,
            u_min=torch.tensor([-2,-2,-2], device=self.device),
            u_max=torch.tensor([2,2,2], device=self.device)
        )

    def pose_callback(self, msg):
        self.robot_pose = msg

    def obstacle_callback(self, msg):
        """Process VDB cloud as obstacles."""
        if self.gp_model.min_bound is None:
            return
            
        # Parse PC2
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        if not points:
            return
            
        pts = np.array(points, dtype=np.float32)
        
        # Map to grid indices
        # idx = (pt - min) / res
        min_b = self.gp_model.min_bound.cpu().numpy()
        res = self.gp_model.resolution
        
        indices = np.floor((pts - min_b) / res).astype(np.int32)
        self.latest_obstacles_indices = indices

    def grid_callback(self, msg):
        """Main trigger: update grid -> run MPPI."""
        # Parse raw data
        # Layout: [Metadata(7) | Mean(N) | Uncert(N)]
        data = np.array(msg.data, dtype=np.float32)
        
        meta = data[0:7]
        # min_x, min_y, min_z, res, nx, ny, nz
        nx, ny, nz = int(meta[4]), int(meta[5]), int(meta[6])
        n_elements = nx * ny * nz
        
        mean_offset = 7
        uncert_offset = 7 + n_elements
        
        mean_data = data[mean_offset : mean_offset + n_elements]
        uncert_data = data[uncert_offset : uncert_offset + n_elements]
        
        # Update GP Model
        self.gp_model.update_grid(mean_data, uncert_data, meta)
        
        # Check for non-zero values Trigger
        # User: "whenever theres non zero values ... do mppi"
        if np.max(np.abs(mean_data)) < 1e-3 and np.max(uncert_data) < 1e-3:
            # self.get_logger().info("GP field zero, skipping MPPI")
            return
            
        self.run_control_loop()

    def run_control_loop(self):
        if self.robot_pose is None:
            return
            
        # Current State
        pos = self.robot_pose.pose.position
        # Vel? Need odom or estimate. Assume 0 for now or track?
        # MPPI needs state [x, y, z, vx, vy, vz]
        state = torch.tensor([pos.x, pos.y, pos.z, 0, 0, 0], device=self.device, dtype=torch.float32)
        
        # Determine Goal
        # "end of fwd traj with the grid cuboid"
        # Find intersection of nominal path with box
        local_goal = self.compute_local_goal(pos)
        
        # Update Cost Params
        self.mppi.running_cost.goal = torch.as_tensor(local_goal, device=self.device, dtype=torch.float32)
        self.mppi.running_cost.gp_model = self.gp_model
        
        # Update Obstacles
        if self.latest_obstacles_indices is not None and self.gp_model.grid_size is not None:
             self.mppi.running_cost.set_obstacle_grid(self.latest_obstacles_indices, self.gp_model.grid_size)
             
        # Nominal Path snippet for cost
        if self.nominal_path_points is not None:
             # Just pass the whole thing or crop. Passing whole is fine for GPU cost.
             self.mppi.running_cost.nominal_path = torch.as_tensor(self.nominal_path_points, device=self.device, dtype=torch.float32)
             
        # Run MPPI
        action = self.mppi.command(state)
        
        # Publish Trajectory
        self.publish_path()

    def compute_local_goal(self, current_pos):
        # Simple logic: Find point on nominal path strictly ahead of robot
        # that is near the edge of the definition box (e.g. 5m away)
        # Or just use the intersection with the grid box
        
        if self.nominal_path_points is None:
            return [current_pos.x + 2.0, current_pos.y, current_pos.z] # Dummy
            
        # Find index of nearest point
        curr_vec = np.array([current_pos.x, current_pos.y, current_pos.z])
        dists = np.linalg.norm(self.nominal_path_points - curr_vec, axis=1)
        idx_min = np.argmin(dists)
        
        # Look ahead until out of box
        box_half = 5.0 # xy
        box_min = curr_vec - box_half
        box_max = curr_vec + box_half
        
        goal = self.nominal_path_points[-1]
        
        for i in range(idx_min, len(self.nominal_path_points)):
            p = self.nominal_path_points[i]
            # Check containment
            if np.any(p < box_min) or np.any(p > box_max):
                goal = p
                break
        
        return goal

    def publish_path(self):
        # Get optimal trajectory from MPPI
        # Since pytorch_mppi command() returns action, we need to roll it out or get cached states?
        # The library might not expose cached states easily unless we modified it.
        # But we can re-roll with the optimal action sequence.
        
        actions = self.mppi.U # (T, nu)
        init_state = torch.tensor([self.robot_pose.pose.position.x, 
                                   self.robot_pose.pose.position.y, 
                                   self.robot_pose.pose.position.z, 0,0,0], device=self.device)
        
        states = [init_state]
        curr = init_state.unsqueeze(0)
        for t in range(actions.shape[0]):
            u = actions[t].unsqueeze(0)
            next_s = self.mppi.dynamics(curr, u)
            states.append(next_s.squeeze(0))
            curr = next_s
            
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for s in states:
            pose = PoseStamped()
            pose.pose.position.x = float(s[0])
            pose.pose.position.y = float(s[1])
            pose.pose.position.z = float(s[2])
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MPPIControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

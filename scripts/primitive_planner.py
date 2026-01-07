#!/usr/bin/env python3
"""
Motion Primitive Planner Node
Path planning using a library of motion primitives evaluated against a Robot-Centric GP Grid.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, ColorRGBA
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Point, Vector3, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import torch
import torch.nn.functional as F
import math
import sys
import os
import json

# ============================================================================
# GRID-BASED GP MODEL (COPIED FROM MPPI NODE)
# ============================================================================

class GridDisturbanceGP(torch.nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.grid_tensor = None  
        self.min_bound = None    
        self.max_bound = None    
        self.resolution = 0.1
        self.grid_size = None    
        
    def update_grid(self, mean_data, uncert_data, metadata):
        min_x, min_y, min_z = metadata[0:3]
        res = metadata[3]
        nx, ny, nz = int(metadata[4]), int(metadata[5]), int(metadata[6])
        self.resolution = res
        self.min_bound = torch.tensor([min_x, min_y, min_z], device=self.device, dtype=self.dtype)
        self.grid_size = torch.tensor([nx, ny, nz], device=self.device, dtype=self.dtype)
        self.max_bound = self.min_bound + self.grid_size * res
        try:
            mean_grid = torch.as_tensor(mean_data, device=self.device, dtype=self.dtype).reshape(nx, ny, nz)
            uncert_grid = torch.as_tensor(uncert_data, device=self.device, dtype=self.dtype).reshape(nx, ny, nz)
            self.grid_tensor = torch.stack([mean_grid, uncert_grid], dim=0).unsqueeze(0) 
        except Exception as e:
            print(f"Error reshaping grid: {e}")

    def normalize_coords(self, pos):
        grid_range = torch.clamp(self.max_bound - self.min_bound, min=1e-6)
        norm_pos = (pos - self.min_bound) / grid_range * 2.0 - 1.0
        return torch.stack([norm_pos[:, 2], norm_pos[:, 1], norm_pos[:, 0]], dim=1)

    def forward_with_uncertainty(self, pos):
        mean = self._sample(pos, channel=0)
        std = self._sample(pos, channel=1)
        return mean, std

    def _sample(self, pos, channel=0):
        if self.grid_tensor is None:
            return torch.zeros(pos.shape[0], device=self.device, dtype=self.dtype)
        N = pos.shape[0]
        norm_coords = self.normalize_coords(pos)
        grid_coords = norm_coords.view(1, 1, 1, N, 3)
        # grid_sample expects input in range [-1, 1]
        sampled = F.grid_sample(self.grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        return sampled[0, channel, 0, 0, :]

# ============================================================================
# MOTION PRIMITIVE LIBRARY
# ============================================================================

class MotionPrimitiveLibrary:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # Generator params
        self.speed = 1.0        # m/s
        self.horizon = 2.0      # seconds
        self.dt = 0.1          # time step
        self.num_steps = int(self.horizon / self.dt)
        
        # Define ranges for primitives
        # Curvature (yaw rate) - extending left and right
        self.yaw_rates = torch.linspace(-0.8, 0.8, 15, device=device, dtype=dtype)
        # Vertical angle (pitch) - extending up and down
        self.pitch_angles = torch.linspace(-0.4, 0.4, 7, device=device, dtype=dtype)
        
        # Create library parameters
        yy, pp = torch.meshgrid(self.yaw_rates, self.pitch_angles, indexing='ij')
        self.prim_ws = yy.flatten()       # Yaw rates
        self.prim_gammas = pp.flatten()   # Pitch angles
        self.num_prims = self.prim_ws.shape[0]
        
    def generate_primitives(self, start_pose):
        """
        Generate motion primitives from the current pose.
        start_pose: [x, y, z, yaw] (Tensor)
        Returns: Tensor of shape (NumPrims, NumSteps, 3) representing (x,y,z) trajectories
        """
        x0, y0, z0, yaw0 = start_pose
        
        # Initialize trajectories
        # Use simple kinematic model:
        # x_dot = v * cos(pitch) * cos(yaw)
        # y_dot = v * cos(pitch) * sin(yaw)
        # z_dot = v * sin(pitch)
        # yaw_dot = w
        
        trajectories = torch.zeros((self.num_prims, self.num_steps, 3), device=self.device, dtype=self.dtype)
        
        # Initial state for all primitives
        curr_x = x0.repeat(self.num_prims)
        curr_y = y0.repeat(self.num_prims)
        curr_z = z0.repeat(self.num_prims)
        curr_yaw = yaw0.repeat(self.num_prims)
        
        v_xy = self.speed * torch.cos(self.prim_gammas)
        v_z = self.speed * torch.sin(self.prim_gammas)
        
        for t in range(self.num_steps):
            # Update position
            curr_x = curr_x + v_xy * torch.cos(curr_yaw) * self.dt
            curr_y = curr_y + v_xy * torch.sin(curr_yaw) * self.dt
            curr_z = curr_z + v_z * self.dt # Constant vertical vel
            
            # Update yaw
            curr_yaw = curr_yaw + self.prim_ws * self.dt
            
            # Store
            trajectories[:, t, 0] = curr_x
            trajectories[:, t, 1] = curr_y
            trajectories[:, t, 2] = curr_z
            
        return trajectories

# ============================================================================
# MOTION PLANNER NODE
# ============================================================================

class MotionPrimitivePlannerNode(Node):
    def __init__(self):
        super().__init__('motion_primitive_planner_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Motion Primitive Planner Node on {self.device}")
        
        self.nominal_path_file = '/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json'
        
        self.gp_model = GridDisturbanceGP(device=self.device)
        self.primitive_lib = MotionPrimitiveLibrary(device=self.device)
        
        self.robot_pose = None
        self.nominal_path_points = None
        self.latest_obstacles_indices = None
        self.obstacle_grid_tensor = None
        
        self.load_nominal_path()
        
        # Publishers & Subscribers
        self.path_pub = self.create_publisher(Path, '/mppi_path', 10)
        self.primitives_pub = self.create_publisher(MarkerArray, '/motion_primitives/candidates', 10)
        
        self.create_subscription(Float32MultiArray, '/gp_grid_raw', self.grid_callback, 10)
        self.create_subscription(PoseStamped, '/robot_1/sensors/front_stereo/pose', self.pose_callback, 10)
        self.create_subscription(PointCloud2, '/semantic_octomap_colored_cloud', self.obstacle_callback, 10)
        
        self.create_timer(0.2, self.run_planner_loop) # Run at 5Hz

    def load_nominal_path(self):
        if os.path.exists(self.nominal_path_file):
            try:
                with open(self.nominal_path_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.nominal_path_points = torch.tensor(data, device=self.device, dtype=torch.float32)
            except Exception as e:
                self.get_logger().error(f"Path load error: {e}")

    def pose_callback(self, msg): 
        self.robot_pose = msg

    def obstacle_callback(self, msg):
        if self.gp_model.min_bound is None: return
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        if not points: return
        pts = np.array(points, dtype=np.float32)
        min_b = self.gp_model.min_bound.cpu().numpy()
        res = self.gp_model.resolution
        # Convert to grid indices
        occupied_indices = np.floor((pts - min_b) / res).astype(np.int32)
        self.update_obstacle_grid(occupied_indices)

    def update_obstacle_grid(self, occupied_indices):
        if len(occupied_indices) == 0:
            self.obstacle_grid_tensor = None
            return
        grid_shape = self.gp_model.grid_size.cpu().numpy()
        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        
        grid = torch.zeros((nx, ny, nz), device=self.device, dtype=torch.float32)
        inds = torch.as_tensor(occupied_indices, device=self.device, dtype=torch.long)
        mask = (inds[:,0]>=0) & (inds[:,0]<nx) & (inds[:,1]>=0) & (inds[:,1]<ny) & (inds[:,2]>=0) & (inds[:,2]<nz)
        valid_inds = inds[mask]
        if len(valid_inds) > 0:
            grid[valid_inds[:,0], valid_inds[:,1], valid_inds[:,2]] = 1.0
        self.obstacle_grid_tensor = grid.unsqueeze(0).unsqueeze(0)

    def grid_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        meta = data[0:7]
        nx, ny, nz = int(meta[4]), int(meta[5]), int(meta[6])
        n_elements = nx * ny * nz
        self.gp_model.update_grid(data[7 : 7+n_elements], data[7+n_elements : 7+2*n_elements], meta)

    def get_yaw_from_pose(self, pose):
        # Extract yaw from quaternion
        q = pose.orientation
        # sin(y) = 2(wz + xy)
        # cos(y) = 1 - 2(y^2 + z^2)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def compute_costs(self, trajectories, goal):
        # trajectories: [NumPrims, NumSteps, 3]
        # goal: [3]
        
        num_prims, num_steps, _ = trajectories.shape
        flat_traj = trajectories.reshape(-1, 3) # [NumPrims*NumSteps, 3]
        
        costs = torch.zeros(num_prims, device=self.device)
        
        # 1. Goal Cost (Distance to goal at last point)
        end_points = trajectories[:, -1, :]
        d_goal = torch.norm(end_points - goal, dim=1)
        costs += 5.0 * d_goal # Weight 5.0
        
        # 2. Reference Path Cost (Average distance to nominal path)
        if self.nominal_path_points is not None:
             # Just check distance of end points to nearest nominal point for efficiency
             # Or check all points? Let's check a few points to be faster? 
             # Let's check middle and end point
             # Broadcost nominal path: [1, N_ref, 3] vs [NumPrims, 1, 3]
             # This can be heavy. Let's simplfy: Dist from end_point to nearest ref point
             dists = torch.cdist(end_points, self.nominal_path_points) # [NumPrims, N_ref]
             min_dists, _ = torch.min(dists, dim=1)
             costs += 2.0 * min_dists
        
        # 3. Obstacle Cost
        if self.obstacle_grid_tensor is not None:
            norm_coords = self.gp_model.normalize_coords(flat_traj)
            grid_coords = norm_coords.view(1, 1, 1, -1, 3)
            samp = F.grid_sample(self.obstacle_grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
            # samp shape: [1, 1, 1, 1, NumPrims*NumSteps] -> view
            obs_vals = samp.view(num_prims, num_steps)
            # Sum up obstacle collision along trajectory
            obs_cost = torch.sum(obs_vals, dim=1)
            costs += 50.0 * obs_cost
            
        # 4. GP Risk Cost
        if self.gp_model.grid_tensor is not None:
            mean, std = self.gp_model.forward_with_uncertainty(flat_traj)
            risk_val = mean + 2.0 * std
            # Filter low risk to save compute or just apply barrier
            # Exponential barrier
            risk_cost_steps = torch.exp(2.0 * risk_val).view(num_prims, num_steps)
            risk_cost = torch.mean(risk_cost_steps, dim=1)
            costs += 15.0 * risk_cost
            
        return costs

    def find_local_goal(self, current_pos):
        if self.nominal_path_points is None:
            # Default forward if no path
            # Assuming robot faces +x roughly or just extend forward
            return torch.tensor([current_pos.x + 5.0, current_pos.y, current_pos.z], device=self.device)
        
        curr_vec = torch.tensor([current_pos.x, current_pos.y, current_pos.z], device=self.device)
        dists = torch.norm(self.nominal_path_points - curr_vec, dim=1)
        min_idx = torch.argmin(dists)
        
        # Lookahead
        lookahead_dist = 6.0
        target_idx = min_idx
        for i in range(min_idx, len(self.nominal_path_points)):
            if torch.norm(self.nominal_path_points[i] - curr_vec) > lookahead_dist:
                target_idx = i
                break
        return self.nominal_path_points[target_idx]

    def run_planner_loop(self):
        if self.robot_pose is None: return
        
        with torch.no_grad():
            pos = self.robot_pose.pose.position
            yaw = self.get_yaw_from_pose(self.robot_pose.pose)
            start_pose_t = torch.tensor([pos.x, pos.y, pos.z, yaw], device=self.device)
            
            # Generate primitives
            trajectories = self.primitive_lib.generate_primitives(start_pose_t)
            
            # Get goal
            goal = self.find_local_goal(pos)
            
            # Compute costs
            costs = self.compute_costs(trajectories, goal)
            
            # Find best
            best_idx = torch.argmin(costs)
            best_traj = trajectories[best_idx]
            
            # Visualize / Publish
            self.publish_best_path(best_traj)
            self.publish_primitives(trajectories, costs)

    def publish_best_path(self, traj):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        scaler_cpu = traj.cpu().numpy()
        for i in range(scaler_cpu.shape[0]):
            p = PoseStamped()
            p.pose.position.x = float(scaler_cpu[i, 0])
            p.pose.position.y = float(scaler_cpu[i, 1])
            p.pose.position.z = float(scaler_cpu[i, 2])
            # Orientation? identity for now
            path_msg.poses.append(p)
        
        self.path_pub.publish(path_msg)

    def publish_primitives(self, trajectories, costs):
        marker_array = MarkerArray()

        # 1. Robust Normalization (Clip outliers to top 90% percentile to preserve color gradient for good paths)
        #    Otherwise, one "infinite" cost obstacle path makes everything else look identical (green).
        min_c = torch.min(costs)
        
        # Calculate 90th percentile to ignore extreme outliers for coloring
        k = int(0.9 * len(costs))
        if k < len(costs) - 1:
            sorted_costs, _ = torch.sort(costs)
            max_c = sorted_costs[k]
        else:
            max_c = torch.max(costs)
            
        # Avoid division by zero
        denom = max_c - min_c
        if denom < 1e-6:
            denom = 1.0
            
        # Normalize to [0, 1] range, clipping anything above max_c to 1.0
        norm_costs = torch.clamp((costs - min_c) / denom, 0.0, 1.0)

        traj_cpu = trajectories.cpu().numpy()
        costs_cpu = norm_costs.cpu().numpy()
        
        # Get index of the best path to highlight it
        best_idx = torch.argmin(costs).item()

        timestamp = self.get_clock().now().to_msg()
        
        # Create a "delete all" marker to clear stale paths if number of primitives changes
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i in range(traj_cpu.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = timestamp
            marker.ns = "motion_primitives"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            
            # Scale: Thicker for best path
            if i == best_idx:
                marker.scale.x = 0.08  # Thick for best
                marker.pose.position.z += 0.05 # Draw slightly above others
            else:
                marker.scale.x = 0.02  # Thin for candidates

            # Color Map: Turbo/Jet style manual implementation
            # Low cost (0.0) -> Green/Blue
            # Med cost (0.5) -> Yellow/Orange
            # High cost (1.0) -> Red
            
            c_val = float(costs_cpu[i])
            
            if i == best_idx:
                # Best path is pure Cyan
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
            else:
                # Gradient from Green (low cost) to Red (high cost)
                # You can tweak this. Here is Green -> Yellow -> Red
                if c_val < 0.5:
                    # Green to Yellow
                    marker.color.r = 2.0 * c_val
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:
                    # Yellow to Red
                    marker.color.r = 1.0
                    marker.color.g = 2.0 * (1.0 - c_val)
                    marker.color.b = 0.0
                
                # Make high cost paths more transparent so they don't clutter view
                marker.color.a = 0.8 - (0.5 * c_val)

            for t in range(traj_cpu.shape[1]):
                p = Point()
                p.x = float(traj_cpu[i, t, 0])
                p.y = float(traj_cpu[i, t, 1])
                p.z = float(traj_cpu[i, t, 2])
                marker.points.append(p)

            marker_array.markers.append(marker)

        self.primitives_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPrimitivePlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Path Manager Module

Handles unified path planning interface supporting two modes:
1. JSON file mode: Load nominal path from JSON and publish to global path topic
2. External planner mode: Listen to external planner's global path topic

Provides a consistent interface for drift detection regardless of path source.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from threading import Lock


@dataclass
class DiscretizedPoint:
    """Discretized trajectory point"""
    position: np.ndarray  # 3D position [x, y, z]
    index: int
    distance_from_start: float


class TrajectoryDiscretizer:
    """Discretizes trajectory based on sampling length"""
    
    def __init__(self, sampling_length: float = 0.1):
        self.sampling_length = sampling_length
    
    def discretize_trajectory(self, points: List[dict]) -> List[DiscretizedPoint]:
        """Discretize trajectory from JSON points"""
        if not points:
            return []
        
        # Convert to numpy array with coordinate convention: forward->X, left->Y, up->Z
        positions = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                             for p in points])
        
        discretized = []
        current_distance = 0.0
        discretized.append(DiscretizedPoint(
            position=positions[0],
            index=0,
            distance_from_start=0.0
        ))
        
        # Walk along trajectory and sample at regular intervals
        for i in range(1, len(positions)):
            segment_length = np.linalg.norm(positions[i] - positions[i-1])
            current_distance += segment_length
            
            # Add points at sampling_length intervals
            while current_distance >= self.sampling_length:
                # Interpolate position along the segment
                alpha = self.sampling_length / segment_length
                interpolated_pos = positions[i-1] + alpha * (positions[i] - positions[i-1])
                
                discretized.append(DiscretizedPoint(
                    position=interpolated_pos,
                    index=len(discretized),
                    distance_from_start=len(discretized) * self.sampling_length
                ))
                
                current_distance -= self.sampling_length
                segment_length -= self.sampling_length
        
        return discretized
    
    def discretize_path_message(self, path_msg) -> List[DiscretizedPoint]:
        """Discretize trajectory from ROS Path message"""
        if not path_msg or len(path_msg.poses) == 0:
            return []
        
        # Convert Path message to discretized points
        points = []
        for i, pose_stamped in enumerate(path_msg.poses):
            point = {
                'position': {
                    'x': float(pose_stamped.pose.position.x),
                    'y': float(pose_stamped.pose.position.y),
                    'z': float(pose_stamped.pose.position.z)
                }
            }
            points.append(point)
        
        return self.discretize_trajectory(points)


class PathManager:
    """Unified path manager for resilience system."""
    
    def __init__(self, node: Node, config: Dict[str, Any]):
        """
        Initialize path manager.
        
        Args:
            node: ROS2 node instance
            config: Path configuration dictionary
        """
        self.node = node
        self.config = config
        self.lock = Lock()
        
        # Discretization configuration
        discretization_config = config.get('discretization', {})
        self.sampling_distance = discretization_config.get('sampling_distance', 0.1)
        self.lookback_window_size = discretization_config.get('lookback_window_size', 20)
        self.default_soft_threshold = discretization_config.get('default_soft_threshold', 0.3)
        
        # Initialize discretizer
        self.discretizer = TrajectoryDiscretizer(self.sampling_distance)
        
        # Path state
        self.nominal_points = []
        self.discretized_nominal = []  # List[DiscretizedPoint]
        self.nominal_np = None  # Initialize as None
        self.soft_threshold = self.default_soft_threshold  # Use default from config
        self.hard_threshold = 0.5
        self.initial_pose = np.array([0.0, 0.0, 0.0])
        self.path_ready = False
        self.last_path_update = 0.0
        
        # Mode configuration
        self.mode = config.get('mode', 'json_file')
        self.global_path_topic = config.get('global_path_topic', '/global_path')
        self.json_config = config.get('json_file', {})
        self.external_config = config.get('external_planner', {})
        
        # Publishers and subscribers
        self.path_publisher = None
        self.path_subscriber = None
        
        # Initialize based on mode
        if self.mode == 'json_file':
            self._init_json_mode()
        elif self.mode == 'external_planner':
            self._init_external_mode()
        else:
            raise ValueError(f"Invalid path mode: {self.mode}")
    
    def _init_json_mode(self):
        """Initialize JSON file mode - load path and publish to topic."""
        try:
            # Load nominal path from JSON file
            nominal_path_file = self.json_config.get('nominal_path_file', 'adjusted_nominal_spline.json')
            
            # Try to find the file in assets directory
            from ament_index_python.packages import get_package_share_directory
            package_dir = get_package_share_directory('resilience')
            json_path = os.path.join(package_dir, 'assets', nominal_path_file)
            
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Nominal path file not found: {json_path}")
            
            # Load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.nominal_points = data['points']
            
            # Use thresholds from JSON if available, otherwise use config defaults
            if 'calibration' in data:
                self.soft_threshold = data['calibration'].get('soft_threshold', self.default_soft_threshold)
                self.hard_threshold = data['calibration'].get('hard_threshold', 0.5)
            else:
                self.soft_threshold = self.default_soft_threshold
                self.hard_threshold = 0.5
            
            # Discretize the trajectory
            self.discretized_nominal = self.discretizer.discretize_trajectory(self.nominal_points)
            
            # Convert discretized points to numpy array for efficient computation
            if self.discretized_nominal:
                self.nominal_np = np.array([point.position for point in self.discretized_nominal])
                self.initial_pose = self.nominal_np[0]
            else:
                self.nominal_np = np.array([])
                self.initial_pose = np.array([0.0, 0.0, 0.0])
            
            self.path_ready = True
            self.node.get_logger().info(f"Loaded nominal path from {json_path} with {len(self.nominal_points)} points")
            self.node.get_logger().info(f"Discretized to {len(self.discretized_nominal)} points with {self.sampling_distance}m sampling")
            
            # Print detailed discretization status
            print("=" * 60)
            print("JSON PATH DISCRETIZATION STATUS")
            print("=" * 60)
            print(f"JSON file: {json_path}")
            print(f"Original path points: {len(self.nominal_points)}")
            print(f"Discretized points: {len(self.discretized_nominal)}")
            print(f"Sampling distance: {self.sampling_distance:.3f}m")
            print(f"Lookback window: {self.lookback_window_size} points")
            print(f"Soft threshold: {self.soft_threshold:.3f}m")
            print(f"Hard threshold: {self.hard_threshold:.3f}m")
            print("=" * 60)
            
            # Create publisher for global path topic
            self.path_publisher = self.node.create_publisher(
                Path, 
                self.global_path_topic, 
                10
            )
            
            # Start publishing the nominal path
            self._publish_nominal_path()
            
            # Set up timer for periodic publishing
            publish_rate = self.json_config.get('publish_rate', 1.0)
            self.publish_timer = self.node.create_timer(
                1.0 / publish_rate, 
                self._publish_nominal_path
            )
            
        except Exception as e:
            self.node.get_logger().error(f"Failed to initialize JSON mode: {e}")
            raise
    
    def _init_external_mode(self):
        """Initialize external planner mode - simple one-time listener for a path message."""
        try:
            # Use thresholds from external config, fallback to defaults
            thresholds_config = self.external_config.get('thresholds', {})
            self.soft_threshold = float(thresholds_config.get('soft_threshold', self.default_soft_threshold))
            self.hard_threshold = float(thresholds_config.get('hard_threshold', 0.5))
            
            # Create subscriber for external global path (one-time trigger upon first valid message)
            self.path_subscriber = self.node.create_subscription(
                Path,
                self.global_path_topic,
                self._external_path_callback,
                10
            )
            
            self.node.get_logger().info(f"Listening for external path on topic: {self.global_path_topic}")
            self.node.get_logger().info(f"Using discretization: {self.sampling_distance}m sampling, {self.lookback_window_size} point lookback")
        except Exception as e:
            self.node.get_logger().error(f"Failed to initialize external mode: {e}")
            raise
    
    def _publish_nominal_path(self):
        """Publish nominal path to global path topic."""
        if not self.path_ready or self.path_publisher is None:
            return
        
        try:
            path_msg = Path()
            path_msg.header = Header()
            path_msg.header.stamp = self.node.get_clock().now().to_msg()
            path_msg.header.frame_id = "map"
            
            for point in self.nominal_points:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = point['position']['x']
                pose.pose.position.y = point['position']['y']
                pose.pose.position.z = point['position']['z']
                pose.pose.orientation.w = 1.0  # Default orientation
                path_msg.poses.append(pose)
            
            self.path_publisher.publish(path_msg)
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing nominal path: {e}")
    
    def _external_path_callback(self, path_msg: Path):
        """Handle external path message (one-time store and trigger)."""
        try:
            # Ignore if we've already set a path (one-time trigger)
            if self.path_ready:
                return
            
            # Validate message contains poses
            if path_msg is None or len(path_msg.poses) == 0:
                self.node.get_logger().warn("Received empty path; waiting for a valid external path...")
                return
            
            with self.lock:
                # Convert Path message to the internal simple format expected elsewhere
                self.nominal_points = []
                for i, pose_stamped in enumerate(path_msg.poses):
                    point = {
                        'position': {
                            'x': float(pose_stamped.pose.position.x),
                            'y': float(pose_stamped.pose.position.y),
                            'z': float(pose_stamped.pose.position.z)
                        }
                    }
                    self.nominal_points.append(point)
                
                # Discretize the external path
                self.discretized_nominal = self.discretizer.discretize_path_message(path_msg)
                
                # Convert discretized points to numpy array for fast distance checks
                if self.discretized_nominal:
                    self.nominal_np = np.array([point.position for point in self.discretized_nominal])
                    self.initial_pose = self.nominal_np[0]
                else:
                    self.nominal_np = np.array([])
                    self.initial_pose = np.array([0.0, 0.0, 0.0])
                
                self.path_ready = True
                self.last_path_update = time.time()
            
            self.node.get_logger().info(f"✓ External path received: {len(self.nominal_points)} points. Path ready.")
            self.node.get_logger().info(f"Discretized to {len(self.discretized_nominal)} points with {self.sampling_distance}m sampling")
            
            # Print detailed discretization status
            print("=" * 60)
            print("EXTERNAL PATH DISCRETIZATION STATUS")
            print("=" * 60)
            print(f"Original path points: {len(self.nominal_points)}")
            print(f"Discretized points: {len(self.discretized_nominal)}")
            print(f"Sampling distance: {self.sampling_distance:.3f}m")
            print(f"Lookback window: {self.lookback_window_size} points")
            print(f"Soft threshold: {self.soft_threshold:.3f}m")
            print(f"Hard threshold: {self.hard_threshold:.3f}m")
            print("=" * 60)
            
            # Notify the main node that path has been updated and enable processing
            try:
                setattr(self.node, 'path_ready', True)
                setattr(self.node, 'disable_drift_detection', False)
                self.node.get_logger().info("Notified main node: path_ready=TRUE, drift detection ENABLED")
                if hasattr(self.node, 'narration_manager'):
                    nominal_points = self.get_nominal_points_as_numpy()
                    if len(nominal_points) > 0:
                        self.node.narration_manager.update_intended_trajectory(nominal_points)
                        self.node.get_logger().info("Updated narration manager with external path")

            except Exception:
                pass
            
            # One-time listener: stop listening after first valid path
            try:
                if self.path_subscriber is not None:
                    self.node.destroy_subscription(self.path_subscriber)
                    self.path_subscriber = None
                    self.node.get_logger().info("Stopped listening for external path (one-time trigger).")
            except Exception:
                pass
        except Exception as e:
            self.node.get_logger().error(f"Error processing external path: {e}")
            import traceback
            traceback.print_exc()
    
    def is_ready(self) -> bool:
        """Check if path manager is ready."""
        return self.path_ready and self.nominal_np is not None and len(self.nominal_points) > 0
    
    def wait_for_path(self, timeout_seconds: float = 30.0) -> bool:
        """
        Wait for path to be ready.
        
        Args:
            timeout_seconds: Maximum time to wait for path
            
        Returns:
            True if path is ready within timeout, False otherwise
        """
        start_time = time.time()
        while not self.is_ready() and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)
        
        if self.is_ready():
            self.node.get_logger().info(f"Path ready after {time.time() - start_time:.1f}s")
            return True
        else:
            self.node.get_logger().error(f"Path not ready after {timeout_seconds}s timeout")
            return False
    
    def get_discretized_nominal_points(self) -> List[DiscretizedPoint]:
        """Get discretized nominal trajectory points."""
        with self.lock:
            return self.discretized_nominal.copy()
    
    def get_discretized_nominal_as_numpy(self) -> np.ndarray:
        """Get discretized nominal trajectory as numpy array for narration manager."""
        with self.lock:
            if self.discretized_nominal and len(self.discretized_nominal) > 0:
                return np.array([point.position for point in self.discretized_nominal])
            else:
                return np.array([])
    
    def get_lookback_window_size(self) -> int:
        """Get lookback window size."""
        return self.lookback_window_size
    
    def get_sampling_distance(self) -> float:
        """Get sampling distance."""
        return self.sampling_distance

    def get_nominal_points(self) -> List[Dict]:
        """Get nominal trajectory points."""
        with self.lock:
            return self.nominal_points.copy()
    
    def get_nominal_points_as_numpy(self) -> np.ndarray:
        """Get nominal trajectory as numpy array for narration manager."""
        with self.lock:
            if self.nominal_np is not None and len(self.nominal_np) > 0:
                return self.nominal_np.copy()
            else:
                return np.array([])
    
    def get_nominal_np(self) -> np.ndarray:
        """Get nominal trajectory as numpy array."""
        with self.lock:
            if self.nominal_np is not None and len(self.nominal_np) > 0:
                return self.nominal_np.copy()
            else:
                return np.array([])
    
    def get_initial_pose(self) -> np.ndarray:
        """Get initial pose."""
        with self.lock:
            return self.initial_pose.copy()
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get drift thresholds."""
        return self.soft_threshold, self.hard_threshold
    
    def compute_drift(self, pos: np.ndarray) -> Tuple[float, int]:
        """Compute drift between current position and nearest nominal point."""
        with self.lock:
            if self.nominal_np is None or len(self.nominal_np) == 0:
                return 0.0, 0
            
            dists = np.linalg.norm(self.nominal_np - pos, axis=1)
            nearest_idx = int(np.argmin(dists))
            drift = dists[nearest_idx]
            return drift, nearest_idx
    
    def is_breach(self, drift: float) -> bool:
        """Check if drift exceeds soft threshold."""
        return drift > self.soft_threshold
    
    def get_mode(self) -> str:
        """Get current path mode."""
        return self.mode
    
    def get_path_topic(self) -> str:
        """Get global path topic name."""
        return self.global_path_topic
    
    def update_thresholds(self, soft_threshold: float, hard_threshold: float):
        """Update drift thresholds dynamically (useful for external planner mode)."""
        with self.lock:
            self.soft_threshold = soft_threshold
            self.hard_threshold = hard_threshold
            self.node.get_logger().info(f"Updated thresholds - soft: {soft_threshold}, hard: {hard_threshold}")
    
    def get_threshold_source(self) -> str:
        """Get the source of current thresholds."""
        if self.mode == 'json_file':
            return "JSON file calibration"
        else:
            return "External planner config" 
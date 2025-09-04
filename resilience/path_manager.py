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
from typing import Optional, List, Dict, Any, Tuple
from threading import Lock


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
        
        # Path state
        self.nominal_points = []
        self.nominal_np = None  # Initialize as None
        self.soft_threshold = 0.1
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
            self.soft_threshold = data['calibration']['soft_threshold']
            self.hard_threshold = data['calibration']['hard_threshold']
            
            # Convert to numpy array for efficient computation
            self.nominal_np = np.array([
                np.array([p['position']['x'], p['position']['y'], p['position']['z']]) 
                for p in self.nominal_points
            ])
            
            if len(self.nominal_np) > 0:
                self.initial_pose = self.nominal_np[0]
            
            self.path_ready = True
            self.node.get_logger().info(f"Loaded nominal path from {json_path} with {len(self.nominal_points)} points")
            
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
        """Initialize external planner mode - listen to external path topic."""
        try:
            # Set default thresholds from config
            thresholds_config = self.external_config.get('thresholds', {})
            self.soft_threshold = thresholds_config.get('soft_threshold', 0.1)
            self.hard_threshold = thresholds_config.get('hard_threshold', 0.25)
            
            self.node.get_logger().info(f"External mode: Using default thresholds - soft: {self.soft_threshold}, hard: {self.hard_threshold}")
            
            # Create subscriber for external global path
            self.path_subscriber = self.node.create_subscription(
                Path,
                self.global_path_topic,
                self._external_path_callback,
                10
            )
            
            self.node.get_logger().info(f"Listening for external path on topic: {self.global_path_topic}")
            
            # Set up timeout checking
            timeout_seconds = self.external_config.get('timeout_seconds', 10.0)
            self.timeout_timer = self.node.create_timer(
                1.0,  # Check every second
                lambda: self._check_external_path_timeout(timeout_seconds)
            )
            
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
        """Handle external path message."""
        try:
            with self.lock:
                # Convert Path message to internal format
                self.nominal_points = []
                for i, pose_stamped in enumerate(path_msg.poses):
                    point = {
                        'index': i,
                        'position': {
                            'x': pose_stamped.pose.position.x,
                            'y': pose_stamped.pose.position.y,
                            'z': pose_stamped.pose.position.z
                        }
                    }
                    self.nominal_points.append(point)
                
                # Convert to numpy array
                self.nominal_np = np.array([
                    np.array([p['position']['x'], p['position']['y'], p['position']['z']]) 
                    for p in self.nominal_points
                ])
                
                if len(self.nominal_np) > 0:
                    self.initial_pose = self.nominal_np[0]
                
                self.path_ready = True
                self.last_path_update = time.time()
                
                self.node.get_logger().info(f"✓ External path received and processed with {len(self.nominal_points)} points")
                
                # Notify the main node that path has been updated
                if hasattr(self.node, 'narration_manager'):
                    nominal_points = self.get_nominal_points_as_numpy()
                    if len(nominal_points) > 0:
                        self.node.narration_manager.update_intended_trajectory(nominal_points)
                        self.node.get_logger().info("Updated narration manager with external path")
                
                # Ensure main node can proceed even if it previously timed out waiting
                try:
                    setattr(self.node, 'path_ready', True)
                    setattr(self.node, 'disable_drift_detection', False)
                except Exception:
                    pass
        except Exception as e:
            self.node.get_logger().error(f"Error processing external path: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_external_path_timeout(self, timeout_seconds: float):
        """Check if external path has timed out."""
        if self.mode != 'external_planner':
            return
        
        current_time = time.time()
        if (not self.path_ready or 
            (current_time - self.last_path_update) > timeout_seconds):
            
            require_path = self.external_config.get('require_path', True)
            if require_path:
                self.node.get_logger().warn(
                    f"No external path received for {timeout_seconds}s on topic {self.global_path_topic}"
                )
    
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
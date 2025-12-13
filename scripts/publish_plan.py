#!/usr/bin/env python3
"""
Publish Plan Node

Simple node that loads a nominal path from JSON file and publishes it to a global path topic.
Used for testing external planner mode in the resilience system.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import json
import os
import time
from typing import List, Dict, Any


class PublishPlanNode(Node):
    """Node that publishes nominal path from JSON file."""
    
    def __init__(self):
        super().__init__('publish_plan_node')
        
        # Declare parameters
        self.declare_parameters('', [
            ('json_file_path', ''),
            ('global_path_topic', '/global_path'),
            ('publish_rate', 1.0),
            ('frame_id', 'map')
        ])
        
        # Get parameters
        self.json_file_path = self.get_parameter('json_file_path').value
        self.global_path_topic = self.get_parameter('global_path_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Load path data
        self.nominal_points = []
        self.load_path_from_json()
        
        # Create publisher
        self.path_publisher = self.create_publisher(
            Path, 
            self.global_path_topic, 
            10
        )
        
        # Create timer for periodic publishing
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate, 
            self.publish_path
        )
        
        self.get_logger().info(f"PublishPlanNode initialized")
        self.get_logger().info(f"  - JSON file: {self.json_file_path}")
        self.get_logger().info(f"  - Topic: {self.global_path_topic}")
        self.get_logger().info(f"  - Rate: {self.publish_rate} Hz")
        self.get_logger().info(f"  - Points: {len(self.nominal_points)}")
        self.get_logger().info(f"  - Frame: {self.frame_id}")
    
    def load_path_from_json(self):
        """Load nominal path from JSON file."""
        try:
            # If no path specified, try to find default file
            if not self.json_file_path:
                from ament_index_python.packages import get_package_share_directory
                package_dir = get_package_share_directory('resilience')
                self.json_file_path = os.path.join(package_dir, 'assets', 'adjusted_nominal_spline.json')
                self.get_logger().info(f"Using default path: {self.json_file_path}")
            
            # Check if file exists
            if not os.path.exists(self.json_file_path):
                raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
            
            # Load the JSON file
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            
            self.nominal_points = data['points']
            
            # Extract thresholds for logging
            if 'calibration' in data:
                soft_threshold = data['calibration'].get('soft_threshold', 'unknown')
                hard_threshold = data['calibration'].get('hard_threshold', 'unknown')
                avg_drift = data['calibration'].get('avg_drift', 'unknown')
                self.get_logger().info(f"  - Soft threshold: {soft_threshold}")
                self.get_logger().info(f"  - Hard threshold: {hard_threshold}")
                self.get_logger().info(f"  - Average drift: {avg_drift}")
            else:
                self.get_logger().warn("  - No calibration data found in JSON")
            
            self.get_logger().info(f"✓ Successfully loaded {len(self.nominal_points)} points from {self.json_file_path}")
            
        except Exception as e:
            self.get_logger().error(f"✗ Failed to load path from JSON: {e}")
            # Create a simple fallback path
            self.create_fallback_path()
    
    def create_fallback_path(self):
        """Create a simple fallback path if JSON loading fails."""
        self.get_logger().warn("Creating fallback path...")
        
        # Create a simple square path
        points = [
            {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}},
            {'position': {'x': 1.0, 'y': 0.0, 'z': 0.0}},
            {'position': {'x': 1.0, 'y': 1.0, 'z': 0.0}},
            {'position': {'x': 0.0, 'y': 1.0, 'z': 0.0}},
            {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}}
        ]
        
        self.nominal_points = points
        self.get_logger().info(f"✓ Created fallback path with {len(self.nominal_points)} points")
    
    def publish_path(self):
        """Publish the nominal path to the global path topic."""
        try:
            path_msg = Path()
            path_msg.header = Header()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = self.frame_id
            
            for i, point in enumerate(self.nominal_points):
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = point['position']['x']
                pose.pose.position.y = point['position']['y']
                pose.pose.position.z = point['position']['z']
                pose.pose.orientation.w = 1.0  # Default orientation
                path_msg.poses.append(pose)
            
            self.path_publisher.publish(path_msg)
            
            # Log first publish with more details
            if not hasattr(self, '_first_publish_logged'):
                self.get_logger().info(f"✓ Published path to {self.global_path_topic}")
                self.get_logger().info(f"  - Frame ID: {self.frame_id}")
                self.get_logger().info(f"  - Points: {len(path_msg.poses)}")
                self.get_logger().info(f"  - First point: ({path_msg.poses[0].pose.position.x:.3f}, {path_msg.poses[0].pose.position.y:.3f}, {path_msg.poses[0].pose.position.z:.3f})")
                self.get_logger().info(f"  - Last point: ({path_msg.poses[-1].pose.position.x:.3f}, {path_msg.poses[-1].pose.position.y:.3f}, {path_msg.poses[-1].pose.position.z:.3f})")
                self.get_logger().info(f"  - First publish at {time.time():.1f}s")
                self._first_publish_logged = True
            
        except Exception as e:
            self.get_logger().error(f"Error publishing path: {e}")
    
    def get_path_info(self) -> Dict[str, Any]:
        """Get information about the loaded path."""
        if not self.nominal_points:
            return {'error': 'No path loaded'}
        
        # Calculate path statistics
        points = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                          for p in self.nominal_points])
        
        total_distance = 0.0
        for i in range(1, len(points)):
            total_distance += np.linalg.norm(points[i] - points[i-1])
        
        return {
            'num_points': len(self.nominal_points),
            'total_distance': total_distance,
            'start_point': points[0].tolist(),
            'end_point': points[-1].tolist(),
            'bounds': {
                'x_min': float(np.min(points[:, 0])),
                'x_max': float(np.max(points[:, 0])),
                'y_min': float(np.min(points[:, 1])),
                'y_max': float(np.max(points[:, 1])),
                'z_min': float(np.min(points[:, 2])),
                'z_max': float(np.max(points[:, 2]))
            }
        }


def main():
    rclpy.init()
    node = PublishPlanNode()
    
    try:
        # Print path information
        path_info = node.get_path_info()
        if 'error' not in path_info:
            node.get_logger().info("Path Information:")
            node.get_logger().info(f"  - Points: {path_info['num_points']}")
            node.get_logger().info(f"  - Total distance: {path_info['total_distance']:.2f}m")
            node.get_logger().info(f"  - Start: {path_info['start_point']}")
            node.get_logger().info(f"  - End: {path_info['end_point']}")
            node.get_logger().info(f"  - Bounds: X[{path_info['bounds']['x_min']:.2f}, {path_info['bounds']['x_max']:.2f}], "
                                 f"Y[{path_info['bounds']['y_min']:.2f}, {path_info['bounds']['y_max']:.2f}], "
                                 f"Z[{path_info['bounds']['z_min']:.2f}, {path_info['bounds']['z_max']:.2f}]")
        
        node.get_logger().info("Starting path publishing...")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down PublishPlanNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
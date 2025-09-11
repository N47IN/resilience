#!/usr/bin/env python3
"""
GP Evolution Visualizer

Provides enhanced visualization of evolving GP fields with temporal information.
Shows how the superposed GP field changes over time as new voxels are added.
"""

import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple
import threading

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String, Header
from visualization_msgs.msg import MarkerArray, Marker


class GPEvolutionVisualizer:
    """Visualizes the evolution of superposed GP fields over time."""
    
    def __init__(self, node, map_frame: str = "map"):
        self.node = node
        self.map_frame = map_frame
        
        # Evolution tracking
        self.evolution_history = {}  # cause_name -> list of (timestamp, voxel_count, gp_stats)
        self.max_history_length = 50  # Keep last 50 evolution steps
        
        # Visualization parameters
        self.evolution_alpha = 0.7  # Transparency for historical data
        self.current_alpha = 1.0    # Full opacity for current data
        
        # Publishers
        self.evolution_cloud_pub = self.node.create_publisher(PointCloud2, '/gp_evolution_field', 10)
        self.evolution_markers_pub = self.node.create_publisher(MarkerArray, '/gp_evolution_markers', 10)
        self.evolution_timeline_pub = self.node.create_publisher(String, '/gp_evolution_timeline', 10)
        
        # Threading
        self.visualization_lock = threading.Lock()
        
        self.node.get_logger().info("GP Evolution Visualizer initialized")
    
    def update_evolution(self, cause_name: str, voxel_count: int, gp_stats: dict):
        """Update evolution history for a cause."""
        try:
            with self.visualization_lock:
                if cause_name not in self.evolution_history:
                    self.evolution_history[cause_name] = []
                
                # Add new evolution step
                evolution_step = {
                    'timestamp': time.time(),
                    'voxel_count': voxel_count,
                    'gp_stats': gp_stats.copy()
                }
                
                self.evolution_history[cause_name].append(evolution_step)
                
                # Keep only recent history
                if len(self.evolution_history[cause_name]) > self.max_history_length:
                    self.evolution_history[cause_name] = self.evolution_history[cause_name][-self.max_history_length:]
                
                # Publish evolution visualization
                self._publish_evolution_visualization(cause_name)
                
        except Exception as e:
            self.node.get_logger().error(f"Error updating evolution for {cause_name}: {e}")
    
    def _publish_evolution_visualization(self, cause_name: str):
        """Publish evolution visualization for a cause."""
        try:
            if cause_name not in self.evolution_history:
                return
            
            history = self.evolution_history[cause_name]
            if len(history) < 2:  # Need at least 2 steps to show evolution
                return
            
            # Create evolution point cloud
            evolution_cloud = self._create_evolution_pointcloud(cause_name, history)
            if evolution_cloud:
                self.evolution_cloud_pub.publish(evolution_cloud)
            
            # Create evolution markers
            evolution_markers = self._create_evolution_markers(cause_name, history)
            if evolution_markers:
                self.evolution_markers_pub.publish(evolution_markers)
            
            # Publish evolution timeline
            timeline_data = self._create_evolution_timeline(cause_name, history)
            if timeline_data:
                timeline_msg = String(data=json.dumps(timeline_data))
                self.evolution_timeline_pub.publish(timeline_msg)
                
        except Exception as e:
            self.node.get_logger().error(f"Error publishing evolution visualization: {e}")
    
    def _create_evolution_pointcloud(self, cause_name: str, history: List[dict]) -> Optional[PointCloud2]:
        """Create point cloud showing GP field evolution."""
        try:
            if len(history) < 2:
                return None
            
            # Get current GP data (assuming it's available from superposition manager)
            # This would need to be passed in or accessed from the superposition manager
            current_gp = self._get_current_gp_data(cause_name)
            if current_gp is None:
                return None
            
            grid_points, gp_values = current_gp
            
            # Create colors based on evolution
            colors = self._create_evolution_colors(gp_values, history)
            
            # Create PointCloud2 message
            header = Header()
            header.stamp = self.node.get_clock().now().to_msg()
            header.frame_id = self.map_frame
            
            # Create structured array with XYZ + RGB
            cloud_data_combined = np.empty(len(grid_points), dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32), 
                ('rgb', np.uint32)
            ])
            
            # Fill in the data
            cloud_data_combined['x'] = grid_points[:, 0]
            cloud_data_combined['y'] = grid_points[:, 1]
            cloud_data_combined['z'] = grid_points[:, 2]
            
            # Pack RGB values as UINT32
            rgb_packed = np.zeros(len(colors), dtype=np.uint32)
            for i, c in enumerate(colors):
                rgb_packed[i] = (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])
            cloud_data_combined['rgb'] = rgb_packed
            
            # Create PointCloud2 message
            cloud_msg = PointCloud2()
            cloud_msg.header = header
            
            # Define the fields
            cloud_msg.fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
            ]
            
            # Set the message properties
            cloud_msg.point_step = 16
            cloud_msg.width = len(grid_points)
            cloud_msg.height = 1
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True
            
            # Set the data
            cloud_msg.data = cloud_data_combined.tobytes()
            
            return cloud_msg
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating evolution point cloud: {e}")
            return None
    
    def _create_evolution_colors(self, gp_values: np.ndarray, history: List[dict]) -> np.ndarray:
        """Create colors that show evolution information."""
        try:
            # Normalize GP values
            gp_min, gp_max = gp_values.min(), gp_values.max()
            if gp_max > gp_min:
                normalized_values = (gp_values - gp_min) / (gp_max - gp_min)
            else:
                normalized_values = np.zeros_like(gp_values)
            
            # Create colors based on evolution
            colors = np.zeros((len(gp_values), 3), dtype=np.uint8)
            
            # Evolution-based coloring:
            # - Red: High intensity, recent evolution
            # - Blue: Low intensity, recent evolution  
            # - Green: Medium intensity, stable regions
            # - Yellow: High intensity, stable regions
            
            for i, value in enumerate(normalized_values):
                if value > 0.7:  # High intensity
                    if len(history) > 5:  # Recent evolution
                        colors[i] = [255, 0, 0]  # Red
                    else:  # Stable
                        colors[i] = [255, 255, 0]  # Yellow
                elif value > 0.3:  # Medium intensity
                    colors[i] = [0, 255, 0]  # Green
                else:  # Low intensity
                    if len(history) > 5:  # Recent evolution
                        colors[i] = [0, 0, 255]  # Blue
                    else:  # Stable
                        colors[i] = [128, 128, 128]  # Gray
            
            return colors
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating evolution colors: {e}")
            return np.zeros((len(gp_values), 3), dtype=np.uint8)
    
    def _create_evolution_markers(self, cause_name: str, history: List[dict]) -> Optional[MarkerArray]:
        """Create markers showing evolution timeline."""
        try:
            if len(history) < 2:
                return None
            
            markers = MarkerArray()
            
            # Create timeline markers
            for i, step in enumerate(history):
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = self.node.get_clock().now().to_msg()
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Position based on evolution step
                marker.pose.position.x = float(i * 0.5)  # Spread along X axis
                marker.pose.position.y = 0.0
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Size based on voxel count
                voxel_count = step['voxel_count']
                marker.scale.x = 0.1 + (voxel_count * 0.01)
                marker.scale.y = 0.1 + (voxel_count * 0.01)
                marker.scale.z = 0.1 + (voxel_count * 0.01)
                
                # Color based on evolution age
                age_factor = i / len(history)
                marker.color.r = 1.0 - age_factor
                marker.color.g = age_factor
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                markers.markers.append(marker)
            
            # Add text marker for cause name
            text_marker = Marker()
            text_marker.header.frame_id = self.map_frame
            text_marker.header.stamp = self.node.get_clock().now().to_msg()
            text_marker.id = len(history)
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = 0.0
            text_marker.pose.position.y = 0.0
            text_marker.pose.position.z = 1.0
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"GP Evolution: {cause_name} ({len(history)} steps)"
            
            markers.markers.append(text_marker)
            
            return markers
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating evolution markers: {e}")
            return None
    
    def _create_evolution_timeline(self, cause_name: str, history: List[dict]) -> Optional[dict]:
        """Create evolution timeline data."""
        try:
            timeline = {
                'cause_name': cause_name,
                'total_steps': len(history),
                'evolution_steps': [],
                'summary': {
                    'initial_voxels': history[0]['voxel_count'] if history else 0,
                    'current_voxels': history[-1]['voxel_count'] if history else 0,
                    'evolution_duration': history[-1]['timestamp'] - history[0]['timestamp'] if len(history) > 1 else 0,
                    'avg_growth_rate': self._calculate_growth_rate(history)
                }
            }
            
            # Add individual steps
            for step in history:
                timeline['evolution_steps'].append({
                    'timestamp': step['timestamp'],
                    'voxel_count': step['voxel_count'],
                    'time_since_start': step['timestamp'] - history[0]['timestamp']
                })
            
            return timeline
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating evolution timeline: {e}")
            return None
    
    def _calculate_growth_rate(self, history: List[dict]) -> float:
        """Calculate average growth rate of voxels over time."""
        try:
            if len(history) < 2:
                return 0.0
            
            total_time = history[-1]['timestamp'] - history[0]['timestamp']
            total_growth = history[-1]['voxel_count'] - history[0]['voxel_count']
            
            if total_time > 0:
                return total_growth / total_time
            else:
                return 0.0
                
        except Exception as e:
            self.node.get_logger().error(f"Error calculating growth rate: {e}")
            return 0.0
    
    def _get_current_gp_data(self, cause_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get current GP data for a cause (placeholder - would be connected to superposition manager)."""
        # This would be connected to the superposition manager
        # For now, return None to indicate no current data
        return None
    
    def get_evolution_statistics(self) -> Dict:
        """Get evolution statistics for all causes."""
        stats = {}
        for cause_name, history in self.evolution_history.items():
            if history:
                stats[cause_name] = {
                    'total_steps': len(history),
                    'initial_voxels': history[0]['voxel_count'],
                    'current_voxels': history[-1]['voxel_count'],
                    'evolution_duration': history[-1]['timestamp'] - history[0]['timestamp'],
                    'growth_rate': self._calculate_growth_rate(history)
                }
        return stats

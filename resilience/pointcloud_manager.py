#!/usr/bin/env python3
"""
Point Cloud Manager Module

Handles 3D point cloud creation, accumulation, and publishing.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2


class PointCloudManager:
    """Manages 3D point cloud creation and accumulation."""
    
    def __init__(self, max_accumulated_points: int = 1000, accumulation_clear_interval: int = 100):
        """Initialize point cloud manager."""
        self.max_accumulated_points = max_accumulated_points
        self.accumulation_clear_interval = accumulation_clear_interval
        
        # Accumulative point cloud storage (bbox centers only)
        self.accumulated_points = []
        self.accumulated_colors = []
        self.accumulated_cluster_ids = []  # Track cluster IDs for consistent colors
        self.frame_count = 0
        
        # Point merging parameters based on bbox size
        self.point_merging_enabled = True
        self.bbox_merging_enabled = True  # Use bbox size for merging
        self.next_cluster_id = 0  # Track next available cluster ID
    
    def bbox_pixels_to_meters(self, bbox: List[int], depth_m: float, 
                             camera_intrinsics: List[float]) -> Tuple[float, float]:
        """Convert bbox size from pixels to meters using camera intrinsics."""
        if camera_intrinsics is None:
            return 0.1, 0.1  # Default small size
        
        fx, fy, cx, cy = camera_intrinsics
        x1, y1, x2, y2 = bbox
        
        # Bbox dimensions in pixels
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        
        # Convert to meters using depth and camera intrinsics
        width_meters = (width_pixels * depth_m) / fx
        height_meters = (height_pixels * depth_m) / fy
        
        return width_meters, height_meters
    
    def get_cluster_color(self, cluster_id: int) -> List[int]:
        """Get distinct color for a cluster ID."""
        # Predefined distinct colors for clusters
        cluster_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [0, 128, 255],  # Light Blue
            [255, 0, 128],  # Pink
            [128, 255, 0],  # Light Green
            [255, 128, 128], # Light Red
            [128, 255, 128], # Light Green
            [128, 128, 255], # Light Blue
            [255, 255, 128], # Light Yellow
        ]
        
        # Cycle through colors if more clusters than colors
        color_idx = cluster_id % len(cluster_colors)
        return cluster_colors[color_idx]
    
    def merge_bbox_based_points(self, new_points: List, new_colors: List, 
                               new_bboxes: List, new_depths: List) -> Tuple[List, List, List]:
        """Merge points based on bbox size - if points are within each other's bbox dimensions."""
        if not self.point_merging_enabled:
            # If no existing points, assign new cluster IDs
            if len(self.accumulated_points) == 0:
                merged_points = []
                merged_colors = []
                merged_cluster_ids = []
                
                for i, (point, bbox, depth) in enumerate(zip(new_points, new_bboxes, new_depths)):
                    cluster_id = self.next_cluster_id
                    cluster_color = self.get_cluster_color(cluster_id)
                    
                    merged_points.append(point)
                    merged_colors.append(cluster_color)
                    merged_cluster_ids.append(cluster_id)
                    self.next_cluster_id += 1
                
                return merged_points, merged_colors, merged_cluster_ids
        
        merged_points = []
        merged_colors = []
        merged_cluster_ids = []
        
        for i, (point, bbox, depth) in enumerate(zip(new_points, new_bboxes, new_depths)):
            # Get bbox dimensions in meters (simplified - would need camera intrinsics)
            bbox_size_m = 0.1  # Default size for merging
            
            point_added = False
            
            # Check against existing points
            for j, (existing_point, existing_cluster_id) in enumerate(zip(self.accumulated_points, self.accumulated_cluster_ids)):
                distance = np.linalg.norm(np.array(point) - np.array(existing_point))
                
                # If distance is within bbox size, merge them (same cluster)
                if distance <= bbox_size_m:
                    # Merge by averaging positions, keep existing cluster ID and color
                    avg_point = [(p1 + p2) / 2 for p1, p2 in zip(point, existing_point)]
                    
                    # Update existing point
                    self.accumulated_points[j] = avg_point
                    point_added = True
                    break
            
            if not point_added:
                # Create new cluster
                cluster_id = self.next_cluster_id
                cluster_color = self.get_cluster_color(cluster_id)
                
                merged_points.append(point)
                merged_colors.append(cluster_color)
                merged_cluster_ids.append(cluster_id)
                self.next_cluster_id += 1
        
        return merged_points, merged_colors, merged_cluster_ids
    
    def create_bbox_center_pointclouds(self, bboxes: List, depth_image: np.ndarray, 
                                     camera_intrinsics: List[float], transform_matrix: np.ndarray,
                                     min_distance: float = 0.5, max_distance: float = 2.0,
                                     min_height: float = 0.15) -> bool:
        """Create 3D points from bbox centers only and accumulate them."""
        try:
            if camera_intrinsics is None:
                return False

            # Early exit if no bboxes
            if not bboxes:
                return False

            fx, fy, cx, cy = camera_intrinsics
            new_points = []
            new_colors = []
            
            # Process each bbox center to 3D point
            new_bboxes = []  # Store bboxes for merging
            new_depths = []  # Store depths for bbox size calculation
            
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get depth at center
                if (center_y < 0 or center_y >= depth_image.shape[0] or 
                    center_x < 0 or center_x >= depth_image.shape[1]):
                    continue
                    
                depth_value = depth_image[center_y, center_x]
                
                # Filter valid depth
                if depth_value <= 0 or np.isnan(depth_value):
                    continue
                    
                depth_m = depth_value / 1000.0 if depth_value > 100 else depth_value
                if not (min_distance <= depth_m <= max_distance):
                    continue
                
                # Project center to 3D camera coordinates
                x = (center_x - cx) * depth_m / fx
                y = (center_y - cy) * depth_m / fy
                center_point = np.array([x, y, depth_m])
                
                # Transform to global frame
                center_point_h = np.concatenate([center_point, np.ones(1)], axis=0)
                center_point_map = (transform_matrix @ center_point_h.T).T[:3]
                
                # Filter by height (remove ground points)
                if center_point_map[2] <= min_height:
                    continue
                
                new_points.append(center_point_map.tolist())
                new_bboxes.append(bbox)
                new_depths.append(depth_m)
            
            # Merge nearby points based on bbox size
            if new_points:
                merged_points, merged_colors, merged_cluster_ids = self.merge_bbox_based_points(
                    new_points, [], new_bboxes, new_depths)
                
                # Add merged points to accumulated storage
                self.accumulated_points.extend(merged_points)
                self.accumulated_colors.extend(merged_colors)
                self.accumulated_cluster_ids.extend(merged_cluster_ids)
                
                # Limit accumulation size
                if len(self.accumulated_points) > self.max_accumulated_points:
                    excess = len(self.accumulated_points) - self.max_accumulated_points
                    self.accumulated_points = self.accumulated_points[excess:]
                    self.accumulated_colors = self.accumulated_colors[excess:]
                    self.accumulated_cluster_ids = self.accumulated_cluster_ids[excess:]
                
                return True

            return False
            
        except Exception as e:
            print(f"Error creating bbox center pointclouds: {e}")
            return False
    
    def create_pointcloud_message(self, header: Header) -> PointCloud2:
        """Create a PointCloud2 message from accumulated points."""
        if len(self.accumulated_points) == 0:
            # Return empty point cloud
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            return pc2.create_cloud(header, fields, [])
        
        all_points = np.array(self.accumulated_points)
        all_colors = np.array(self.accumulated_colors)
        
        # Downsample if too many points
        if len(all_points) > 10000:
            idx = np.random.choice(len(all_points), 10000, replace=False)
            all_points = all_points[idx]
            all_colors = all_colors[idx]
        
        # Create point cloud message
        return self.create_simple_colored_pointcloud(all_points, all_colors, header)
    
    def create_simple_colored_pointcloud(self, points: np.ndarray, colors: np.ndarray, 
                                       header: Header) -> PointCloud2:
        """Create a simple colored PointCloud2 message."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        if len(points) == 0:
            return pc2.create_cloud(header, fields, [])

        # Ensure colors are 3D (RGB)
        if len(colors.shape) == 2 and colors.shape[1] == 3:
            colors = colors.reshape(-1, 3)
        elif len(colors.shape) == 1 and colors.shape[0] == 3:
            colors = np.tile(colors, (len(points), 1))

        # Vectorized RGB packing
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b

        # Convert to float
        rgb_float = rgb_packed.view(np.float32)

        # Create cloud points
        cloud_points = np.column_stack([points, rgb_float])

        return pc2.create_cloud(header, fields, cloud_points.tolist())
    
    def clear_old_accumulated_points(self):
        """Clear half of accumulated points to prevent memory buildup."""
        if len(self.accumulated_points) > self.max_accumulated_points // 2:
            half_size = len(self.accumulated_points) // 2
            self.accumulated_points = self.accumulated_points[half_size:]
            self.accumulated_colors = self.accumulated_colors[half_size:]
            self.accumulated_cluster_ids = self.accumulated_cluster_ids[half_size:]
    
    def increment_frame_count(self):
        """Increment frame count and clear accumulation periodically."""
        self.frame_count += 1
        
        # Clear accumulation periodically
        if self.frame_count % self.accumulation_clear_interval == 0:
            self.clear_old_accumulated_points()
    
    def get_accumulated_points_count(self) -> int:
        """Get number of accumulated points."""
        return len(self.accumulated_points)
    
    def clear_all_points(self):
        """Clear all accumulated points."""
        self.accumulated_points = []
        self.accumulated_colors = []
        self.accumulated_cluster_ids = []
        self.frame_count = 0 
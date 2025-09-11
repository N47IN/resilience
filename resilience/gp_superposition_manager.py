#!/usr/bin/env python3
"""
GP Superposition Manager

Handles real-time superposition of multiple per-voxel GPs for the same semantic cause.
Provides evolving visualization of the superposed disturbance field.
"""

import numpy as np
import threading
import time
import json
import os
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple
import cv2

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String, Header


class GPSuperpositionManager:
    """Manages real-time superposition of per-voxel GPs for semantic causes."""
    
    def __init__(self, node, map_frame: str = "map", update_interval: float = 1.0, voxel_helper=None, 
                 grid_extension: float = 1.5, grid_resolution: float = 0.15, max_grid_distance: float = 2.0):
        self.node = node
        self.map_frame = map_frame
        self.update_interval = update_interval
        self.voxel_helper = voxel_helper  # Reference to the voxel mapping helper
        self.grid_extension = grid_extension  # How far to extend grid from voxels (meters)
        self.grid_resolution = grid_resolution  # Grid resolution (meters)
        self.max_grid_distance = max_grid_distance  # Max distance for grid points from voxels (meters)
        
        # GP storage per cause
        self.cause_gps = {}  # cause_name -> list of gp_params
        self.cause_voxel_positions = {}  # cause_name -> list of voxel positions
        
        # Superposed GP results
        self.superposed_gps = {}  # cause_name -> (grid_points, gp_values)
        self.last_update_time = {}  # cause_name -> timestamp
        
        # Threading
        self.gp_queue = Queue()
        self.computation_lock = threading.Lock()
        self.running = True
        
        # Background processing thread
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        
        # Publishers
        self.superposed_gp_pub = self.node.create_publisher(PointCloud2, '/superposed_gp_field', 10)
        self.evolution_stats_pub = self.node.create_publisher(String, '/gp_evolution_stats', 10)
        
        self.node.get_logger().info("GP Superposition Manager initialized")
    
    def add_voxel_gp(self, voxel_position: np.ndarray, gp_params: dict, cause_name: str, buffer_dir: str):
        """Add a new voxel GP to the superposition for a given cause."""
        try:
            with self.computation_lock:
                if cause_name not in self.cause_gps:
                    self.cause_gps[cause_name] = []
                    self.cause_voxel_positions[cause_name] = []
                
                # Store GP parameters and voxel position
                gp_data = {
                    'params': gp_params,
                    'position': voxel_position.copy(),
                    'buffer_dir': buffer_dir,
                    'timestamp': time.time()
                }
                
                self.cause_gps[cause_name].append(gp_data)
                self.cause_voxel_positions[cause_name].append(voxel_position.copy())
                
                # Queue for background processing
                self.gp_queue.put((cause_name, gp_data))
                
                self.node.get_logger().info(f"Added voxel GP for cause '{cause_name}' (total: {len(self.cause_gps[cause_name])})")
                
        except Exception as e:
            self.node.get_logger().error(f"Error adding voxel GP: {e}")
    
    def _background_processor(self):
        """Background thread for processing GP superposition."""
        while self.running:
            try:
                # Process queued GPs
                if not self.gp_queue.empty():
                    self._process_queued_gps()
                
                # Periodic updates for all causes
                self._periodic_superposition_update()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.node.get_logger().error(f"Error in background GP processor: {e}")
                time.sleep(1.0)
    
    def _process_queued_gps(self):
        """Process all queued GP updates."""
        try:
            processed_count = 0
            while not self.gp_queue.empty():
                try:
                    cause_name, gp_data = self.gp_queue.get_nowait()
                    self._update_superposed_gp(cause_name)
                    processed_count += 1
                except Empty:
                    break
            
            if processed_count > 0:
                self.node.get_logger().info(f"Processed {processed_count} queued GP updates")
                
        except Exception as e:
            self.node.get_logger().error(f"Error processing queued GPs: {e}")
    
    def _periodic_superposition_update(self):
        """Periodic update of superposed GPs for all causes."""
        current_time = time.time()
        
        for cause_name in self.cause_gps.keys():
            last_update = self.last_update_time.get(cause_name, 0)
            
            # Update if enough time has passed
            if (current_time - last_update) >= self.update_interval:
                self._update_superposed_gp(cause_name)
                self.last_update_time[cause_name] = current_time
    
    def _update_superposed_gp(self, cause_name: str):
        """Update the superposed GP for a specific cause."""
        try:
            if cause_name not in self.cause_gps or not self.cause_gps[cause_name]:
                return
            
            # Get all voxel GPs for this cause
            voxel_gps = self.cause_gps[cause_name]
            voxel_positions = self.cause_voxel_positions[cause_name]
            
            if not voxel_gps:
                return
            
            # For semantic_disturbance, create grid around ALL semantic voxels
            if cause_name == "semantic_disturbance":
                # Get all semantic voxels from the voxel mapping helper
                all_semantic_voxels = self._get_all_semantic_voxels()
                if len(all_semantic_voxels) > 0:
                    # Create prediction grid around all semantic voxels
                    grid_points = self._create_prediction_grid_around_voxels(all_semantic_voxels)
                else:
                    # Fallback to GP voxel positions
                    grid_points = self._create_prediction_grid_around_voxels(voxel_positions)
            else:
                # For other causes, use the original behavior
                grid_points = self._create_prediction_grid_around_voxels(voxel_positions)
            
            if len(grid_points) == 0:
                return
            
            # Compute superposed GP values
            superposed_values = self._compute_superposed_gp_values(grid_points, voxel_gps)
            
            # Store results
            self.superposed_gps[cause_name] = (grid_points, superposed_values)
            
            # Publish visualization
            self._publish_superposed_visualization(cause_name, grid_points, superposed_values)
            
            # Publish evolution stats
            self._publish_evolution_stats(cause_name, len(voxel_gps))
            
            self.node.get_logger().info(f"Updated superposed GP for '{cause_name}': {len(grid_points)} grid points, {len(voxel_gps)} voxel GPs")
            
        except Exception as e:
            self.node.get_logger().error(f"Error updating superposed GP for {cause_name}: {e}")
    
    def _create_prediction_grid_around_voxels(self, voxel_positions: List[np.ndarray]) -> np.ndarray:
        """Create a 3D prediction grid around the given voxel positions."""
        try:
            if not voxel_positions:
                return np.array([])
            
            # Convert to numpy array
            positions = np.array(voxel_positions)
            
            # Find bounding box
            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)
            center = (min_coords + max_coords) / 2.0
            
            # Calculate the extent of the actual voxels
            voxel_extent = np.max(max_coords - min_coords)
            
            # Use smaller extension: max of grid_extension or 1.5x the voxel extent
            extension = min(self.grid_extension, max(0.5, voxel_extent * 1.5))
            
            # Create tighter grid around voxels
            x_range = np.arange(min_coords[0] - extension, max_coords[0] + extension, self.grid_resolution)
            y_range = np.arange(min_coords[1] - extension, max_coords[1] + extension, self.grid_resolution)
            z_range = np.arange(min_coords[2] - extension, max_coords[2] + extension, self.grid_resolution)
            
            # Create meshgrid
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            
            # Filter grid points to keep only those close to voxels
            filtered_grid_points = self._filter_grid_points_near_voxels(grid_points, positions, self.max_grid_distance)
            
            self.node.get_logger().info(f"Created tight prediction grid: {len(filtered_grid_points)} points around {len(voxel_positions)} voxels "
                                     f"(extension: {extension:.2f}m, resolution: {self.grid_resolution:.2f}m)")
            
            return filtered_grid_points
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating prediction grid: {e}")
            return np.array([])
    
    def _filter_grid_points_near_voxels(self, grid_points: np.ndarray, voxel_positions: np.ndarray, max_distance: float = 2.0) -> np.ndarray:
        """Filter grid points to keep only those within max_distance of any voxel."""
        try:
            if len(grid_points) == 0 or len(voxel_positions) == 0:
                return grid_points
            
            # For each grid point, find minimum distance to any voxel
            filtered_points = []
            
            for grid_point in grid_points:
                # Calculate distances to all voxels
                distances = np.linalg.norm(voxel_positions - grid_point, axis=1)
                min_distance = np.min(distances)
                
                # Keep point if it's within max_distance of any voxel
                if min_distance <= max_distance:
                    filtered_points.append(grid_point)
            
            filtered_points = np.array(filtered_points)
            
            self.node.get_logger().info(f"Filtered grid points: {len(grid_points)} -> {len(filtered_points)} "
                                     f"(kept points within {max_distance}m of voxels)")
            
            return filtered_points
            
        except Exception as e:
            self.node.get_logger().error(f"Error filtering grid points: {e}")
            return grid_points
    
    def _enhance_contrast(self, values: np.ndarray, contrast_factor: float = 2.0) -> np.ndarray:
        """Enhance contrast by stretching the middle values."""
        try:
            # Apply contrast enhancement using power function
            # This makes low values lower and high values higher
            enhanced_values = np.power(values, 1.0 / contrast_factor)
            
            # Ensure values stay in [0, 1] range
            enhanced_values = np.clip(enhanced_values, 0.0, 1.0)
            
            return enhanced_values
            
        except Exception as e:
            self.node.get_logger().error(f"Error enhancing contrast: {e}")
            return values
    
    def _compute_superposed_gp_values(self, grid_points: np.ndarray, voxel_gps: List[dict]) -> np.ndarray:
        """Compute superposed GP values at grid points."""
        try:
            superposed_values = np.zeros(len(grid_points))
            
            for gp_data in voxel_gps:
                gp_params = gp_data['params']
                voxel_position = gp_data['position']
                
                # Compute GP contribution at each grid point
                gp_values = self._predict_gp_at_points(grid_points, gp_params, voxel_position)
                superposed_values += gp_values
            
            return superposed_values
            
        except Exception as e:
            self.node.get_logger().error(f"Error computing superposed GP values: {e}")
            return np.zeros(len(grid_points))
    
    def _predict_gp_at_points(self, grid_points: np.ndarray, gp_params: dict, voxel_position: np.ndarray) -> np.ndarray:
        """Predict GP values at grid points using voxel GP parameters."""
        try:
            # Extract GP parameters
            lxy = gp_params.get('lxy', 0.5)
            lz = gp_params.get('lz', 0.5)
            A = gp_params.get('A', 1.0)
            b = gp_params.get('b', 0.0)
            
            # Compute distances from grid points to voxel position
            distances = np.linalg.norm(grid_points - voxel_position, axis=1)
            
            # Apply anisotropic scaling (simplified - use average length scale)
            avg_length = (lxy + lz) / 2.0
            
            # Compute RBF contribution
            rbf_values = A * np.exp(-(distances**2) / (2 * avg_length**2)) + b
            
            return rbf_values
            
        except Exception as e:
            self.node.get_logger().error(f"Error predicting GP at points: {e}")
            return np.zeros(len(grid_points))
    
    def _publish_superposed_visualization(self, cause_name: str, grid_points: np.ndarray, gp_values: np.ndarray):
        """Publish superposed GP visualization as colored point cloud."""
        try:
            if len(grid_points) == 0 or len(gp_values) == 0:
                return
            
            # Create colored point cloud
            colored_cloud = self._create_superposed_colored_pointcloud(grid_points, gp_values, cause_name)
            
            if colored_cloud:
                self.superposed_gp_pub.publish(colored_cloud)
                
        except Exception as e:
            self.node.get_logger().error(f"Error publishing superposed visualization: {e}")
    
    def _create_superposed_colored_pointcloud(self, grid_points: np.ndarray, gp_values: np.ndarray, 
                                            cause_name: str) -> Optional[PointCloud2]:
        """Create colored point cloud from superposed GP field."""
        try:
            # Normalize GP values to [0, 1] for coloring
            gp_min, gp_max = gp_values.min(), gp_values.max()
            if gp_max > gp_min:
                normalized_values = (gp_values - gp_min) / (gp_max - gp_min)
            else:
                normalized_values = np.zeros_like(gp_values)
            
            # Apply contrast enhancement - stretch the middle values
            normalized_values = self._enhance_contrast(normalized_values)
            
            # Create high-contrast color map (blue to red: low to high intensity)
            colors = np.zeros((len(grid_points), 3), dtype=np.uint8)
            
            # High-contrast color mapping with better visibility
            # Low values: Deep blue (0, 0, 255)
            # High values: Bright red (255, 0, 0)
            # Medium values: Bright yellow (255, 255, 0)
            
            for i, value in enumerate(normalized_values):
                if value < 0.33:  # Low values: Blue to Cyan
                    # Blue to Cyan transition
                    blue_intensity = int(255 * (1 - value * 3))
                    green_intensity = int(255 * value * 3)
                    colors[i] = [0, green_intensity, blue_intensity]
                elif value < 0.66:  # Medium values: Cyan to Yellow
                    # Cyan to Yellow transition
                    local_value = (value - 0.33) / 0.33
                    red_intensity = int(255 * local_value)
                    green_intensity = 255
                    blue_intensity = int(255 * (1 - local_value))
                    colors[i] = [red_intensity, green_intensity, blue_intensity]
                else:  # High values: Yellow to Red
                    # Yellow to Red transition
                    local_value = (value - 0.66) / 0.34
                    red_intensity = 255
                    green_intensity = int(255 * (1 - local_value))
                    blue_intensity = 0
                    colors[i] = [red_intensity, green_intensity, blue_intensity]
            
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
            cloud_msg.point_step = 16  # 4 bytes per float * 4 fields
            cloud_msg.width = len(grid_points)
            cloud_msg.height = 1
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True
            
            # Set the data
            cloud_msg.data = cloud_data_combined.tobytes()
            
            return cloud_msg
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating superposed colored point cloud: {e}")
            return None
    
    def _publish_evolution_stats(self, cause_name: str, voxel_count: int):
        """Publish evolution statistics for the GP field."""
        try:
            stats = {
                'cause_name': cause_name,
                'voxel_gp_count': voxel_count,
                'timestamp': time.time(),
                'status': 'evolving' if voxel_count > 1 else 'initial'
            }
            
            stats_msg = String(data=json.dumps(stats))
            self.evolution_stats_pub.publish(stats_msg)
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing evolution stats: {e}")
    
    def get_superposed_gp(self, cause_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the current superposed GP for a cause."""
        return self.superposed_gps.get(cause_name)
    
    def get_cause_statistics(self) -> Dict:
        """Get statistics about all tracked causes."""
        stats = {}
        for cause_name in self.cause_gps.keys():
            stats[cause_name] = {
                'voxel_count': len(self.cause_gps[cause_name]),
                'has_superposed_gp': cause_name in self.superposed_gps,
                'last_update': self.last_update_time.get(cause_name, 0)
            }
        return stats
    
    def _get_all_semantic_voxels(self) -> List[np.ndarray]:
        """Get all semantic voxel positions from the voxel mapping helper."""
        try:
            if self.voxel_helper is None:
                return []
            
            # Check if semantic mapper is available
            if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
                return []
            
            semantic_mapper = self.voxel_helper.semantic_mapper
            semantic_voxel_positions = []
            
            with semantic_mapper.voxel_semantics_lock:
                for voxel_key in semantic_mapper.voxel_semantics.keys():
                    # Convert voxel key to world position
                    voxel_position = self._get_voxel_center_from_key(voxel_key)
                    if voxel_position is not None:
                        semantic_voxel_positions.append(voxel_position)
            
            self.node.get_logger().info(f"Found {len(semantic_voxel_positions)} semantic voxels for GP superposition")
            return semantic_voxel_positions
            
        except Exception as e:
            self.node.get_logger().error(f"Error getting semantic voxels: {e}")
            return []
    
    def _get_voxel_center_from_key(self, voxel_key: tuple) -> Optional[np.ndarray]:
        """Get voxel center position from voxel key."""
        try:
            vx, vy, vz = voxel_key
            
            # Get voxel resolution from voxel helper
            voxel_resolution = 0.1  # Default
            if self.voxel_helper and hasattr(self.voxel_helper, 'voxel_resolution'):
                voxel_resolution = self.voxel_helper.voxel_resolution
            
            # Convert voxel coordinates to world coordinates
            world_x = vx * voxel_resolution
            world_y = vy * voxel_resolution
            world_z = vz * voxel_resolution
            
            return np.array([world_x, world_y, world_z], dtype=np.float32)
            
        except Exception as e:
            self.node.get_logger().warn(f"Error getting voxel center for key {voxel_key}: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the superposition manager."""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        self.node.get_logger().info("GP Superposition Manager shutdown")

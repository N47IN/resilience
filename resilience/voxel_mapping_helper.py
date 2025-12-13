#!/usr/bin/env python3
"""
Voxel Mapping Helper

Basic helper class for voxel mapping and semantic labeling.
Provides the interface needed by the octomap node.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3
from resilience.semantic_voxel_mapper import SemanticVoxelMapper


class VoxelMappingHelper:
    """Basic helper class for voxel mapping and semantic labeling."""
    
    def __init__(self, 
                 voxel_resolution: float = 0.1,
                 probability_hit: float = 0.7,
                 probability_miss: float = 0.4,
                 occupancy_threshold: float = 0.5,
                 embedding_dim: int = 1152,
                 enable_semantic_mapping: bool = True,
                 semantic_similarity_threshold: float = 0.6,
                 map_frame: str = 'map',
                 verbose: bool = False):
        """
        Initialize voxel mapping helper.
        
        Args:
            voxel_resolution: Size of each voxel in meters
            probability_hit: Probability update for occupied voxels
            probability_miss: Probability update for free voxels
            occupancy_threshold: Threshold for occupancy
            embedding_dim: Expected embedding dimension
            enable_semantic_mapping: Whether to enable semantic mapping
            semantic_similarity_threshold: Threshold for semantic similarity
            map_frame: Frame ID for the map
            verbose: Whether to print verbose messages
        """
        self.voxel_resolution = float(voxel_resolution)
        self.probability_hit = float(probability_hit)
        self.probability_miss = float(probability_miss)
        self.occupancy_threshold = float(occupancy_threshold)
        self.embedding_dim = int(embedding_dim)
        self.enable_semantic_mapping = bool(enable_semantic_mapping)
        self.semantic_similarity_threshold = float(semantic_similarity_threshold)
        self.map_frame = str(map_frame)
        self.verbose = bool(verbose)
        
        # Voxel storage
        self.voxels = {}  # (vx, vy, vz) -> occupancy_probability
        self.voxel_timestamps = {}  # (vx, vy, vz) -> last_update_time
        
        # Semantic mapping
        if self.enable_semantic_mapping:
            self.semantic_mapper = SemanticVoxelMapper(
                similarity_threshold=self.semantic_similarity_threshold,
                embedding_dim=self.embedding_dim,
                verbose=self.verbose
            )
        else:
            self.semantic_mapper = None
        
        if self.verbose:
            print(f"VoxelMappingHelper initialized:")
            print(f"  - Voxel resolution: {self.voxel_resolution}m")
            print(f"  - Semantic mapping: {'ENABLED' if self.enable_semantic_mapping else 'DISABLED'}")
            print(f"  - Map frame: {self.map_frame}")
    
    def get_voxel_key_from_point(self, world_point: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel key."""
        vx = int(np.round(world_point[0] / self.voxel_resolution))
        vy = int(np.round(world_point[1] / self.voxel_resolution))
        vz = int(np.round(world_point[2] / self.voxel_resolution))
        return (vx, vy, vz)
    
    def update_map(self, points_world: np.ndarray, origin: np.ndarray, 
                   hit_confidences: Optional[np.ndarray] = None, 
                   voxel_embeddings: Optional[np.ndarray] = None):
        """Update voxel map with new points."""
        try:
            current_time = time.time()
            
            for i, point in enumerate(points_world):
                voxel_key = self.get_voxel_key_from_point(point)
                
                # Update occupancy probability
                if voxel_key not in self.voxels:
                    self.voxels[voxel_key] = 0.5  # Initialize with 0.5 (unknown)
                
                # Simple occupancy update
                if hit_confidences is not None and i < len(hit_confidences):
                    confidence = hit_confidences[i]
                else:
                    confidence = 1.0
                
                # Update probability
                if confidence > 0.5:
                    self.voxels[voxel_key] = min(0.95, self.voxels[voxel_key] + self.probability_hit * confidence)
                else:
                    self.voxels[voxel_key] = max(0.05, self.voxels[voxel_key] - self.probability_miss * (1 - confidence))
                
                self.voxel_timestamps[voxel_key] = current_time
            
            if self.verbose and len(points_world) > 0:
                print(f"Updated voxel map with {len(points_world)} points, total voxels: {len(self.voxels)}")
                
        except Exception as e:
            if self.verbose:
                print(f"Error updating voxel map: {e}")
    
    def create_markers(self, max_markers: int = 10000, use_cube_list: bool = True) -> MarkerArray:
        """Create visualization markers for occupied voxels."""
        try:
            markers = MarkerArray()
            
            # Filter occupied voxels
            occupied_voxels = [
                (key, prob) for key, prob in self.voxels.items() 
                if prob > self.occupancy_threshold
            ]
            
            if len(occupied_voxels) > max_markers:
                occupied_voxels = occupied_voxels[:max_markers]
            
            if use_cube_list:
                # Create single marker with multiple cubes
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = time.time()  # This will be fixed by the calling node
                marker.ns = "voxels"
                marker.id = 0
                marker.type = Marker.CUBE_LIST
                marker.action = Marker.ADD
                
                # Set scale for all cubes
                marker.scale.x = self.voxel_resolution * 0.9
                marker.scale.y = self.voxel_resolution * 0.9
                marker.scale.z = self.voxel_resolution * 0.9
                
                # Add cubes for each occupied voxel
                for voxel_key, prob in occupied_voxels:
                    vx, vy, vz = voxel_key
                    world_x = vx * self.voxel_resolution
                    world_y = vy * self.voxel_resolution
                    world_z = vz * self.voxel_resolution
                    
                    # Create point
                    point = Point()
                    point.x = world_x
                    point.y = world_y
                    point.z = world_z
                    marker.points.append(point)
                    
                    # Create color based on occupancy probability
                    color = ColorRGBA()
                    if prob > 0.8:
                        color.r = 1.0  # Red for high occupancy
                        color.g = 0.0
                        color.b = 0.0
                    elif prob > 0.6:
                        color.r = 1.0  # Orange for medium occupancy
                        color.g = 0.5
                        color.b = 0.0
                    else:
                        color.r = 1.0  # Yellow for low occupancy
                        color.g = 1.0
                        color.b = 0.0
                    color.a = 0.8
                    marker.colors.append(color)
                
                markers.markers.append(marker)
                
            else:
                # Create individual markers for each voxel
                for i, (voxel_key, prob) in enumerate(occupied_voxels):
                    marker = Marker()
                    marker.header.frame_id = self.map_frame
                    marker.header.stamp = time.time()  # This will be fixed by the calling node
                    marker.ns = "voxels"
                    marker.id = i
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    vx, vy, vz = voxel_key
                    marker.pose.position.x = vx * self.voxel_resolution
                    marker.pose.position.y = vy * self.voxel_resolution
                    marker.pose.position.z = vz * self.voxel_resolution
                    
                    marker.scale.x = self.voxel_resolution * 0.9
                    marker.scale.y = self.voxel_resolution * 0.9
                    marker.scale.z = self.voxel_resolution * 0.9
                    
                    # Color based on occupancy
                    if prob > 0.8:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                    elif prob > 0.6:
                        marker.color.r = 1.0
                        marker.color.g = 0.5
                        marker.color.b = 0.0
                    else:
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                    color.a = 0.8
                    
                    markers.markers.append(marker)
            
            if self.verbose:
                print(f"Created {len(markers.markers)} markers for {len(occupied_voxels)} occupied voxels")
            
            return markers
            
        except Exception as e:
            if self.verbose:
                print(f"Error creating markers: {e}")
            return MarkerArray()
    
    def create_colored_cloud(self, max_points: int = 10000) -> Optional[PointCloud2]:
        """Create colored point cloud from occupied voxels."""
        try:
            # Filter occupied voxels
            occupied_voxels = [
                (key, prob) for key, prob in self.voxels.items() 
                if prob > self.occupancy_threshold
            ]
            
            if len(occupied_voxels) > max_points:
                occupied_voxels = occupied_voxels[:max_points]
            
            if not occupied_voxels:
                return None
            
            # Convert to points and colors
            points = []
            colors = []
            
            for voxel_key, prob in occupied_voxels:
                vx, vy, vz = voxel_key
                world_x = vx * self.voxel_resolution
                world_y = vy * self.voxel_resolution
                world_z = vz * self.voxel_resolution
                
                points.append([world_x, world_y, world_z])
                
                # Color based on occupancy
                if prob > 0.8:
                    colors.append([1.0, 0.0, 0.0])  # Red
                elif prob > 0.6:
                    colors.append([1.0, 0.5, 0.0])  # Orange
                else:
                    colors.append([1.0, 1.0, 0.0])  # Yellow
            
            # Create PointCloud2 message
            from sensor_msgs_py import point_cloud2 as pc2
            
            points_np = np.array(points, dtype=np.float32)
            header = Header()
            header.frame_id = self.map_frame
            
            cloud_msg = pc2.create_cloud_xyz32(header, points_np)
            
            if self.verbose:
                print(f"Created colored cloud with {len(points)} points")
            
            return cloud_msg
            
        except Exception as e:
            if self.verbose:
                print(f"Error creating colored cloud: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get voxel mapping statistics."""
        try:
            total_voxels = len(self.voxels)
            occupied_voxels = sum(1 for prob in self.voxels.values() if prob > self.occupancy_threshold)
            free_voxels = sum(1 for prob in self.voxels.values() if prob < (1 - self.occupancy_threshold))
            unknown_voxels = total_voxels - occupied_voxels - free_voxels
            
            stats = {
                'total_voxels': total_voxels,
                'occupied_voxels': occupied_voxels,
                'free_voxels': free_voxels,
                'unknown_voxels': unknown_voxels,
                'occupancy_rate': occupied_voxels / max(total_voxels, 1),
                'voxel_resolution': self.voxel_resolution,
                'map_frame': self.map_frame,
                'semantic_mapping_enabled': self.enable_semantic_mapping
            }
            
            if self.semantic_mapper:
                semantic_stats = self.semantic_mapper.get_semantic_statistics()
                stats['semantic_mapping'] = semantic_stats
            
            return stats
            
        except Exception as e:
            return {'error': f'Failed to get statistics: {e}'}
    
    def load_vlm_embeddings_from_buffers(self, buffers_directory: str) -> int:
        """Load VLM embeddings from buffer directories."""
        try:
            if not self.semantic_mapper:
                return 0
            
            loaded_count = 0
            # This would implement loading embeddings from buffer files
            # For now, return 0 as placeholder
            
            if self.verbose:
                print(f"Loaded {loaded_count} VLM embeddings from buffers")
            
            return loaded_count
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading VLM embeddings: {e}")
            return 0 
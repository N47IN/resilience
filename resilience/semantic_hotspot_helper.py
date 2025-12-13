#!/usr/bin/env python3
"""
Semantic Hotspot Helper

Comprehensive helper that maintains a queue of all hotspots and uses the exact same 
depth-to-world-points pipeline as the octomap node to color existing voxels.
"""

import numpy as np
import cv2
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
from collections import deque


class SemanticHotspotHelper:
    """Comprehensive helper that colors existing voxels using the same pipeline as depth processing."""
    
    def __init__(self, node, voxel_helper, config: Dict[str, Any] = None):
        """
        Initialize semantic hotspot helper.
        
        Args:
            node: ROS2 node instance (octomap node)
            voxel_helper: VoxelMappingHelper instance
            config: Configuration dictionary
        """
        self.node = node
        self.voxel_helper = voxel_helper
        self.bridge = CvBridge()
        
        # Load configuration 
        self.config = config.get('semantic_bridge', {}) if config else {}
        self.enable_semantic = self.config.get('enable_semantic_mapping', True)
        
        # COMPREHENSIVE QUEUE: Store ALL hotspots for comprehensive processing
        self.hotspot_queue = deque(maxlen=50)  # Keep last 50 hotspots
        self.processed_hotspots = set()  # Track processed hotspots to avoid duplicates
        
        # Queue for hotspots that arrive before voxels are created
        self.pending_hotspots = []
        self.max_queue_size = 20  # Increased queue size for comprehensive processing
        self.retry_interval = 0.1  # seconds
        self.last_retry_time = 0.0
        
        # Semantic color mapping for different VLM answers
        self._semantic_colors = {}
        self._color_palette = [
            [1.0, 0.0, 0.0],    # Red
            [0.0, 1.0, 0.0],    # Green  
            [0.0, 0.0, 1.0],    # Blue
            [1.0, 1.0, 0.0],    # Yellow
            [1.0, 0.0, 1.0],    # Magenta
            [0.0, 1.0, 1.0],    # Cyan
            [1.0, 0.5, 0.0],    # Orange
            [0.5, 0.0, 1.0],    # Purple
            [0.5, 0.5, 0.0],    # Olive
            [0.0, 0.5, 0.5],    # Teal
            [1.0, 0.8, 0.0],    # Gold
            [0.8, 0.0, 0.8],    # Pink
        ]
        
        if self.enable_semantic:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info("Comprehensive semantic hotspot helper initialized")
    
    def _get_semantic_color(self, vlm_answer: str) -> List[float]:
        """Get or assign a unique color for a VLM answer."""
        if vlm_answer not in self._semantic_colors:
            # Assign next available color from palette
            color_idx = len(self._semantic_colors) % len(self._color_palette)
            self._semantic_colors[vlm_answer] = self._color_palette[color_idx]
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info(f"Assigned color {self._semantic_colors[vlm_answer]} to VLM answer: '{vlm_answer}'")
        
        return self._semantic_colors[vlm_answer]
    
    def process_hotspot_message(self, msg: String) -> bool:
        """Process incoming semantic hotspot message and add to comprehensive queue."""
        try:
            if not self.enable_semantic:
                return False
                
            data = json.loads(msg.data)
            
            if data.get('type') != 'similarity_hotspots':
                return False
            
            vlm_answer = data.get('vlm_answer')
            pose = data.get('pose', {})
            mask_shape = data.get('mask_shape')
            mask_data = data.get('mask_data')
            threshold_used = data.get('threshold_used', 0.6)
            depth_data = data.get('depth_data')
            stats = data.get('stats', {})
            
            if not vlm_answer or not mask_data:
                return False
            
            # Reconstruct binary hotspot mask
            mask = np.array(mask_data, dtype=np.uint8).reshape(mask_shape)
            
            # Get depth data if available
            depth_mask = None
            if depth_data and 'depth_mask' in depth_data and 'depth_shape' in depth_data:
                try:
                    depth_mask_list = depth_data['depth_mask']
                    depth_shape = tuple(depth_data['depth_shape'])
                    depth_mask = np.array(depth_mask_list, dtype=np.float32).reshape(depth_shape)
                except Exception:
                    depth_mask = None
            
            # Create pose object for the node's existing functions
            from geometry_msgs.msg import PoseStamped
            pose_msg = PoseStamped()
            pose_msg.pose.position.x = float(pose.get('x', 0))
            pose_msg.pose.position.y = float(pose.get('y', 0))
            pose_msg.pose.position.z = float(pose.get('z', 0))
            pose_msg.pose.orientation.w = 1.0  # Identity quaternion
            
            # Create unique identifier for this hotspot
            hotspot_id = f"{vlm_answer}_{pose.get('x', 0):.3f}_{pose.get('y', 0):.3f}_{pose.get('z', 0):.3f}_{time.time():.3f}"
            
            # Add to comprehensive queue if not already processed
            if hotspot_id not in self.processed_hotspots:
                hotspot_data = {
                    'id': hotspot_id,
                    'vlm_answer': vlm_answer,
                    'mask': mask,
                    'depth_mask': depth_mask,
                    'pose_msg': pose_msg,
                    'threshold_used': threshold_used,
                    'stats': stats,
                    'timestamp': time.time(),
                    'retry_count': 0
                }
                
                self.hotspot_queue.append(hotspot_data)
                self.processed_hotspots.add(hotspot_id)
                
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().info(f"Added hotspot '{vlm_answer}' to comprehensive queue (queue size: {len(self.hotspot_queue)})")
            
            # Try to process immediately
            success = self._color_voxels_using_octomap_pipeline(vlm_answer, mask, depth_mask, pose_msg, threshold_used, stats)
            
            if not success:
                # Queue for later processing if voxels don't exist yet
                self._queue_hotspot_for_retry(vlm_answer, mask, depth_mask, pose_msg, threshold_used, stats)
            
            return success
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error processing hotspot message: {e}")
            return False
    
    def _queue_hotspot_for_retry(self, vlm_answer: str, mask: np.ndarray, depth_mask: Optional[np.ndarray], 
                                pose_msg, threshold_used: float, stats: Dict):
        """Queue a hotspot for later processing when voxels might be available."""
        if len(self.pending_hotspots) >= self.max_queue_size:
            # Remove oldest hotspot
            self.pending_hotspots.pop(0)
        
        hotspot_data = {
            'vlm_answer': vlm_answer,
            'mask': mask,
            'depth_mask': depth_mask,
            'pose_msg': pose_msg,
            'threshold_used': threshold_used,
            'stats': stats,
            'timestamp': time.time(),
            'retry_count': 0
        }
        
        self.pending_hotspots.append(hotspot_data)
        
        if hasattr(self.node, 'get_logger'):
            self.node.get_logger().info(f"Queued hotspot '{vlm_answer}' for later processing (queue size: {len(self.pending_hotspots)})")
    
    def process_comprehensive_hotspots(self):
        """Process ALL hotspots in the comprehensive queue for maximum information coverage."""
        try:
            if not self.hotspot_queue:
                return
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info(f"Processing comprehensive hotspot queue: {len(self.hotspot_queue)} hotspots")
            
            # Process all hotspots in the queue
            processed_count = 0
            failed_count = 0
            
            for hotspot_data in list(self.hotspot_queue):  # Copy list to avoid modification during iteration
                try:
                    success = self._color_voxels_using_octomap_pipeline(
                        hotspot_data['vlm_answer'],
                        hotspot_data['mask'],
                        hotspot_data['depth_mask'],
                        hotspot_data['pose_msg'],
                        hotspot_data['threshold_used'],
                        hotspot_data['stats']
                    )
                    
                    if success:
                        processed_count += 1
                        # Remove from queue after successful processing
                        self.hotspot_queue.remove(hotspot_data)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    if hasattr(self.node, 'get_logger'):
                        self.node.get_logger().warn(f"Error processing comprehensive hotspot: {e}")
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info(f"Comprehensive processing: {processed_count} successful, {failed_count} failed, {len(self.hotspot_queue)} remaining")
                
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error in comprehensive hotspot processing: {e}")
    
    def process_pending_hotspots(self):
        """Process any pending hotspots that were queued due to missing voxels."""
        current_time = time.time()
        if current_time - self.last_retry_time < self.retry_interval:
            return
        
        self.last_retry_time = current_time
        
        if not self.pending_hotspots:
            return
        
        # Try to process pending hotspots
        still_pending = []
        
        for hotspot_data in self.pending_hotspots:
            # Limit retry attempts
            if hotspot_data['retry_count'] >= 5:  # Max 5 retries
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().warn(f"Giving up on hotspot '{hotspot_data['vlm_answer']}' after 5 retries")
                continue
            
            # Try to process
            success = self._color_voxels_using_octomap_pipeline(
                hotspot_data['vlm_answer'],
                hotspot_data['mask'],
                hotspot_data['depth_mask'],
                hotspot_data['pose_msg'],
                hotspot_data['threshold_used'],
                hotspot_data['stats']
            )
            
            if success:
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().info(f"Successfully processed queued hotspot '{hotspot_data['vlm_answer']}'")
            else:
                # Increment retry count and keep in queue
                hotspot_data['retry_count'] += 1
                still_pending.append(hotspot_data)
        
        # Update queue
        self.pending_hotspots = still_pending
        
        if len(self.pending_hotspots) > 0 and hasattr(self.node, 'get_logger'):
            self.node.get_logger().debug(f"Still have {len(self.pending_hotspots)} pending hotspots")
    
    def force_comprehensive_processing(self):
        """Force processing of all hotspots in the comprehensive queue."""
        try:
            if not self.hotspot_queue:
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().info("No hotspots in comprehensive queue to process")
                return
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info(f"Force processing comprehensive hotspot queue: {len(self.hotspot_queue)} hotspots")
            
            # Process all hotspots in the queue
            processed_count = 0
            failed_count = 0
            
            for hotspot_data in list(self.hotspot_queue):  # Copy list to avoid modification during iteration
                try:
                    success = self._color_voxels_using_octomap_pipeline(
                        hotspot_data['vlm_answer'],
                        hotspot_data['mask'],
                        hotspot_data['depth_mask'],
                        hotspot_data['pose_msg'],
                        hotspot_data['threshold_used'],
                        hotspot_data['stats']
                    )
                    
                    if success:
                        processed_count += 1
                        # Remove from queue after successful processing
                        self.hotspot_queue.remove(hotspot_data)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    if hasattr(self.node, 'get_logger'):
                        self.node.get_logger().warn(f"Error processing comprehensive hotspot: {e}")
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info(f"Force comprehensive processing complete: {processed_count} successful, {failed_count} failed, {len(self.hotspot_queue)} remaining")
                
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error in force comprehensive processing: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all queues."""
        return {
            'comprehensive_queue_size': len(self.hotspot_queue),
            'pending_hotspots': len(self.pending_hotspots),
            'total_processed': len(self.processed_hotspots),
            'queue_contents': [
                {
                    'vlm_answer': h['vlm_answer'],
                    'timestamp': h['timestamp'],
                    'retry_count': h.get('retry_count', 0)
                } for h in list(self.hotspot_queue)[:10]  # Show first 10
            ]
        }
    
    def _color_voxels_using_octomap_pipeline(self, vlm_answer: str, hotspot_mask: np.ndarray, 
                                           depth_mask: Optional[np.ndarray], pose_msg, 
                                           threshold_used: float, stats: Dict) -> bool:
        """Color existing voxels using the EXACT same pipeline as the octomap node depth processing."""
        try:
            if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
                return False
            
            # Get hotspot pixel coordinates
            hotspot_coords = np.where(hotspot_mask > 0)
            if len(hotspot_coords[0]) == 0:
                return False
            
            # Use depth data or fallback to estimated depth
            if depth_mask is not None:
                # Use actual depth where available
                depth_m = depth_mask
            else:
                # Create estimated depth image
                estimated_depth = self.config.get('estimated_hotspot_depth', 1.5)
                depth_m = np.full(hotspot_mask.shape, estimated_depth, dtype=np.float32)
                # Only keep depths where we have hotspots
                depth_m[hotspot_mask == 0] = 0.0
            
            # Use the node's EXACT _depth_to_world_points function
            if hasattr(self.node, '_depth_to_world_points') and hasattr(self.node, 'camera_intrinsics'):
                points_world, u_indices, v_indices = self.node._depth_to_world_points(
                    depth_m, self.node.camera_intrinsics, pose_msg
                )
                
                if points_world is None or len(points_world) == 0:
                    return False
                
                # Filter to only include points that correspond to hotspot pixels
                hotspot_world_points = []
                for i, (u, v) in enumerate(zip(u_indices, v_indices)):
                    if v < hotspot_mask.shape[0] and u < hotspot_mask.shape[1]:
                        if hotspot_mask[v, u] > 0:  # This pixel is a hotspot
                            hotspot_world_points.append(points_world[i])
                
                if not hotspot_world_points:
                    return False
                
                hotspot_world_points = np.array(hotspot_world_points)
                
                # Get voxel keys for these world points using the voxel helper's function
                voxel_keys = set()
                for world_point in hotspot_world_points:
                    voxel_key = self.voxel_helper.get_voxel_key_from_point(world_point)
                    # CRITICAL: Only color voxels that already exist (were created by depth processing)
                    # This prevents creating new voxels and ensures we only color existing ones
                    if voxel_key in self.voxel_helper.voxels:
                        voxel_keys.add(voxel_key)
                
                if not voxel_keys:
                    # No existing voxels found - this will be queued for retry
                    return False
                
                # Add semantic information to existing voxels
                similarity_score = stats.get('avg_similarity', threshold_used + 0.1)
                semantic_mapper = self.voxel_helper.semantic_mapper
                
                with semantic_mapper.voxel_semantics_lock:
                    for voxel_key in voxel_keys:
                        semantic_info = {
                            'vlm_answer': vlm_answer,
                            'similarity': similarity_score,
                            'threshold_used': threshold_used,
                            'detection_method': 'binary_threshold_hotspot',
                            'depth_used': depth_mask is not None,
                            'timestamp': time.time(),
                            'color': self._get_semantic_color(vlm_answer)
                        }
                        semantic_mapper.voxel_semantics[voxel_key] = semantic_info
                
                if hasattr(self.node, 'get_logger'):
                    depth_info = " with depth" if depth_mask is not None else f" (estimated depth)"
                    self.node.get_logger().info(f"Colored {len(voxel_keys)} existing voxels for '{vlm_answer}' (threshold: {threshold_used}){depth_info}")
                
                return True
            else:
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().error("Node missing required depth processing functions")
                return False
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error coloring voxels from hotspot: {e}")
            return False
    
    def get_semantic_colored_cloud(self, max_points: int = 10000) -> Optional[PointCloud2]:
        """Create a colored point cloud with semantic colors for semantic voxels."""
        try:
            if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
                return None
            
            semantic_mapper = self.voxel_helper.semantic_mapper
            
            # Get all voxels with semantic information
            with semantic_mapper.voxel_semantics_lock:
                semantic_voxels = semantic_mapper.voxel_semantics.copy()
            
            if not semantic_voxels:
                return None
            
            # Convert voxel keys to 3D points and colors
            points = []
            colors = []
            
            voxel_resolution = getattr(self.voxel_helper, 'voxel_resolution', 0.1)
            
            for voxel_key, semantic_info in semantic_voxels.items():
                if len(points) >= max_points:
                    break
                
                # Only include voxels that exist in the main voxel map
                if voxel_key not in self.voxel_helper.voxels:
                    continue
                
                # Convert voxel key to world coordinates
                vx, vy, vz = voxel_key
                world_x = vx * voxel_resolution
                world_y = vy * voxel_resolution
                world_z = vz * voxel_resolution
                
                points.append([world_x, world_y, world_z])
                
                # Use semantic color
                semantic_color = semantic_info.get('color', [0.0, 1.0, 0.0])  # Default green
                colors.append(semantic_color + [1.0])  # Add alpha
            
            if not points:
                return None
            
            # Create PointCloud2 message
            points_array = np.array(points, dtype=np.float32)
            colors_array = np.array(colors, dtype=np.float32)
            
            # Create structured array for PointCloud2
            point_cloud_data = []
            for i in range(len(points)):
                point_cloud_data.append((
                    points_array[i][0], points_array[i][1], points_array[i][2],
                    int(colors_array[i][0] * 255),  # R
                    int(colors_array[i][1] * 255),  # G
                    int(colors_array[i][2] * 255),  # B
                ))
            
            # Convert to PointCloud2
            import sensor_msgs_py.point_cloud2 as pc2
            from sensor_msgs.msg import PointField
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
                PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
                PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
            ]
            
            cloud_msg = pc2.create_cloud(Header(), fields, point_cloud_data)
            cloud_msg.header.frame_id = getattr(self.voxel_helper, 'map_frame', 'map')
            cloud_msg.header.stamp = self.node.get_clock().now().to_msg()
            
            return cloud_msg
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error creating semantic colored cloud: {e}")
            return None
    
    def get_semantic_stats(self) -> Dict[str, Any]:
        """Get semantic mapping statistics."""
        try:
            if hasattr(self.voxel_helper, 'semantic_mapper') and self.voxel_helper.semantic_mapper:
                semantic_mapper = self.voxel_helper.semantic_mapper
                
                with semantic_mapper.voxel_semantics_lock:
                    semantic_voxels = semantic_mapper.voxel_semantics.copy()
                
                # Count by VLM answer
                vlm_counts = {}
                total_semantic_voxels = len(semantic_voxels)
                
                for voxel_key, semantic_info in semantic_voxels.items():
                    vlm_answer = semantic_info.get('vlm_answer', 'unknown')
                    vlm_counts[vlm_answer] = vlm_counts.get(vlm_answer, 0) + 1
                
                total_voxels = len(self.voxel_helper.voxels)
                
                return {
                    'total_voxels': total_voxels,
                    'semantic_voxels': total_semantic_voxels,
                    'semantic_percentage': (total_semantic_voxels / max(total_voxels, 1)) * 100,
                    'vlm_answer_counts': vlm_counts,
                    'unique_vlm_answers': len(vlm_counts),
                    'colors_assigned': len(self._semantic_colors),
                    'pending_hotspots': len(self.pending_hotspots),
                    'comprehensive_queue_size': len(self.hotspot_queue),
                    'total_processed': len(self.processed_hotspots)
                }
            return {'semantic_mapping': 'disabled'}
        except Exception as e:
            return {'error': f'Failed to get semantic stats: {e}'} 
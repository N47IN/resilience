#!/usr/bin/env python3
"""
Clean Semantic Bridge with Binary Threshold Filtering

Efficient bridge that directly applies binary threshold to similarity maps
and publishes both binary mask images for RViz and pose data for voxel mapping.
"""

import numpy as np
import json
import time
import cv2
from typing import Dict, List, Optional, Any, Tuple
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SemanticHotspotPublisher:
    """Clean publisher that applies binary threshold to similarity maps and publishes results."""
    
    def __init__(self, node, config: Dict[str, Any] = None):
        self.node = node
        self.bridge = CvBridge()
        
        # Load configuration
        self.config = config.get('semantic_bridge', {}) if config else {}
        self.hotspot_threshold = self.config.get('hotspot_similarity_threshold', 0.6)
        self.min_area = self.config.get('min_hotspot_area', 100)
        self.publish_rate_limit = self.config.get('publish_rate_limit', 2.0)
        
        # Rate limiting
        self.last_publish_time = 0.0
        
        # Publishers
        self.hotspot_data_pub = self.node.create_publisher(String, '/semantic_hotspots', 10)
        self.hotspot_mask_pub = self.node.create_publisher(Image, '/semantic_hotspot_mask', 10)
        self.hotspot_overlay_pub = self.node.create_publisher(Image, '/semantic_hotspot_overlay', 10)
        
        if hasattr(self.node, 'get_logger'):
            self.node.get_logger().info(f"Semantic bridge initialized - threshold: {self.hotspot_threshold}")
    
    def publish_similarity_hotspots(self, vlm_answer: str, similarity_map: np.ndarray, 
                                   pose: np.ndarray, timestamp: float, 
                                   original_image: Optional[np.ndarray] = None,
                                   depth_image: Optional[np.ndarray] = None) -> bool:
        """
        Apply binary threshold to similarity map and publish hotspots + visualization.
        
        Args:
            vlm_answer: VLM answer being processed
            similarity_map: 2D similarity map (H, W) with values in [0, 1]
            pose: Robot pose [x, y, z] when similarity was computed
            timestamp: Timestamp of the data
            original_image: Original RGB image for overlay visualization (optional)
            depth_image: Depth image for accurate 3D projection (optional)
            
        Returns:
            True if hotspots were published, False otherwise
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_publish_time < (1.0 / self.publish_rate_limit):
                return False
            
            # Apply binary threshold directly to similarity map
            hotspot_mask = similarity_map > self.hotspot_threshold
            
            if not np.any(hotspot_mask):
                return False  # No hotspots
            
            # Filter small regions using connected components
            hotspot_mask_uint8 = (hotspot_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(hotspot_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create filtered mask with only significant regions
            filtered_mask = np.zeros_like(hotspot_mask, dtype=np.uint8)
            hotspot_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    cv2.fillPoly(filtered_mask, [contour], 255)
                    hotspot_count += 1
            
            if hotspot_count == 0:
                return False  # No significant hotspots
            
            # Compute statistics
            hotspot_pixels = np.sum(filtered_mask > 0)
            hotspot_similarity_values = similarity_map[hotspot_mask]
            
            # Publish binary mask image for RViz visualization
            self._publish_hotspot_mask_image(filtered_mask, vlm_answer, timestamp)
            
            # Publish overlay image if original image provided
            if original_image is not None:
                self._publish_hotspot_overlay(original_image, filtered_mask, vlm_answer, timestamp)
            
            # Create structured data message for voxel mapping
            hotspot_data = {
                'type': 'similarity_hotspots',
                'vlm_answer': vlm_answer,
                'timestamp': timestamp,
                'pose': {
                    'x': float(pose[0]),
                    'y': float(pose[1]), 
                    'z': float(pose[2])
                },
                'mask_shape': filtered_mask.shape,
                'mask_data': filtered_mask.flatten().tolist(),  # Binary mask as list
                'threshold_used': self.hotspot_threshold,
                'has_depth_data': depth_image is not None,
                'depth_shape': depth_image.shape if depth_image is not None else None,
                'stats': {
                    'hotspot_count': hotspot_count,
                    'hotspot_pixels': int(hotspot_pixels),
                    'max_similarity': float(np.max(hotspot_similarity_values)),
                    'avg_similarity': float(np.mean(hotspot_similarity_values)),
                    'threshold': self.hotspot_threshold
                }
            }
            
            # If we have depth data, include it for accurate 3D mapping
            if depth_image is not None:
                # Extract depth values only for hotspot regions to reduce data size
                hotspot_depth_values = depth_image[filtered_mask > 0]
                if len(hotspot_depth_values) > 0:
                    # Send depth statistics and sample values
                    hotspot_data['depth_stats'] = {
                        'min_depth': float(np.min(hotspot_depth_values)),
                        'max_depth': float(np.max(hotspot_depth_values)),
                        'mean_depth': float(np.mean(hotspot_depth_values)),
                        'depth_units': 'meters'
                    }
                    
                    # For very large depth images, we could downsample or send key depth values
                    # For now, just send the depth image dimensions and availability flag
                    hotspot_data['depth_available'] = True
                else:
                    hotspot_data['depth_available'] = False
            else:
                hotspot_data['depth_available'] = False
            
            # Publish structured data
            msg = String(data=json.dumps(hotspot_data))
            self.hotspot_data_pub.publish(msg)
            
            self.last_publish_time = current_time
            
            if hasattr(self.node, 'get_logger'):
                depth_info = f" + depth" if depth_image is not None else ""
                self.node.get_logger().info(f"Published {hotspot_count} hotspots for '{vlm_answer}' (>{self.hotspot_threshold}){depth_info}")
            
            return True
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing hotspots: {e}")
            return False
    
    def _publish_hotspot_mask_image(self, mask: np.ndarray, vlm_answer: str, timestamp: float):
        """Publish binary hotspot mask as ROS Image for RViz visualization."""
        try:
            # Convert binary mask to 3-channel image for better visibility
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask > 0] = [0, 255, 0]  # Green hotspots
            
            # Convert to ROS Image message
            mask_msg = self.bridge.cv2_to_imgmsg(mask_rgb, encoding='rgb8')
            mask_msg.header.stamp = self.node.get_clock().now().to_msg()
            mask_msg.header.frame_id = 'camera_link'
            
            self.hotspot_mask_pub.publish(mask_msg)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing mask image: {e}")
    
    def _publish_hotspot_overlay(self, original_image: np.ndarray, mask: np.ndarray, 
                                vlm_answer: str, timestamp: float):
        """Publish overlay of hotspots on original image for RViz visualization."""
        try:
            # Create overlay
            overlay = original_image.copy()
            
            # Apply green overlay where hotspots are detected
            hotspot_pixels = mask > 0
            overlay[hotspot_pixels] = cv2.addWeighted(
                overlay[hotspot_pixels], 0.7,
                np.full_like(overlay[hotspot_pixels], [0, 255, 0]), 0.3,
                0
            )
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Hotspots: {vlm_answer}"
            cv2.putText(overlay, text, (10, 30), font, 0.7, (0, 255, 0), 2)
            
            # Convert to ROS Image message
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
            overlay_msg.header.stamp = self.node.get_clock().now().to_msg()
            overlay_msg.header.frame_id = 'camera_link'
            
            self.hotspot_overlay_pub.publish(overlay_msg)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing overlay image: {e}")


class SemanticHotspotSubscriber:
    """Clean subscriber for receiving binary hotspot masks in voxel mapper."""
    
    def __init__(self, node, voxel_helper, config: Dict[str, Any] = None):
        self.node = node
        self.voxel_helper = voxel_helper
        self.bridge = CvBridge()
        
        # Load configuration 
        self.config = config.get('semantic_bridge', {}) if config else {}
        self.enable_semantic = self.config.get('enable_semantic_mapping', True)
        
        # Subscribe to hotspots
        if self.enable_semantic:
            self.node.create_subscription(
                String, '/semantic_hotspots', 
                self.hotspot_callback, 10
            )
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info("Subscribed to semantic hotspots")
    
    def hotspot_callback(self, msg: String):
        """Process incoming binary hotspot masks."""
        try:
            if not self.enable_semantic:
                return
                
            data = json.loads(msg.data)
            
            if data.get('type') != 'similarity_hotspots':
                return
            
            vlm_answer = data.get('vlm_answer')
            pose = data.get('pose', {})
            mask_shape = data.get('mask_shape')
            mask_data = data.get('mask_data')
            threshold_used = data.get('threshold_used', 0.6)
            stats = data.get('stats', {})
            
            if not vlm_answer or not mask_data:
                return
            
            # Reconstruct binary mask
            mask = np.array(mask_data, dtype=np.uint8).reshape(mask_shape)
            
            # Apply semantic labeling to voxels
            depth_available = data.get('depth_available', False)
            self._apply_hotspot_to_voxels(vlm_answer, mask, pose, threshold_used, stats, depth_available)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error processing hotspots: {e}")
    
    def _apply_hotspot_to_voxels(self, vlm_answer: str, mask: np.ndarray, 
                                pose: Dict, threshold_used: float, stats: Dict,
                                depth_available: bool = False):
        """Apply semantic labels to voxels based on binary hotspot mask."""
        try:
            if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
                return
            
            semantic_mapper = self.voxel_helper.semantic_mapper
            
            # Get hotspot pixel coordinates
            hotspot_coords = np.where(mask > 0)
            
            if len(hotspot_coords[0]) == 0:
                return
            
            # Check if we have depth data available
            has_depth = depth_available
            
            if has_depth:
                # TODO: Extract depth values from the message
                # For now, we'll need to modify the message structure to include depth values
                # This would require sending depth data along with the mask
                print(f"Depth data available for '{vlm_answer}' - would use actual depth for 3D mapping")
                # Fall back to estimated depth for now
                estimated_depth = self.config.get('estimated_hotspot_depth', 1.5)
            else:
                # Use configured depth estimation as fallback
                estimated_depth = self.config.get('estimated_hotspot_depth', 1.5)
                print(f"No depth data available for '{vlm_answer}' - using estimated depth: {estimated_depth}m")
            
            # Camera intrinsics - could be passed via config or use reasonable defaults
            intrinsics = self.config.get('camera_intrinsics', [186.0, 186.0, mask.shape[1]//2, mask.shape[0]//2])
            fx, fy, cx, cy = intrinsics
            
            # Convert hotspot pixels to world coordinates
            voxel_keys = set()
            robot_pos = np.array([pose.get('x', 0), pose.get('y', 0), pose.get('z', 0)])
            
            for i in range(len(hotspot_coords[0])):
                v, u = hotspot_coords[0][i], hotspot_coords[1][i]
                
                # Pixel to camera coordinates
                cam_x = (u - cx) * estimated_depth / fx
                cam_y = (v - cy) * estimated_depth / fy
                cam_z = estimated_depth
                
                # Simple world transformation (camera at robot position)
                # TODO: Could be enhanced with proper camera-to-base transformation
                world_point = robot_pos + np.array([cam_x, cam_y, cam_z])
                
                # Get voxel key
                if hasattr(self.voxel_helper, 'get_voxel_key_from_point'):
                    voxel_key = self.voxel_helper.get_voxel_key_from_point(world_point)
                    voxel_keys.add(voxel_key)
            
            # Mark voxels as semantic with high confidence since they passed binary threshold
            similarity_score = stats.get('avg_similarity', threshold_used + 0.1)
            
            with semantic_mapper.voxel_semantics_lock:
                for voxel_key in voxel_keys:
                    semantic_info = {
                        'vlm_answer': vlm_answer,
                        'similarity': similarity_score,
                        'threshold_used': threshold_used,
                        'detection_method': 'binary_threshold_hotspot',
                        'depth_used': has_depth,
                        'timestamp': time.time()
                    }
                    semantic_mapper.voxel_semantics[voxel_key] = semantic_info
            
            if hasattr(self.node, 'get_logger'):
                depth_info = f" with depth" if has_depth else f" (estimated depth: {estimated_depth}m)"
                self.node.get_logger().info(f"Applied semantic labels to {len(voxel_keys)} voxels for '{vlm_answer}' (threshold: {threshold_used}){depth_info}")
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error applying hotspots to voxels: {e}")
    
    def get_semantic_stats(self) -> Dict[str, Any]:
        """Get semantic mapping statistics."""
        try:
            if hasattr(self.voxel_helper, 'semantic_mapper') and self.voxel_helper.semantic_mapper:
                return self.voxel_helper.semantic_mapper.get_semantic_statistics()
            return {'semantic_mapping': 'disabled'}
        except Exception:
            return {'error': 'Failed to get semantic stats'} 
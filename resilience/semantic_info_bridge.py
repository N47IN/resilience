#!/usr/bin/env python3
"""
Clean Semantic Bridge with Binary Threshold Filtering

Simplified bridge that applies a binary threshold to similarity maps and
publishes only the hotspot mask and the original RGB image timestamp.
"""

import numpy as np
import json
import time
import cv2
from typing import Dict, List, Optional, Any, Tuple
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SemanticHotspotPublisher:
    """Publisher that applies binary threshold to similarity maps and publishes results."""
    
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
                                   timestamp: float,
                                   original_image: Optional[np.ndarray] = None) -> bool:
        """
        Publish binary hotspot mask built from a similarity map along with original RGB timestamp.
        
        Args:
            vlm_answer: VLM answer being processed
            similarity_map: 2D similarity map (H, W) with values in [0, 1]
            timestamp: Original RGB image timestamp (float seconds)
            original_image: Original RGB image for overlay visualization (optional)
            
        Returns:
            True if hotspots were published, False otherwise
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_publish_time < (1.0 / self.publish_rate_limit):
                return False
            
            # Validate input data
            if similarity_map is None or similarity_map.size == 0:
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().warn(f"Invalid similarity map for '{vlm_answer}'")
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
            hotspot_pixels = int(np.sum(filtered_mask > 0))
            hotspot_similarity_values = similarity_map[hotspot_mask]
            
            # Publish binary mask image for RViz visualization
            self._publish_hotspot_mask_image(filtered_mask, vlm_answer, timestamp)
            
            # Publish overlay image if original image provided
            if original_image is not None:
                self._publish_hotspot_overlay(original_image, filtered_mask, vlm_answer, timestamp)
            
            # Create structured data message for voxel mapping (timestamp-only, no pose/depth)
            hotspot_data = {
                'type': 'similarity_hotspots',
                'vlm_answer': vlm_answer,
                'timestamp': float(timestamp),
                'mask_shape': list(filtered_mask.shape),
                'mask_data': filtered_mask.flatten().tolist(),  # Binary mask as list
                'threshold_used': float(self.hotspot_threshold),
                'stats': {
                    'hotspot_count': int(hotspot_count),
                    'hotspot_pixels': int(hotspot_pixels),
                    'max_similarity': float(np.max(hotspot_similarity_values)),
                    'avg_similarity': float(np.mean(hotspot_similarity_values)),
                    'threshold': float(self.hotspot_threshold)
                }
            }
            
            # Publish structured data
            msg = String(data=json.dumps(hotspot_data))
            self.hotspot_data_pub.publish(msg)
            
            self.last_publish_time = current_time
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info(f"Published {hotspot_count} hotspots for '{vlm_answer}' (>{self.hotspot_threshold}) @ {timestamp:.6f}")
            
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
    """Subscriber for receiving binary hotspot masks in voxel mapper."""
    
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
            
            # This subscriber remains available for direct application if needed by a helper.
            # The octomap node will primarily consume the JSON via its own subscriber.
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error processing hotspots: {e}") 
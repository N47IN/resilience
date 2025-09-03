#!/usr/bin/env python3
"""
Semantic Depth OctoMap ROS2 Node

Simplified node that uses VoxelMappingHelper for all heavy lifting.
Subscribes to depth, pose, and semantic info to create semantic voxel maps.
Maintains timestamped buffers for depth frames and poses to align with hotspot masks
received via the semantic bridge using original RGB timestamps.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

import numpy as np
from cv_bridge import CvBridge
import time
import json
import math
from typing import Optional, List
import sensor_msgs_py.point_cloud2 as pc2
import threading
import cv2
import os

# Import helpers
try:
	from resilience.voxel_mapping_helper import VoxelMappingHelper
	HELPERS_AVAILABLE = True
except ImportError:
	HELPERS_AVAILABLE = False


class SemanticDepthOctoMapNode(Node):
	"""Simplified semantic depth octomap node using helpers."""

	def __init__(self):
		super().__init__('semantic_depth_octomap_node')

		# Parameters
		self.declare_parameters('', [
			('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered'),
			('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info'),
			('pose_topic', '/robot_1/sensors/front_stereo/pose'),
			('map_frame', 'map'),
			('voxel_resolution', 0.1),
			('max_range', 5.0),
			('min_range', 0.1),
			('probability_hit', 0.7),
			('probability_miss', 0.4),
			('occupancy_threshold', 0.5),
			('publish_markers', True),
			('publish_stats', True),
			('publish_colored_cloud', True),
			('use_cube_list_markers', True),
			('max_markers', 30000),
			('marker_publish_rate', 1.0),
			('stats_publish_rate', 1.0),
			('pose_is_base_link', True),
			('apply_optical_frame_rotation', True),
			('cam_to_base_rpy_deg', [0.0, 0.0, 0.0]),
			('cam_to_base_xyz', [0.0, 0.0, 0.0]),
			('embedding_dim', 1152),
			('enable_semantic_mapping', True),
			('semantic_similarity_threshold', 0.6),
			('buffers_directory', '/home/navin/ros2_ws/src/buffers'),
			('bridge_queue_max_size', 100),
			('bridge_queue_process_interval', 0.1),
			('enable_voxel_mapping', True),
			('sync_buffer_seconds', 2.0),
			# New: inactivity detection and export
			('inactivity_threshold_seconds', 2.5),
			('semantic_export_directory', '/home/navin/ros2_ws/src/buffers')
		])

		params = self.get_parameters([
			'depth_topic', 'camera_info_topic', 'pose_topic',
			'map_frame', 'voxel_resolution', 'max_range', 'min_range', 'probability_hit',
			'probability_miss', 'occupancy_threshold', 'publish_markers', 'publish_stats',
			'publish_colored_cloud', 'use_cube_list_markers', 'max_markers', 'marker_publish_rate', 'stats_publish_rate',
			'pose_is_base_link', 'apply_optical_frame_rotation', 'cam_to_base_rpy_deg', 'cam_to_base_xyz', 'embedding_dim',
			'enable_semantic_mapping', 'semantic_similarity_threshold', 'buffers_directory', 'bridge_queue_max_size', 'bridge_queue_process_interval',
			'enable_voxel_mapping', 'sync_buffer_seconds', 'inactivity_threshold_seconds', 'semantic_export_directory'
		])

		# Extract parameter values
		(self.depth_topic, self.camera_info_topic, self.pose_topic,
		 self.map_frame, self.voxel_resolution, self.max_range, self.min_range, self.prob_hit,
		 self.prob_miss, self.occ_thresh, self.publish_markers, self.publish_stats, self.publish_colored_cloud,
		 self.use_cube_list_markers, self.max_markers, self.marker_publish_rate, self.stats_publish_rate,
		 self.pose_is_base_link, self.apply_optical_frame_rotation, self.cam_to_base_rpy_deg, self.cam_to_base_xyz,
		 self.embedding_dim, self.enable_semantic_mapping, self.semantic_similarity_threshold,
		 self.buffers_directory, self.bridge_queue_max_size, self.bridge_queue_process_interval,
		 self.enable_voxel_mapping, self.sync_buffer_seconds, self.inactivity_threshold_seconds, self.semantic_export_directory) = [p.value for p in params]

		# State
		self.bridge = CvBridge()
		self.camera_intrinsics = None
		self.latest_pose = None
		self.last_marker_pub = 0.0
		self.last_stats_pub = 0.0
		# Inactivity detection state
		self.last_data_time = time.time()
		self.semantic_pcd_exported = False

		# Timestamped buffers for sync
		self.depth_buffer = []  # list of tuples (timestamp_float, depth_np_float_meters)
		self.pose_buffer = []   # list of tuples (timestamp_float, PoseStamped)
		self.sync_buffer_duration = float(self.sync_buffer_seconds)
		self.sync_lock = threading.Lock()

		# Queue system for bridge messages
		self.bridge_message_queue = []
		self.bridge_queue_lock = threading.Lock()
		self.max_queue_size = int(self.bridge_queue_max_size)
		self.queue_processing_interval = float(self.bridge_queue_process_interval)
		self.last_queue_process_time = 0.0

		# Initialize voxel mapping helper
		if not HELPERS_AVAILABLE:
			self.get_logger().error("VoxelMappingHelper not available!")
			return

		self.voxel_helper = VoxelMappingHelper(
			voxel_resolution=float(self.voxel_resolution),
			probability_hit=float(self.prob_hit),
			probability_miss=float(self.prob_miss),
			occupancy_threshold=float(self.occ_thresh),
			embedding_dim=int(self.embedding_dim),
			enable_semantic_mapping=bool(self.enable_semantic_mapping),
			semantic_similarity_threshold=float(self.semantic_similarity_threshold),
			map_frame=str(self.map_frame),
			verbose=False
		)

		# Load existing embeddings
		if isinstance(self.buffers_directory, str) and len(self.buffers_directory) > 0:
			loaded = self.voxel_helper.load_vlm_embeddings_from_buffers(self.buffers_directory)
			if loaded > 0:
				self.get_logger().info(f"Loaded {loaded} VLM embeddings from buffers")

		# Precompute transforms
		self.R_opt_to_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
		self.R_cam_to_base_extra = self._rpy_deg_to_rot(self.cam_to_base_rpy_deg)
		self.t_cam_to_base_extra = np.array(self.cam_to_base_xyz, dtype=np.float32)

		# QoS
		sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

		# Subscribers
		self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
		self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
		self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)
		# Subscribe to semantic hotspots (timestamp-only bridge)
		if self.enable_semantic_mapping and self.enable_voxel_mapping:
			self.create_subscription(String, '/semantic_hotspots', self.semantic_hotspot_callback, 10)

		# Publishers
		self.marker_pub = self.create_publisher(MarkerArray, '/semantic_octomap_markers', 10) if self.publish_markers else None
		self.stats_pub = self.create_publisher(String, '/semantic_octomap_stats', 10) if self.publish_stats else None
		self.cloud_pub = self.create_publisher(PointCloud2, '/semantic_octomap_colored_cloud', 10) if self.publish_colored_cloud else None
		self.semantic_only_pub = self.create_publisher(PointCloud2, '/semantic_voxels_only', 10) if self.publish_colored_cloud else None

		self.get_logger().info(
			f"SemanticDepthOctoMapNode initialized:\n"
			f"  - Voxel resolution: {self.voxel_resolution}m\n"
			f"  - Voxel mapping: {'ENABLED' if self.enable_voxel_mapping else 'DISABLED'}\n"
			f"  - Semantic mapping: {'ENABLED' if self.enable_semantic_mapping else 'DISABLED'}\n"
			f"  - Bridge queue: max_size={self.max_queue_size}, process_interval={self.queue_processing_interval}s\n"
			f"  - Sync buffers: duration={self.sync_buffer_duration}s\n"
			f"  - Voxel helper ready: {hasattr(self, 'voxel_helper') and self.voxel_helper is not None}\n"
			f"  - Semantic mapper ready: {hasattr(self.voxel_helper, 'semantic_mapper') and self.voxel_helper.semantic_mapper is not None}\n"
			f"  - Topics: depth={self.depth_topic}, pose={self.pose_topic}"
		)

		# Timer for inactivity detection (non-intrusive)
		self.inactivity_timer = self.create_timer(0.5, self._check_inactivity_timer_cb)

	def camera_info_callback(self, msg: CameraInfo):
		if self.camera_intrinsics is None:
			self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
			self.get_logger().info(f"Camera intrinsics set: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")
		# Update activity
		self.last_data_time = time.time()

	def pose_callback(self, msg: PoseStamped):
		self.latest_pose = msg
		# Push into pose buffer with timestamp
		try:
			pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
			with self.sync_lock:
				self.pose_buffer.append((pose_time, msg))
				self._prune_sync_buffers()
		except Exception:
			pass
		# Update activity
		self.last_data_time = time.time()

	def semantic_hotspot_callback(self, msg: String):
		"""Queue incoming semantic hotspot messages for batch processing."""
		try:
			if not self.enable_semantic_mapping or not self.enable_voxel_mapping:
				return
			
			# Add message to queue with timestamp
			with self.bridge_queue_lock:
				# Prevent queue from growing too large
				if len(self.bridge_message_queue) >= self.max_queue_size:
					# Remove oldest message
					self.bridge_message_queue.pop(0)
					self.get_logger().warn(f"Bridge message queue full, dropped oldest message")
				
				# Add new message with timestamp
				self.bridge_message_queue.append({
					'msg_data': msg.data,
					'received_time': time.time()
				})
				
				queue_size = len(self.bridge_message_queue)
			
			self.get_logger().info(f"Queued hotspot message (queue size: {queue_size})")
			# Update activity
			self.last_data_time = time.time()
			
		except Exception as e:
			self.get_logger().error(f"Error queuing semantic hotspot message: {e}")
			import traceback
			traceback.print_exc()
	
	def _process_bridge_message_queue(self):
		"""Process all queued bridge messages in batch."""
		try:
			# Get all messages from queue
			messages_to_process = []
			with self.bridge_queue_lock:
				if not self.bridge_message_queue:
					return
				
				# Copy all messages and clear the queue
				messages_to_process = self.bridge_message_queue.copy()
				self.bridge_message_queue.clear()
			
			if not messages_to_process:
				return
			
			self.get_logger().info(f"Processing {len(messages_to_process)} queued bridge messages")
			
			# Process each message
			processed_count = 0
			for msg_info in messages_to_process:
				try:
					success = self._process_single_bridge_message(msg_info['msg_data'])
					if success:
						processed_count += 1
				except Exception as e:
					self.get_logger().warn(f"Failed to process bridge message: {e}")
			
			self.get_logger().info(f"Successfully processed {processed_count}/{len(messages_to_process)} bridge messages")
			
		except Exception as e:
			self.get_logger().error(f"Error processing bridge message queue: {e}")
			import traceback
			traceback.print_exc()
	
	def _process_single_bridge_message(self, msg_data: str) -> bool:
		"""Process a single bridge message and apply to voxel map by timestamp lookup."""
		try:
			# Parse the JSON message
			data = json.loads(msg_data)
			
			if data.get('type') == 'merged_similarity_hotspots':
				return self._process_merged_hotspot_message(data)
			elif data.get('type') == 'similarity_hotspots':
				return self._process_single_hotspot_message(data)
			else:
				return False
			
		except Exception as e:
			self.get_logger().error(f"Error processing single bridge message: {e}")
			return False

	def _process_merged_hotspot_message(self, data: dict) -> bool:
		"""Process merged hotspot message with color-based VLM answer association."""
		try:
			mask_shape = data.get('mask_shape')
			mask_data = data.get('mask_data')
			vlm_info = data.get('vlm_info', {})
			rgb_timestamp = float(data.get('timestamp', 0.0))
			
			if not mask_data or not mask_shape or rgb_timestamp <= 0.0:
				self.get_logger().warn(f"Incomplete merged hotspot data")
				return False
			
			# Reconstruct colored mask
			merged_mask = np.array(mask_data, dtype=np.uint8).reshape(tuple(mask_shape))
			
			# Lookup closest depth frame and pose by timestamp
			depth_image, pose_msg, used_ts = self._lookup_depth_and_pose(rgb_timestamp)
			if depth_image is None or pose_msg is None:
				self.get_logger().warn(f"No matching depth/pose found for timestamp {rgb_timestamp:.6f}")
				return False
			
			# Process each VLM answer based on color
			processed_count = 0
			for vlm_answer, info in vlm_info.items():
				color = info.get('color', [0, 0, 0])
				
				# Create binary mask for this VLM answer based on color
				vlm_mask = np.all(merged_mask == color, axis=2)
				
				if np.any(vlm_mask):
					success = self._process_hotspot_with_depth(
						vlm_mask, pose_msg, depth_image, vlm_answer, 
						info.get('hotspot_threshold', 0.6), 
						{'hotspot_pixels': info.get('hotspot_pixels', 0)}, 
						rgb_timestamp, used_ts
					)
					if success:
						processed_count += 1
			
			self.get_logger().info(f"Processed {processed_count}/{len(vlm_info)} VLM answers from merged hotspots")
			return processed_count > 0
			
		except Exception as e:
			self.get_logger().error(f"Error processing merged hotspot message: {e}")
			return False

	def _process_single_hotspot_message(self, data: dict) -> bool:
		"""Process single hotspot message (legacy support)."""
		try:
			vlm_answer = data.get('vlm_answer')
			mask_shape = data.get('mask_shape')
			mask_data = data.get('mask_data')
			threshold_used = float(data.get('threshold_used', 0.6))
			stats = data.get('stats', {})
			rgb_timestamp = float(data.get('timestamp', 0.0))
			
			if not vlm_answer or not mask_data or not mask_shape or rgb_timestamp <= 0.0:
				self.get_logger().warn(f"Incomplete hotspot data: vlm_answer={vlm_answer}, mask_shape={mask_shape}, ts={rgb_timestamp}")
				return False
			
			# Reconstruct binary mask
			mask = np.array(mask_data, dtype=np.uint8).reshape(tuple(mask_shape))
			
			# Lookup closest depth frame and pose by timestamp
			depth_image, pose_msg, used_ts = self._lookup_depth_and_pose(rgb_timestamp)
			if depth_image is None or pose_msg is None:
				self.get_logger().warn(f"No matching depth/pose found for timestamp {rgb_timestamp:.6f}")
				return False
			
			# Process hotspot mask with retrieved depth and pose
			return self._process_hotspot_with_depth(mask, pose_msg, depth_image, vlm_answer, threshold_used, stats, rgb_timestamp, used_ts)
			
		except Exception as e:
			self.get_logger().error(f"Error processing single hotspot message: {e}")
			return False

	def _lookup_depth_and_pose(self, target_ts: float):
		"""Find closest depth frame and pose to target timestamp within buffer window."""
		with self.sync_lock:
			best_depth = None
			best_pose = None
			best_depth_dt = float('inf')
			best_pose_dt = float('inf')
			best_depth_ts = None
			best_pose_ts = None
			
			# Find closest depth
			for ts, depth in self.depth_buffer:
				dt = abs(ts - target_ts)
				if dt < best_depth_dt and dt <= self.sync_buffer_duration:
					best_depth_dt = dt
					best_depth = depth
					best_depth_ts = ts
			
			# Find closest pose
			for ts, pose in self.pose_buffer:
				dt = abs(ts - target_ts)
				if dt < best_pose_dt and dt <= self.sync_buffer_duration:
					best_pose_dt = dt
					best_pose = pose
					best_pose_ts = ts
			
			# Return if both found
			if best_depth is not None and best_pose is not None:
				return best_depth, best_pose, (best_depth_ts, best_pose_ts)
			
			return None, None, (None, None)

	def _process_hotspot_with_depth(self, mask: np.ndarray, pose: PoseStamped, depth_m: np.ndarray,
								   vlm_answer: str, threshold: float, stats: dict, rgb_ts: float, used_ts: tuple) -> bool:
		"""Project hotspot mask using matched depth and pose; update voxel map and semantics."""
		try:
			if self.camera_intrinsics is None:
				self.get_logger().warn("No camera intrinsics available for hotspot processing")
				return False
			
			# Get hotspot pixel coordinates
			hotspot_coords = np.where(mask > 0)
			if len(hotspot_coords[0]) == 0:
				self.get_logger().warn("No hotspot pixels found in mask")
				return False
			
			# Build a depth image using only hotspot pixels (others zero)
			h, w = mask.shape
			depth_hot = np.zeros((h, w), dtype=np.float32)
			# Ensure depth_m shape matches mask; if not, resize
			if depth_m.shape != (h, w):
				depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
				depth_hot[hotspot_coords] = depth_resized[hotspot_coords]
			else:
				depth_hot[hotspot_coords] = depth_m[hotspot_coords]
			
			# Convert depth hotspots to world points
			points_world, u_indices, v_indices = self._depth_to_world_points(depth_hot, self.camera_intrinsics, pose)
			if points_world is None or len(points_world) == 0:
				self.get_logger().warn("Failed to project hotspot points to world coordinates")
				return False
			
			# Range filter
			origin = self._pose_position(pose)
			dist = np.linalg.norm(points_world - origin, axis=1)
			mask_range = (dist >= float(self.min_range)) & (dist <= float(self.max_range))
			points_world = points_world[mask_range]
			
			if points_world.size == 0:
				self.get_logger().warn("All hotspot points filtered out by range constraints")
				return False
			
			# Update voxel map with hotspot points
			self.voxel_helper.update_map(points_world, origin, hit_confidences=None, voxel_embeddings=None)
			
			# Apply semantic labels to voxels
			self._apply_semantic_labels_to_voxels(points_world, vlm_answer, threshold, stats)
			
			self.get_logger().info(
				f"Applied {len(points_world)} hotspot points to voxel map for '{vlm_answer}' "
				f"(rgb_ts={rgb_ts:.6f}, depth_ts={used_ts[0]}, pose_ts={used_ts[1]})"
			)
			return True
			
		except Exception as e:
			self.get_logger().error(f"Error processing hotspot with depth: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def _apply_semantic_labels_to_voxels(self, points_world: np.ndarray, vlm_answer: str,
										 threshold: float, stats: dict):
		"""Apply semantic labels to voxels based on hotspot points."""
		try:
			if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
				self.get_logger().warn("Semantic mapper not available")
				return
			
			semantic_mapper = self.voxel_helper.semantic_mapper
			
			# Get voxel keys for the world points
			voxel_keys = set()
			for point in points_world:
				if hasattr(self.voxel_helper, 'get_voxel_key_from_point'):
					voxel_key = self.voxel_helper.get_voxel_key_from_point(point)
					voxel_keys.add(voxel_key)
			
			# Mark voxels as semantic with high confidence since they passed binary threshold
			similarity_score = stats.get('avg_similarity', threshold + 0.1)
			
			with semantic_mapper.voxel_semantics_lock:
				for voxel_key in voxel_keys:
					semantic_info = {
						'vlm_answer': vlm_answer,
						'similarity': similarity_score,
						'threshold_used': threshold,
						'detection_method': 'binary_threshold_hotspot',
						'depth_used': True,
						'timestamp': time.time()
					}
					semantic_mapper.voxel_semantics[voxel_key] = semantic_info
			
			self.get_logger().info(f"✓ Applied semantic labels to {len(voxel_keys)} voxels for '{vlm_answer}' - these will appear RED in the colored cloud")
			
		except Exception as e:
			self.get_logger().error(f"Error applying semantic labels to voxels: {e}")
	
	def depth_callback(self, msg: Image):
		if self.camera_intrinsics is None:
			self.get_logger().warn("No camera intrinsics received yet")
			return

		# Convert and store depth with timestamp in meters
		try:
			depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
			depth_m = self._depth_to_meters(depth, msg.encoding)
			if depth_m is None:
				return
			
			depth_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
			with self.sync_lock:
				self.depth_buffer.append((depth_time, depth_m))
				self._prune_sync_buffers()
			
			# Regular mapping: project full depth using latest pose (when available)
			if self.latest_pose is not None:
				points_world, u_indices, v_indices = self._depth_to_world_points(depth_m, self.camera_intrinsics, self.latest_pose)
				if points_world is not None and len(points_world) > 0:
					origin = self._pose_position(self.latest_pose)
					dist = np.linalg.norm(points_world - origin, axis=1)
					mask = (dist >= float(self.min_range)) & (dist <= float(self.max_range))
					points_world = points_world[mask]
					if points_world.size > 0:
						self.voxel_helper.update_map(points_world, origin, hit_confidences=None, voxel_embeddings=None)
			
		except Exception as e:
			self.get_logger().error(f"Error storing depth frame: {e}")

		# Activity update
		self.last_data_time = time.time()

		# Process queue periodically
		current_time = time.time()
		if (current_time - self.last_queue_process_time) >= self.queue_processing_interval:
			self._process_bridge_message_queue()
			self.last_queue_process_time = current_time

		# Periodic publishing
		self._periodic_publishing()

	def _prune_sync_buffers(self):
		"""Keep only recent entries within sync window."""
		cutoff = time.time() - self.sync_buffer_duration
		# Depth buffer is keyed by message stamps; we cannot use wall time; simply limit by length and by age heuristics
		max_entries = 200
		if len(self.depth_buffer) > max_entries:
			self.depth_buffer = self.depth_buffer[-max_entries:]
		if len(self.pose_buffer) > max_entries:
			self.pose_buffer = self.pose_buffer[-max_entries:]

	def _depth_to_meters(self, depth, encoding: str):
		try:
			enc = (encoding or '').lower()
			if '16uc1' in enc or 'mono16' in enc:
				return depth.astype(np.float32) / 1000.0
			elif '32fc1' in enc or 'float32' in enc:
				return depth.astype(np.float32)
			else:
				return depth.astype(np.float32) / 1000.0
		except Exception:
			return None

	def _depth_to_world_points(self, depth_m: np.ndarray, intrinsics, pose: PoseStamped):
		try:
			fx, fy, cx, cy = intrinsics
			h, w = depth_m.shape
			u, v = np.meshgrid(np.arange(w), np.arange(h))
			z = depth_m
			valid = np.isfinite(z) & (z > 0.0)
			if not np.any(valid):
				return None, None, None

			u, v, z = u[valid], v[valid], z[valid]
			x = (u - cx) * z / fx
			y = (v - cy) * z / fy
			pts_cam = np.stack([x, y, z], axis=1)

			# Transform to base if needed
			if bool(self.pose_is_base_link):
				pts_cam = pts_cam @ (self.R_opt_to_base.T if bool(self.apply_optical_frame_rotation) else np.eye(3, dtype=np.float32))
				pts_cam = pts_cam @ self.R_cam_to_base_extra.T + self.t_cam_to_base_extra

			# World transform
			R_world = self._quat_to_rot(self._pose_quat(pose))
			p_world = self._pose_position(pose)
			pts_world = pts_cam @ R_world.T + p_world
			return pts_world, u, v
		except Exception:
			return None, None, None

	def _create_semantic_colored_cloud(self, max_points: int) -> Optional[PointCloud2]:
		"""Create a colored point cloud that shows semantic voxels with VLM answer colors."""
		try:
			# Get all voxels from the helper using the correct attribute
			all_voxels = list(self.voxel_helper.voxels.keys())
			if not all_voxels:
				return None
			
			# Limit the number of points
			if len(all_voxels) > max_points:
				all_voxels = all_voxels[:max_points]
			
			# Create point cloud data
			points = []
			colors = []
			semantic_count = 0
			regular_count = 0
			
			# Check if semantic mapper is available
			has_semantic_mapper = (hasattr(self.voxel_helper, 'semantic_mapper') and 
						  self.voxel_helper.semantic_mapper is not None)
			
			if has_semantic_mapper:
				semantic_mapper = self.voxel_helper.semantic_mapper
				with semantic_mapper.voxel_semantics_lock:
					for voxel_key in all_voxels:
						# Get voxel center position using the correct method
						voxel_center = self._get_voxel_center_from_key(voxel_key)
						if voxel_center is None:
							continue
						
						# Check if this voxel has semantic information
						if voxel_key in semantic_mapper.voxel_semantics:
							# Get VLM answer and use consistent color
							semantic_info = semantic_mapper.voxel_semantics[voxel_key]
							vlm_answer = semantic_info.get('vlm_answer', 'unknown')
							
							# Use consistent color based on VLM answer
							color = self._get_vlm_answer_color(vlm_answer)
							semantic_count += 1
						else:
							# Regular voxel - use default color (gray)
							color = [128, 128, 128]  # Gray for regular voxels
							regular_count += 1
						
						points.append(voxel_center)
						colors.append(color)
			else:
				# No semantic mapper - just create regular colored voxels
				for voxel_key in all_voxels:
					# Get voxel center position using the correct method
					voxel_center = self._get_voxel_center_from_key(voxel_key)
					if voxel_center is None:
						continue
					
					# Regular voxel - use default color (gray)
					color = [128, 128, 128]  # Gray for all voxels
					regular_count += 1
					
					points.append(voxel_center)
					colors.append(color)
			
			if not points:
				return None
			
			# Log the coloring information
			if has_semantic_mapper and semantic_count > 0:
				self.get_logger().info(f"Creating colored cloud: {semantic_count} semantic voxels (colored by VLM answer), {regular_count} regular voxels (GRAY)")
			else:
				self.get_logger().info(f"Creating colored cloud: {regular_count} regular voxels (GRAY)")
			
			# Convert to numpy arrays
			points_array = np.array(points, dtype=np.float32)
			colors_array = np.array(colors, dtype=np.uint8)
			
			# Create PointCloud2 message with proper structure
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.map_frame
			
			# Create structured array with XYZ + RGB
			cloud_data_combined = np.empty(len(points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), 
				('rgb', np.uint32)
			])
			
			# Fill in the data
			cloud_data_combined['x'] = points_array[:, 0]
			cloud_data_combined['y'] = points_array[:, 1]
			cloud_data_combined['z'] = points_array[:, 2]
			
			# Pack RGB values as UINT32 (standard for PointCloud2 RGB)
			rgb_packed = np.zeros(len(colors_array), dtype=np.uint32)
			for i, c in enumerate(colors_array):
				rgb_packed[i] = (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])
			cloud_data_combined['rgb'] = rgb_packed
			
			# Create PointCloud2 message with proper fields from the start
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields properly - use UINT32 for rgb to ensure RViz compatibility
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, rgb)
			cloud_msg.width = len(points)  # Set correct width
			cloud_msg.height = 1  # Set height to 1 for organized point cloud
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data_combined.tobytes()
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating semantic colored cloud: {e}")
			import traceback
			traceback.print_exc()
			return None

	def _get_vlm_answer_color(self, vlm_answer: str) -> List[int]:
		"""Get consistent color for VLM answer (same as bridge)."""
		# Use same color palette as semantic bridge
		color_palette = [
			[255, 0, 0],    # Red
			[0, 255, 0],    # Green
			[0, 0, 255],    # Blue
			[255, 255, 0],  # Yellow
			[255, 0, 255],  # Magenta
			[0, 255, 255],  # Cyan
			[255, 128, 0],  # Orange
			[128, 0, 255],  # Purple
			[128, 128, 0],  # Olive
			[0, 128, 128],  # Teal
			[128, 0, 128],  # Maroon
			[255, 165, 0],  # Orange Red
			[75, 0, 130],   # Indigo
			[240, 230, 140], # Khaki
			[255, 20, 147]  # Deep Pink
		]
		
		# Simple hash-based color assignment
		hash_val = hash(vlm_answer) % len(color_palette)
		return color_palette[hash_val]

	def _get_voxel_center_from_key(self, voxel_key: tuple) -> Optional[np.ndarray]:
		"""Get voxel center position from voxel key."""
		try:
			vx, vy, vz = voxel_key
			
			# Convert voxel coordinates to world coordinates
			world_x = vx * self.voxel_resolution
			world_y = vy * self.voxel_resolution
			world_z = vz * self.voxel_resolution
			
			return np.array([world_x, world_y, world_z], dtype=np.float32)
			
		except Exception as e:
			self.get_logger().warn(f"Error getting voxel center for key {voxel_key}: {e}")
			return None

	def _create_semantic_only_cloud(self) -> Optional[PointCloud2]:
		"""Create a point cloud containing all accumulated semantic voxels (red)."""
		try:
			if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
				return None
			
			semantic_mapper = self.voxel_helper.semantic_mapper
			
			# Get all accumulated semantic voxels
			with semantic_mapper.voxel_semantics_lock:
				semantic_voxel_keys = list(semantic_mapper.voxel_semantics.keys())
			
			if not semantic_voxel_keys:
				return None
			
			# Create point cloud data for all accumulated semantic voxels
			points = []
			for voxel_key in semantic_voxel_keys:
				voxel_center = self._get_voxel_center_from_key(voxel_key)
				if voxel_center is not None:
					points.append(voxel_center)
			
			if not points:
				return None
			
			# Convert to numpy array
			points_array = np.array(points, dtype=np.float32)
			
			# Create PointCloud2 message with XYZ only (no RGB needed for semantic-only)
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.map_frame
			
			# Create structured array with just XYZ
			cloud_data = np.empty(len(points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32)
			])
			
			# Fill in the data
			cloud_data['x'] = points_array[:, 0]
			cloud_data['y'] = points_array[:, 1]
			cloud_data['z'] = points_array[:, 2]
			
			# Create PointCloud2 message
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields (XYZ only)
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 12  # 4 bytes per float * 3 fields (x, y, z)
			cloud_msg.width = len(points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data.tobytes()
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating semantic-only cloud: {e}")
			return None
	
	def _periodic_publishing(self):
		now = time.time()
		
		# Process bridge message queue regularly
		if (now - self.last_queue_process_time) >= self.queue_processing_interval:
			self._process_bridge_message_queue()
			self.last_queue_process_time = now
		
		if self.marker_pub and (now - self.last_marker_pub) >= float(self.marker_publish_rate):
			markers = self.voxel_helper.create_markers(int(self.max_markers), bool(self.use_cube_list_markers))
			
			# Fix timestamps for all markers
			current_time = self.get_clock().now().to_msg()
			for marker in markers.markers:
				marker.header.stamp = current_time
			
			self.marker_pub.publish(markers)
			marker_count = len(markers.markers) if hasattr(markers, 'markers') else 0
			self.get_logger().info(f"Published {marker_count} voxel markers")
			self.last_marker_pub = now
		
		if self.cloud_pub:
			try:
				# Try to create semantic-aware colored cloud first
				semantic_cloud = self._create_semantic_colored_cloud(int(self.max_markers))
				if semantic_cloud:
					self.cloud_pub.publish(semantic_cloud)
					self.get_logger().debug(f"Published semantic colored cloud with {len(semantic_cloud.data)//16} points")
				else:
					# Fallback to regular colored cloud
					self.get_logger().debug("Semantic cloud creation returned None, falling back to regular colored cloud")
					cloud = self.voxel_helper.create_colored_cloud(int(self.max_markers))
					if cloud:
						self.cloud_pub.publish(cloud)
						self.get_logger().debug("Published regular colored cloud")
					else:
						self.get_logger().warn("Both semantic and regular colored cloud creation returned None")
			except Exception as e:
				self.get_logger().warn(f"Failed to create semantic colored cloud: {e}")
				# Try fallback to regular colored cloud even if semantic failed
				try:
					cloud = self.voxel_helper.create_colored_cloud(int(self.max_markers))
					if cloud:
						self.cloud_pub.publish(cloud)
						self.get_logger().debug("Published regular colored cloud as fallback")
				except Exception as fallback_e:
					self.get_logger().error(f"Both semantic and regular colored cloud creation failed: {fallback_e}")
		
		# Publish semantic-only point cloud (XYZ only, no RGB)
		if self.semantic_only_pub:
			try:
				semantic_only_cloud = self._create_semantic_only_cloud()
				if semantic_only_cloud:
					self.semantic_only_pub.publish(semantic_only_cloud)
					self.get_logger().debug(f"Published semantic-only cloud with {len(semantic_only_cloud.data)//12} points")
			except Exception as e:
				self.get_logger().warn(f"Failed to create semantic-only cloud: {e}")
		
		if self.stats_pub and (now - self.last_stats_pub) >= float(self.stats_publish_rate):
			# Get statistics from voxel helper
			stats = self.voxel_helper.get_statistics()
			
			# Add semantic mapping status and counts
			semantic_voxel_count = 0
			queue_size = 0
			if hasattr(self.voxel_helper, 'semantic_mapper') and self.voxel_helper.semantic_mapper:
				with self.voxel_helper.semantic_mapper.voxel_semantics_lock:
					semantic_voxel_count = len(self.voxel_helper.semantic_mapper.voxel_semantics)
			
			with self.bridge_queue_lock:
				queue_size = len(self.bridge_message_queue)
			
			stats['semantic_mapping'] = {
				'enabled': self.enable_semantic_mapping,
				'status': 'active' if self.enable_semantic_mapping else 'disabled',
				'semantic_voxel_count': semantic_voxel_count,
				'total_voxel_count': stats.get('total_voxels', 0),
				'bridge_queue_size': queue_size,
				'queue_max_size': self.max_queue_size
			}
			
			self.stats_pub.publish(String(data=json.dumps(stats)))
			self.last_stats_pub = now

	def _pose_position(self, pose: PoseStamped):
		return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)

	def _pose_quat(self, pose: PoseStamped):
		q = pose.pose.orientation
		return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

	def _quat_to_rot(self, q: np.ndarray):
		x, y, z, w = q
		n = x*x + y*y + z*z + w*w
		if n < 1e-8:
			return np.eye(3, dtype=np.float32)
		s = 2.0 / n
		xx, yy, zz = x*x*s, y*y*s, z*z*s
		xy, xz, yz = x*y*s, x*z*s, y*z*s
		wx, wy, wz = w*x*s, w*y*s, w*z*s
		return np.array([
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)]
		], dtype=np.float32)

	def _rpy_deg_to_rot(self, rpy_deg):
		try:
			roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
			cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
			Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
			Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
			Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
			return Rz @ Ry @ Rx
		except Exception:
			return np.eye(3, dtype=np.float32)

	# ==== Inactivity detection and PCD export ====
	def _check_inactivity_timer_cb(self):
		try:
			now = time.time()
			inactivity = now - self.last_data_time
			threshold = float(self.inactivity_threshold_seconds)
			# Export only once per node lifetime
			if (not self.semantic_pcd_exported) and inactivity >= threshold:
				export_dir = self._resolve_export_directory()
				os.makedirs(export_dir, exist_ok=True)
				ts_str = time.strftime('%Y%m%d_%H%M%S')
				filename = f"semantic_voxels_{ts_str}.pcd"
				filepath = os.path.join(export_dir, filename)
				count = self._save_semantic_voxels_to_pcd(filepath)
				if count > 0:
					self.get_logger().info(f"✓ Exported {count} semantic voxel points to PCD: {filepath}")
				else:
					self.get_logger().warn(f"No semantic voxels to export (skipped writing): {filepath}")
				
				self.semantic_pcd_exported = True
		except Exception as e:
			self.get_logger().error(f"Inactivity check error: {e}")

	def _save_semantic_voxels_to_pcd(self, filepath: str) -> int:
		"""Write accumulated semantic voxels as PCD using Open3D (XYZ)."""
		try:
			if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
				return 0
			semantic_mapper = self.voxel_helper.semantic_mapper
			with semantic_mapper.voxel_semantics_lock:
				voxel_keys = list(semantic_mapper.voxel_semantics.keys())
			if not voxel_keys:
				return 0
			points = []
			for key in voxel_keys:
				center = self._get_voxel_center_from_key(key)
				if center is not None:
					points.append(center)
			if not points:
				return 0
			pts = np.array(points, dtype=np.float32)
			try:
				import open3d as o3d
				pcd = o3d.geometry.PointCloud()
				pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
				ok = o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)
				return int(pts.shape[0] if ok else 0)
			except Exception:
				# Fallback to simple ASCII PCD header if Open3D not available
				with open(filepath, 'w') as f:
					f.write("# .PCD v0.7 - Point Cloud Data file\n")
					f.write("VERSION 0.7\n")
					f.write("FIELDS x y z\n")
					f.write("SIZE 4 4 4\n")
					f.write("TYPE F F F\n")
					f.write("COUNT 1 1 1\n")
					f.write(f"WIDTH {pts.shape[0]}\n")
					f.write("HEIGHT 1\n")
					f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
					f.write(f"POINTS {pts.shape[0]}\n")
					f.write("DATA ascii\n")
					for p in pts:
						f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
				return pts.shape[0]
		except Exception as e:
			self.get_logger().error(f"Failed to save semantic voxels to PCD: {e}")
			return 0

	def _resolve_export_directory(self) -> str:
		"""Return latest run_* dir under buffers if available; else configured directory."""
		try:
			base_dir = str(self.semantic_export_directory or '').strip()
			root = base_dir
			if base_dir.endswith('/buffers') and os.path.isdir(base_dir):
				candidates = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('run_')]
				if candidates:
					candidates.sort(key=lambda p: os.path.getmtime(p))
					return candidates[-1]
			return root
		except Exception:
			return str(self.semantic_export_directory)



def main():
	rclpy.init()
	node = SemanticDepthOctoMapNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main() 
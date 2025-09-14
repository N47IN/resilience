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

# Optional GP helper
try:
	from resilience.voxel_gp_helper import DisturbanceFieldHelper
	GP_HELPER_AVAILABLE = True
except ImportError:
	GP_HELPER_AVAILABLE = False

# Simple GP Manager - no complex superposition needed

# Optional PathManager for global path access
try:
	from resilience.path_manager import PathManager
	PATH_MANAGER_AVAILABLE = True
except ImportError:
	PATH_MANAGER_AVAILABLE = False


class SemanticDepthOctoMapNode(Node):
	"""Simplified semantic depth octomap node using helpers."""

	def __init__(self):
		super().__init__('semantic_depth_octomap_node')

		# Professional startup message
		self.get_logger().info("=" * 60)
		self.get_logger().info("SEMANTIC OCTOMAP SYSTEM INITIALIZING")
		self.get_logger().info("=" * 60)

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
			('semantic_export_directory', '/home/navin/ros2_ws/src/buffers'),
			('mapping_config_path', ''),
			('nominal_path', '/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json'),
			('main_config_path', '')
		])

		params = self.get_parameters([
			'depth_topic', 'camera_info_topic', 'pose_topic',
			'map_frame', 'voxel_resolution', 'max_range', 'min_range', 'probability_hit',
			'probability_miss', 'occupancy_threshold', 'publish_markers', 'publish_stats',
			'publish_colored_cloud', 'use_cube_list_markers', 'max_markers', 'marker_publish_rate', 'stats_publish_rate',
			'pose_is_base_link', 'apply_optical_frame_rotation', 'cam_to_base_rpy_deg', 'cam_to_base_xyz', 'embedding_dim',
			'enable_semantic_mapping', 'semantic_similarity_threshold', 'buffers_directory', 'bridge_queue_max_size', 'bridge_queue_process_interval',
			'enable_voxel_mapping', 'sync_buffer_seconds', 'inactivity_threshold_seconds', 'semantic_export_directory', 'mapping_config_path', 'nominal_path', 'main_config_path'
		])

		# Extract parameter values
		(self.depth_topic, self.camera_info_topic, self.pose_topic,
		 self.map_frame, self.voxel_resolution, self.max_range, self.min_range, self.prob_hit,
		 self.prob_miss, self.occ_thresh, self.publish_markers, self.publish_stats, self.publish_colored_cloud,
		 self.use_cube_list_markers, self.max_markers, self.marker_publish_rate, self.stats_publish_rate,
		 self.pose_is_base_link, self.apply_optical_frame_rotation, self.cam_to_base_rpy_deg, self.cam_to_base_xyz,
		 self.embedding_dim, self.enable_semantic_mapping, self.semantic_similarity_threshold,
		 self.buffers_directory, self.bridge_queue_max_size, self.bridge_queue_process_interval,
		 self.enable_voxel_mapping, self.sync_buffer_seconds, self.inactivity_threshold_seconds,
		 self.semantic_export_directory, self.mapping_config_path, self.nominal_path, self.main_config_path) = [p.value for p in params]

		# Read nominal path separately (optional for GP)
		self.nominal_path = self.get_parameter('nominal_path').value
		self.main_config_path = self.get_parameter('main_config_path').value

		# Load topic configuration from mapping config
		self.load_topic_configuration()

	def load_topic_configuration(self):
		"""Load topic configuration from mapping config file."""
		try:
			import yaml
			if self.mapping_config_path:
				config_path = self.mapping_config_path
			else:
				# Use default config path
				from ament_index_python.packages import get_package_share_directory
				package_dir = get_package_share_directory('resilience')
				config_path = os.path.join(package_dir, 'config', 'mapping_config.yaml')
			
			with open(config_path, 'r') as f:
				config = yaml.safe_load(f)
			
			# Extract topic configuration
			topics = config.get('topics', {})
			
			# Input topics
			self.depth_topic = topics.get('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered')
			self.camera_info_topic = topics.get('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info')
			self.pose_topic = topics.get('pose_topic', '/robot_1/sensors/front_stereo/pose')
			self.semantic_hotspots_topic = topics.get('semantic_hotspots_topic', '/semantic_hotspots')
			self.semantic_hotspot_mask_topic = topics.get('semantic_hotspot_mask_topic', '/semantic_hotspot_mask')
			
			# Output topics
			self.semantic_octomap_markers_topic = topics.get('semantic_octomap_markers_topic', '/semantic_octomap_markers')
			self.semantic_octomap_stats_topic = topics.get('semantic_octomap_stats_topic', '/semantic_octomap_stats')
			self.semantic_octomap_colored_cloud_topic = topics.get('semantic_octomap_colored_cloud_topic', '/semantic_octomap_colored_cloud')
			self.semantic_voxels_only_topic = topics.get('semantic_voxels_only_topic', '/semantic_voxels_only')
			
			self.get_logger().info(f"Topic configuration loaded from: {config_path}")
			
		except Exception as e:
			self.get_logger().warn(f"Using default topic configuration: {e}")
			# Fallback to default topics
			self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
			self.camera_info_topic = '/robot_1/sensors/front_stereo/left/camera_info'
			self.pose_topic = '/robot_1/sensors/front_stereo/pose'
			self.semantic_hotspots_topic = '/semantic_hotspots'
			self.semantic_hotspot_mask_topic = '/semantic_hotspot_mask'
			self.semantic_octomap_markers_topic = '/semantic_octomap_markers'
			self.semantic_octomap_stats_topic = '/semantic_octomap_stats'
			self.semantic_octomap_colored_cloud_topic = '/semantic_octomap_colored_cloud'
			self.semantic_voxels_only_topic = '/semantic_voxels_only'
			self.get_logger().info("Using default topic configuration")

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
		self.mask_buffer = []   # list of tuples (timestamp_float, merged_mask_rgb_uint8)
		self.sync_buffer_duration = float(self.sync_buffer_seconds)
		self.sync_lock = threading.Lock()

		# GP fitting state
		self.gp_fit_lock = threading.Lock()
		self.gp_fitting_active = False

		# Simple GP state - use same parameters for all semantic voxels
		self.global_gp_params = None  # Will store the latest GP parameters
		self.last_gp_update_time = 0.0
		self.gp_update_interval = 1.0  # Update every 1 second
		
		# GP computation thread
		self.gp_computation_thread = None
		self.gp_thread_lock = threading.Lock()
		self.gp_thread_running = False
		
		# Adaptive radius parameters
		self.min_radius = 0.5  # Minimum radius around voxels
		self.max_radius = 2.0  # Maximum radius around voxels
		self.base_radius = 1.0  # Base radius for computation

		# Optional: initialize PathManager for accessing global path (non-blocking)
		self.path_manager = None
		if PATH_MANAGER_AVAILABLE:
			try:
				# Load path_mode config from main config
				path_config = None
				if isinstance(self.main_config_path, str) and len(self.main_config_path) > 0:
					import yaml
					with open(self.main_config_path, 'r') as f:
						cfg = yaml.safe_load(f)
					path_config = cfg.get('path_mode', {}) if isinstance(cfg, dict) else {}
				else:
					try:
						from ament_index_python.packages import get_package_share_directory
						package_dir = get_package_share_directory('resilience')
						default_main = os.path.join(package_dir, 'config', 'main_config.yaml')
						import yaml
						with open(default_main, 'r') as f:
							cfg = yaml.safe_load(f)
						path_config = cfg.get('path_mode', {}) if isinstance(cfg, dict) else {}
					except Exception:
						path_config = {}
				self.path_manager = PathManager(self, path_config)
				self.get_logger().info("PathManager initialized for nominal path access (non-blocking)")
				# Announce nominal source preference at startup
				if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0:
					self.get_logger().info(f"Nominal source preference: GLOBAL PATH via PathManager when available; fallback FILE: {self.nominal_path}")
				else:
					self.get_logger().info("Nominal source preference: GLOBAL PATH via PathManager; no file fallback provided")
			except Exception as e:
				self.get_logger().warn(f"Failed to initialize PathManager: {e}")
				if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0:
					self.get_logger().info(f"Nominal source: FILE only: {self.nominal_path}")
				else:
					self.get_logger().warn("Nominal source: NONE available (no PathManager, no file)")
		else:
			# No PathManager available
			if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0:
				self.get_logger().info(f"Nominal source: FILE only: {self.nominal_path}")
			else:
				self.get_logger().warn("Nominal source: NONE available (PathManager not available, no file path)")

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

		# Simple GP visualization - no complex superposition needed
		self.get_logger().info("Simple GP visualization system initialized")
		
		# Start GP computation thread
		self._start_gp_computation_thread()

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
		# Subscribe to semantic hotspots and mask image
		if self.enable_semantic_mapping and self.enable_voxel_mapping:
			self.create_subscription(String, self.semantic_hotspots_topic, self.semantic_hotspot_callback, 10)
			self.create_subscription(Image, self.semantic_hotspot_mask_topic, self.semantic_hotspot_mask_callback, 10)

		# Publishers
		self.marker_pub = self.create_publisher(MarkerArray, self.semantic_octomap_markers_topic, 10) if self.publish_markers else None
		self.stats_pub = self.create_publisher(String, self.semantic_octomap_stats_topic, 10) if self.publish_stats else None
		self.cloud_pub = self.create_publisher(PointCloud2, self.semantic_octomap_colored_cloud_topic, 10) if self.publish_colored_cloud else None
		self.semantic_only_pub = self.create_publisher(PointCloud2, self.semantic_voxels_only_topic, 10) if self.publish_colored_cloud else None
		# GP visualization and costmap publishers
		self.gp_visualization_pub = self.create_publisher(PointCloud2, '/gp_field_visualization', 10)
		self.costmap_pub = self.create_publisher(PointCloud2, '/semantic_costmap', 10)

		self.get_logger().info("=" * 60)
		self.get_logger().info("SEMANTIC OCTOMAP SYSTEM READY")
		self.get_logger().info("=" * 60)
		
		self.get_logger().info(f"Mapping Configuration:")
		self.get_logger().info(f"   Voxel resolution: {self.voxel_resolution}m")
		self.get_logger().info(f"   Max range: {self.max_range}m")
		self.get_logger().info(f"   Min range: {self.min_range}m")
		
		self.get_logger().info(f"Feature Status:")
		self.get_logger().info(f"   Voxel mapping: {'ENABLED' if self.enable_voxel_mapping else 'DISABLED'}")
		self.get_logger().info(f"   Semantic mapping: {'ENABLED' if self.enable_semantic_mapping else 'DISABLED'}")
		self.get_logger().info(f"   Voxel helper: {'READY' if hasattr(self, 'voxel_helper') and self.voxel_helper is not None else 'NOT READY'}")
		
		self.get_logger().info(f"Topics:")
		self.get_logger().info(f"   Depth: {self.depth_topic}")
		self.get_logger().info(f"   Pose: {self.pose_topic}")
		self.get_logger().info(f"   Semantic hotspots: {self.semantic_hotspots_topic}")
		
		self.get_logger().info("=" * 60)

		# Timer for inactivity detection (non-intrusive)

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

	def semantic_hotspot_mask_callback(self, msg: Image):
		"""Buffer the merged hotspot mask image keyed by its stamp time."""
		try:
			mask_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
			mask_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
			with self.sync_lock:
				self.mask_buffer.append((mask_time, mask_rgb))
				self._prune_sync_buffers()
			# Update activity
			self.last_data_time = time.time()
		except Exception as e:
			self.get_logger().warn(f"Failed to buffer hotspot mask image: {e}")

	def semantic_hotspot_callback(self, msg: String):
		"""Queue incoming semantic hotspot metadata for batch processing."""
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
			else:
				return False
			
		except Exception as e:
			self.get_logger().error(f"Error processing single bridge message: {e}")
			return False
	
	def _process_merged_hotspot_message(self, data: dict) -> bool:
		"""Process merged hotspot metadata; fetch mask image by timestamp and apply."""
		try:
			is_narration = data.get('is_narration')
			vlm_info = data.get('vlm_info', {})
			rgb_timestamp = float(data.get('timestamp', 0.0))
			buffer_id = data.get('buffer_id')  # Extract buffer_id
			
			if rgb_timestamp <= 0.0:
				self.get_logger().warn(f"Incomplete hotspot data (no timestamp)")
				return False
			
			# Lookup merged mask image by timestamp
			merged_mask = self._lookup_mask(rgb_timestamp)
			if merged_mask is None:
				self.get_logger().warn(f"No matching hotspot mask found for timestamp {rgb_timestamp:.6f}")
				return False
			
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
						rgb_timestamp, used_ts, is_narration, buffer_id
					)
					if success:
						processed_count += 1
						if len(vlm_info) == 1:
							self.get_logger().info(f"NARRATION HOTSPOT PROCESSED: '{vlm_answer}' with {info.get('hotspot_pixels', 0)} pixels")
			self.get_logger().info(f"Processed {processed_count}/{len(vlm_info)} VLM answers from merged hotspots")
			return processed_count > 0
			
		except Exception as e:
			self.get_logger().error(f"Error processing merged hotspot message: {e}")
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
	
	def _lookup_mask(self, target_ts: float) -> Optional[np.ndarray]:
		"""Find closest merged mask image to target timestamp within buffer window."""
		with self.sync_lock:
			best_mask = None
			best_dt = float('inf')
			for ts, mask in self.mask_buffer:
				dt = abs(ts - target_ts)
				if dt < best_dt and dt <= self.sync_buffer_duration:
					best_dt = dt
					best_mask = mask
			return best_mask
	
	def _process_hotspot_with_depth(self, mask: np.ndarray, pose: PoseStamped, depth_m: np.ndarray,
								   vlm_answer: str, threshold: float, stats: dict, rgb_ts: float, used_ts: tuple, is_narration: bool, buffer_id: str = None) -> bool:
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
			if is_narration:
				# Save PCD to the specific buffer directory if buffer_id is provided
				if buffer_id:
					buffer_dir = self._save_narration_pcd_to_specific_buffer(buffer_id, points_world)
				else:
					buffer_dir, pcd_path = self.save_points_to_latest_nested_subfolder("/home/navin/ros2_ws/src/buffers", points_world)
				
				# Check if poses.npy is available before starting GP fit
				if buffer_dir is not None and GP_HELPER_AVAILABLE:
					# Use voxelized points for GP fitting to avoid memory issues
					voxelized_points = self._voxelize_pointcloud(points_world, float(self.voxel_resolution), max_points=200)
					self._check_and_start_gp_fit_if_ready(buffer_dir, voxelized_points)

			self.voxel_helper.update_map(points_world, origin, hit_confidences=None, voxel_embeddings=None)

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
	
	def _voxelize_pointcloud(self, points: np.ndarray, voxel_size: float, max_points: int = 200) -> np.ndarray:
		"""
		Voxelize a point cloud by taking the centroid of points within each voxel.
		This reduces the number of points while preserving the spatial distribution.
		If still too many points after voxelization, randomly sample down to max_points.
		"""
		if len(points) == 0:
			return points
		
		# Convert points to voxel coordinates
		voxel_coords = np.floor(points / voxel_size).astype(np.int32)
		
		# Find unique voxels and their indices
		unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
		
		# Compute centroids for each voxel
		voxelized_points = []
		for i in range(len(unique_voxels)):
			# Find all points belonging to this voxel
			voxel_mask = inverse_indices == i
			voxel_points = points[voxel_mask]
			
			# Take centroid
			centroid = np.mean(voxel_points, axis=0)
			voxelized_points.append(centroid)
		
		voxelized_points = np.array(voxelized_points)
		
		# If still too many points, randomly sample down
		if len(voxelized_points) > max_points:
			indices = np.random.choice(len(voxelized_points), size=max_points, replace=False)
			voxelized_points = voxelized_points[indices]
			self.get_logger().info(f"Voxelized {len(points)} points to {len(voxelized_points)} points (voxel_size={voxel_size:.3f}m, sampled to max {max_points})")
		else:
			self.get_logger().info(f"Voxelized {len(points)} points to {len(voxelized_points)} points (voxel_size={voxel_size:.3f}m)")
		
		return voxelized_points

	def save_points_to_latest_nested_subfolder(self, known_folder: str,
	                                      points_world: np.ndarray,
	                                      filename: str = "points.pcd"):
		"""
    	Find the latest subfolder1 inside known_folder, then the latest subfolder2 inside it,
    	and save points_world as a binary PCD file in subfolder2.
    	Voxelizes the points first to reduce density for GP fitting.
    	"""
    	# Helper to save PCD
		def _save_pcd(points: np.ndarray, out_path: str):
			pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
			mask = np.isfinite(pts).all(axis=1)
			pts = pts[mask]
			header = (
    	        "# .PCD v0.7 - Point Cloud Data file format\n"
    	        "VERSION 0.7\n"
    	        "FIELDS x y z\n"
    	        "SIZE 4 4 4\n"
    	        "TYPE F F F\n"
    	        "COUNT 1 1 1\n"
    	        f"WIDTH {pts.shape[0]}\n"
    	        "HEIGHT 1\n"
    	        "VIEWPOINT 0 0 0 1 0 0 0\n"
    	        f"POINTS {pts.shape[0]}\n"
    	        "DATA binary\n"
    	    )
			with open(out_path, "wb") as f:
				f.write(header.encode("ascii"))
				f.write(pts.astype("<f4").tobytes())
			print(f"Saved {pts.shape[0]} voxelized points to {out_path}")

		# Voxelize points before saving to reduce density for GP fitting
		voxelized_points = self._voxelize_pointcloud(points_world, float(self.voxel_resolution), max_points=200)

    	# Step 1: find latest subfolder1
		subfolders1 = [os.path.join(known_folder, d) for d in os.listdir(known_folder)
		               if os.path.isdir(os.path.join(known_folder, d))]
		if not subfolders1:
			print(f"No subfolders found inside {known_folder}")
			return None, None
		latest_subfolder1 = max(subfolders1, key=os.path.getmtime)		
    	# Step 2: find latest subfolder2 inside latest_subfolder1
		subfolders2 = [os.path.join(latest_subfolder1, d) for d in os.listdir(latest_subfolder1)
		               if os.path.isdir(os.path.join(latest_subfolder1, d))]
		if not subfolders2:
			print(f"No subfolders found inside {latest_subfolder1}")
			return None, None
		latest_subfolder2 = max(subfolders2, key=os.path.getmtime)		
		# Step 3: save voxelized PCD inside latest_subfolder2
		save_path = os.path.join(latest_subfolder2, filename)
		_save_pcd(voxelized_points, save_path)
		return latest_subfolder2, save_path

	def _save_narration_pcd_to_specific_buffer(self, buffer_id: str, points_world: np.ndarray) -> str:
		"""Save narration PCD to a specific buffer directory."""
		try:
			# Find the buffer directory by looking for the buffer_id in the buffers directory
			buffers_base_dir = "/home/navin/ros2_ws/src/buffers"
			
			# Look for the latest run directory
			run_dirs = [d for d in os.listdir(buffers_base_dir) if os.path.isdir(os.path.join(buffers_base_dir, d)) and d.startswith('run_')]
			if not run_dirs:
				self.get_logger().warn(f"No run directories found in {buffers_base_dir}")
				return None
			
			latest_run_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(buffers_base_dir, d)))
			run_path = os.path.join(buffers_base_dir, latest_run_dir)
			
			# Look for the specific buffer directory
			buffer_dir = os.path.join(run_path, buffer_id)
			if not os.path.exists(buffer_dir):
				self.get_logger().warn(f"Buffer directory {buffer_dir} not found")
				return None
			
			# Voxelize points before saving to reduce density for GP fitting
			voxelized_points = self._voxelize_pointcloud(points_world, float(self.voxel_resolution), max_points=200)
			
			# Save PCD to the specific buffer directory
			pcd_path = os.path.join(buffer_dir, "points.pcd")
			self._save_pcd_file(voxelized_points, pcd_path)
			
			self.get_logger().info(f"Saved narration PCD to specific buffer: {buffer_dir}")
			return buffer_dir
			
		except Exception as e:
			self.get_logger().error(f"Error saving narration PCD to specific buffer: {e}")
			return None
	
	def _save_pcd_file(self, points: np.ndarray, out_path: str):
		"""Save points as a binary PCD file."""
		try:
			pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
			mask = np.isfinite(pts).all(axis=1)
			pts = pts[mask]
			header = (
				"# .PCD v0.7 - Point Cloud Data file format\n"
				"VERSION 0.7\n"
				"FIELDS x y z\n"
				"SIZE 4 4 4\n"
				"TYPE F F F\n"
				"COUNT 1 1 1\n"
				f"WIDTH {pts.shape[0]}\n"
				"HEIGHT 1\n"
				"VIEWPOINT 0 0 0 1 0 0 0\n"
				f"POINTS {pts.shape[0]}\n"
				"DATA binary\n"
			)
			with open(out_path, "wb") as f:
				f.write(header.encode("ascii"))
				f.write(pts.astype("<f4").tobytes())
			self.get_logger().info(f"Saved {pts.shape[0]} voxelized points to {out_path}")
		except Exception as e:
			self.get_logger().error(f"Error saving PCD file: {e}")

	def _check_and_start_gp_fit_if_ready(self, buffer_dir: str, pointcloud_xyz: np.ndarray):
		"""Check if poses.npy is available and start GP fitting if ready."""
		try:
			# Check if poses.npy exists in the buffer directory
			poses_path = os.path.join(buffer_dir, 'poses.npy')
			if not os.path.exists(poses_path):
				self.get_logger().info(f"poses.npy not yet available in {buffer_dir}, skipping GP fit for now")
				return
			
			# Check if poses.npy has data
			try:
				poses_data = np.load(poses_path)
				if len(poses_data) == 0:
					self.get_logger().info(f"poses.npy is empty in {buffer_dir}, skipping GP fit for now")
					return
			except Exception as e:
				self.get_logger().warn(f"Error reading poses.npy from {buffer_dir}: {e}")
				return
			
			# Both PCD and poses are available, start GP fitting
			self.get_logger().info(f"Both PCD and poses.npy available in {buffer_dir}, starting GP fit")
			self._start_background_gp_fit(buffer_dir, pointcloud_xyz)
			
		except Exception as e:
			self.get_logger().warn(f"Error checking GP fit readiness: {e}")

	def _start_background_gp_fit(self, buffer_dir: str, pointcloud_xyz: np.ndarray):
		"""Start GP fitting in a background thread if not already running."""
		try:
			with self.gp_fit_lock:
				if self.gp_fitting_active:
					self.get_logger().info("GP fit already running; skipping new request")
					return
				self.gp_fitting_active = True
			args = (buffer_dir, np.array(pointcloud_xyz, dtype=np.float32))
			threading.Thread(target=self._run_gp_fit_task, args=args, daemon=True).start()
		except Exception as e:
			self.get_logger().warn(f"Failed to start GP fit thread: {e}")

	def _run_gp_fit_task(self, buffer_dir: str, pointcloud_xyz: np.ndarray):
		"""Run GP fitting and save parameters to buffer directory."""
		try:
			self.get_logger().info(f"Starting GP fit for buffer: {buffer_dir}")
			helper = DisturbanceFieldHelper()
			# Try to get nominal XYZ from PathManager if available and ready
			nominal_xyz = None
			try:
				if self.path_manager is not None and hasattr(self.path_manager, 'get_nominal_points_as_numpy'):
					nominal_xyz = self.path_manager.get_nominal_points_as_numpy()
					if nominal_xyz is not None and len(nominal_xyz) == 0:
						nominal_xyz = None
			except Exception:
				pass
			# Announce which nominal will be used for this GP fit
			if nominal_xyz is not None:
				self.get_logger().info(f"GP nominal source: GLOBAL PATH (points={len(nominal_xyz)})")
			elif isinstance(self.nominal_path, str) and len(self.nominal_path) > 0:
				self.get_logger().info(f"GP nominal source: FILE {self.nominal_path}")
			else:
				self.get_logger().warn("GP nominal source: NONE (using actual-only baseline)")
			result = helper.fit_from_pointcloud_and_buffer(
				pointcloud_xyz=pointcloud_xyz,
				buffer_dir=buffer_dir,
				nominal_path=(None if nominal_xyz is not None else (self.nominal_path if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0 else None)),
				nominal_xyz=nominal_xyz
			)
			fit = result.get('fit', {})
			opt = fit.get('optimization_result') if isinstance(fit, dict) else None
			o = {
				'fit_params': {
					'lxy': fit.get('lxy'),
					'lz': fit.get('lz'),
					'A': fit.get('A'),
					'b': fit.get('b'),
					'mse': fit.get('mse'),
					'rmse': fit.get('rmse'),
					'mae': fit.get('mae'),
					'r2_score': fit.get('r2_score')
				},
				'optimization': ({
					'nit': getattr(opt, 'nit', None),
					'nfev': getattr(opt, 'nfev', None),
					'success': getattr(opt, 'success', None),
					'message': getattr(opt, 'message', None)
				} if opt is not None else None),
				'metadata': {
					'timestamp': time.time(),
					'buffer_dir': buffer_dir,
					'nominal_path': self.nominal_path,
					'used_nominal_source': ('path_manager' if nominal_xyz is not None else 'file' if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0 else 'none')
				}
			}
			out_path = os.path.join(buffer_dir, 'voxel_gp_fit.json')
			with open(out_path, 'w') as f:
				json.dump(o, f, indent=2)
			self.get_logger().info(f"Saved GP fit parameters to {out_path}")
			
		except Exception as e:
			self.get_logger().error(f"GP fit task failed: {e}")
			import traceback
			traceback.print_exc()
		finally:
			with self.gp_fit_lock:
				self.gp_fitting_active = False
			
			# Store the latest GP parameters for global use
			if result and 'fit' in result:
				self.global_gp_params = result['fit']
				self.get_logger().info(f"Updated global GP parameters: lxy={self.global_gp_params.get('lxy', 0):.3f}, lz={self.global_gp_params.get('lz', 0):.3f}, A={self.global_gp_params.get('A', 0):.3f}")
			
			# After GP fitting is complete, create and publish visualization
			self._create_and_publish_gp_visualization(buffer_dir)

	def _create_and_publish_gp_visualization(self, buffer_dir: str):
		"""Create and publish GP field visualization using the SAME optimized method as semantic voxels."""
		try:
			if not GP_HELPER_AVAILABLE:
				return
			
			# Load GP fit parameters
			gp_fit_path = os.path.join(buffer_dir, 'voxel_gp_fit.json')
			if not os.path.exists(gp_fit_path):
				self.get_logger().warn(f"GP fit file not found: {gp_fit_path}")
				return
			
			with open(gp_fit_path, 'r') as f:
				gp_data = json.load(f)
			
			fit_params = gp_data.get('fit_params', {})
			if not fit_params:
				self.get_logger().warn("No GP fit parameters found")
				return
			
			# Load cause points from PCD (these are semantic voxels from narration)
			pcd_path = os.path.join(buffer_dir, 'points.pcd')
			if not os.path.exists(pcd_path):
				self.get_logger().warn(f"PCD file not found: {pcd_path}")
				return
			
			cause_points = self._load_pcd_points(pcd_path)
			if cause_points.size == 0:
				self.get_logger().warn("No cause points loaded from PCD")
				return
			
			# Use the SAME optimized method as semantic voxels
			# Calculate adaptive radius based on cause points
			adaptive_radius = self._calculate_adaptive_radius(cause_points)
			
			# Create FAST, adaptive grids around cause points (same as semantic voxels)
			grid_points = self._create_fast_adaptive_gp_grid(cause_points, adaptive_radius)
			if len(grid_points) == 0:
				return
			
			# Predict GP field values using OPTIMIZED method (same as semantic voxels)
			gp_values = self._predict_gp_field_fast(grid_points, cause_points, fit_params)
			
			# Create colored point cloud for visualization (same as semantic voxels)
			colored_cloud = self._create_gp_colored_pointcloud(grid_points, gp_values)
			if colored_cloud:
				self.gp_visualization_pub.publish(colored_cloud)
			
			# Create costmap (same as semantic voxels)
			costmap_cloud = self._create_costmap_pointcloud(grid_points, gp_values)
			if costmap_cloud:
				self.costmap_pub.publish(costmap_cloud)
			
			self.get_logger().info(f"Published cause.pcd GP visualization + costmap: {len(grid_points)} points, {len(cause_points)} cause voxels, radius={adaptive_radius:.2f}m (SAME method as semantic voxels)")
			
		except Exception as e:
			self.get_logger().error(f"Error creating GP visualization: {e}")
			import traceback
			traceback.print_exc()
	
	def _load_pcd_points(self, pcd_path: str) -> np.ndarray:
		"""Load points from PCD file."""
		try:
			# Simple PCD loader for binary format
			with open(pcd_path, 'rb') as f:
				# Skip header
				header_lines = []
				while True:
					line = f.readline().decode('ascii')
					header_lines.append(line)
					if line.startswith('DATA binary'):
						break
				
				# Find POINTS count
				points_count = 0
				for line in header_lines:
					if line.startswith('POINTS'):
						points_count = int(line.split()[1])
						break
				
				if points_count == 0:
					return np.array([])
				
				# Read binary data (3 floats per point: x, y, z)
				points_data = f.read(points_count * 3 * 4)  # 4 bytes per float
				points = np.frombuffer(points_data, dtype=np.float32).reshape(-1, 3)
				
				return points
				
		except Exception as e:
			self.get_logger().error(f"Error loading PCD points: {e}")
			return np.array([])
	
	def _create_gp_prediction_grid(self, cause_points: np.ndarray, grid_size: float = 2.0, resolution: float = 0.1) -> np.ndarray:
		"""Create a 3D grid around the cause points for GP prediction."""
		try:
			# Find bounding box of cause points
			min_coords = np.min(cause_points, axis=0)
			max_coords = np.max(cause_points, axis=0)
			center = (min_coords + max_coords) / 2.0
			
			# Extend bounding box by grid_size
			extent = np.max(max_coords - min_coords) + grid_size
			half_extent = extent / 2.0
			
			# Create grid
			x_range = np.arange(center[0] - half_extent, center[0] + half_extent, resolution)
			y_range = np.arange(center[1] - half_extent, center[1] + half_extent, resolution)
			z_range = np.arange(center[2] - half_extent, center[2] + half_extent, resolution)
			
			# Create meshgrid
			X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
			grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
			
			self.get_logger().info(f"Created GP prediction grid: {len(grid_points)} points around cause center {center}")
			return grid_points
			
		except Exception as e:
			self.get_logger().error(f"Error creating GP prediction grid: {e}")
			return np.array([])
	
	def _predict_gp_field(self, grid_points: np.ndarray, cause_points: np.ndarray, fit_params: dict) -> np.ndarray:
		"""Predict GP field values at grid points using the SAME anisotropic RBF method as cause.pcd."""
		try:
			# Extract GP parameters
			lxy = fit_params.get('lxy', 0.5)
			lz = fit_params.get('lz', 0.5)
			A = fit_params.get('A', 1.0)
			b = fit_params.get('b', 0.0)
			
			# Use the EXACT SAME anisotropic RBF computation as cause.pcd system
			phi = self._sum_of_anisotropic_rbf(grid_points, cause_points, lxy, lz)
			
			# Apply the learned parameters: disturbance = A * phi + b (same as cause.pcd)
			predictions = A * phi + b
			
			self.get_logger().info(f"GP field prediction using ANISOTROPIC RBF: min={predictions.min():.3f}, max={predictions.max():.3f}")
			return predictions
			
		except Exception as e:
			self.get_logger().error(f"Error predicting GP field: {e}")
			return np.zeros(len(grid_points))
	
	def _sum_of_anisotropic_rbf(self, grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
		"""Compute phi(x) = sum_j exp(-0.5 * [((dx/lxy)^2 + (dy/lxy)^2 + (dz/lz)^2)] ) for all grid points.
		This is the EXACT SAME function used in the cause.pcd system for computing disturbance fields.
		"""
		try:
			if centers.size == 0:
				return np.zeros(grid_points.shape[0], dtype=float)
			
			num_points = grid_points.shape[0]
			phi = np.zeros(num_points, dtype=float)
			chunk = 200000  # Process in chunks for memory efficiency
			
			# Precompute inverse squared length scales (same as cause.pcd)
			inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
			inv_lz2 = 1.0 / (lz * lz + 1e-12)
			
			# Process grid points in chunks (same as cause.pcd)
			for start in range(0, num_points, chunk):
				end = min(num_points, start + chunk)
				gp_chunk = grid_points[start:end]
				
				# Broadcast centers over chunk for efficient computation (same as cause.pcd)
				dx = gp_chunk[:, None, 0] - centers[None, :, 0]
				dy = gp_chunk[:, None, 1] - centers[None, :, 1]
				dz = gp_chunk[:, None, 2] - centers[None, :, 2]
				
				# Compute anisotropic distance squared (same as cause.pcd)
				d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
				
				# Compute RBF contributions and sum over all centers (same as cause.pcd)
				np.exp(-0.5 * d2, out=d2)
				phi[start:end] = np.sum(d2, axis=1)
			
			return phi
			
		except Exception as e:
			self.get_logger().error(f"Error computing anisotropic RBF: {e}")
			return np.zeros(grid_points.shape[0], dtype=float)
	
	def _create_gp_colored_pointcloud(self, grid_points: np.ndarray, gp_values: np.ndarray) -> Optional[PointCloud2]:
		"""Create colored point cloud from GP field predictions."""
		try:
			if len(grid_points) == 0 or len(gp_values) == 0:
				return None
			
			# Normalize GP values to [0, 1] for coloring
			gp_min, gp_max = gp_values.min(), gp_values.max()
			if gp_max > gp_min:
				normalized_values = (gp_values - gp_min) / (gp_max - gp_min)
			else:
				normalized_values = np.zeros_like(gp_values)
			
			# Create BRIGHT, HIGH-CONTRAST color map with proper gradient
			colors = np.zeros((len(grid_points), 3), dtype=np.uint8)
			
			# High-contrast colormap: Dark Blue -> Cyan -> Yellow -> Bright Red
			# This gives much better visibility and contrast
			for i, value in enumerate(normalized_values):
				if value < 0.25:  # Low values: Dark Blue to Cyan
					local_val = value / 0.25
					colors[i] = [0, int(255 * local_val), 255]  # Blue to Cyan
				elif value < 0.5:  # Medium-low: Cyan to Green
					local_val = (value - 0.25) / 0.25
					colors[i] = [0, 255, int(255 * (1 - local_val))]  # Cyan to Green
				elif value < 0.75:  # Medium-high: Green to Yellow
					local_val = (value - 0.5) / 0.25
					colors[i] = [int(255 * local_val), 255, 0]  # Green to Yellow
				else:  # High values: Yellow to Bright Red
					local_val = (value - 0.75) / 0.25
					colors[i] = [255, int(255 * (1 - local_val)), 0]  # Yellow to Red
			
			# Create PointCloud2 message
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
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
			
			# Pack RGB values as UINT32 (standard for PointCloud2 RGB)
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
			cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, rgb)
			cloud_msg.width = len(grid_points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data_combined.tobytes()
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating GP colored point cloud: {e}")
			import traceback
			traceback.print_exc()
			return None
	
	def _start_gp_computation_thread(self):
		"""Start the GP computation thread."""
		try:
			with self.gp_thread_lock:
				if self.gp_thread_running:
					return
				self.gp_thread_running = True
			
			self.gp_computation_thread = threading.Thread(target=self._gp_computation_worker, daemon=True)
			self.gp_computation_thread.start()
			self.get_logger().info("GP computation thread started")
			
		except Exception as e:
			self.get_logger().error(f"Error starting GP computation thread: {e}")
			with self.gp_thread_lock:
				self.gp_thread_running = False
	
	def _gp_computation_worker(self):
		"""Background worker thread for GP computation and visualization."""
		try:
			while self.gp_thread_running:
				current_time = time.time()
				
				# Check if it's time to update GP visualization
				if (current_time - self.last_gp_update_time) >= self.gp_update_interval:
					self._update_semantic_gp_visualization()
					self.last_gp_update_time = current_time
				
				# Sleep for a short time to avoid busy waiting
				time.sleep(0.1)
				
		except Exception as e:
			self.get_logger().error(f"Error in GP computation worker: {e}")
			import traceback
			traceback.print_exc()
		finally:
			with self.gp_thread_lock:
				self.gp_thread_running = False
	
	def _update_semantic_gp_visualization(self):
		"""Update GP visualization for all semantic voxels using global GP parameters - optimized for speed."""
		try:
			if self.global_gp_params is None:
				return
			
			# Get all semantic voxels (these are like the cause points)
			semantic_voxels = self._get_all_semantic_voxels()
			if len(semantic_voxels) == 0:
				return
			
			# Convert to numpy array (like loading cause points from PCD)
			semantic_points = np.array(semantic_voxels)
			
			# Calculate adaptive radius based on voxel density
			adaptive_radius = self._calculate_adaptive_radius(semantic_points)
			
			# Create FAST, adaptive grids around semantic voxel clusters
			grid_points = self._create_fast_adaptive_gp_grid(semantic_points, adaptive_radius)
			if len(grid_points) == 0:
				return
			
			# Predict GP field values using OPTIMIZED method
			gp_values = self._predict_gp_field_fast(grid_points, semantic_points, self.global_gp_params)
			
			# Create colored point cloud for visualization
			colored_cloud = self._create_gp_colored_pointcloud(grid_points, gp_values)
			if colored_cloud:
				self.gp_visualization_pub.publish(colored_cloud)
			
			# Create costmap (same data, different interpretation)
			costmap_cloud = self._create_costmap_pointcloud(grid_points, gp_values)
			if costmap_cloud:
				self.costmap_pub.publish(costmap_cloud)
			
			self.get_logger().info(f"Published GP visualization + costmap: {len(grid_points)} points, {len(semantic_voxels)} voxels, radius={adaptive_radius:.2f}m")
			
		except Exception as e:
			self.get_logger().error(f"Error updating semantic GP visualization: {e}")
			import traceback
			traceback.print_exc()
	
	def _calculate_adaptive_radius(self, semantic_points: np.ndarray) -> float:
		"""Calculate adaptive radius based on voxel density and distribution."""
		try:
			if len(semantic_points) < 2:
				return self.base_radius
			
			# Calculate average distance between voxels
			distances = []
			for i in range(len(semantic_points)):
				for j in range(i + 1, len(semantic_points)):
					dist = np.linalg.norm(semantic_points[i] - semantic_points[j])
					distances.append(dist)
			
			if len(distances) == 0:
				return self.base_radius
			
			avg_distance = np.mean(distances)
			
			# Adaptive radius: smaller for dense clusters, larger for sparse voxels
			adaptive_radius = max(self.min_radius, min(self.max_radius, avg_distance * 0.8))
			
			return adaptive_radius
			
		except Exception as e:
			self.get_logger().warn(f"Error calculating adaptive radius: {e}")
			return self.base_radius
	
	def _create_fast_adaptive_gp_grid(self, semantic_points: np.ndarray, radius: float) -> np.ndarray:
		"""Create FAST, adaptive grid around semantic voxel clusters."""
		try:
			if len(semantic_points) == 0:
				return np.array([])
			
			# Use coarser resolution for speed (0.2m)
			resolution = 0.2
			
			# Find bounding box of all semantic voxels
			min_coords = np.min(semantic_points, axis=0)
			max_coords = np.max(semantic_points, axis=0)
			
			# Use adaptive radius for extension
			extent = np.max(max_coords - min_coords) + radius
			half_extent = extent / 2.0
			center = (min_coords + max_coords) / 2.0
			
			# Create FAST grid with coarser resolution
			x_range = np.arange(center[0] - half_extent, center[0] + half_extent, resolution)
			y_range = np.arange(center[1] - half_extent, center[1] + half_extent, resolution)
			z_range = np.arange(center[2] - half_extent, center[2] + half_extent, resolution)
			
			# Create meshgrid
			X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
			grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
			
			# FAST filtering using vectorized operations
			filtered_grid_points = self._filter_grid_points_fast(grid_points, semantic_points, radius)
			
			return filtered_grid_points
			
		except Exception as e:
			self.get_logger().error(f"Error creating fast adaptive GP grid: {e}")
			return np.array([])
	
	def _filter_grid_points_fast(self, grid_points: np.ndarray, voxel_positions: np.ndarray, max_distance: float) -> np.ndarray:
		"""FAST filtering using vectorized operations."""
		try:
			if len(grid_points) == 0 or len(voxel_positions) == 0:
				return grid_points
			
			# Vectorized distance computation - much faster
			distances = np.linalg.norm(grid_points[:, np.newaxis, :] - voxel_positions[np.newaxis, :, :], axis=2)
			min_distances = np.min(distances, axis=1)
			
			# Keep points within max_distance
			mask = min_distances <= max_distance
			filtered_points = grid_points[mask]
			
			return filtered_points
			
		except Exception as e:
			self.get_logger().error(f"Error in fast grid filtering: {e}")
			return grid_points
	
	def _predict_gp_field_fast(self, grid_points: np.ndarray, cause_points: np.ndarray, fit_params: dict) -> np.ndarray:
		"""FAST GP field prediction using optimized anisotropic RBF."""
		try:
			# Extract GP parameters
			lxy = fit_params.get('lxy', 0.5)
			lz = fit_params.get('lz', 0.5)
			A = fit_params.get('A', 1.0)
			b = fit_params.get('b', 0.0)
			
			# Use OPTIMIZED anisotropic RBF computation
			phi = self._sum_of_anisotropic_rbf_fast(grid_points, cause_points, lxy, lz)
			
			# Apply the learned parameters: disturbance = A * phi + b
			predictions = A * phi + b
			
			return predictions
			
		except Exception as e:
			self.get_logger().error(f"Error in fast GP prediction: {e}")
			return np.zeros(len(grid_points))
	
	def _sum_of_anisotropic_rbf_fast(self, grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
		"""OPTIMIZED anisotropic RBF computation for speed."""
		try:
			if centers.size == 0:
				return np.zeros(grid_points.shape[0], dtype=float)
			
			# Precompute inverse squared length scales
			inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
			inv_lz2 = 1.0 / (lz * lz + 1e-12)
			
			# Vectorized computation - much faster than chunked approach
			dx = grid_points[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
			dy = grid_points[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
			dz = grid_points[:, np.newaxis, 2] - centers[np.newaxis, :, 2]
			
			# Compute anisotropic distance squared
			d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
			
			# Compute RBF contributions and sum over all centers
			phi = np.sum(np.exp(-0.5 * d2), axis=1)
			
			return phi
			
		except Exception as e:
			self.get_logger().error(f"Error in fast anisotropic RBF: {e}")
			return np.zeros(grid_points.shape[0], dtype=float)
	
	def _create_costmap_pointcloud(self, grid_points: np.ndarray, gp_values: np.ndarray) -> Optional[PointCloud2]:
		"""Create costmap point cloud with ACTUAL disturbance values for motion planning."""
		try:
			if len(grid_points) == 0 or len(gp_values) == 0:
				return None
			
			# Use ACTUAL GP disturbance values (not normalized) for motion planning
			# These are the real disturbance magnitudes that motion planning needs
			disturbance_values = gp_values.astype(np.float32)
			
			# Create PointCloud2 message with XYZ + disturbance values
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.map_frame
			
			# Create structured array with XYZ + disturbance value
			cloud_data_combined = np.empty(len(grid_points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), 
				('disturbance', np.float32)
			])
			
			# Fill in the data
			cloud_data_combined['x'] = grid_points[:, 0]
			cloud_data_combined['y'] = grid_points[:, 1]
			cloud_data_combined['z'] = grid_points[:, 2]
			cloud_data_combined['disturbance'] = disturbance_values
			
			# Create PointCloud2 message
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields - XYZ + disturbance value
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='disturbance', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, disturbance)
			cloud_msg.width = len(grid_points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data_combined.tobytes()
			
			self.get_logger().info(f"Published costmap with ACTUAL disturbance values: min={disturbance_values.min():.3f}, max={disturbance_values.max():.3f}")
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating costmap point cloud: {e}")
			return None
	
	def _create_tight_semantic_gp_grid(self, semantic_points: np.ndarray, grid_size: float = 0.8, resolution: float = 0.15) -> np.ndarray:
		"""Create tight, smaller grids around semantic voxel clusters with REDUCED resolution for efficiency."""
		try:
			if len(semantic_points) == 0:
				return np.array([])
			
			# Find bounding box of all semantic voxels
			min_coords = np.min(semantic_points, axis=0)
			max_coords = np.max(semantic_points, axis=0)
			
			# Use smaller extension for tighter grid
			extent = np.max(max_coords - min_coords) + grid_size
			half_extent = extent / 2.0
			center = (min_coords + max_coords) / 2.0
			
			# Create REDUCED resolution grid (0.15m instead of 0.08m)
			x_range = np.arange(center[0] - half_extent, center[0] + half_extent, resolution)
			y_range = np.arange(center[1] - half_extent, center[1] + half_extent, resolution)
			z_range = np.arange(center[2] - half_extent, center[2] + half_extent, resolution)
			
			# Create meshgrid
			X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
			grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
			
			# Filter grid points to keep only those close to semantic voxels (within 1.0m)
			filtered_grid_points = self._filter_grid_points_near_voxels(grid_points, semantic_points, max_distance=1.0)
			
			self.get_logger().info(f"Created TIGHT semantic GP grid: {len(filtered_grid_points)} points around {len(semantic_points)} semantic voxels (grid_size={grid_size}m, resolution={resolution}m)")
			return filtered_grid_points
			
		except Exception as e:
			self.get_logger().error(f"Error creating tight semantic GP grid: {e}")
			return np.array([])
	
	def _filter_grid_points_near_voxels(self, grid_points: np.ndarray, voxel_positions: np.ndarray, max_distance: float = 1.0) -> np.ndarray:
		"""Filter grid points to keep only those within max_distance of any semantic voxel."""
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
			
			self.get_logger().info(f"Filtered grid points: {len(grid_points)} -> {len(filtered_points)} (kept points within {max_distance}m of semantic voxels)")
			
			return filtered_points
			
		except Exception as e:
			self.get_logger().error(f"Error filtering grid points: {e}")
			return grid_points
	
	def _get_all_semantic_voxels(self) -> List[np.ndarray]:
		"""Get all semantic voxel positions with deduplication and revoxelization to avoid double counting."""
		try:
			if not hasattr(self.voxel_helper, 'semantic_mapper') or not self.voxel_helper.semantic_mapper:
				return []
			
			semantic_mapper = self.voxel_helper.semantic_mapper
			semantic_voxel_positions = []
			
			with semantic_mapper.voxel_semantics_lock:
				for voxel_key in semantic_mapper.voxel_semantics.keys():
					voxel_position = self._get_voxel_center_from_key(voxel_key)
					if voxel_position is not None:
						semantic_voxel_positions.append(voxel_position)
			
			if not semantic_voxel_positions:
				return []
			
			# Convert to numpy array for processing
			semantic_points = np.array(semantic_voxel_positions)
			
			# DEDUPLICATION: Remove duplicate voxels to avoid double counting
			# This happens when the same semantic voxels come from both direct mapping and narration
			deduplicated_points = self._deduplicate_semantic_voxels(semantic_points)
			
			# REVOXELIZATION: Maintain nearly same ratio by clustering nearby voxels
			# This prevents misleading double points while preserving spatial distribution
			revoxelized_points = self._revoxelize_semantic_voxels(deduplicated_points)
			
			self.get_logger().info(f"Semantic voxels: {len(semantic_points)} → {len(deduplicated_points)} (dedup) → {len(revoxelized_points)} (revoxelized)")
			
			return revoxelized_points.tolist()
			
		except Exception as e:
			self.get_logger().error(f"Error getting semantic voxels: {e}")
			return []
	
	def _deduplicate_semantic_voxels(self, semantic_points: np.ndarray) -> np.ndarray:
		"""Remove duplicate voxels that are too close together (within voxel resolution)."""
		try:
			if len(semantic_points) <= 1:
				return semantic_points
			
			# Use voxel resolution as the deduplication threshold
			dedup_threshold = self.voxel_resolution * 0.5  # Half voxel resolution for safety
			
			# Simple deduplication: keep first occurrence of each unique voxel
			deduplicated = []
			used_indices = set()
			
			for i, point in enumerate(semantic_points):
				if i in used_indices:
					continue
				
				# Find all points within dedup_threshold of this point
				distances = np.linalg.norm(semantic_points - point, axis=1)
				close_indices = np.where(distances <= dedup_threshold)[0]
				
				# Mark all close points as used
				used_indices.update(close_indices)
				
				# Add the first (representative) point
				deduplicated.append(point)
			
			return np.array(deduplicated)
			
		except Exception as e:
			self.get_logger().error(f"Error deduplicating semantic voxels: {e}")
			return semantic_points
	
	def _revoxelize_semantic_voxels(self, semantic_points: np.ndarray) -> np.ndarray:
		"""Revoxelize semantic voxels to maintain nearly same ratio while avoiding double counting."""
		try:
			if len(semantic_points) <= 1:
				return semantic_points
			
			# Use a slightly larger voxel size for revoxelization to cluster nearby voxels
			revoxel_size = self.voxel_resolution * 1.5  # 1.5x voxel resolution
			
			# Convert points to revoxel coordinates
			revoxel_coords = np.floor(semantic_points / revoxel_size).astype(np.int32)
			
			# Find unique revoxels and their indices
			unique_revoxels, inverse_indices = np.unique(revoxel_coords, axis=0, return_inverse=True)
			
			# Compute centroids for each revoxel
			revoxelized_points = []
			for i in range(len(unique_revoxels)):
				# Find all points belonging to this revoxel
				voxel_mask = inverse_indices == i
				voxel_points = semantic_points[voxel_mask]
				
				# Take centroid
				centroid = np.mean(voxel_points, axis=0)
				revoxelized_points.append(centroid)
			
			return np.array(revoxelized_points)
			
		except Exception as e:
			self.get_logger().error(f"Error revoxelizing semantic voxels: {e}")
			return semantic_points

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
		# Depth/mask/pose buffers capped by length (heuristic) to bound memory
		max_entries = 200
		if len(self.depth_buffer) > max_entries:
			self.depth_buffer = self.depth_buffer[-max_entries:]
		if len(self.pose_buffer) > max_entries:
			self.pose_buffer = self.pose_buffer[-max_entries:]
		if len(self.mask_buffer) > max_entries:
			self.mask_buffer = self.mask_buffer[-max_entries:]

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





def main():
	rclpy.init()
	node = SemanticDepthOctoMapNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		# Cleanup GP computation thread
		if hasattr(node, 'gp_thread_running') and node.gp_thread_running:
			with node.gp_thread_lock:
				node.gp_thread_running = False
			if node.gp_computation_thread and node.gp_computation_thread.is_alive():
				node.gp_computation_thread.join(timeout=2.0)
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main() 
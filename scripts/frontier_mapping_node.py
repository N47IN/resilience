#!/usr/bin/env python3
"""
Semantic Depth VDB Mapping ROS2 Node

Simplified node that uses RayFronts SemanticRayFrontiersMap for efficient 3D mapping.
Subscribes to depth, pose, and semantic info to create semantic voxel maps using OpenVDB.
Maintains timestamped buffers for depth frames and poses to align with hotspot masks
received via the semantic bridge using original RGB timestamps.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.logging import LoggingSeverity

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

import numpy as np
import torch
from cv_bridge import CvBridge
import time
import json
import math
from typing import Optional, List, Dict
import sensor_msgs_py.point_cloud2 as pc2
import threading
import cv2
import os
import bisect

# Import RayFronts VDB mapping
try:
	import sys
	sys.path.append('/home/navin/ros2_ws/src/resilience/RayFronts')
	from rayfronts.mapping.semantic_ray_frontiers_map import SemanticRayFrontiersMap
	from rayfronts import geometry3d as g3d
	import rayfronts_cpp
	VDB_AVAILABLE = True
except ImportError as e:
	print(f"RayFronts VDB not available: {e}")
	VDB_AVAILABLE = False

# Optional GP helper
try:
	from resilience.voxel_gp_helper import DisturbanceFieldHelper
	GP_HELPER_AVAILABLE = True
except ImportError:
	GP_HELPER_AVAILABLE = False

# Optional PathManager for global path access
try:
	from resilience.path_manager import PathManager
	PATH_MANAGER_AVAILABLE = True
except ImportError:
	PATH_MANAGER_AVAILABLE = False


class _ZeroImageEncoder:
	def __init__(self, embed_dim: int, device: str):
		self.embed_dim = embed_dim
		self.device = device

	def encode_image_to_vector(self, rgb_img: torch.Tensor) -> torch.Tensor:
		batch = rgb_img.shape[0]
		return torch.zeros(batch, self.embed_dim, device=rgb_img.device, dtype=rgb_img.dtype)

	def encode_image_to_feat_map(self, rgb_img: torch.Tensor) -> torch.Tensor:
		batch, _, h, w = rgb_img.shape
		return torch.zeros(batch, self.embed_dim, h, w, device=rgb_img.device, dtype=rgb_img.dtype)

	def align_spatial_features_with_language(self, feat: torch.Tensor) -> torch.Tensor:
		return feat


class SemanticDepthOctoMapNode(Node):
	"""Simplified semantic depth VDB mapping node using RayFronts SemanticRayFrontiersMap."""

	def __init__(self):
		super().__init__('semantic_depth_vdb_mapping_node')
		self.get_logger().set_level(LoggingSeverity.WARN)

		# Professional startup message
		self.get_logger().info("=" * 60)
		self.get_logger().info("SEMANTIC VDB MAPPING SYSTEM INITIALIZING")
		self.get_logger().info("=" * 60)

		if not VDB_AVAILABLE:
			self.get_logger().error("RayFronts VDB not available! Please check installation.")
			return

		# Parameters
		self.declare_parameters('', [
			('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered'),
			('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info'),
			('pose_topic', '/robot_1/sensors/front_stereo/pose'),
			('map_frame', 'map'),
			('voxel_resolution', 0.1),
			('max_range', 1.5),
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
			('enable_voxel_mapping', True),
			('sync_buffer_seconds', 2.0),
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
			'enable_semantic_mapping', 'semantic_similarity_threshold', 'buffers_directory',
			'enable_voxel_mapping', 'sync_buffer_seconds', 'inactivity_threshold_seconds', 'semantic_export_directory', 'mapping_config_path', 'nominal_path', 'main_config_path'
		])

		# Extract parameter values
		(self.depth_topic, self.camera_info_topic, self.pose_topic,
		 self.map_frame, self.voxel_resolution, self.max_range, self.min_range, self.prob_hit,
		 self.prob_miss, self.occ_thresh, self.publish_markers, self.publish_stats, self.publish_colored_cloud,
		 self.use_cube_list_markers, self.max_markers, self.marker_publish_rate, self.stats_publish_rate,
		 self.pose_is_base_link, self.apply_optical_frame_rotation, self.cam_to_base_rpy_deg, self.cam_to_base_xyz,
			self.embedding_dim, self.enable_semantic_mapping, self.semantic_similarity_threshold,
			self.buffers_directory,
			self.enable_voxel_mapping, self.sync_buffer_seconds, self.inactivity_threshold_seconds,
		 self.semantic_export_directory, self.mapping_config_path, self.nominal_path, self.main_config_path) = [p.value for p in params]

		# Read nominal path separately (optional for GP)
		self.nominal_path = self.get_parameter('nominal_path').value
		self.main_config_path = self.get_parameter('main_config_path').value

		# Load topic configuration from mapping config
		self.load_topic_configuration()
		
		# Initialize state variables
		self.bridge = CvBridge()
		self.camera_intrinsics = None
		self.latest_pose = None
		self.last_marker_pub = 0.0
		self.last_stats_pub = 0.0
		self.last_data_time = time.time()
		self.semantic_pcd_exported = False
		
		# Timestamped buffers for sync
		self.depth_buffer = []
		self.pose_buffer = []
		self.mask_buffer = []
		self.sync_buffer_duration = float(self.sync_buffer_seconds)
		self.sync_lock = threading.Lock()
		
		# Cache for latest buffer subfolder (avoid repeated file system calls)
		self._cached_latest_subfolder = None
		self._cached_subfolder_time = 0.0
		self._subfolder_cache_ttl = 1.0  # Refresh cache every 1 second
		
		# GP fitting state
		self.gp_fit_lock = threading.Lock()
		self.gp_fitting_active = False
		self.global_gp_params = None
		self.global_nominal_points = None  # Store nominal points for uncertainty computation
		self.global_disturbances = None  # Store disturbances for uncertainty computation
		self.last_gp_update_time = 0.0
		self.gp_update_interval = 1.0
		self.gp_computation_thread = None
		self.gp_thread_lock = threading.Lock()
		self.gp_thread_running = False
		self.min_radius = 0.5
		self.max_radius = 2.0
		self.base_radius = 1.0
		
		# PathManager initialization
		self.path_manager = None
		if PATH_MANAGER_AVAILABLE:
			try:
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
			except Exception as e:
				self.get_logger().warn(f"Failed to initialize PathManager: {e}")
		
		# Simple event-driven processing - messages processed directly in callback
		self._latest_pose_rays = None  # (origin_world np.array(3,), dirs np.array(N,3))
		
		# Initialize unified VDB mapper (occupancy + frontiers + rays)
		self._initialize_vdb_mapper()
		
		# Create alias for backward compatibility
		self.rf_sem_map = self.vdb_mapper
		
		# Semantic voxel tracking with RayFronts-style confidence accumulation
		self.semantic_voxels = {}  # voxel_key -> {'vlm_answer': str, 'similarity': float, 'timestamp': float, 'position': np.array, 'confidence': float}
		self.semantic_voxels_lock = threading.Lock()
		
		# Temporal confirmation: track observations for each voxel
		self.semantic_voxel_observations = {}  # voxel_key -> [{'vlm_answer': str, 'timestamp': float, 'frame_id': int}, ...]
		self.narration_confirmation_threshold = 1  # Narration: instant confirmation (1 frame)
		self.operational_confirmation_threshold = 2  # Operational: require 2 frames for noise rejection (non-blocking, incremental)
		self.semantic_observation_max_age = 5.0  # Keep observations for 5 seconds
		self.frame_counter = 0  # Track unique frames for operational hotspots
		
		# OPTIMIZED: Incremental spatial observation counts for fast threshold checks
		# Structure: (voxel_key, vlm_answer) -> {'count': int, 'unique_frames': set, 'last_update': float}
		# Updated incrementally when observations are added (O(1) threshold checks)
		self.spatial_observation_counts = {}  # (voxel_key, vlm_answer) -> {'count': int, 'unique_frames': set, 'last_update': float}
		# Accumulated pose-ray bins (match RayFronts behavior)
		self.pose_rays_orig_angles = None
		self.pose_rays_feats_cnt = None
		
		# Load existing embeddings
		if isinstance(self.buffers_directory, str) and len(self.buffers_directory) > 0:
			self.get_logger().info(f"Buffers directory: {self.buffers_directory}")
		
		# Simple GP visualization
		self.get_logger().info("Simple GP visualization system initialized")
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
		if self.enable_semantic_mapping and self.enable_voxel_mapping:
			self.create_subscription(String, self.semantic_hotspots_topic, self.semantic_hotspot_callback, 10)
			self.create_subscription(Image, self.semantic_hotspot_mask_topic, self.semantic_hotspot_mask_callback, 10)
		
		# Publishers
		self.marker_pub = self.create_publisher(MarkerArray, self.semantic_octomap_markers_topic, 10) if self.publish_markers else None
		self.stats_pub = self.create_publisher(String, self.semantic_octomap_stats_topic, 10) if self.publish_stats else None
		self.cloud_pub = self.create_publisher(PointCloud2, self.semantic_octomap_colored_cloud_topic, 10) if self.publish_colored_cloud else None
		self.semantic_only_pub = self.create_publisher(PointCloud2, self.semantic_voxels_only_topic, 10) if self.publish_colored_cloud else None
		
		# Registry query publishers/subscribers for real-time access
		self.registry_query_pub = self.create_publisher(String, '/cause_registry/query', 10)
		self.registry_response_sub = self.create_subscription(String, '/cause_registry/response', self._handle_registry_response, 10)
		self.pending_registry_queries = {}  # query_id -> callback
		self.registry_query_counter = 0
		self.registry_query_lock = threading.Lock()
		self.gp_visualization_pub = self.create_publisher(PointCloud2, '/gp_field_visualization', 10)
		self.costmap_pub = self.create_publisher(PointCloud2, '/semantic_costmap', 10)
		self.gp_uncertainty_pub = self.create_publisher(PointCloud2, '/gp_uncertainty_field', 10)
		# New: frontiers and rays publishers
		self.frontiers_pub = self.create_publisher(PointCloud2, '/vdb_frontiers', 10)
		self.mask_frontiers_pub = self.create_publisher(PointCloud2, '/mask_frontiers', 10)
		self.mask_rays_pub = self.create_publisher(MarkerArray, '/mask_rays', 10)
		
		self.get_logger().info("=" * 60)
		self.get_logger().info("SEMANTIC VDB MAPPING SYSTEM READY")
		self.get_logger().info("=" * 60)
		self.get_logger().info(f"Mapping Configuration:")
		self.get_logger().info(f"   Mapper: SemanticRayFrontiersMap (OpenVDB)")
		self.get_logger().info(f"   Device: {self.vdb_mapper.device}")
		self.get_logger().info(f"   Voxel resolution: {self.voxel_resolution}m")
		self.get_logger().info(f"   Max range: {self.max_range}m")
		self.get_logger().info(f"   Min range: {self.min_range}m")
		self.get_logger().info(f"Feature Status:")
		self.get_logger().info(f"   VDB occupancy mapping: ENABLED")
		self.get_logger().info(f"   Semantic mapping: {'ENABLED' if self.enable_semantic_mapping else 'DISABLED'}")
		self.get_logger().info(f"   VDB mapper: {'READY' if hasattr(self, 'vdb_mapper') and self.vdb_mapper is not None else 'NOT READY'}")
		self.get_logger().info(f"Topics:")
		self.get_logger().info(f"   Depth: {self.depth_topic}")
		self.get_logger().info(f"   Pose: {self.pose_topic}")
		self.get_logger().info(f"   Semantic hotspots: {self.semantic_hotspots_topic}")
		self.get_logger().info("=" * 60)

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

	def _initialize_vdb_mapper(self):
		"""Initialize the RayFronts VDB occupancy mapper with noise-robust parameters."""
		try:
			# Create dummy intrinsics (will be updated when camera info is received)
			dummy_intrinsics = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
			
			# Initialize SemanticRayFrontiersMap as the primary VDB mapper
			# This includes occupancy, frontiers, and rays all in one
			self.vdb_mapper = SemanticRayFrontiersMap(
				intrinsics_3x3=dummy_intrinsics,
				device=("cuda" if torch.cuda.is_available() else "cpu"),
				visualizer=None,
				clip_bbox=None,
				encoder=None,
				feat_compressor=None,
				interp_mode="bilinear",
				max_pts_per_frame=2000,  # Increased for better coverage
				vox_size=float(self.voxel_resolution),
				vox_accum_period=2,  # Accumulate over 2 frames for smoother updates
				max_empty_pts_per_frame=2000,  # Increased for better free space clearing
				max_rays_per_frame=2000,
				max_depth_sensing=2.5,  # 1.5m for voxelization and frontiers
				max_empty_cnt=8,  # Increased: require more evidence before removing voxels (reduces flicker)
				max_occ_cnt=7,  # Increased: require more confirmation before marking occupied (reduces noise)
				occ_observ_weight=3,  # Reduced: less aggressive updates per observation (smoother)
				occ_thickness=3,  # Increased: thicker occupied surface (more robust)
				occ_pruning_tolerance=5,  # Increased: more forgiving pruning (keeps stable voxels)
				occ_pruning_period=3,  # Increased: prune less frequently (more stable map)
				sem_pruning_thresh=0,
				sem_pruning_period=1,
				fronti_neighborhood_r=1,
				fronti_min_unobserved=4,
				fronti_min_empty=2,
				fronti_min_occupied=0,
				fronti_subsampling=4,
				fronti_subsampling_min_fronti=10,
				ray_accum_period=1,
				ray_accum_phase=0,
				angle_bin_size=30.0,
				ray_erosion=1,
				ray_tracing=True,
				global_encoding=True,
				zero_depth_mode=False,
				infer_direction=False,
			)
			
			# Set dummy encoder for SemanticRayFrontiersMap
			if self.vdb_mapper is not None:
				self.vdb_mapper.encoder = _ZeroImageEncoder(self.embedding_dim, self.vdb_mapper.device)
			
			self.get_logger().info(f"VDB SemanticRayFrontiersMap initialized (device: {self.vdb_mapper.device})")
			self.get_logger().info(f"Unified mapper settings:")
			self.get_logger().info(f"   Max depth sensing: 1.5m (voxelization and frontiers)")
			self.get_logger().info(f"   Empty count: 8 (stable free space)")
			self.get_logger().info(f"   Occupied count: 7 (confirmed occupancy)")
			self.get_logger().info(f"   Observation weight: 3 (smooth updates)")
			self.get_logger().info(f"   Surface thickness: 3 voxels (robust surfaces)")
			
		except Exception as e:
			self.get_logger().error(f"Failed to initialize VDB mapper: {e}")
			import traceback
			traceback.print_exc()
			raise

	def camera_info_callback(self, msg: CameraInfo):
		if self.camera_intrinsics is None:
			# Update VDB mapper intrinsics
			intrinsics = torch.tensor([
				[msg.k[0], msg.k[1], msg.k[2]],
				[msg.k[3], msg.k[4], msg.k[5]],
				[msg.k[6], msg.k[7], msg.k[8]]
			], dtype=torch.float32)
			
			self.vdb_mapper.intrinsics_3x3 = intrinsics.to(self.vdb_mapper.device)
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
		"""Process incoming semantic hotspot metadata directly in callback."""
		try:
			if not self.enable_semantic_mapping or not self.enable_voxel_mapping:
				return
			
			# Process message directly in background thread (non-blocking)
			threading.Thread(
				target=self._process_single_bridge_message,
				args=(msg.data,),
				daemon=True
			).start()
			
			# Update activity
			self.last_data_time = time.time()
			
		except Exception as e:
			self.get_logger().error(f"Error processing semantic hotspot message: {e}")
			import traceback
			traceback.print_exc()
	
	def _process_single_bridge_message(self, msg_data: str) -> bool:
		"""Process a single bridge message and apply to voxel map by timestamp lookup."""
		try:
			# Parse the JSON message
			time_start = time.time()
			data = json.loads(msg_data)
			json_load_time = time.time() - time_start
			self.get_logger().warn(f"Time taken to load JSON: {json_load_time}")
			if data.get('type') == 'merged_similarity_hotspots':
				return self._process_merged_hotspot_message(data)
			else:
				return False
			
		except Exception as e:
			self.get_logger().error(f"Error processing single bridge message: {e}")
			return False
	
	def _precompute_color_indices(self, merged_mask: np.ndarray, vlm_info: dict) -> dict:
		"""Pre-compute pixel indices for each color once (fixes bottleneck #1).
		
		Returns dict mapping vlm_answer -> (v_coords, u_coords) numpy arrays.
		"""
		color_to_indices = {}
		h, w = merged_mask.shape[:2]
		
		# Vectorized approach: flatten and find matches
		mask_flat = merged_mask.reshape(-1, 3)  # (H*W, 3)
		
		for vlm_answer, info in vlm_info.items():
			color = np.array(info.get('color', [0, 0, 0]), dtype=np.uint8)
			# Vectorized comparison (much faster than per-pixel loop)
			matches = np.all(mask_flat == color, axis=1)
			if np.any(matches):
				indices = np.where(matches)[0]
				v_coords = indices // w
				u_coords = indices % w
				color_to_indices[vlm_answer] = (v_coords, u_coords)
			else:
				color_to_indices[vlm_answer] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
		
		return color_to_indices
	
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
			start = time.time()
			
			# Lookup merged mask image by timestamp
			merged_mask = self._lookup_mask(rgb_timestamp)
			mask_lookup_time = time.time() - start
			self.get_logger().warn(f"Time taken to lookup mask: {mask_lookup_time}")
			if merged_mask is None:
				self.get_logger().warn(f"No matching hotspot mask found for timestamp {rgb_timestamp:.6f}")
				return False
			
			# Lookup closest depth frame and pose by timestamp
			depth_image, pose_msg, used_ts = self._lookup_depth_and_pose(rgb_timestamp)
			depth_lookup_time = time.time() - start - mask_lookup_time
			self.get_logger().warn(f"Time taken to lookup depth: {depth_lookup_time}")
			if depth_image is None or pose_msg is None:
				self.get_logger().warn(f"No matching depth/pose found for timestamp {rgb_timestamp:.6f}")
				return False
			
			# OPTIMIZATION: Pre-compute color indices once (fixes bottleneck #1)
			color_to_indices = self._precompute_color_indices(merged_mask, vlm_info)
			
			# Process each VLM answer using pre-computed indices
			processed_count = 0
			for vlm_answer, info in vlm_info.items():
				if vlm_answer not in color_to_indices:
					continue
				
				v_coords, u_coords = color_to_indices[vlm_answer]
				if len(v_coords) == 0:
					continue
				
				# Create sparse mask directly from indices (much faster than full image comparison)
				h, w = merged_mask.shape[:2]
				vlm_mask = np.zeros((h, w), dtype=bool)
				vlm_mask[v_coords, u_coords] = True
				
				vlm_mask_time = time.time() - start - mask_lookup_time - depth_lookup_time
				self.get_logger().debug(f"Time taken to create vlm mask: {vlm_mask_time:.4f}s (optimized)")
				
				success = self._process_hotspot_with_depth(
					vlm_mask, pose_msg, depth_image, vlm_answer, 
					info.get('hotspot_threshold', 0.6), 
					{'hotspot_pixels': info.get('hotspot_pixels', 0)}, 
					rgb_timestamp, used_ts, is_narration, buffer_id
				)
				hotspot_processing_time = time.time() - start - mask_lookup_time - depth_lookup_time - vlm_mask_time
				self.get_logger().debug(f"Time taken to process hotspot: {hotspot_processing_time:.4f}s")
				if success:
					processed_count += 1
					if len(vlm_info) == 1:
						self.get_logger().info(f"NARRATION HOTSPOT PROCESSED: '{vlm_answer}' with {info.get('hotspot_pixels', 0)} pixels")
			self.get_logger().info(f"Processed {processed_count}/{len(vlm_info)} VLM answers from merged hotspots")
			total_time = time.time() - start
			self.get_logger().warn(f"Total time taken to process merged hotspot: {total_time}")
			return processed_count > 0
			
		except Exception as e:
			self.get_logger().error(f"Error processing merged hotspot message: {e}")
			return False
	
	def _lookup_depth_and_pose(self, target_ts: float):
		"""Find closest depth frame and pose to target timestamp within buffer window using binary search."""
		with self.sync_lock:
			# Optimized binary search for depth
			best_depth, best_depth_ts = self._binary_search_closest(
				self.depth_buffer, target_ts, self.sync_buffer_duration
			)
			
			# Optimized binary search for pose
			best_pose, best_pose_ts = self._binary_search_closest(
				self.pose_buffer, target_ts, self.sync_buffer_duration
			)
			
			# Return if both found
			if best_depth is not None and best_pose is not None:
				return best_depth, best_pose, (best_depth_ts, best_pose_ts)
			
			return None, None, (None, None)
	
	def _binary_search_closest(self, buffer: List, target_ts: float, max_dt: float):
		"""Binary search to find closest timestamp entry in sorted buffer. Returns (data, timestamp) or (None, None).
		
		Optimized O(log n) lookup using numpy's searchsorted for better performance.
		"""
		if not buffer:
			return None, None
		
		# Fast path: if buffer is very small, linear search is faster
		if len(buffer) < 5:
			best_data = None
			best_ts = None
			best_dt = float('inf')
			for ts, data in buffer:
				dt = abs(ts - target_ts)
				if dt < best_dt and dt <= max_dt:
					best_dt = dt
					best_data = data
					best_ts = ts
			return best_data, best_ts
		
		# Use numpy for efficient timestamp extraction and binary search
		# Convert to numpy array once - much faster than list comprehension for large buffers
		timestamps = np.array([ts for ts, _ in buffer], dtype=np.float64)
		
		# Use numpy's searchsorted - optimized C implementation, faster than bisect for numpy arrays
		idx = np.searchsorted(timestamps, target_ts, side='left')
		
		# Check candidate positions: idx-1, idx (if exists)
		best_data = None
		best_ts = None
		best_dt = float('inf')
		
		# Check element at idx (if exists)
		if idx < len(buffer):
			ts, data = buffer[idx]
			dt = abs(ts - target_ts)
			if dt < best_dt and dt <= max_dt:
				best_dt = dt
				best_data = data
				best_ts = ts
		
		# Check element before idx (if exists)
		if idx > 0:
			ts, data = buffer[idx - 1]
			dt = abs(ts - target_ts)
			if dt < best_dt and dt <= max_dt:
				best_dt = dt
				best_data = data
				best_ts = ts
		
		return best_data, best_ts
	
	def _lookup_mask(self, target_ts: float) -> Optional[np.ndarray]:
		"""Find closest merged mask image to target timestamp within buffer window using binary search."""
		with self.sync_lock:
			best_mask, _ = self._binary_search_closest(
				self.mask_buffer, target_ts, self.sync_buffer_duration
			)
			return best_mask
	
	def _process_hotspot_with_depth(self, mask: np.ndarray, pose: PoseStamped, depth_m: np.ndarray,
								   vlm_answer: str, threshold: float, stats: dict, rgb_ts: float, used_ts: tuple, is_narration: bool, buffer_id: str = None) -> bool:
		"""Project hotspot mask using matched depth and pose; update voxel map and semantics."""
		try:
			if self.camera_intrinsics is None:
				self.get_logger().warn("No camera intrinsics available for hotspot processing")
				return False
			
			# Get hotspot pixel coordinates
			v_coords, u_coords = np.where(mask > 0)
			if len(u_coords) == 0:
				self.get_logger().warn("No hotspot pixels found in mask")
				return False
			
			# Extract only hotspot pixels from depth (no full array creation)
			h, w = mask.shape
			if depth_m.shape != (h, w):
				depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
				depth_values = depth_resized[v_coords, u_coords]
			else:
				depth_values = depth_m[v_coords, u_coords]
			
			# Filter valid depth values
			valid_mask = np.isfinite(depth_values) & (depth_values > 0.0)
			if not np.any(valid_mask):
				self.get_logger().warn("No valid depth values in hotspot")
				return False
			
			# Only process valid hotspot pixels directly (skip meshgrid)
			u_valid = u_coords[valid_mask]
			v_valid = v_coords[valid_mask]
			z_valid = depth_values[valid_mask]
			
			# Convert to world points using only hotspot pixels
			points_world = self._depth_to_world_points_sparse(u_valid, v_valid, z_valid, self.camera_intrinsics, pose)
			if points_world is None or len(points_world) == 0:
				self.get_logger().warn("Failed to project hotspot points to world coordinates")
				return False
			
			if points_world is None or len(points_world) == 0:
				self.get_logger().warn("Failed to project hotspot points to world coordinates")
				return False
			
			# Range filter using squared distance (faster than norm)
			origin = self._pose_position(pose)
			diff = points_world - origin
			dist_sq = np.sum(diff * diff, axis=1)
			min_range_sq = float(self.min_range) * float(self.min_range)
			max_range_sq = 10.0 * 10.0
			mask_range = (dist_sq >= min_range_sq) & (dist_sq <= max_range_sq)
			points_world_near = points_world[mask_range]
			if points_world_near.size == 0:
				self.get_logger().debug("Hotspot points beyond semantic max_range; skipping semantic voxel update but continuing with ray casting")
			
			# GP fitting for narration hotspots (background thread)
			if is_narration and points_world_near.size > 0:
				buffer_dir, pcd_path = self.save_points_to_latest_nested_subfolder("/home/navin/ros2_ws/src/buffers", points_world_near)
				if buffer_dir is not None and GP_HELPER_AVAILABLE:
					voxelized_points = self._voxelize_pointcloud(points_world_near, float(self.voxel_resolution), max_points=200)
					self._check_and_start_gp_fit_if_ready(buffer_dir, voxelized_points, vlm_answer)

			# Build depth image with only hotspot pixels (same as tmp.py) - prepare for threading
			h, w = mask.shape
			depth_hot = np.zeros((h, w), dtype=np.float32)
			if depth_m.shape != (h, w):
				depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
				depth_hot[mask > 0] = depth_resized[mask > 0]
			else:
				depth_hot[mask > 0] = depth_m[mask > 0]
			
			# Update VDB map with semantic hotspot using masked depth - run in separate thread (optimized)
			mask_copy = mask.copy()
			depth_hot_copy = depth_hot.copy()
			pose_copy = PoseStamped()
			pose_copy.header = pose.header
			pose_copy.pose = pose.pose
			
			threading.Thread(
				target=self._update_semantic_vdb_mapping,
				args=(mask_copy, depth_hot_copy, pose_copy),
				daemon=True
			).start()

			# Semantic label application - run in separate thread to avoid blocking
			if points_world_near.size > 0:
				points_copy = points_world_near.copy()
				threading.Thread(
					target=self._update_semantic_voxels,
					args=(points_copy, vlm_answer, threshold, stats, is_narration),
					daemon=True
				).start()
				near_count = points_world_near.shape[0]
			else:
				near_count = 0

			hotspot_type = "NARRATION" if is_narration else "OPERATIONAL"
			self.get_logger().info(
				f"Applied hotspot processing for '{vlm_answer}' (within_range={near_count}, rgb_ts={rgb_ts:.6f}, depth_ts={used_ts[0]}, pose_ts={used_ts[1]}, type={hotspot_type})"
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
		
		OPTIMIZED: Uses vectorized numpy operations instead of Python loops.
		"""
		if len(points) == 0:
			return points
		
		# Convert points to voxel coordinates
		voxel_coords = np.floor(points / voxel_size).astype(np.int32)
		
		# Find unique voxels and their inverse indices
		unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
		
		# OPTIMIZED: Vectorized centroid computation using bincount approach
		# This avoids Python loops and boolean masking per voxel
		num_voxels = len(unique_voxels)
		
		# Compute centroids using cumsum trick for each coordinate dimension
		voxelized_points = np.zeros((num_voxels, points.shape[1]), dtype=points.dtype)
		voxel_counts = np.bincount(inverse_indices, minlength=num_voxels)
		
		# For each dimension, compute sum of points per voxel, then divide by count
		for dim in range(points.shape[1]):
			# Sum points per voxel using bincount
			sums = np.bincount(inverse_indices, weights=points[:, dim], minlength=num_voxels)
			# Avoid division by zero
			nonzero_mask = voxel_counts > 0
			voxelized_points[nonzero_mask, dim] = sums[nonzero_mask] / voxel_counts[nonzero_mask]
		
		# Random sampling if too many points (deterministic seed for reproducibility)
		if len(voxelized_points) > max_points:
			# Use deterministic sampling instead of random for reproducibility
			step = len(voxelized_points) / max_points
			indices = np.arange(0, len(voxelized_points), step, dtype=np.int32)[:max_points]
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

		# Use cached subfolder if available and recent
		current_time = time.time()
		if (self._cached_latest_subfolder is not None and 
		    os.path.exists(self._cached_latest_subfolder) and
		    (current_time - self._cached_subfolder_time) < self._subfolder_cache_ttl):
			latest_subfolder2 = self._cached_latest_subfolder
		else:
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
			# Cache the result
			self._cached_latest_subfolder = latest_subfolder2
			self._cached_subfolder_time = current_time		
		# Step 3: save voxelized PCD inside latest_subfolder2
		save_path = os.path.join(latest_subfolder2, filename)
		arr = np.mean(voxelized_points, axis=0)
		with open(os.path.join(latest_subfolder2, "mean_cause.json"), "w") as f:
			json.dump(arr.tolist(), f)
		_save_pcd(voxelized_points, save_path)
		return latest_subfolder2, save_path


	def _check_and_start_gp_fit_if_ready(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
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
			self._start_background_gp_fit(buffer_dir, pointcloud_xyz, cause_name)
			
		except Exception as e:
			self.get_logger().warn(f"Error checking GP fit readiness: {e}")

	def _start_background_gp_fit(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
		"""Start GP fitting in a background thread if not already running."""
		try:
			with self.gp_fit_lock:
				if self.gp_fitting_active:
					self.get_logger().info("GP fit already running; skipping new request")
					return
				self.gp_fitting_active = True
			args = (buffer_dir, np.array(pointcloud_xyz, dtype=np.float32), cause_name)
			threading.Thread(target=self._run_gp_fit_task, args=args, daemon=True).start()
		except Exception as e:
			self.get_logger().warn(f"Failed to start GP fit thread: {e}")

	def _run_gp_fit_task(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
		"""Run GP fitting and save parameters to buffer directory."""
		result = None
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
					'r2_score': fit.get('r2_score'),
					'sigma2': fit.get('sigma2'),  # Noise variance for uncertainty
					'nll': fit.get('nll')  # Negative log-likelihood
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
			
			# Update cause registry with GP params if cause_name is available
			if cause_name and result and 'fit' in result:
				self._update_registry_gp_params(cause_name, buffer_dir, result['fit'])
			
		except Exception as e:
			self.get_logger().error(f"GP fit task failed: {e}")
			import traceback
			traceback.print_exc()
		finally:
			with self.gp_fit_lock:
				self.gp_fitting_active = False
			
			# Store the latest GP parameters and training data for global use
			if result and 'fit' in result:
				self.global_gp_params = result['fit']
				self.global_nominal_points = result.get('nominal_used')  # Store for uncertainty computation
				self.global_disturbances = result.get('disturbances')  # Store for uncertainty computation
				self.get_logger().info(f"Updated global GP parameters: lxy={self.global_gp_params.get('lxy', 0):.3f}, lz={self.global_gp_params.get('lz', 0):.3f}, A={self.global_gp_params.get('A', 0):.3f}")
			
			# After GP fitting is complete, create and publish visualization
			self._create_and_publish_gp_visualization(buffer_dir, result)

	def _create_and_publish_gp_visualization(self, buffer_dir: str, result: Optional[Dict] = None):
		"""Create and publish GP field visualization and epistemic uncertainty field."""
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
			
			# Compute and publish epistemic uncertainty field
			if result is not None:
				nominal_points = result.get('nominal_used')
				disturbances = result.get('disturbances')
				if nominal_points is not None and disturbances is not None and len(nominal_points) > 0:
					uncertainty_std = self._compute_epistemic_uncertainty(
						grid_points, cause_points, fit_params, nominal_points, disturbances
					)
					if uncertainty_std is not None:
						uncertainty_cloud = self._create_uncertainty_pointcloud(grid_points, uncertainty_std)
						if uncertainty_cloud:
							self.gp_uncertainty_pub.publish(uncertainty_cloud)
							self.get_logger().info(f"Published GP epistemic uncertainty field: {len(grid_points)} points")
			
			self.get_logger().info(f"Published cause.pcd GP visualization + costmap: {len(grid_points)} points, {len(cause_points)} cause voxels, radius={adaptive_radius:.2f}m (SAME method as semantic voxels)")
			
		except Exception as e:
			self.get_logger().error(f"Error creating GP visualization: {e}")
			import traceback
			traceback.print_exc()
	
	def _update_registry_gp_params(self, cause_name: str, buffer_dir: str, fit: Dict):
		"""Update cause registry with GP params via ROS topic query.
		
		First queries registry by name to get vec_id, then uses vec_id for update.
		This ensures we're working with embedding-indexed entries, not text names.
		"""
		try:
			# Step 1: Query registry by name to get vec_id
			query_get = {
				'type': 'get_by_name',
				'name': cause_name
			}
			
			# Store callback to handle response (use default args to avoid closure issues)
			query_id = f"gp_update_{time.time()}_{id(self)}"
			def make_callback(bd, f, cn):
				return lambda resp: self._handle_gp_update_response(resp, bd, f, cn)
			with self.registry_query_lock:
				self.pending_registry_queries[query_id] = make_callback(buffer_dir, fit, cause_name)
			
			query_get['query_id'] = query_id
			query_msg = String(data=json.dumps(query_get))
			self.registry_query_pub.publish(query_msg)
			self.get_logger().info(f"Querying registry for vec_id of '{cause_name}' before GP update")
			
		except Exception as e:
			self.get_logger().warn(f"Failed to query registry for '{cause_name}': {e}")
	
	def _handle_gp_update_response(self, response: Dict, buffer_dir: str, fit: Dict, cause_name: str):
		"""Handle registry query response and update GP params using vec_id."""
		try:
			if not response.get('success'):
				self.get_logger().warn(f"Registry query failed for '{cause_name}': {response.get('message')}")
				return
			
			vec_id = response.get('vec_id')
			if not vec_id:
				self.get_logger().warn(f"No vec_id in registry response for '{cause_name}'")
				return
			
			# Step 2: Update GP params using vec_id (embedding-indexed)
			buffer_id = os.path.basename(buffer_dir) if buffer_dir else None
			gp_params = {
				'lxy': fit.get('lxy'),
				'lz': fit.get('lz'),
				'A': fit.get('A'),
				'b': fit.get('b'),
				'mse': fit.get('mse'),
				'rmse': fit.get('rmse'),
				'mae': fit.get('mae'),
				'r2_score': fit.get('r2_score'),
				'timestamp': time.time(),
				'buffer_id': buffer_id
			}
			
			query_set = {
				'type': 'set_gp',
				'vec_id': vec_id,  # Use vec_id instead of name
				'gp_params': gp_params
			}
			
			query_msg = String(data=json.dumps(query_set))
			self.registry_query_pub.publish(query_msg)
			self.get_logger().info(f"Published GP params update to registry for vec_id '{vec_id}' (cause: '{cause_name}')")
			
		except Exception as e:
			self.get_logger().warn(f"Failed to update registry GP params: {e}")
	
	def _handle_registry_response(self, msg):
		"""Handle registry query responses and route to appropriate callbacks."""
		try:
			response = json.loads(msg.data)
			query_id = response.get('query_id')
			
			if query_id and query_id in self.pending_registry_queries:
				# Route to callback
				callback = self.pending_registry_queries.pop(query_id)
				callback(response)
			else:
				# No callback, just log
				if response.get('success'):
					self.get_logger().debug(f"Registry query succeeded: {response.get('message', 'OK')}")
				else:
					self.get_logger().warn(f"Registry query failed: {response.get('message', 'Unknown error')}")
		except Exception as e:
			self.get_logger().warn(f"Error handling registry response: {e}")
	
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
			
			# Compute and publish epistemic uncertainty if training data is available
			if (self.global_nominal_points is not None and self.global_disturbances is not None and 
			    len(self.global_nominal_points) > 0 and len(self.global_disturbances) > 0):
				uncertainty_std = self._compute_epistemic_uncertainty(
					grid_points, semantic_points, self.global_gp_params,
					self.global_nominal_points, self.global_disturbances
				)
				if uncertainty_std is not None:
					uncertainty_cloud = self._create_uncertainty_pointcloud(grid_points, uncertainty_std)
					if uncertainty_cloud:
						self.gp_uncertainty_pub.publish(uncertainty_cloud)
			
			self.get_logger().info(f"Published GP visualization + costmap: {len(grid_points)} points, {len(semantic_voxels)} voxels, radius={adaptive_radius:.2f}m")
			
		except Exception as e:
			self.get_logger().error(f"Error updating semantic GP visualization: {e}")
			import traceback
			traceback.print_exc()
	
	def _calculate_adaptive_radius(self, semantic_points: np.ndarray) -> float:
		"""Calculate adaptive radius using nearest neighbor (O(N log N) instead of O(N²))."""
		try:
			if len(semantic_points) < 2:
				return self.base_radius
			
			# Use nearest neighbor distance (much faster than all pairs - fixes O(N²) bottleneck)
			try:
				from scipy.spatial import cKDTree
				tree = cKDTree(semantic_points)
				# Query for 2 nearest neighbors (self + 1 neighbor)
				k = min(2, len(semantic_points))
				distances, _ = tree.query(semantic_points, k=k)
				
				if distances.ndim == 2:
					# Get distance to nearest neighbor (skip self)
					nn_distances = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
				else:
					# Single point case
					nn_distances = distances if isinstance(distances, np.ndarray) else np.array([distances])
				
				avg_distance = np.mean(nn_distances)
				
			except ImportError:
				# Fallback: sample subset of points for speed if scipy not available
				if len(semantic_points) > 100:
					sample_idx = np.random.choice(len(semantic_points), 100, replace=False)
					sample_points = semantic_points[sample_idx]
					# Compute pairwise distances for sample only
					from scipy.spatial.distance import pdist
					distances = pdist(sample_points)
					avg_distance = np.mean(distances)
				else:
					# Small dataset, use original approach
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
		"""FAST filtering using KD-tree (O(N log M) instead of O(N*M) full distance matrix)."""
		try:
			if len(grid_points) == 0 or len(voxel_positions) == 0:
				return grid_points
			
			# Use KD-tree for O(N log M) instead of O(N*M) full distance matrix
			try:
				from scipy.spatial import cKDTree
				# Build KD-tree once (O(M log M))
				tree = cKDTree(voxel_positions)
				
				# Query all grid points (O(N log M))
				distances, _ = tree.query(grid_points, k=1)
				
				# Filter points within max_distance
				mask = distances <= max_distance
				filtered_points = grid_points[mask]
				
				return filtered_points
				
			except ImportError:
				# Fallback: chunked computation to avoid large memory allocation
				chunk_size = 10000
				mask = np.zeros(len(grid_points), dtype=bool)
				
				for i in range(0, len(grid_points), chunk_size):
					chunk = grid_points[i:i+chunk_size]
					distances = np.linalg.norm(
						chunk[:, np.newaxis, :] - voxel_positions[np.newaxis, :, :], 
						axis=2
					)
					min_distances = np.min(distances, axis=1)
					mask[i:i+chunk_size] = min_distances <= max_distance
				
				return grid_points[mask]
			
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
	
	def _compute_epistemic_uncertainty(self, grid_points: np.ndarray, cause_points: np.ndarray, 
	                                   fit_params: dict, nominal_points: np.ndarray, 
	                                   disturbances: np.ndarray) -> Optional[np.ndarray]:
		"""
		Compute epistemic uncertainty (standard deviation) at query points using Bayesian linear regression.
		
		Uncertainty = sqrt(sigma² * (1 + v^T * (X^T X)^-1 * v))
		where v = [phi(x), 1] is the feature vector at query point x.
		
		This captures uncertainty in A and b parameters given fixed lxy, lz.
		
		Args:
			grid_points: (N, 3) query points
			cause_points: (M, 3) cause points
			fit_params: Dictionary with lxy, lz, sigma2
			nominal_points: (K, 3) training points where disturbances were measured
			disturbances: (K,) observed disturbance magnitudes
		
		Returns:
			(N,) predictive standard deviation (uncertainty)
		"""
		try:
			lxy = fit_params.get('lxy')
			lz = fit_params.get('lz')
			sigma2_noise = fit_params.get('sigma2')
			
			if lxy is None or lz is None or sigma2_noise is None:
				self.get_logger().warn("Missing GP parameters for uncertainty computation")
				return None
			
			if len(nominal_points) == 0 or len(disturbances) == 0:
				self.get_logger().warn("No training data for uncertainty computation")
				return None
			
			# 1. Compute training feature matrix X
			phi_train = self._sum_of_anisotropic_rbf_fast(nominal_points, cause_points, lxy, lz)
			X_train = np.column_stack([phi_train, np.ones(len(phi_train))])  # (K, 2)
			
			# 2. Compute parameter covariance: Cov(A, b) = sigma² * (X^T X)^-1
			XtX = X_train.T @ X_train
			XtX[0, 0] += 1e-6  # Regularization for stability
			XtX[1, 1] += 1e-6
			
			try:
				XtX_inv = np.linalg.inv(XtX)
				Cov_params = sigma2_noise * XtX_inv  # (2, 2)
			except np.linalg.LinAlgError:
				# Fallback: just return noise level
				return np.full(len(grid_points), np.sqrt(sigma2_noise))
			
			# 3. Compute phi at query points
			phi_query = self._sum_of_anisotropic_rbf_fast(grid_points, cause_points, lxy, lz)
			
			# 4. Epistemic variance: v^T * Cov * v where v = [phi, 1]
			epistemic_var = (Cov_params[0, 0] * phi_query**2 + 
			                 2 * Cov_params[0, 1] * phi_query + 
			                 Cov_params[1, 1])
			
			# 5. Total variance = epistemic + aleatoric
			total_variance = epistemic_var + sigma2_noise
			
			# Return standard deviation
			return np.sqrt(np.maximum(total_variance, 0.0))
			
		except Exception as e:
			self.get_logger().error(f"Error computing epistemic uncertainty: {e}")
			import traceback
			traceback.print_exc()
			return None
	
	def _create_uncertainty_pointcloud(self, grid_points: np.ndarray, uncertainty_std: np.ndarray) -> Optional[PointCloud2]:
		"""
		Create point cloud for epistemic uncertainty visualization.
		
		Similar to _create_costmap_pointcloud but for uncertainty values instead of disturbance.
		
		Args:
			grid_points: (N, 3) query points
			uncertainty_std: (N,) uncertainty standard deviation values
		
		Returns:
			PointCloud2 message with XYZ + uncertainty values
		"""
		try:
			if len(grid_points) == 0 or len(uncertainty_std) == 0:
				return None
			
			# Use actual uncertainty std values for visualization
			uncertainty_values = uncertainty_std.astype(np.float32)
			
			# Create PointCloud2 message with XYZ + uncertainty values
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.map_frame
			
			# Create structured array with XYZ + uncertainty value
			cloud_data_combined = np.empty(len(grid_points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), 
				('uncertainty', np.float32)
			])
			
			# Fill in the data
			cloud_data_combined['x'] = grid_points[:, 0]
			cloud_data_combined['y'] = grid_points[:, 1]
			cloud_data_combined['z'] = grid_points[:, 2]
			cloud_data_combined['uncertainty'] = uncertainty_values
			
			# Create PointCloud2 message
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields - XYZ + uncertainty value
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='uncertainty', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, uncertainty)
			cloud_msg.width = len(grid_points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data_combined.tobytes()
			
			self.get_logger().info(f"Created uncertainty point cloud: min={uncertainty_values.min():.3f}, max={uncertainty_values.max():.3f}")
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating uncertainty point cloud: {e}")
			return None
	
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
		"""Return stored semantic voxel positions without further processing."""
		try:
			semantic_voxel_positions: List[np.ndarray] = []
			with self.semantic_voxels_lock:
				for semantic_info in self.semantic_voxels.values():
					voxel_position = semantic_info.get('position')
					if voxel_position is not None:
						semantic_voxel_positions.append(voxel_position)
			return semantic_voxel_positions
		except Exception as e:
			self.get_logger().error(f"Error getting semantic voxels: {e}")
			return []
	
	def _get_neighboring_voxel_keys(self, voxel_key: tuple) -> List[tuple]:
		"""Get voxel key and its 26 neighbors (3x3x3 cube)."""
		vx, vy, vz = voxel_key
		neighbors = []
		for dx in [-1, 0, 1]:
			for dy in [-1, 0, 1]:
				for dz in [-1, 0, 1]:
					neighbors.append((vx + dx, vy + dy, vz + dz))
		return neighbors
	
	def _increment_spatial_observation_counts(self, voxel_key: tuple, vlm_answer: str, frame_id: int, timestamp: float):
		"""OPTIMIZED: Incrementally update spatial observation counts for all 27 neighbors (including self).
		
		This maintains pre-computed counts so threshold checks are O(1) instead of O(neighbors * observations).
		"""
		neighbors = self._get_neighboring_voxel_keys(voxel_key)
		current_time = time.time()
		
		for nkey in neighbors:
			key = (nkey, vlm_answer)
			if key not in self.spatial_observation_counts:
				self.spatial_observation_counts[key] = {
					'count': 0,
					'unique_frames': set(),
					'last_update': current_time
				}
			
			entry = self.spatial_observation_counts[key]
			
			# Increment count (for narration)
			entry['count'] += 1
			
			# Add unique frame (for operational)
			if frame_id is not None:
				entry['unique_frames'].add(frame_id)
			
			entry['last_update'] = current_time
	
	def _cleanup_old_spatial_counts(self, current_time: float):
		"""Periodically cleanup old entries from spatial_observation_counts."""
		# Only cleanup if dict is getting large (avoid overhead on every call)
		if len(self.spatial_observation_counts) < 1000:
			return
		
		# Remove entries older than max_age
		keys_to_remove = []
		for key, entry in self.spatial_observation_counts.items():
			if (current_time - entry['last_update']) > self.semantic_observation_max_age:
				keys_to_remove.append(key)
		
		for key in keys_to_remove:
			del self.spatial_observation_counts[key]
	
	def _get_observation_count_fast(self, voxel_key: tuple, vlm_answer: str) -> int:
		"""OPTIMIZED: Fast O(1) lookup for observation count with spatial support."""
		key = (voxel_key, vlm_answer)
		entry = self.spatial_observation_counts.get(key)
		if entry is None:
			return 0
		
		# Check if entry is still valid (not expired)
		current_time = time.time()
		if (current_time - entry['last_update']) > self.semantic_observation_max_age:
			return 0
		
		return entry['count']
	
	def _get_unique_frames_count_fast(self, voxel_key: tuple, vlm_answer: str) -> int:
		"""OPTIMIZED: Fast O(1) lookup for unique frames count with spatial support."""
		key = (voxel_key, vlm_answer)
		entry = self.spatial_observation_counts.get(key)
		if entry is None:
			return 0
		
		# Check if entry is still valid (not expired)
		current_time = time.time()
		if (current_time - entry['last_update']) > self.semantic_observation_max_age:
			return 0
		
		return len(entry['unique_frames'])
	
	def _apply_semantic_labels_to_voxels(self, points_world: np.ndarray, vlm_answer: str,
									 threshold: float, stats: dict, is_narration: bool = False):
		"""Apply semantic labels with temporal+spatial confirmation."""
		try:
			current_time = time.time()
			
			# Increment frame counter for operational hotspots
			if not is_narration:
				self.frame_counter += 1
			
			# Vectorized voxel key computation (much faster than loop)
			voxel_coords = np.floor(points_world / self.voxel_resolution).astype(np.int32)
			voxel_keys = set(tuple(coord) for coord in voxel_coords)
			
			# Add observations (cleanup only when list gets too long to avoid per-voxel overhead)
			frame_id = 0 if is_narration else self.frame_counter
			obs_data = {
				'vlm_answer': vlm_answer,
				'timestamp': current_time,
				'frame_id': frame_id,
				'similarity': stats.get('avg_similarity', threshold + 0.1)
			}
			
			# OPTIMIZED: Incrementally update spatial observation counts for all voxels
			# This pre-computes counts so threshold checks are O(1) instead of O(neighbors * observations)
			for voxel_key in voxel_keys:
				if voxel_key not in self.semantic_voxel_observations:
					self.semantic_voxel_observations[voxel_key] = []
				
				self.semantic_voxel_observations[voxel_key].append(obs_data)
				
				# Incrementally update spatial counts for all 27 neighbors (including self)
				# This makes threshold checks O(1) instead of scanning all neighbors
				self._increment_spatial_observation_counts(
					voxel_key, vlm_answer, 
					frame_id if not is_narration else None,  # Only track frames for operational
					current_time
				)
				
				# Only cleanup if list is getting long (reduces overhead)
				if len(self.semantic_voxel_observations[voxel_key]) > 20:
					self.semantic_voxel_observations[voxel_key] = [
						obs for obs in self.semantic_voxel_observations[voxel_key]
						if (current_time - obs['timestamp']) <= self.semantic_observation_max_age
					]
			
			# Periodic cleanup of old spatial counts (only if dict is large)
			self._cleanup_old_spatial_counts(current_time)
			
			# Apply different confirmation logic based on hotspot type
			confirmation_threshold = self.narration_confirmation_threshold if is_narration else self.operational_confirmation_threshold
			
			# NON-BLOCKING MULTI-FRAME CONFIRMATION:
			# - Observations are added immediately (non-blocking)
			# - Frame counts are tracked incrementally as new frames arrive
			# - Voxels are confirmed automatically when threshold is reached (no waiting/blocking)
			# - This provides noise rejection while maintaining low latency
			with self.semantic_voxels_lock:
				confirmed_count = 0
				for voxel_key in voxel_keys:
					if is_narration:
						# FAST: O(1) lookup instead of scanning 27 neighbors
						observation_count = self._get_observation_count_fast(voxel_key, vlm_answer)
						meets_threshold = observation_count >= confirmation_threshold
						confidence = observation_count
					else:
						# FAST: O(1) lookup - checks unique frames seen so far (incremental, non-blocking)
						unique_frames = self._get_unique_frames_count_fast(voxel_key, vlm_answer)
						meets_threshold = unique_frames >= confirmation_threshold
						confidence = unique_frames
					
					if meets_threshold:
						similarity_score = stats.get('avg_similarity', threshold + 0.1)
						
						semantic_info = {
							'vlm_answer': vlm_answer,
							'similarity': similarity_score,
							'threshold_used': threshold,
							'detection_method': 'binary_threshold_hotspot',
							'depth_used': True,
							'timestamp': current_time,
							'position': self._get_voxel_center_from_key(voxel_key),
							'confidence': confidence,
							'is_narration': is_narration
						}
						self.semantic_voxels[voxel_key] = semantic_info
						confirmed_count += 1
			
			hotspot_type = "narration" if is_narration else "operational"
			self.get_logger().info(
				f"Semantic observation ({hotspot_type}): {len(voxel_keys)} voxels for '{vlm_answer}', "
				f"{confirmed_count} newly confirmed (threshold: {confirmation_threshold})"
			)
			
		except Exception as e:
			self.get_logger().error(f"Error applying semantic labels to voxels: {e}")
	
	def _get_voxel_key_from_point(self, point) -> tuple:
		"""Convert world point to voxel key. Handles both numpy arrays and torch tensors."""
		# Convert torch tensor to numpy if needed
		if torch.is_tensor(point):
			point = point.cpu().numpy()
		
		# Ensure it's a numpy array
		if not isinstance(point, np.ndarray):
			point = np.array(point)
		
		voxel_coords = np.floor(point / self.voxel_resolution).astype(np.int32)
		return tuple(voxel_coords)
		
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
			
			# Regular VDB occupancy mapping: run in separate thread to avoid blocking
			if self.latest_pose is not None:
				# Make a copy of depth and pose for thread safety
				depth_copy = depth_m.copy()
				pose_copy = PoseStamped()
				pose_copy.header = self.latest_pose.header
				pose_copy.pose = self.latest_pose.pose
				
				# Run regular mapping in background thread
				threading.Thread(
					target=self._update_regular_mapping,
					args=(depth_copy, pose_copy),
					daemon=True
				).start()
			
		except Exception as e:
			self.get_logger().error(f"Error storing depth frame: {e}")

		# Activity update
		self.last_data_time = time.time()

		# Periodic publishing (includes deferred frontier computation)
		self._periodic_publishing()
		
		# Compute regular frontiers periodically (not on every depth frame to reduce contention)
		now = time.time()
		if not hasattr(self, 'last_frontier_compute_time'):
			self.last_frontier_compute_time = 0.0
		if (now - self.last_frontier_compute_time) >= 0.1:  # Every 0.5s instead of every frame
			self._compute_and_publish_regular_frontiers()
			self.last_frontier_compute_time = now

	def _update_regular_mapping(self, depth_m: np.ndarray, pose: PoseStamped):
		"""Update regular VDB occupancy mapping in a separate thread."""
		try:
			# Convert to torch tensors
			device = self.vdb_mapper.device
			depth_tensor = torch.from_numpy(depth_m).float().unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
			
			# Create dummy RGB (VDB needs it but we're focusing on occupancy)
			h, w = depth_m.shape
			rgb_tensor = torch.zeros(1, 3, h, w, dtype=torch.float32).to(device)
			
			# Convert pose to 4x4 matrix
			pose_4x4 = self._pose_to_4x4_matrix(pose)
			
			# Process with VDB mapper for regular occupancy
			update_info = self.vdb_mapper.process_posed_rgbd(
				rgb_img=rgb_tensor,
				depth_img=depth_tensor,
				pose_4x4=pose_4x4
			)
		except Exception as e:
			self.get_logger().warn(f"VDB mapping error: {e}")

	def _update_semantic_vdb_mapping(self, mask: np.ndarray, depth_hot: np.ndarray, pose: PoseStamped):
		"""Update VDB map with semantic hotspot using masked depth in a separate thread (optimized)."""
		try:
			device = self.vdb_mapper.device
			h, w = mask.shape
			
			# Ensure minimum image size
			if h < 1 or w < 1:
				return
			
			# Create tensors with batch size 1 (critical for indexing)
			depth_tensor = torch.from_numpy(depth_hot).float().unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
			rgb_tensor = torch.zeros(1, 3, h, w, dtype=torch.float32).to(device)
			pose_4x4 = self._pose_to_4x4_matrix(pose)
			
			# Ensure pose_4x4 has correct batch dimension (1x4x4)
			if pose_4x4.dim() == 2:
				pose_4x4 = pose_4x4.unsqueeze(0)
			elif pose_4x4.shape[0] != 1:
				pose_4x4 = pose_4x4[:1]
			
			# Process with VDB mapper for semantic occupancy
			update_info = self.vdb_mapper.process_posed_rgbd(
				rgb_img=rgb_tensor,
				depth_img=depth_tensor,
				pose_4x4=pose_4x4
			)
			
			# Process rays/frontiers with conf_map (same as tmp.py)
			try:
				# Prepare masked depth for rays-only beyond max_range
				depth_for_rays = np.zeros_like(depth_hot, dtype=np.float32)
				masked = (mask > 0)
				if self.camera_intrinsics is not None:
					# Use original depth_m if available, otherwise use depth_hot
					# For rays, we want pixels beyond max_range or missing depth
					masked_depth_vals = depth_hot[masked]
					threshold = float(self.max_range)
					beyond_or_missing = (masked_depth_vals <= 0.0) | (masked_depth_vals > threshold)
					dr = np.zeros_like(masked_depth_vals, dtype=np.float32)
					dr[beyond_or_missing] = np.inf
					depth_for_rays[masked] = dr
					mask_far = np.zeros_like(depth_for_rays, dtype=bool)
					mask_far[masked] = beyond_or_missing
					
					if np.any(mask_far):
						try:
							far_v, far_u = np.where(mask_far)
							fx, fy, cx, cy = self.camera_intrinsics
							fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
							u = far_u.astype(np.float32)
							v = far_v.astype(np.float32)
							dir_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
							dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True) + 1e-9
							pose_mat = self._pose_to_4x4_matrix(pose).detach().cpu().numpy()[0]
							R_world_cam = pose_mat[:3, :3]
							origin_world = pose_mat[:3, 3]
							dir_world = dir_cam @ R_world_cam.T
							dir_world /= np.linalg.norm(dir_world, axis=1, keepdims=True) + 1e-9
							self._latest_pose_rays = (origin_world, dir_world)
						except Exception:
							self._latest_pose_rays = None
					else:
						# Fallback: derive rays from all masked pixels (sampled)
						try:
							if np.any(masked):
								fx, fy, cx, cy = self.camera_intrinsics
								fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
								all_v, all_u = np.where(masked)
								max_samples = 800
								if all_u.shape[0] > max_samples:
									idx = np.random.choice(all_u.shape[0], size=max_samples, replace=False)
									all_u = all_u[idx]
									all_v = all_v[idx]
								u = all_u.astype(np.float32)
								v = all_v.astype(np.float32)
								dir_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
								dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True) + 1e-9
								pose_mat = self._pose_to_4x4_matrix(pose).detach().cpu().numpy()[0]
								R_world_cam = pose_mat[:3, :3]
								origin_world = pose_mat[:3, 3]
								dir_world = dir_cam @ R_world_cam.T
								dir_world /= np.linalg.norm(dir_world, axis=1, keepdims=True) + 1e-9
								self._latest_pose_rays = (origin_world, dir_world)
							else:
								self._latest_pose_rays = None
						except Exception:
							self._latest_pose_rays = None
					
					# Process rays with conf_map to restrict to mask
					rgb_dummy = torch.zeros(1, 3, depth_for_rays.shape[0], depth_for_rays.shape[1], dtype=torch.float32, device=device)
					depth_masked_t = torch.from_numpy(depth_for_rays).float().unsqueeze(0).unsqueeze(0).to(device)
					pose_4x4_rf = self._pose_to_4x4_matrix(pose).to(device)
					conf_map_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
					
					# Update intrinsics if available
					fx, fy, cx, cy = self.camera_intrinsics
					self.vdb_mapper.intrinsics_3x3 = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
					
					# Process rays (no encoding used due to global_encoding=True)
					self.vdb_mapper.process_posed_rgbd(rgb_dummy, depth_masked_t, pose_4x4_rf, conf_map=conf_map_t, feat_img=None)
					# Publish mask-specific frontiers and rays immediately (same as tmp.py)
					self._publish_mask_frontiers_and_rays()
			except Exception as e:
				self.get_logger().warn(f"Mask rays/frontiers processing failed: {e}")
				
		except Exception as e:
			self.get_logger().warn(f"VDB semantic mapping error: {e}")

	def _update_semantic_voxels(self, points_world: np.ndarray, vlm_answer: str, threshold: float, 
								 stats: dict, is_narration: bool):
		"""Update semantic voxel labels in a separate thread (optimized)."""
		try:
			if points_world.size > 0:
				self._apply_semantic_labels_to_voxels(points_world, vlm_answer, threshold, stats, is_narration)
		except Exception as e:
			self.get_logger().warn(f"Semantic voxel update error: {e}")

	def _compute_and_publish_regular_frontiers(self):
		try:
			if self.vdb_mapper is None or self.vdb_mapper.is_empty():
				return
			# Derive active bbox from current occupied points
			pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.vdb_mapper.occ_map_vdb)
			if torch.is_tensor(pc_xyz_occ_size):
				pc_xyz_occ_size = pc_xyz_occ_size.cpu().numpy()
			if pc_xyz_occ_size.shape[0] == 0:
				return
			xyz = pc_xyz_occ_size[:, :3]
			bbox_min = torch.from_numpy(np.min(xyz, axis=0)).float().to(self.vdb_mapper.device)
			bbox_max = torch.from_numpy(np.max(xyz, axis=0)).float().to(self.vdb_mapper.device)
			# Update frontiers for the whole active region
			self.vdb_mapper.update_frontiers(bbox_min, bbox_max)
			# Publish as PointCloud2
			if self.vdb_mapper.frontiers is not None and self.vdb_mapper.frontiers.shape[0] > 0:
				frontiers_np = self.vdb_mapper.frontiers.detach().cpu().numpy()
				cloud = self._create_cloud_xyz(frontiers_np)
				if cloud is not None:
					self.frontiers_pub.publish(cloud)
		except Exception as e:
			self.get_logger().warn(f"Regular frontiers publishing failed: {e}")

	def _create_cloud_xyz(self, points: np.ndarray) -> Optional[PointCloud2]:
		try:
			if points is None or len(points) == 0:
				return None
			pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.map_frame
			cloud_data = np.empty(pts.shape[0], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
			cloud_data['x'] = pts[:, 0]
			cloud_data['y'] = pts[:, 1]
			cloud_data['z'] = pts[:, 2]
			msg = PointCloud2()
			msg.header = header
			msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)
			]
			msg.point_step = 12
			msg.width = pts.shape[0]
			msg.height = 1
			msg.row_step = msg.point_step * msg.width
			msg.is_dense = True
			msg.data = cloud_data.tobytes()
			return msg
		except Exception:
			return None

	def _prune_sync_buffers(self):
		"""Keep only recent entries within sync window."""
		cutoff = time.time() - self.sync_buffer_duration
		# Depth/mask/pose buffers capped by length (heuristic) to bound memory
		max_entries = 50
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
		# BOTTLENECK FUNCTION: This entire function is inefficient for sparse depth images
		# Creates meshgrid for entire image (H x W), then filters - should process only valid pixels
		try:
			fx, fy, cx, cy = intrinsics
			h, w = depth_m.shape
			# BOTTLENECK: np.meshgrid() creates full H x W coordinate arrays even for sparse depth
			# For 640x480 image, creates 307,200 coordinate pairs, most of which are discarded
			u, v = np.meshgrid(np.arange(w), np.arange(h))
			z = depth_m
			# BOTTLENECK: Valid mask computation on full image
			valid = np.isfinite(z) & (z > 0.0)
			if not np.any(valid):
				return None, None, None

			# BOTTLENECK: Indexing full arrays to extract valid pixels (memory intensive)
			u, v, z = u[valid], v[valid], z[valid]
			x = (u - cx) * z / fx
			y = (v - cy) * z / fy
			pts_cam = np.stack([x, y, z], axis=1)

			# Transform to base if needed
			# BOTTLENECK: Matrix multiplication for all points (even if most are zeros)
			if bool(self.pose_is_base_link):
				pts_cam = pts_cam @ (self.R_opt_to_base.T if bool(self.apply_optical_frame_rotation) else np.eye(3, dtype=np.float32))
				pts_cam = pts_cam @ self.R_cam_to_base_extra.T + self.t_cam_to_base_extra

			# World transform
			# BOTTLENECK: Another matrix multiplication for all points
			R_world = self._quat_to_rot(self._pose_quat(pose))
			p_world = self._pose_position(pose)
			pts_world = pts_cam @ R_world.T + p_world
			return pts_world, u, v
		except Exception:
			return None, None, None

	def _depth_to_world_points_sparse(self, u: np.ndarray, v: np.ndarray, z: np.ndarray, intrinsics, pose: PoseStamped):
		"""Optimized version that only processes sparse hotspot pixels (no meshgrid)."""
		try:
			fx, fy, cx, cy = intrinsics
			# Direct computation for sparse pixels
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
			return pts_world
		except Exception:
			return None

	def _create_semantic_colored_cloud(self, max_points: int) -> Optional[PointCloud2]:
		"""Create a colored point cloud that shows both regular occupancy voxels and semantic voxels."""
		try:
			# Get occupancy voxels from VDB
			if self.vdb_mapper.is_empty():
				# If VDB is empty, only show semantic voxels
				with self.semantic_voxels_lock:
					if not self.semantic_voxels:
						return None
					
					points = []
					colors = []
					for voxel_key, semantic_info in self.semantic_voxels.items():
						voxel_center = semantic_info['position']
						if voxel_center is not None:
							points.append(voxel_center)
							vlm_answer = semantic_info.get('vlm_answer', 'unknown')
							color = self._get_vlm_answer_color(vlm_answer)
							colors.append(color)
			else:
				# Get occupancy data from VDB
				pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.vdb_mapper.occ_map_vdb)
				
				# Convert to numpy if it's a torch tensor
				if torch.is_tensor(pc_xyz_occ_size):
					pc_xyz_occ_size = pc_xyz_occ_size.cpu().numpy()
				
				# Filter occupied voxels
				occupied_mask = pc_xyz_occ_size[:, -2] > 0
				occupied_points_data = pc_xyz_occ_size[occupied_mask]
				
				# OPTIMIZED: Copy semantic voxels once with single lock acquisition
				# This avoids thousands of lock acquisitions inside the loop
				with self.semantic_voxels_lock:
					semantic_voxels_copy = dict(self.semantic_voxels)  # Fast shallow copy
				
				# Create point cloud data
				points = []
				colors = []
				semantic_count = 0
				regular_count = 0
				
				# Add regular occupancy voxels
				for point_data in occupied_points_data:
					point = point_data[:3]  # xyz
					voxel_key = self._get_voxel_key_from_point(point)
					
					# FAST: Check semantic voxels from copy (no lock needed)
					if voxel_key in semantic_voxels_copy:
						# Semantic voxel - use VLM answer color
						semantic_info = semantic_voxels_copy[voxel_key]
						vlm_answer = semantic_info.get('vlm_answer', 'unknown')
						color = self._get_vlm_answer_color(vlm_answer)
						semantic_count += 1
					else:
						# Regular occupancy voxel - use gray
						color = [128, 128, 128]
						regular_count += 1
					
					points.append(point)
					colors.append(color)
					
					# Limit points
					if len(points) >= max_points:
						break
				
				# Add any semantic voxels that aren't in VDB occupancy
				# Use the copy we already have (no lock needed)
				for voxel_key, semantic_info in semantic_voxels_copy.items():
					if len(points) >= max_points:
						break
					# Check if this semantic voxel is already added
					voxel_center = semantic_info['position']
					if voxel_center is not None:
						# Simple check: if voxel_key not in occupancy voxels
						# (This is approximate, but good enough for visualization)
						vlm_answer = semantic_info.get('vlm_answer', 'unknown')
						color = self._get_vlm_answer_color(vlm_answer)
						points.append(voxel_center)
						colors.append(color)
						semantic_count += 1
			
			if not points:
				return None
			
			# Log the coloring information
			if 'semantic_count' in locals() and semantic_count > 0:
				self.get_logger().info(f"Creating VDB colored cloud: {semantic_count} semantic voxels (colored by VLM answer), {regular_count if 'regular_count' in locals() else 0} regular voxels (GRAY)")
			else:
				self.get_logger().info(f"Creating VDB colored cloud: {len(points)} voxels")
			
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
	
	def _create_vdb_markers(self, max_markers: int) -> Optional[MarkerArray]:
		"""Create visualization markers from VDB occupancy data."""
		try:
			if self.vdb_mapper.is_empty():
				return None
			
			# Get occupancy data from VDB
			pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.vdb_mapper.occ_map_vdb)
			
			# Convert to numpy if it's a torch tensor
			if torch.is_tensor(pc_xyz_occ_size):
				pc_xyz_occ_size = pc_xyz_occ_size.cpu().numpy()
			
			# Filter occupied voxels
			occupied_mask = pc_xyz_occ_size[:, -2] > 0
			occupied_points = pc_xyz_occ_size[occupied_mask]
			
			if len(occupied_points) == 0:
				return None
			
			# Limit number of markers
			if len(occupied_points) > max_markers:
				indices = np.random.choice(len(occupied_points), max_markers, replace=False)
				occupied_points = occupied_points[indices]
			
			# Create marker array
			marker_array = MarkerArray()
			
			if bool(self.use_cube_list_markers):
				# Create single CUBE_LIST marker for all voxels
				marker = Marker()
				marker.header.frame_id = self.map_frame
				marker.header.stamp = self.get_clock().now().to_msg()
				marker.ns = "vdb_occupancy"
				marker.id = 0
				marker.type = Marker.CUBE_LIST
				marker.action = Marker.ADD
				marker.scale.x = float(self.voxel_resolution)
				marker.scale.y = float(self.voxel_resolution)
				marker.scale.z = float(self.voxel_resolution)
				
				for point_data in occupied_points:
					p = Point()
					p.x, p.y, p.z = float(point_data[0]), float(point_data[1]), float(point_data[2])
					marker.points.append(p)
					
					# Check if semantic voxel
					voxel_key = self._get_voxel_key_from_point(point_data[:3])
					with self.semantic_voxels_lock:
						if voxel_key in self.semantic_voxels:
							# Semantic voxel - use VLM answer color
							semantic_info = self.semantic_voxels[voxel_key]
							vlm_answer = semantic_info.get('vlm_answer', 'unknown')
							color_rgb = self._get_vlm_answer_color(vlm_answer)
							color = ColorRGBA()
							color.r, color.g, color.b, color.a = color_rgb[0]/255.0, color_rgb[1]/255.0, color_rgb[2]/255.0, 1.0
						else:
							# Regular voxel - gray
							color = ColorRGBA()
							color.r, color.g, color.b, color.a = 0.5, 0.5, 0.5, 0.8
					marker.colors.append(color)
				
				marker_array.markers.append(marker)
			else:
				# Create individual cube markers
				for i, point_data in enumerate(occupied_points):
					marker = Marker()
					marker.header.frame_id = self.map_frame
					marker.header.stamp = self.get_clock().now().to_msg()
					marker.ns = "vdb_occupancy"
					marker.id = i
					marker.type = Marker.CUBE
					marker.action = Marker.ADD
					marker.pose.position.x = float(point_data[0])
					marker.pose.position.y = float(point_data[1])
					marker.pose.position.z = float(point_data[2])
					marker.pose.orientation.w = 1.0
					marker.scale.x = float(self.voxel_resolution)
					marker.scale.y = float(self.voxel_resolution)
					marker.scale.z = float(self.voxel_resolution)
					
					# Check if semantic voxel
					voxel_key = self._get_voxel_key_from_point(point_data[:3])
					with self.semantic_voxels_lock:
						if voxel_key in self.semantic_voxels:
							# Semantic voxel - use VLM answer color
							semantic_info = self.semantic_voxels[voxel_key]
							vlm_answer = semantic_info.get('vlm_answer', 'unknown')
							color_rgb = self._get_vlm_answer_color(vlm_answer)
							marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgb[0]/255.0, color_rgb[1]/255.0, color_rgb[2]/255.0, 1.0
						else:
							# Regular voxel - gray
							marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.5, 0.5, 0.5, 0.8
					
					marker_array.markers.append(marker)
			
			return marker_array
			
		except Exception as e:
			self.get_logger().error(f"Error creating VDB markers: {e}")
			return None
	
	def _create_semantic_only_cloud(self) -> Optional[PointCloud2]:
		"""Create a point cloud containing all accumulated semantic voxels."""
		try:
			# Get all accumulated semantic voxels
			with self.semantic_voxels_lock:
				if not self.semantic_voxels:
					return None
				
				# Create point cloud data for all accumulated semantic voxels
				points = []
				for voxel_key, semantic_info in self.semantic_voxels.items():
					voxel_center = semantic_info['position']
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
		
		# if self.marker_pub and (now - self.last_marker_pub) >= float(self.marker_publish_rate):
		# 	markers = self._create_vdb_markers(int(self.max_markers))
			
		# 	if markers is not None:
		# 		# Fix timestamps for all markers
		# 		current_time = self.get_clock().now().to_msg()
		# 		for marker in markers.markers:
		# 			marker.header.stamp = current_time
				
		# 		self.marker_pub.publish(markers)
		# 		marker_count = len(markers.markers) if hasattr(markers, 'markers') else 0
		# 		self.get_logger().info(f"Published {marker_count} VDB voxel markers")
		# 	self.last_marker_pub = now
		
		if self.cloud_pub:
			try:
				# Create VDB-based semantic-aware colored cloud
				semantic_cloud = self._create_semantic_colored_cloud(int(self.max_markers))
				if semantic_cloud:
					self.cloud_pub.publish(semantic_cloud)
					self.get_logger().debug(f"Published VDB semantic colored cloud with {len(semantic_cloud.data)//16} points")
				else:
					self.get_logger().debug("VDB cloud creation returned None (map may be empty)")
			except Exception as e:
				self.get_logger().warn(f"Failed to create VDB colored cloud: {e}")
		
		# Publish semantic-only point cloud (XYZ only, no RGB)
		if self.semantic_only_pub:
			try:
				semantic_only_cloud = self._create_semantic_only_cloud()
				if semantic_only_cloud:
					self.semantic_only_pub.publish(semantic_only_cloud)
					self.get_logger().debug(f"Published semantic-only cloud with {len(semantic_only_cloud.data)//12} points")
			except Exception as e:
				self.get_logger().warn(f"Failed to create semantic-only cloud: {e}")
		
		# if self.stats_pub and (now - self.last_stats_pub) >= float(self.stats_publish_rate):
			# Get statistics from VDB mapper
			try:
				if not self.vdb_mapper.is_empty():
					pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.vdb_mapper.occ_map_vdb)
					
					# Convert to numpy if it's a torch tensor
					if torch.is_tensor(pc_xyz_occ_size):
						pc_xyz_occ_size = pc_xyz_occ_size.cpu().numpy()
					
					occupied_mask = pc_xyz_occ_size[:, -2] > 0
					total_voxels = int(np.sum(occupied_mask))
				else:
					total_voxels = 0
			except:
				total_voxels = 0
			
			# Add semantic mapping status and counts
			semantic_voxel_count = 0
			
			with self.semantic_voxels_lock:
				semantic_voxel_count = len(self.semantic_voxels)
			
			stats = {
				'mapper_type': 'VDB OccupancyMap',
				'total_voxels': total_voxels,
				'voxel_resolution': float(self.voxel_resolution),
				'semantic_mapping': {
					'enabled': self.enable_semantic_mapping,
					'status': 'active' if self.enable_semantic_mapping else 'disabled',
					'semantic_voxel_count': semantic_voxel_count
				}
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

	def _pose_to_4x4_matrix(self, pose: PoseStamped) -> torch.Tensor:
		"""Convert PoseStamped to 4x4 transformation matrix with proper coordinate frame handling."""
		# Position
		p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
		
		# Orientation (quaternion to rotation matrix)
		q = np.array([pose.pose.orientation.x, pose.pose.orientation.y, 
					 pose.pose.orientation.z, pose.pose.orientation.w], dtype=np.float32)
		R = self._quat_to_rot(q)
		
		# Apply coordinate frame transformation if pose is in base_link frame
		if bool(self.pose_is_base_link):
			# Transform from base_link to camera frame
			# This is the inverse of the transformation used in depth projection
			
			# Step 1: Transform pose from base_link to camera frame
			# Apply camera-to-base transformation (inverse)
			p = p - self.t_cam_to_base_extra
			R = R @ self.R_cam_to_base_extra
			
			# Step 2: Apply optical frame rotation (inverse)
			if bool(self.apply_optical_frame_rotation):
				R = R @ self.R_opt_to_base
		
		# Create 4x4 matrix
		T = np.eye(4, dtype=np.float32)
		T[:3, :3] = R
		T[:3, 3] = p
		
		# Convert to tensor and move to same device as mapper
		device = self.vdb_mapper.device
		return torch.from_numpy(T).unsqueeze(0).to(device)  # 1x4x4

	def _publish_mask_frontiers_and_rays(self):
		try:
			if self.vdb_mapper is None:
				return
			# Mask frontiers
			if self.vdb_mapper.frontiers is not None and self.vdb_mapper.frontiers.shape[0] > 0:
				frontiers_np = self.vdb_mapper.frontiers.detach().cpu().numpy()
				cloud = self._create_cloud_xyz(frontiers_np)
				if cloud is not None:
					self.mask_frontiers_pub.publish(cloud)
			# Rays as arrows
			def _offset_origin(base: np.ndarray, direction: np.ndarray, idx: int) -> np.ndarray:
				offset_dir = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=np.float32))
				if np.linalg.norm(offset_dir) < 1e-6:
					offset_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
				offset_dir /= np.linalg.norm(offset_dir) + 1e-9
				offset_mag = float(self.voxel_resolution) * 0.3 * ((idx % 5) - 2)
				return base + offset_dir * offset_mag

			if (self.vdb_mapper.global_rays_orig_angles is not None and
				self.vdb_mapper.global_rays_orig_angles.shape[0] > 0):
				msg = MarkerArray()
				now = self.get_clock().now().to_msg()
				# Publish ALL global angle rays without clustering to avoid dropping true positives
				data = self.vdb_mapper.global_rays_orig_angles.detach().cpu().numpy()
				length = 0.75
				for i, row in enumerate(data):
					x, y, z, theta_deg, phi_deg = row
					theta = np.deg2rad(theta_deg)
					phi = np.deg2rad(phi_deg)
					dir_world = np.array([
						np.cos(theta) * np.sin(phi),
						np.sin(theta) * np.sin(phi),
						np.cos(phi)
					], dtype=np.float32)
					dir_world /= np.linalg.norm(dir_world) + 1e-9
					start = _offset_origin(np.array([x, y, z], dtype=np.float32), dir_world, i)
					end = start + dir_world * length
					m = Marker()
					m.header.frame_id = self.map_frame
					m.header.stamp = now
					m.ns = "mask_rays_frontier"
					m.id = i
					m.type = Marker.ARROW
					m.action = Marker.ADD
					m.scale.x = float(self.voxel_resolution) * 0.4
					m.scale.y = float(self.voxel_resolution) * 0.6
					m.scale.z = float(self.voxel_resolution) * 0.6
					m.color.r = 1.0
					m.color.g = 0.3
					m.color.b = 0.0
					m.color.a = 0.95
					m.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
						Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
					msg.markers.append(m)
				if len(msg.markers) > 0:
					self.mask_rays_pub.publish(msg)

			if self._latest_pose_rays is not None:
				origin_world, dir_world = self._latest_pose_rays
				now = self.get_clock().now().to_msg()
				# Use the same binning method as RayFronts for consistency
				try:
					# Convert numpy to torch tensors
					dirs_torch = torch.from_numpy(dir_world).float().to(self.vdb_mapper.device)
					origin_torch = torch.from_numpy(origin_world).float().to(self.vdb_mapper.device)
					
					# Convert cartesian to spherical coordinates (same as RayFronts)
					r, theta, phi = g3d.cartesian_to_spherical(
						dirs_torch[:, 0], dirs_torch[:, 1], dirs_torch[:, 2])
					
					# Create ray_orig_angle format: [x, y, z, theta_deg, phi_deg]
					ray_orig_angle = torch.cat([
						origin_torch.repeat(dir_world.shape[0], 1),  # Duplicate origin for each ray
						torch.rad2deg(theta).unsqueeze(-1),
						torch.rad2deg(phi).unsqueeze(-1)
					], dim=-1)
					
					# Create dummy features with uniform weights (1.0 for all)
					dummy_feat_weights = torch.ones(ray_orig_angle.shape[0], 1, 
						device=self.vdb_mapper.device, dtype=torch.float32)
					
					# Accumulate bins similar to SemanticRayFrontiersMap (weighted_mean)
					# Use weights only (no extra features). Shape Nx1 with last column as weight.
					weights_only = torch.ones(ray_orig_angle.shape[0], 1, device=self.vdb_mapper.device, dtype=torch.float32)
					if self.pose_rays_orig_angles is None:
						self.pose_rays_orig_angles, self.pose_rays_feats_cnt = g3d.bin_rays(
							ray_orig_angle,
							vox_size=float(self.voxel_resolution),
							bin_size=self.vdb_mapper.angle_bin_size,
							feat=weights_only,
							aggregation="weighted_mean"
						)
					else:
						self.pose_rays_orig_angles, self.pose_rays_feats_cnt = g3d.add_weighted_binned_rays(
							self.pose_rays_orig_angles,
							self.pose_rays_feats_cnt,
							ray_orig_angle,
							weights_only,
							vox_size=float(self.voxel_resolution),
							bin_size=self.vdb_mapper.angle_bin_size
						)
					
					# Convert accumulated bins back to numpy for visualization
					binned_rays_np = self.pose_rays_orig_angles.detach().cpu().numpy()
					
					# Extract origins and angles
					origins_np = binned_rays_np[:, :3]
					theta_deg = binned_rays_np[:, 3]
					phi_deg = binned_rays_np[:, 4]
					
					# Convert spherical back to cartesian directions
					theta_rad = np.deg2rad(theta_deg)
					phi_rad = np.deg2rad(phi_deg)
					sin_phi = np.sin(phi_rad)
					dirs_np = np.stack([
						np.cos(theta_rad) * sin_phi,
						np.sin(theta_rad) * sin_phi,
						np.cos(phi_rad)
					], axis=1)
					
					# Normalize directions
					dirs_np = dirs_np / (np.linalg.norm(dirs_np, axis=1, keepdims=True) + 1e-9)
					
					# Create visualization
					length = 0.75
					msg_pose = MarkerArray()
					for i in range(len(binned_rays_np)):
						start = origins_np[i]
						end = start + dirs_np[i] * length
						m = Marker()
						m.header.frame_id = self.map_frame
						m.header.stamp = now
						m.ns = "mask_rays_pose"
						m.id = i
						m.type = Marker.ARROW
						m.action = Marker.ADD
						m.scale.x = float(self.voxel_resolution) * 0.4
						m.scale.y = float(self.voxel_resolution) * 0.6
						m.scale.z = float(self.voxel_resolution) * 0.6
						m.color.r = 0.0
						m.color.g = 0.7
						m.color.b = 1.0
						m.color.a = 0.95
						m.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
							Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
						msg_pose.markers.append(m)
					
					if len(msg_pose.markers) > 0:
						self.mask_rays_pub.publish(msg_pose)
						
				except Exception as e:
					self.get_logger().warn(f"Pose ray binning failed: {e}")
					import traceback
					traceback.print_exc()
		except Exception as e:
			self.get_logger().warn(f"Publishing mask rays/frontiers failed: {e}")





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


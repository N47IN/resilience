#!/usr/bin/env python3
"""
Semantic Depth VDB Mapping ROS2 Node

Simplified node that uses RayFronts SemanticRayFrontiersMap for efficient 3D mapping.
Subscribes to depth, pose, and semantic info to create semantic voxel maps using OpenVDB.
Maintains timestamped buffers for depth frames and poses to align with hotspot masks
received via the semantic bridge using original RGB timestamps.
"""

import rclpy
import matplotlib.cm as cm  
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
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import numpy as np
from collections import deque
import torch
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
import time
import json
import math
from typing import Optional, List, Dict
import sensor_msgs_py.point_cloud2 as pc2
import threading
from concurrent.futures import ThreadPoolExecutor
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
from resilience.voxel_gp_helper import _sum_of_anisotropic_rbf_fast

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
			('voxel_resolution', 0.2),
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
			('marker_publish_rate', 20.0),
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
		from collections import deque

		
		self.depth_buffer_data = deque(maxlen=100)
		self.depth_buffer_ts = deque(maxlen=100)

		self.pose_buffer_data = deque(maxlen=200)
		self.pose_buffer_ts = deque(maxlen=200)

		self.mask_buffer_data = deque(maxlen=50) # Smaller buffer usually okay for masks
		self.mask_buffer_ts = deque(maxlen=50)
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
		self.mask_buffer = []
		self.sync_buffer_duration = float(self.sync_buffer_seconds)
		self.sync_lock = threading.Lock()
		
		# Thread pool for processing semantic hotspots
		self.hotspot_executor = ThreadPoolExecutor(max_workers=3)
		
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
		self.gp_update_interval = 0.75
		self.gp_computation_thread = None
		self.gp_thread_lock = threading.Lock()
		self.gp_thread_running = False
		self.min_radius = 0.5
		self.max_radius = 2.0
		self.base_radius = 1.0
		
		# Robot-centric 3D grid parameters (NEW)
		self.robot_grid_size_xy = 5.0  # 10m x 10m in XY plane
		self.robot_grid_size_z = 3.0    # 4m in Z axis
		self.robot_grid_resolution = 0.2  # 0.2m voxel resolution
		self.robot_position = None  # Current robot position (from latest_pose)
		
		# GPU tensor for mean and uncertainty fields (Channel=2, Depth, Height, Width)
		try:
			import torch
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			self.gp_grid_tensor = None  # Will be initialized on first update
			self.TORCH_AVAILABLE = True
			self.get_logger().info(f"PyTorch GPU tensor backend: {self.device}")
		except ImportError:
			self.TORCH_AVAILABLE = False
			self.get_logger().warn("PyTorch not available, using NumPy fallback")
		
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
		self.operational_confirmation_threshold = 3  # Operational: require 2 frames for noise rejection (non-blocking, incremental)
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
		# Raw GP grid publisher for control
		self.gp_grid_raw_pub = self.create_publisher(Float32MultiArray, '/gp_grid_raw', 10)
		self.voxel_resolution = 0.2
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
				ray_tracing=False,
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
		# Update robot position for robot-centric grid (NEW)
		self.robot_position = np.array([
			msg.pose.position.x,
			msg.pose.position.y,
			msg.pose.position.z
		], dtype=np.float32)
		# Push into pose buffer with timestamp
		
		pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.pose_buffer_data.append(msg)
		self.pose_buffer_ts.append(pose_time)
		self.last_data_time = time.time()

	def semantic_hotspot_mask_callback(self, msg: Image):
		"""Buffer the merged hotspot mask image keyed by its stamp time."""
		mask_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
		mask_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.mask_buffer_data.append(mask_rgb)
		self.mask_buffer_ts.append(mask_time)
		self.last_data_time = time.time()


	def semantic_hotspot_callback(self, msg: String):
		"""Process incoming semantic hotspot metadata using an efficient thread pool."""
		if not self.enable_semantic_mapping or not self.enable_voxel_mapping:
			return

		# Submit the task to the pool instead of spawning a new thread
		self.hotspot_executor.submit(self._process_single_bridge_message, msg.data)
		
		self.last_data_time = time.time()

	def _process_single_bridge_message(self, msg_data: str) -> bool:
		"""Process a single bridge message and apply to voxel map by timestamp lookup."""
		
		# Parse the JSON message
		time_start = time.time()
		data = json.loads(msg_data)
		json_load_time = time.time() - time_start
		self.get_logger().warn(f"Time taken to load JSON: {json_load_time}")
		if data.get('type') == 'merged_similarity_hotspots':
			return self._process_merged_hotspot_message(data)
		else:
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
			depth_image, pose_msg = self._lookup_depth_and_pose(rgb_timestamp)
			depth_lookup_time = time.time() - start - mask_lookup_time
			self.get_logger().warn(f"Time taken to lookup depth: {depth_lookup_time}")
			if depth_image is None or pose_msg is None:
				self.get_logger().warn(f"No matching depth/pose found for timestamp {rgb_timestamp:.6f}")
				return False
			
			try:
				# Convert ROS Image message to numpy array (float32 for depth)
				depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='32FC1')
			except Exception as e:
				self.get_logger().error(f"Failed to convert depth message: {e}")
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
				used_ts = 1.0
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
	
	def _lookup_depth_and_pose(self, target_ts):
		# Pass the dual deques for depth
		depth_data, depth_ts = self._binary_search_closest(
			self.depth_buffer_ts, self.depth_buffer_data, target_ts, 1
		)
	
		# Pass the dual deques for pose
		pose_data, pose_ts = self._binary_search_closest(
			self.pose_buffer_ts, self.pose_buffer_data, target_ts, 1
		)
	
		return depth_data, pose_data
	
	def _binary_search_closest(self, ts_deque: deque, data_deque: deque, target_ts: float, max_dt: float):
		"""Vectorized search across synchronized deques."""
		if not ts_deque:
			return None, None
		ts_array = np.array(ts_deque)
		idx = np.searchsorted(ts_array, target_ts)
		candidates = []
		if idx < len(ts_array): candidates.append(idx)
		if idx > 0: candidates.append(idx - 1)
	
		if not candidates:
			return None, None
	
		diffs = np.abs(ts_array[candidates] - target_ts)
		best_relative_idx = np.argmin(diffs)
		best_idx = candidates[best_relative_idx]
		if diffs[best_relative_idx] <= max_dt:
			return data_deque[best_idx], ts_array[best_idx]
	
		return None, None
	
	def _lookup_mask(self, target_ts: float) -> Optional[np.ndarray]:
		"""Find closest merged mask image using optimized dual-deque binary search."""
		with self.sync_lock:
			# Pass the separate timestamp and data deques
			best_mask, _ = self._binary_search_closest(
				self.mask_buffer_ts, 
				self.mask_buffer_data, 
				target_ts, 
				self.sync_buffer_duration
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
					self._check_and_start_gp_fit(buffer_dir, voxelized_points, vlm_answer)
 				
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
			
			# Prepare masked depth for rays-only beyond max_range
			depth_for_rays = np.zeros_like(depth_hot, dtype=np.float32)
			masked = (mask > 0)
			if self.camera_intrinsics is not None:
				# Use original depth_m if available, otherwise use depth_hot
				# For rays, we want pixels beyond max_range or missing depth
				masked_depth_vals = depth_hot[masked]
				threshold = 5.0
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
				
				
				
				# Update intrinsics if available
				fx, fy, cx, cy = self.camera_intrinsics
				self.vdb_mapper.intrinsics_3x3 = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
				# self.vdb_mapper.process_posed_rgbd(rgb_dummy, depth_masked_t, pose_4x4_rf, conf_map=conf_map_t, feat_img=None)
				# Publish mask-specific frontiers and rays immediately (same as tmp.py)
				self._publish_mask_frontiers_and_rays()
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
				f"Applied hotspot processing for '{vlm_answer}' (within_range={near_count}, rgb_ts={rgb_ts:.6f}, type={hotspot_type})"
			)
			return True
			
		except Exception as e:
			self.get_logger().error(f"Error processing hotspot with depth: {e}")
			import traceback
			traceback.print_exc()
			return False
	


	def _voxelize_pointcloud(self, points: np.ndarray, voxel_size: float, max_points: int = 200) -> np.ndarray:
		"""
		High-performance voxelization using Open3D (C++ backend).
		Reduces point density by averaging points within a spatial grid.
		"""
		if points.shape[0] == 0:
			return points

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
		voxelized_points = np.asarray(downsampled_pcd.points)
		num_voxelized = voxelized_points.shape[0]
		if num_voxelized > max_points:
			step = num_voxelized / max_points
			indices = np.arange(0, num_voxelized, step, dtype=np.int32)[:max_points]
			voxelized_points = voxelized_points[indices]
	
			self.get_logger().info(
				f"O3D Voxelized {points.shape[0]} -> {num_voxelized} points. "
				f"Sampled to {max_points} (voxel_size={voxel_size:.3f}m)"
			)
		return voxelized_points

	def save_points_to_latest_nested_subfolder(self, known_folder: str, 
										  points_world: np.ndarray, 
										  filename: str = "points.pcd"):
		"""
		Finds latest nested subfolders and saves points as a binary PCD using Open3D.
		"""
		if points_world.size == 0:
			return None, None
	
		voxelized_points = self._voxelize_pointcloud(points_world, float(self.voxel_resolution), max_points=200)
	
		current_time = time.time()
		if (self._cached_latest_subfolder and os.path.exists(self._cached_latest_subfolder) and 
			(current_time - self._cached_subfolder_time) < self._subfolder_cache_ttl):
			latest_subfolder2 = self._cached_latest_subfolder
		else:
			try:
				# Find latest subfolder1
				s1 = [os.path.join(known_folder, d) for d in os.listdir(known_folder) if os.path.isdir(os.path.join(known_folder, d))]
				if not s1: return None, None
				latest_s1 = max(s1, key=os.path.getmtime)
	
				# Find latest subfolder2
				s2 = [os.path.join(latest_s1, d) for d in os.listdir(latest_s1) if os.path.isdir(os.path.join(latest_s1, d))]
				if not s2: return None, None
				latest_subfolder2 = max(s2, key=os.path.getmtime)
	
				# Cache it
				self._cached_latest_subfolder = latest_subfolder2
				self._cached_subfolder_time = current_time
			except Exception as e:
				self.get_logger().error(f"Folder search failed: {e}")
				return None, None
	
		# 3. Save JSON Metadata (Mean position of the hazard)
		save_path = os.path.join(latest_subfolder2, filename)
		mean_pos = np.mean(voxelized_points, axis=0).tolist()
		with open(os.path.join(latest_subfolder2, "mean_cause.json"), "w") as f:
			json.dump(mean_pos, f)
	
		# 4. Save PCD using Open3D (Binary format is default and much faster)
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(voxelized_points)
		o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
	
		self.get_logger().info(f"O3D saved {len(voxelized_points)} points to {save_path}")
		return latest_subfolder2, save_path

	def _check_and_start_gp_fit(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
		"""Check if poses.npy is available and start GP fitting if ready."""
		poses_path = os.path.join(buffer_dir, 'poses.npy')
		if not os.path.exists(poses_path):
			self.get_logger().info(f"poses.npy not yet available in {buffer_dir}, skipping GP fit for now")
			return
		
		try:
			poses_data = np.load(poses_path)
			if len(poses_data) == 0:
				self.get_logger().info(f"poses.npy is empty in {buffer_dir}, skipping GP fit for now")
				return
		except Exception as e:
			self.get_logger().warn(f"Error reading poses.npy from {buffer_dir}: {e}")
			return
		
		self.get_logger().info(f"Both PCD and poses.npy available in {buffer_dir}, starting GP fit")
		self._start_background_gp_fit(buffer_dir, pointcloud_xyz, cause_name)
			

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
			
			# Store the latest GP parameters for global use
			if result and 'fit' in result:
				self.global_gp_params = result['fit']
				self.global_nominal_points = result.get('nominal_used')  # Store for uncertainty computation
				self.global_disturbances = result.get('disturbances')  # Store for uncertainty computation
				self.get_logger().info(f"Updated global GP parameters: lxy={self.global_gp_params.get('lxy', 0):.3f}, lz={self.global_gp_params.get('lz', 0):.3f}, A={self.global_gp_params.get('A', 0):.3f}")
			

	
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
		"""Load points from PCD file using Open3D for high compatibility and speed."""
		try:
			pcd = o3d.io.read_point_cloud(pcd_path)
			if pcd.is_empty():
				return np.array([], dtype=np.float32)
			return np.asarray(pcd.points, dtype=np.float32)
	
		except Exception as e:
			self.get_logger().error(f"Error loading PCD points with Open3D: {e}")
			return np.array([], dtype=np.float32)

	def _create_gp_colored_pointcloud(self, grid_points: np.ndarray, gp_values: np.ndarray) -> Optional[PointCloud2]:
		"""Create colored point cloud from GP field predictions using vectorized operations."""
		if len(grid_points) == 0 or len(gp_values) == 0:
			return None

		gp_min, gp_max = gp_values.min(), gp_values.max()
		if gp_max > gp_min:
			normalized_values = (gp_values - gp_min) / (gp_max - gp_min)
		else:
			normalized_values = np.zeros_like(gp_values)

		colors_rgba = cm.turbo(normalized_values)
		colors_uint8 = (colors_rgba[:, :3] * 255).astype(np.uint32)
		rgb_packed = (colors_uint8[:, 0] << 16) | (colors_uint8[:, 1] << 8) | colors_uint8[:, 2]

		cloud_data = np.empty(len(grid_points), dtype=[
			('x', np.float32), ('y', np.float32), ('z', np.float32), 
			('rgb', np.uint32)
		])

		cloud_data['x'] = grid_points[:, 0].astype(np.float32)
		cloud_data['y'] = grid_points[:, 1].astype(np.float32)
		cloud_data['z'] = grid_points[:, 2].astype(np.float32)
		cloud_data['rgb'] = rgb_packed

		# 5. Assemble PointCloud2 Message
		cloud_msg = PointCloud2()
		cloud_msg.header.stamp = self.get_clock().now().to_msg()
		cloud_msg.header.frame_id = self.map_frame

		cloud_msg.fields = [
			pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
			pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
			pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
			pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
		]

		cloud_msg.point_step = 16
		cloud_msg.width = len(grid_points)
		cloud_msg.height = 1
		cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
		cloud_msg.is_dense = True
		cloud_msg.data = cloud_data.tobytes()
		return cloud_msg
	
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
		"""
		Update GP visualization using a ROBOT-CENTRIC 3D grid.
		
		Instead of predicting near cause points, this creates a 3D grid around the robot
		(e.g., 10m × 10m × 4m) and predicts GP mean and epistemic uncertainty on this grid.
		Results are stored in a GPU tensor (Channel=2, Depth, Height, Width) where:
		  - Channel 0 = GP Mean
		  - Channel 1 = Epistemic Uncertainty
		
		Update rate: 2-5 Hz (asynchronous)
		"""
		try:
			if self.global_gp_params is None:
				return
			
			# Check if robot position is available
			if self.robot_position is None:
				self.get_logger().warn("Robot position not available yet, skipping GP update")
				return
			
			# Get all semantic voxels (cause points)
			semantic_voxels = self._get_all_semantic_voxels()
			if len(semantic_voxels) == 0:
				return
			
			# Convert to numpy array (like loading cause points from PCD)
			semantic_points = np.array(semantic_voxels)
			
			# ============================================================
			# ROBOT-CENTRIC 3D GRID GENERATION (NEW APPROACH)
			# ============================================================
			grid_points, grid_shape = self._create_robot_centric_3d_grid()
			if len(grid_points) == 0:
				return
			
			# Predict GP mean on robot-centric grid
			gp_mean = self._predict_gp_field_fast(grid_points, semantic_points, self.global_gp_params)
			
			# Compute epistemic uncertainty on robot-centric grid
			uncertainty_std = None
			if (self.global_nominal_points is not None and self.global_disturbances is not None and 
				len(self.global_nominal_points) > 0 and len(self.global_disturbances) > 0):
				uncertainty_std = self._compute_epistemic_uncertainty(
					grid_points, semantic_points, self.global_gp_params,
					self.global_nominal_points, self.global_disturbances
				)
			
			
			# ============================================================
			# STORE IN GPU TENSOR (Channel=2, Depth, Height, Width)
			# ============================================================
			self._update_gp_gpu_tensor(gp_mean, uncertainty_std, grid_shape)
			

			colored_cloud = self._create_gp_colored_pointcloud(grid_points, gp_mean)
			if colored_cloud:
				self.gp_visualization_pub.publish(colored_cloud)
			
			# Publish costmap (mean disturbance values)
			costmap_cloud = self._create_costmap_pointcloud(grid_points, gp_mean)
			if costmap_cloud:
				self.costmap_pub.publish(costmap_cloud)
			
			# Publish epistemic uncertainty field
			if uncertainty_std is not None:
				uncertainty_cloud = self._create_uncertainty_pointcloud(grid_points, uncertainty_std)
				if uncertainty_cloud:
					self.gp_uncertainty_pub.publish(uncertainty_cloud)
			
			# Publish raw grid for control node
			self._publish_raw_gp_grid(gp_mean, uncertainty_std, grid_shape, grid_points)

			self.get_logger().info(
				f"Published robot-centric GP fields: {len(grid_points)} points, "
				f"grid_shape={grid_shape}, robot_pos=[{self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}, {self.robot_position[2]:.2f}], "
				f"mean_range=[{gp_mean.min():.3f}, {gp_mean.max():.3f}], "
				f"uncertainty_range=[{uncertainty_std.min():.3f}, {uncertainty_std.max():.3f}]"
			)
			
		except Exception as e:
			self.get_logger().error(f"Error updating robot-centric GP visualization: {e}")
			import traceback
			traceback.print_exc()
	

	def _create_robot_centric_3d_grid(self):
		"""
		Optimized 3D grid generation using flattened coordinate arrays.
		Grid size: 10m × 10m (XY) × 4m (Z) centered on the robot.
		"""
		try:
			if self.robot_position is None:
				self.get_logger().warn("Robot position not available for grid generation")
				return np.array([], dtype=np.float32), (0, 0, 0)
	
			# 1. Define bounds
			rx, ry, rz = self.robot_position
			h_xy = self.robot_grid_size_xy / 2.0
			h_z = self.robot_grid_size_z / 2.0
			res = self.robot_grid_resolution
	
			# 2. Use linspace for stability or arange for exact resolution
			# np.arange can sometimes have 'off-by-one' errors with floating points
			x_c = np.arange(rx - h_xy, rx + h_xy, res, dtype=np.float32)
			y_c = np.arange(ry - h_xy, ry + h_xy, res, dtype=np.float32)
			z_c = np.arange(rz - h_z, rz + h_z, res, dtype=np.float32)
	
			# 3. Memory Efficient Meshgrid
			# Using indexing='ij' is correct for (D, H, W) mapping
			X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing='ij')
	
			# 4. Ravel is faster than flatten() as it returns a view when possible
			grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
	
			grid_shape = (len(x_c), len(y_c), len(z_c))
	
			return grid_points, grid_shape
	
		except Exception as e:
			self.get_logger().error(f"Error creating robot-centric 3D grid: {e}")
			return np.array([], dtype=np.float32), (0, 0, 0)
	
	def _update_gp_gpu_tensor(self, gp_mean, uncertainty_std, grid_shape):
		"""
		Update GPU tensor with GP mean and uncertainty fields.
		
		Tensor format: (Channel=2, Depth, Height, Width)
		  - Channel 0: GP Mean
		  - Channel 1: Epistemic Uncertainty
		
		Args:
			gp_mean: (N,) GP mean predictions
			uncertainty_std: (N,) Epistemic uncertainty predictions
			grid_shape: (D, H, W) grid dimensions
		"""
		try:
			if not self.TORCH_AVAILABLE:
				# Skip GPU tensor update if PyTorch not available
				return
			
			import torch
			
			# Reshape flattened arrays to 3D grid
			D, H, W = grid_shape
			mean_grid = gp_mean.reshape(D, H, W).astype(np.float32)
			uncertainty_grid = uncertainty_std.reshape(D, H, W).astype(np.float32)
			
			# Stack into (2, D, H, W) tensor
			# Channel 0 = Mean, Channel 1 = Uncertainty
			combined_grid = np.stack([mean_grid, uncertainty_grid], axis=0)
			
			# Convert to PyTorch tensor and move to GPU
			self.gp_grid_tensor = torch.from_numpy(combined_grid).to(self.device)
			
			self.get_logger().info(
				f"Updated GPU tensor: shape={self.gp_grid_tensor.shape}, "
				f"device={self.device}, "
				f"mean_range=[{mean_grid.min():.3f}, {mean_grid.max():.3f}], "
				f"uncertainty_range=[{uncertainty_grid.min():.3f}, {uncertainty_grid.max():.3f}]"
			)
			
		except Exception as e:
			import traceback
			traceback.print_exc()

	def _publish_raw_gp_grid(self, gp_mean, uncertainty_std, grid_shape, grid_points):
		"""Publish raw GP grid data for control node."""
		try:
			# grid_shape is (Nx, Ny, Nz) - corresponding to coords
			# gp_mean is flattened (N,)
			
			msg = Float32MultiArray()
			
			# Encode metadata in the layout using labels or dimensions
			# Dim 0: Meta [min_x, min_y, min_z, res, size_x, size_y, size_z]
			# We'll just put metadata as the first few elements of the data array, or use a structured approach
			# Let's pack metadata as a prefix to the data. 
			
			if self.robot_position is None:
				return
				
			# Recalculate bounds from robot position and fixed params
			half_size_xy = self.robot_grid_size_xy / 2.0
			half_size_z = self.robot_grid_size_z / 2.0
			min_x = self.robot_position[0] - half_size_xy
			min_y = self.robot_position[1] - half_size_xy
			min_z = self.robot_position[2] - half_size_z
			
			# Metadata header: 7 floats
			metadata = [
				min_x, min_y, min_z, 
				self.robot_grid_resolution, 
				float(grid_shape[0]), float(grid_shape[1]), float(grid_shape[2])
			]
			
			# Concatenate: Metadata + Mean + Uncertainty
			# Note: gp_mean and uncertainty_std are flattened
			data_list = metadata + gp_mean.tolist() + uncertainty_std.tolist()
			
			msg.data = data_list
			
			# Describe layout
			# Dim 0: Metadata (7)
			# Dim 1: Mean (N)
			# Dim 2: Uncertainty (N)
			# This isn't a standard multiarray layout, but the receiver will know how to parse it.
			# Or we can strictly use dimensions to describe the grid, but we need the origin offset.
			
			dim0 = MultiArrayDimension(label="metadata", size=7, stride=7)
			dim1 = MultiArrayDimension(label="mean", size=len(gp_mean), stride=len(gp_mean))
			dim2 = MultiArrayDimension(label="uncertainty", size=len(uncertainty_std), stride=len(uncertainty_std))
			msg.layout.dim = [dim0, dim1, dim2]
			
			self.gp_grid_raw_pub.publish(msg)
			
		except Exception as e:
			self.get_logger().error(f"Error publishing raw GP grid: {e}")
	
	
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
			phi = _sum_of_anisotropic_rbf_fast(grid_points, cause_points, lxy, lz)
			
			# Apply the learned parameters: disturbance = A * phi + b
			predictions = A * phi + b
			
			return predictions
			
		except Exception as e:
			self.get_logger().error(f"Error in fast GP prediction: {e}")
			return np.zeros(len(grid_points))
	
	
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
			phi_train = _sum_of_anisotropic_rbf_fast(nominal_points, cause_points, lxy, lz)
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
			phi_query = _sum_of_anisotropic_rbf_fast(grid_points, cause_points, lxy, lz)
			
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
		Creates a PointCloud2 with a sharp 'Inferno' colormap and percentile normalization.
		"""
		try:
			if len(grid_points) == 0 or len(uncertainty_std) == 0:
				return None

			# 1. Percentile Normalization (The secret to the "Sharp" look)
			# Instead of min/max, use percentiles to ignore outliers and boost contrast
			u_min = np.percentile(uncertainty_std, 5)
			u_max = np.percentile(uncertainty_std, 95)
	
			# Avoid division by zero
			diff = u_max - u_min if u_max > u_min else 1.0
			normalized_values = np.clip((uncertainty_std - u_min) / diff, 0, 1)

			# 2. Apply Sharp Colormap (inferno or magma)
			# 'inferno' goes: Black -> Purple -> Red -> Bright Yellow
			colors_mapped = cm.inferno(normalized_values) 

			# Create structured array for PointCloud2
			cloud_data_combined = np.empty(len(grid_points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)
			])
	
			cloud_data_combined['x'] = grid_points[:, 0]
			cloud_data_combined['y'] = grid_points[:, 1]
			cloud_data_combined['z'] = grid_points[:, 2]

			# 3. Fast RGB Packing
			# Pack RGBA (normalized 0-1) into a single UINT32 for RViz
			r = (colors_mapped[:, 0] * 255).astype(np.uint32)
			g = (colors_mapped[:, 1] * 255).astype(np.uint32)
			b = (colors_mapped[:, 2] * 255).astype(np.uint32)
	
			# Bit-shift to pack into UINT32 (R << 16 | G << 8 | B)
			rgb_packed = (r << 16) | (g << 8) | b
			cloud_data_combined['rgb'] = rgb_packed

			# 4. Standard PointCloud2 Message Creation
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.map_frame
	
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
			]
			cloud_msg.point_step = 16
			cloud_msg.width = len(grid_points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			cloud_msg.data = cloud_data_combined.tobytes()

			return cloud_msg

		except Exception as e:
			self.get_logger().error(f"Error creating sharp uncertainty cloud: {e}")
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
			self.depth_buffer_data.append(msg)
			self.depth_buffer_ts.append(depth_time)
			
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

	def _update_semantic_voxels(self, points_world: np.ndarray, vlm_answer: str, threshold: float, 
								 stats: dict, is_narration: bool):
		"""Update semantic voxel labels in a separate thread (optimized)."""
		try:
			if points_world.size > 0:
				self._apply_semantic_labels_to_voxels(points_world, vlm_answer, threshold, stats, is_narration)
		except Exception as e:
			self.get_logger().warn(f"Semantic voxel update error: {e}")

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

			# Rays as arrows
			def _offset_origin(base: np.ndarray, direction: np.ndarray, idx: int) -> np.ndarray:
				offset_dir = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=np.float32))
				if np.linalg.norm(offset_dir) < 1e-6:
					offset_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
				offset_dir /= np.linalg.norm(offset_dir) + 1e-9
				offset_mag = float(self.voxel_resolution) * 0.3 * ((idx % 5) - 2)
				return base + offset_dir * offset_mag

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

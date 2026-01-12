#!/usr/bin/env python3
"""
Resilience Main Node - Clean NARadio Pipeline

Simplified node focused on drift detection, NARadio processing, and semantic mapping.
Removed YOLO/SAM and historical analysis components for lightweight operation.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import json
import threading
import time
import cv2
import warnings
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
warnings.filterwarnings('ignore')

from resilience.path_manager import PathManager
from resilience.naradio_processor import NARadioProcessor
from resilience.narration_manager import NarrationManager
from resilience.risk_buffer import RiskBufferManager
from resilience.pointcloud_utils import depth_to_meters as pc_depth_to_meters, depth_mask_to_world_points, voxelize_pointcloud, create_cloud_xyz
from resilience.cause_registry import CauseRegistry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


class ResilienceNode(Node):
    """Resilience Node - Clean NARadio pipeline with drift detection and semantic mapping."""
    
    def __init__(self):
        super().__init__('resilience_node')

        # Professional startup message
        self.get_logger().info("=" * 60)
        self.get_logger().info("RESILIENCE SYSTEM INITIALIZING")
        self.get_logger().info("=" * 60)

        # Track recent VLM answers for smart semantic processing
        self.recent_vlm_answers = {}  # vlm_answer -> timestamp
        self.cause_registry = CauseRegistry()
        self.cause_registry_snapshot_path = None
        
        # OPTIMIZATION: Debounce registry snapshot saves to reduce I/O
        self.last_registry_save_time = 0.0
        self.registry_save_debounce_interval = 2.0  # Save at most once every 2 seconds
        
        # FIX: Track causes that have already had narration masks published (by vec_id for canonical identity)
        # Prevents double voxel publishing for the same or similar causes (similarity >0.8)
        # Since cause_registry merges similar causes to the same vec_id, this handles both exact and similar matches
        self.narration_published_vec_ids = set()  # Set of vec_ids that have narration masks published
        
        # Store RGB images with timestamps for hotspot publishing
        self.rgb_images_with_timestamps = []  # [(rgb_msg, timestamp)]
        self.max_rgb_buffer = 50  # Keep last 50 RGB images
        
        self.declare_parameters('', [
            ('flip_y_axis', False),
            ('use_tf', False),
            ('radio_model_version', 'radio_v2.5-b'),
            ('radio_lang_model', 'siglip'),
            ('radio_input_resolution', 512),
            ('enable_naradio_visualization', True),
            ('enable_combined_segmentation', True),
            ('main_config_path', ''),
            ('mapping_config_path', ''),
            ('enable_voxel_mapping', True),
            ('pose_is_base_link', True)
        ])

        param_values = self.get_parameters([
            'flip_y_axis', 'use_tf',
            'radio_model_version', 'radio_lang_model', 'radio_input_resolution',
            'pose_is_base_link',
            'enable_naradio_visualization', 'enable_combined_segmentation',
            'main_config_path', 'mapping_config_path', 'enable_voxel_mapping'
        ])
        
        (self.flip_y_axis, self.use_tf,
         self.radio_model_version, self.radio_lang_model, self.radio_input_resolution,
         self.pose_is_base_link, self.enable_naradio_visualization, self.enable_combined_segmentation,
         self.main_config_path, self.mapping_config_path, self.enable_voxel_mapping
        ) = [p.value for p in param_values]

        # Load topic configuration from main config
        self.load_topic_configuration()

    def load_topic_configuration(self):
        """Load topic configuration from main config file."""
        try:
            import yaml
            if self.main_config_path:
                config_path = self.main_config_path
            else:
                # Use default config path
                from ament_index_python.packages import get_package_share_directory
                package_dir = get_package_share_directory('resilience')
                config_path = os.path.join(package_dir, 'config', 'main_config.yaml')
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract topic configuration
            topics = config.get('topics', {})
            
            # Input topics
            self.rgb_topic = topics.get('rgb_topic', '/robot_1/sensors/front_stereo/right/image')
            self.depth_topic = topics.get('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered')
            self.pose_topic = topics.get('pose_topic', '/robot_1/sensors/front_stereo/pose')
            self.camera_info_topic = topics.get('camera_info_topic', '/robot_1/sensors/front_stereo/right/camera_info')
            self.vlm_answer_topic = topics.get('vlm_answer_topic', '/vlm_answer')
            
            # Output topics
            self.drift_narration_topic = topics.get('drift_narration_topic', '/drift_narration')
            self.narration_text_topic = topics.get('narration_text_topic', '/narration_text')
            self.naradio_image_topic = topics.get('naradio_image_topic', '/naradio_image')
            self.narration_image_topic = topics.get('narration_image_topic', '/narration_image')
            self.vlm_similarity_map_topic = topics.get('vlm_similarity_map_topic', '/vlm_similarity_map')
            self.vlm_similarity_colored_topic = topics.get('vlm_similarity_colored_topic', '/vlm_similarity_colored')
            self.vlm_objects_legend_topic = topics.get('vlm_objects_legend_topic', '/vlm_objects_legend')
            
            # Extract path configuration
            self.path_config = config.get('path_mode', {})
            
            self.get_logger().info(f"Topic configuration loaded from: {config_path}")
            self.get_logger().info(f"Path mode: {self.path_config.get('mode', 'json_file')}")
            
        except Exception as e:
            pass
        #     self.get_logger().warn(f"Using default topic configuration: {e}")
        #     # Fallback to default topics
        #     self.rgb_topic = '/robot_1/sensors/front_stereo/right/image'
        #     self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
        #     self.pose_topic = '/robot_1/sensors/front_stereo/pose'
        #     self.camera_info_topic = '/robot_1/sensors/front_stereo/right/camera_info'
        #     self.vlm_answer_topic = '/vlm_answer'
        #     self.drift_narration_topic = '/drift_narration'
        #     self.narration_text_topic = '/narration_text'
        #     self.naradio_image_topic = '/naradio_image'
        #     self.narration_image_topic = '/narration_image'
        #     self.vlm_similarity_map_topic = '/vlm_similarity_map'
        #     self.vlm_similarity_colored_topic = '/vlm_similarity_colored'
        #     self.vlm_objects_legend_topic = '/vlm_objects_legend'
            
        #     # Default path configuration
        #     self.path_config = {'mode': 'json_file', 'global_path_topic': '/global_path'}
        #     self.get_logger().info("Using default topic configuration")

        self.init_components()
        
        self.last_breach_state = False
        self.current_breach_active = False
        
        if self.use_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        else:
            self.tf_broadcaster = None
            self.tf_buffer = None
            self.tf_listener = None

        self.bridge = CvBridge()
        self.camera_intrinsics = [186.24478149414062, 186.24478149414062, 238.66322326660156, 141.6264190673828]
        self.camera_info_received = False

        self.init_publishers()
        self.init_subscriptions()
        self.init_services()

        self.last_rgb_msg = None
        self.last_depth_msg = None
        self.last_pose = None
        self.last_pose_time = None
        self.lock = threading.Lock()
        self.breach_idx = None
        
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.latest_pose = None
        self.latest_pose_time = None
        
        self.naradio_processing_lock = threading.Lock()
        self.naradio_is_processing = False
        self.naradio_running = True
        
        self.detection_pose = None
        self.detection_pose_time = None
        
        self.image_buffer = []
        self.max_buffer_size = 150
        self.rolling_image_buffer = []
        self.rolling_buffer_duration = 1.0
        
        self.transform_matrix_cache = None
        self.last_transform_time = 0
        self.transform_cache_duration = 0.1

        self.init_risk_buffer_manager()
        self.init_semantic_bridge()
        
        # Wait for path to be ready before starting functionality
        self.wait_for_path_ready()
        
        self.start_naradio_thread()
        
        self.print_initialization_status()

        # PointCloud worker thread state (only if direct_mapping is enabled)
    

    def wait_for_path_ready(self):
        """Wait for path to be ready before starting main functionality."""
        self.get_logger().info("Waiting for path to be ready...")
        
        timeout_seconds = 10.0  # Default timeout
        if self.path_manager.get_mode() == 'external_planner':
            timeout_seconds = self.path_config.get('external_planner', {}).get('timeout_seconds', 30.0)
        
        # For external planner mode, check periodically instead of blocking
        if self.path_manager.get_mode() == 'external_planner':
            self.get_logger().info(f"External planner mode: Waiting up to {timeout_seconds}s for path...")            
            
            if self.path_manager.is_ready():
                self.get_logger().info("External path received - starting main functionality")
                self.path_ready = True
                
                # Update narration manager with path points
                nominal_points = self.path_manager.get_nominal_points_as_numpy()
                if len(nominal_points) > 0:
                    self.narration_manager.update_intended_trajectory(nominal_points)
                    self.get_logger().info("Updated narration manager with external path points")
            else:
                self.get_logger().warn("External path not received within timeout")
                self.path_ready = False
                self.disable_drift_detection = True

    def can_proceed_with_drift_detection(self) -> bool:
        """Check if drift detection can proceed."""
        return (hasattr(self, 'path_ready') and 
                self.path_ready and 
                (not hasattr(self, 'disable_drift_detection') or 
                 not self.disable_drift_detection))

    def init_publishers(self):
        """Initialize publishers."""
        publishers = [
            (self.drift_narration_topic, String, 10),
            (self.narration_text_topic, String, 10),
            (self.naradio_image_topic, Image, 10),
            (self.narration_image_topic, Image, 10)
        ]
        
        self.narration_pub, self.narration_text_pub, self.naradio_image_pub, \
        self.narration_image_pub = [self.create_publisher(msg_type, topic, qos) 
                                   for topic, msg_type, qos in publishers]
        
        if self.enable_combined_segmentation:
            vlm_publishers = [
                (self.vlm_similarity_map_topic, Image, 10),
                (self.vlm_similarity_colored_topic, Image, 10),
                (self.vlm_objects_legend_topic, String, 10)
            ]
            self.original_mask_pub, self.refined_mask_pub, self.segmentation_legend_pub = \
                [self.create_publisher(msg_type, topic, qos) for topic, msg_type, qos in vlm_publishers]

    def init_subscriptions(self):
        """Initialize subscriptions."""
        subscriptions = [
            (self.rgb_topic, Image, self.rgb_callback, 1),
            (self.depth_topic, Image, self.depth_callback, 1),
            (self.pose_topic, PoseStamped, self.pose_callback, 10),
            (self.camera_info_topic, CameraInfo, self.camera_info_callback, 1),
            (self.vlm_answer_topic, String, self.vlm_answer_callback, 10)
        ]
        
        for topic, msg_type, callback, qos in subscriptions:
            self.create_subscription(msg_type, topic, callback, qos)
    
    def init_services(self):
        """Initialize ROS services for registry access (JSON over String messages)."""
        self.registry_query_sub = self.create_subscription(
            String,
            '/cause_registry/query',
            self._handle_registry_query,
            10
        )
        
        self.registry_response_pub = self.create_publisher(
            String,
            '/cause_registry/response',
            10
        )
        
        self.get_logger().info("Cause registry services initialized (topic-based)")
    
    def _handle_registry_query(self, msg):
        """Handle registry query via JSON message."""
        try:
            query = json.loads(msg.data)
            query_type = query.get('type')
            query_id = query.get('query_id')  # For async callback routing
            response_data = {'success': False, 'message': 'Unknown query type'}
            
            if query_id:
                response_data['query_id'] = query_id
            
            if query_type == 'get_by_name':
                name = query.get('name')
                entry = self.cause_registry.get_entry_by_name(name) if name else None
                if entry:
                    response_data = {
                        'success': True,
                        'vec_id': entry.vec_id,
                        'names': entry.names,
                        'color_rgb': entry.color_rgb,
                        'embedding': entry.embedding.tolist(),
                        'enhanced_embedding': entry.enhanced_embedding.tolist() if entry.enhanced_embedding is not None else None,
                        'gp_params': self._gp_params_to_dict(entry.gp_params),
                        'metadata': entry.metadata,
                        'stats': entry.stats
                    }
                    if query_id:
                        response_data['query_id'] = query_id
                else:
                    response_data = {'success': False, 'message': 'Entry not found'}
                    if query_id:
                        response_data['query_id'] = query_id
            
            elif query_type == 'get_by_vec_id':
                vec_id = query.get('vec_id')
                entry = self.cause_registry.get_entry_by_vec_id(vec_id) if vec_id else None
                if entry:
                    response_data = {
                        'success': True,
                        'vec_id': entry.vec_id,
                        'names': entry.names,
                        'color_rgb': entry.color_rgb,
                        'embedding': entry.embedding.tolist(),
                        'enhanced_embedding': entry.enhanced_embedding.tolist() if entry.enhanced_embedding is not None else None,
                        'gp_params': self._gp_params_to_dict(entry.gp_params),
                        'metadata': entry.metadata,
                        'stats': entry.stats
                    }
                    if query_id:
                        response_data['query_id'] = query_id
                else:
                    response_data = {'success': False, 'message': 'Entry not found'}
                    if query_id:
                        response_data['query_id'] = query_id
            
            elif query_type == 'set_gp':
                name = query.get('name')
                vec_id = query.get('vec_id')
                gp_data = query.get('gp_params', {})
                
                entry = None
                if vec_id:
                    # Prefer vec_id (embedding-indexed) over name
                    entry = self.cause_registry.get_entry_by_vec_id(vec_id)
                elif name:
                    entry = self.cause_registry.get_entry_by_name(name)
                
                if entry:
                    from resilience.cause_registry import GPParams
                    gp_params = GPParams(
                        lxy=gp_data.get('lxy'),
                        lz=gp_data.get('lz'),
                        A=gp_data.get('A'),
                        b=gp_data.get('b'),
                        mse=gp_data.get('mse'),
                        rmse=gp_data.get('rmse'),
                        mae=gp_data.get('mae'),
                        r2_score=gp_data.get('r2_score'),
                        timestamp=gp_data.get('timestamp'),
                        buffer_id=gp_data.get('buffer_id')
                    )
                    # Use vec_id directly if available, otherwise fall back to name
                    if vec_id:
                        # Update via vec_id (embedding-indexed)
                        success = self.cause_registry.set_gp_params(
                            entry.names[0] if entry.names else '',
                            gp_params
                        )
                    else:
                        # Fallback to name lookup
                        success = self.cause_registry.set_gp_params(
                            name or (entry.names[0] if entry.names else ''),
                            gp_params
                        )
                    if success:
                        self.save_cause_registry_snapshot()
                        response_data = {'success': True, 'message': 'GP params updated'}
                    else:
                        response_data = {'success': False, 'message': 'Failed to update GP params'}
                else:
                    response_data = {'success': False, 'message': 'Entry not found'}
                
                if query_id:
                    response_data['query_id'] = query_id
            
            # Publish response
            response_msg = String(data=json.dumps(response_data))
            self.registry_response_pub.publish(response_msg)
            
        except Exception as e:
            error_response = {'success': False, 'message': f'Error: {str(e)}'}
            self.registry_response_pub.publish(String(data=json.dumps(error_response)))
    
    def _gp_params_to_dict(self, gp_params):
        """Convert GPParams to dict for JSON serialization."""
        if gp_params is None:
            return None
        return {
            'lxy': gp_params.lxy,
            'lz': gp_params.lz,
            'A': gp_params.A,
            'b': gp_params.b,
            'mse': gp_params.mse,
            'rmse': gp_params.rmse,
            'mae': gp_params.mae,
            'r2_score': gp_params.r2_score,
            'timestamp': gp_params.timestamp,
            'buffer_id': gp_params.buffer_id
        }

    def print_initialization_status(self):
        """Print initialization status."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("RESILIENCE SYSTEM READY")
        self.get_logger().info("=" * 60)
        
        self.get_logger().info(f"Path Configuration:")
        self.get_logger().info(f"   Mode: {self.path_manager.get_mode()}")
        self.get_logger().info(f"   Topic: {self.path_manager.get_path_topic()}")
        self.get_logger().info(f"   Status: {'READY' if hasattr(self, 'path_ready') and self.path_ready else 'NOT READY'}")
        
        if hasattr(self, 'disable_drift_detection'):
            self.get_logger().info(f"Drift Detection: {'ENABLED' if not self.disable_drift_detection else 'DISABLED'}")
        
        soft_thresh, hard_thresh = self.path_manager.get_thresholds()
        self.get_logger().info(f"Thresholds: Soft={soft_thresh:.3f}m, Hard={hard_thresh:.3f}m")
        
        self.get_logger().info(f"NARadio Processing: {'READY' if self.naradio_processor.is_ready() else 'NOT READY'}")
        self.get_logger().info(f"Voxel Mapping: {'ENABLED' if self.enable_voxel_mapping else 'DISABLED'}")
        
        vlm_enabled = (self.enable_combined_segmentation and 
                      hasattr(self, 'naradio_processor') and 
                      self.naradio_processor.is_segmentation_ready())
        self.get_logger().info(f"VLM Similarity: {'ENABLED' if vlm_enabled else 'DISABLED'}")
        
        if vlm_enabled:
            all_objects = self.naradio_processor.get_all_objects()
            self.get_logger().info(f"   Objects loaded: {len(all_objects)}")
            
        config = self.naradio_processor.segmentation_config
        prefer_enhanced = config['segmentation'].get('prefer_enhanced_embeddings', True)
        self.get_logger().info(f"Embedding Method: {'ENHANCED' if prefer_enhanced else 'TEXT'}")
        
        self.get_logger().info("=" * 60)


    def init_components(self):
        """Initialize resilience components."""
        self.get_logger().info("Initializing system components...")
        
        # Initialize path manager with unified interface
        self.path_manager = PathManager(self, self.path_config)
        self.get_logger().info("Path Manager initialized")
        
        # Get thresholds from path manager
        soft_threshold, hard_threshold = self.path_manager.get_thresholds()
        
        # Wait for path to be ready and print discretization results
        if self.path_manager.wait_for_path(timeout_seconds=2.0):
            discretized_points = self.path_manager.get_discretized_nominal_points()
            self.get_logger().info(f"Path loaded: {len(discretized_points)} points, {self.path_manager.get_sampling_distance():.3f}m sampling")
        else:
            self.get_logger().warn("Path not ready within timeout - will retry during operation")
        
        try:
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.main_config_path if self.main_config_path else None,
                cause_registry=self.cause_registry
            )
            
            # Read voxel mapping parameters from main config (non-blocking)
            self.enable_voxel_mapping = False  # Default value
            self.direct_mapping = False  # Default value
            if (self.naradio_processor.is_ready() and 
                hasattr(self.naradio_processor, 'segmentation_config')):
                try:
                    self.enable_voxel_mapping = self.naradio_processor.segmentation_config.get('enable_voxel_mapping', False)
                    self.direct_mapping = self.naradio_processor.segmentation_config.get('direct_mapping', False)
                except Exception as e:
                    self.get_logger().warn(f"Could not read voxel mapping parameters from config: {e}")
                    self.enable_voxel_mapping = False
                    self.direct_mapping = False
            else:
                self.get_logger().warn("NARadio processor not ready, using default voxel mapping: False, direct mapping: False")
                
        except Exception as e:
            self.get_logger().error(f"Error initializing NARadio processor: {e}")
            import traceback
            traceback.print_exc()
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.main_config_path if self.main_config_path else None,
                cause_registry=self.cause_registry
            )
        
        # Initialize narration manager with discretization parameters
        lookback_window_size = self.path_manager.get_lookback_window_size()
        sampling_distance = self.path_manager.get_sampling_distance()
        self.narration_manager = NarrationManager(
            soft_threshold, 
            hard_threshold, 
            lookback_window_size=lookback_window_size,
            sampling_distance=sampling_distance
        )
        self.get_logger().info("Narration Manager initialized")
        
        # Ensure voxel mapping parameters are always set (final fallback)
        if not hasattr(self, 'enable_voxel_mapping'):
            self.enable_voxel_mapping = False
            self.get_logger().info("Voxel mapping parameter not set, using default: False")
        if not hasattr(self, 'direct_mapping'):
            self.direct_mapping = False
            self.get_logger().info("Direct mapping parameter not set, using default: False")
        
        # Set nominal trajectory points if available (use discretized data)
        nominal_points = self.path_manager.get_discretized_nominal_as_numpy()
        if len(nominal_points) > 0:
            self.narration_manager.set_intended_trajectory(nominal_points)
            self.get_logger().info(f"Narration manager initialized with {len(nominal_points)} discretized points")
        else:
            self.get_logger().warn("No discretized nominal points available for narration manager")
    
    def init_risk_buffer_manager(self):
        """Initialize risk buffer manager."""
        try:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            unique_id = str(uuid.uuid4())[:8]
            self.risk_buffer_save_dir = '/home/navin/ros2_ws/src/buffers'
            os.makedirs(self.risk_buffer_save_dir, exist_ok=True)
            
            self.current_run_dir = os.path.join(self.risk_buffer_save_dir, f"run_{run_timestamp}_{unique_id}")
            os.makedirs(self.current_run_dir, exist_ok=True)
            
            self.risk_buffer_manager = RiskBufferManager(save_directory=self.current_run_dir)
            print(f"Buffer directory: {self.current_run_dir}")
            
            self.node_id = f"resilience_{unique_id}"
            self.cause_registry_snapshot_path = os.path.join(self.current_run_dir, "cause_registry.json")
            self.save_cause_registry_snapshot(force=True)  # Force initial save
            
        except Exception as e:
            print(f"Error initializing risk buffer manager: {e}")
            self.risk_buffer_manager = None
    
    def init_semantic_bridge(self):
        """Initialize semantic hotspot bridge for communication with octomap."""
        try:
            if self.enable_voxel_mapping:
                # Load semantic bridge config from main config
                main_config = getattr(self.naradio_processor, 'segmentation_config', {})
                
                from resilience.semantic_info_bridge import SemanticHotspotPublisher
                self.semantic_bridge = SemanticHotspotPublisher(self, main_config)
                print("Semantic bridge initialized")
            else:
                self.semantic_bridge = None
                print("Semantic bridge disabled")
        except Exception as e:
            print(f"Error initializing semantic bridge: {e}")
            self.semantic_bridge = None
    
    def start_naradio_thread(self):
        """Start the parallel NARadio processing thread."""
        if not hasattr(self, 'naradio_thread') or self.naradio_thread is None or not self.naradio_thread.is_alive():
            self.naradio_running = True
            self.naradio_thread = threading.Thread(target=self.naradio_processing_loop, daemon=True)
            self.naradio_thread.start()
            print("NARadio processing thread started")
        else:
            print("NARadio thread already running")

    def rgb_callback(self, msg):
        """Store RGB message with timestamp for hotspot publishing and sync buffers."""
        msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception:
            return
        # Store RGB image with timestamp for hotspot publishing
        self.rgb_images_with_timestamps.append((msg, msg_timestamp))
        if len(self.rgb_images_with_timestamps) > self.max_rgb_buffer:
            self.rgb_images_with_timestamps.pop(0)

        
        with self.processing_lock:
            self.latest_rgb_msg = msg
            if self.latest_pose is not None:
                self.detection_pose = self.latest_pose.copy()
                self.detection_pose_time = self.latest_pose_time
        
        self.image_buffer.append((cv_image, msg_timestamp, msg))
        
        if len(self.image_buffer) > self.max_buffer_size:
            self.image_buffer.pop(0)
        
        current_system_time = time.time()
        self.rolling_image_buffer.append((cv_image, current_system_time, msg))
        
        while self.rolling_image_buffer and (current_system_time - self.rolling_image_buffer[0][1]) > self.rolling_buffer_duration:
            self.rolling_image_buffer.pop(0)
        
    def depth_callback(self, msg):
        """Store latest depth message and push into depth buffer."""
        with self.processing_lock:
            self.latest_depth_msg = msg
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = pc_depth_to_meters(depth_img, msg.encoding)
        except Exception:
            return
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def pose_callback(self, msg):
        """Process pose and trigger detection with consolidated pose updates."""
        # Always compute and print drift, even if detection is disabled
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        drift = 0.0
        nearest_idx = -1
        if self.path_manager.is_ready():
            drift, nearest_idx = self.path_manager.compute_drift(pos)
        
        # Gate the rest of processing on readiness
        if not self.can_proceed_with_drift_detection():
            return
        
        # Check if path manager is ready
        if not self.path_manager.is_ready():
            print("Path manager not ready, skipping pose processing")
            return
        
        self.breach_idx = nearest_idx
        
        with self.lock:
            self.latest_pose = pos
            self.latest_pose_time = pose_time
            self.last_pose = pos
            self.last_pose_time = pose_time
            
            self.narration_manager.add_actual_point(pos, pose_time, self.flip_y_axis)

        breach_now = self.path_manager.is_breach(drift)
        
        breach_started = not self.last_breach_state and breach_now
        breach_ended = self.last_breach_state and not breach_now
        
        if breach_started:
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            
            self.get_logger().warn(f"BREACH STARTED - Drift: {drift:.3f}m (threshold: {self.path_manager.get_thresholds()[0]:.3f}m)")
            
            # Start new buffer when breach begins
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
            
            self.narration_manager.queue_breach_event('start', pose_time)
            
        elif breach_ended:
            self.last_breach_state = False
            self.current_breach_active = False
            
            self.get_logger().info(f"BREACH ENDED - Drift: {drift:.3f}m (threshold: {self.path_manager.get_thresholds()[0]:.3f}m)")
            
            # Freeze buffers when breach ends
            if self.risk_buffer_manager:
                frozen_buffers = self.risk_buffer_manager.freeze_active_buffers(pose_time)
            
            self.narration_manager.queue_breach_event('end', pose_time)
            
        elif breach_now and not self.current_breach_active:
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            
            self.get_logger().warn(f"BREACH DETECTED - Drift: {drift:.3f}m (threshold: {self.path_manager.get_thresholds()[0]:.3f}m)")
            
            # Start new buffer when breach is detected
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
            
            self.narration_manager.queue_breach_event('start', pose_time)
        
        if not breach_started and not breach_ended and not (breach_now and not self.current_breach_active):
            self.last_breach_state = breach_now
        
        with self.lock:
            if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0 and self.current_breach_active:
                self.risk_buffer_manager.add_pose(pose_time, pos, drift)
                

        if self.current_breach_active and not self.narration_manager.get_narration_sent():
            narration = self.narration_manager.check_for_narration(pose_time, self.breach_idx)
            if narration:
                self.publish_narration_with_image(narration)
                self.narration_pub.publish(String(data=narration))

    def publish_narration_with_image(self, narration_text):
        """Publish both narration text and accompanying image together"""
        if not self.image_buffer:
            self.narration_text_pub.publish(String(data=narration_text))
            return
            
        newest_timestamp = self.image_buffer[-1][1]
        current_time = newest_timestamp
        
        target_time_offset = 1.0
        
        if self.image_buffer:
            target_time = current_time - target_time_offset
            
            oldest_timestamp = self.image_buffer[0][1] if self.image_buffer else current_time
            available_time_back = current_time - oldest_timestamp
            
            if available_time_back < target_time_offset:
                target_time = oldest_timestamp
                actual_offset = available_time_back
            else:
                actual_offset = target_time_offset
            
            closest_image = None
            closest_msg = None
            min_time_diff = float('inf')
            
            for image, timestamp, msg in self.image_buffer:
                time_diff = abs(timestamp - target_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_image = image
                    closest_msg = msg
            
            if closest_image is not None and closest_msg is not None:
                # Get the original image timestamp from the message
                original_image_timestamp = closest_msg.header.stamp.sec + closest_msg.header.stamp.nanosec * 1e-9
                
                self.save_narration_image_to_buffer(closest_image, narration_text, current_time)
                
                # Store narration data in active buffers for VLM processing
                # Include the original image timestamp for proper semantic mapping
                if self.risk_buffer_manager:
                    self.risk_buffer_manager.store_narration_data_with_timestamp(
                        closest_image, narration_text, current_time, original_image_timestamp
                    )
                
                image_msg = self.bridge.cv2_to_imgmsg(closest_image, encoding='rgb8')
                image_msg.header.stamp = closest_msg.header.stamp
                image_msg.header.frame_id = closest_msg.header.frame_id
                self.narration_image_pub.publish(image_msg)
                
                self.narration_text_pub.publish(String(data=narration_text))
            else:
                self.narration_text_pub.publish(String(data=narration_text))
        else:
            self.narration_text_pub.publish(String(data=narration_text))

    def load_enhanced_embedding_from_buffer(self, buffer_dir: str, vlm_answer: str) -> Optional[np.ndarray]:
        """Load enhanced embedding from buffer directory."""
        try:
            embeddings_dir = os.path.join(buffer_dir, 'enhanced_embeddings')
            if not os.path.exists(embeddings_dir):
                return None
            
            # Look for enhanced embedding files for this VLM answer
            safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
            embedding_files = [f for f in os.listdir(embeddings_dir) 
                             if f.startswith(f"enhanced_embedding_{safe_vlm_name}") and f.endswith('.npy')]
            
            if not embedding_files:
                return None
            
            # Get the most recent embedding file
            embedding_files.sort()
            latest_embedding_file = embedding_files[-1]
            embedding_path = os.path.join(embeddings_dir, latest_embedding_file)
            
            # Load the enhanced embedding
            enhanced_embedding = np.load(embedding_path)
            
            return enhanced_embedding
            
        except Exception as e:
            print(f"Error loading enhanced embedding: {e}")
            return None

    def save_narration_image_to_buffer(self, image, narration_text, current_time):
        """Save narration image to the current buffer directory."""
        try:
            with self.lock:
                if len(self.risk_buffer_manager.active_buffers) == 0:
                    print("No active buffers to save narration image to")
                    return
            
            current_buffer = self.risk_buffer_manager.active_buffers[-1]
            
            buffer_dir = os.path.join(self.current_run_dir, current_buffer.buffer_id)
            narration_dir = os.path.join(buffer_dir, 'narration')
            os.makedirs(narration_dir, exist_ok=True)
            
            timestamp_str = f"{current_time:.3f}"
            image_filename = f"narration_image_{timestamp_str}.png"
            image_path = os.path.join(narration_dir, image_filename)
            cv2.imwrite(image_path, image)
            
        except Exception as e:
            print(f"Error saving narration image to buffer: {e}")
            import traceback
            traceback.print_exc()

    def publish_narration_hotspot_mask(self, narration_image: np.ndarray, vlm_answer: str, 
                                      original_image_timestamp: float, buffer_id: str) -> bool:
        """
        Publish narration image hotspot mask through semantic bridge for semantic voxel mapping.
        
        Args:
            narration_image: The narration image that was used for similarity processing
            vlm_answer: The VLM answer/cause that was identified
            original_image_timestamp: Timestamp when the original image was recorded (not narration time)
            buffer_id: Buffer ID for tracking
            
        Returns:
            True if successfully published, False otherwise
        """
        try:
            if not hasattr(self, 'semantic_bridge') or self.semantic_bridge is None:
                return False
            
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                return False
            
            # Narration stage: ALWAYS compute similarity using text-based path to bootstrap enhanced embedding
            similarity_result = self.naradio_processor.process_vlm_similarity_visualization_optimized(
                narration_image, vlm_answer, feat_map_np=None
            )
            
            if not similarity_result or 'similarity_map' not in similarity_result:
                print(f"Failed to compute similarity for narration image")
                return False
            
            # Extract similarity map and apply binary threshold
            similarity_map = similarity_result['similarity_map']
            threshold = similarity_result.get('threshold_used', 0.6)
            
            # Create binary hotspot mask
            hotspot_mask = (similarity_map > threshold).astype(np.uint8)
            
            if not np.any(hotspot_mask):
                print(f"No hotspots found in narration image for '{vlm_answer}'")
                return False
            
            # Create single VLM hotspot dictionary (same format as merged hotspots)
            vlm_hotspots = {vlm_answer: hotspot_mask}
            
            # Publish through semantic bridge with original image timestamp
            success = self.semantic_bridge.publish_merged_hotspots(
                vlm_hotspots=vlm_hotspots,
                timestamp=original_image_timestamp, narration=True,  # Use original image timestamp
                original_image=narration_image,
                buffer_id=buffer_id
            )
            
            if success:
                print(f"✓ Published narration hotspot mask for '{vlm_answer}' through semantic bridge")
                print(f"  Original timestamp: {original_image_timestamp:.6f}")
                print(f"  Hotspot pixels: {int(np.sum(hotspot_mask))}")
                print(f"  Threshold: {threshold:.3f}")
                return True
            else:
                print(f"✗ Failed to publish narration hotspot mask through semantic bridge")
                return False
                
        except Exception as e:
            print(f"Error publishing narration hotspot mask: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_ros_timestamp(self, msg):
        """Extract ROS timestamp as float from message header."""
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return time.time()


    def naradio_processing_loop(self):
        """OPTIMIZED: Parallel NARadio processing loop with frame skipping and reduced overhead."""
        print("NARadio processing loop started (OPTIMIZED)")
        
        last_memory_cleanup = time.time()
        memory_cleanup_interval = 60.0
        
        # OPTIMIZATION: Frame skip counter
        process_every_n_frames = 2  # Process every 2nd frame for predictive similarity
        frame_counter = 0
        
        # OPTIMIZATION: Cached RGB conversion
        last_rgb_msg_id = None
        cached_rgb_image = None
        
        while rclpy.ok() and self.naradio_running:
            try:
                current_time = time.time()
                
                # OPTIMIZATION: Less frequent memory cleanup
                if current_time - last_memory_cleanup > memory_cleanup_interval:
                    self.naradio_processor.cleanup_memory()
                    last_memory_cleanup = current_time
                
                # Check if processor is ready
                if not self.naradio_processor.is_ready():
                    time.sleep(0.1)
                    continue
                
                with self.processing_lock:
                    if self.latest_rgb_msg is None:
                        time.sleep(0.01)
                        continue
                    
                    rgb_msg = self.latest_rgb_msg
                    depth_msg = self.latest_depth_msg
                    pose_for_semantic = self.latest_pose.copy() if self.latest_pose is not None else None
                
                # OPTIMIZATION: Frame skipping for predictive similarity
                frame_counter += 1
                if frame_counter % process_every_n_frames != 0:
                    time.sleep(0.01)
                    continue
                
                # OPTIMIZATION: Cache RGB conversion to avoid repeated cv_bridge calls
                rgb_msg_id = id(rgb_msg)
                if rgb_msg_id != last_rgb_msg_id:
                    try:
                        cached_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
                        last_rgb_msg_id = rgb_msg_id
                    except Exception as e:
                        time.sleep(0.01)
                        continue
                
                rgb_image = cached_rgb_image
                
                # OPTIMIZATION: Skip visualization completely
                feat_map_np, _ = self.naradio_processor.process_features_optimized(
                    rgb_image, 
                    need_visualization=False,
                    reuse_features=True
                )
                
                # OPTIMIZATION: Only process if we have dynamic objects and features
                if (self.enable_combined_segmentation and 
                    self.naradio_processor.is_segmentation_ready() and
                    self.naradio_processor.dynamic_objects and
                    feat_map_np is not None):
                    
                    vlm_answers = self.naradio_processor.dynamic_objects
                    
                    # OPTIMIZATION: Use fast version
                    vlm_hotspots = self.naradio_processor.create_merged_hotspot_masks_fast(
                        rgb_image, vlm_answers, feat_map_np=feat_map_np)
                    
                    if vlm_hotspots and len(vlm_hotspots) > 0:
                        rgb_timestamp = self._get_ros_timestamp(rgb_msg)
                        self.semantic_bridge.publish_merged_hotspots(
                            vlm_hotspots=vlm_hotspots,
                            timestamp=rgb_timestamp,
                            original_image=None  # OPTIMIZATION: Skip overlay for predictive
                        )
                
                # OPTIMIZATION: Adaptive sleep based on processing load
                time.sleep(0.005)  # Reduced from 0.05s
                            
            except Exception as e:
                time.sleep(0.05)  
    
    def camera_info_callback(self, msg):
        """Handle camera info to get intrinsics."""
        with self.lock:
            if not self.camera_info_received:
                self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
                self.camera_info_received = True

    def vlm_answer_callback(self, msg):
        """Handle VLM answers for cause analysis and buffer association."""
        try:
            data = msg.data.strip()
            if not data or "VLM Error" in data or "VLM not available" in data:
                return
            
            # Parse JSON list: [{"name": str, "score": float}, ...]
            try:
                top_objects = json.loads(data)
                if not isinstance(top_objects, list):
                    top_objects = []  # Fallback for old format
            except json.JSONDecodeError:
                # Fallback: treat as old format "answer|score"
                parts = data.split('|')
                if len(parts) == 2:
                    top_objects = [{"name": parts[0], "score": float(parts[1])}]
                else:
                    top_objects = [{"name": data, "score": 1.0}]
            
            if not top_objects:
                return
            
            obj_list_str = ', '.join([f"{obj['name']} ({obj['score']:.4f})" for obj in top_objects])
            self.get_logger().info(f"VLM TOP OBJECTS RECEIVED: {obj_list_str}")
            
            # Process all top 4 objects: store in registry with confidence=score, only add high-score ones to processor
            primary_cause = None
            for obj_data in top_objects:
                vlm_answer = obj_data['name']
                score = float(obj_data.get('score', 0.0))
                
                # Track as recent
                self.recent_vlm_answers[vlm_answer] = time.time()
                
                # Store all in registry with confidence=score
                # Only add to processor (dynamic_objects) for predictive similarity if score > 0.8
                if hasattr(self, 'naradio_processor') and self.naradio_processor.is_ready():
                    if score > 0.8:
                        # Add to processor for predictive similarity (also adds to registry)
                        success = self.naradio_processor.add_vlm_object(vlm_answer)
                        if success:
                            self.cause_registry.record_detection(vlm_answer, score)
                            # Verify it's in dynamic_objects
                            if vlm_answer in self.naradio_processor.dynamic_objects:
                                self.get_logger().info(f"✓ '{vlm_answer}' added for predictive similarity (score: {score:.4f} > 0.8)")
                            else:
                                self.get_logger().warn(f"✗ '{vlm_answer}' not in dynamic_objects after add_vlm_object")
                        else:
                            self.get_logger().warn(f"Failed to add '{vlm_answer}' to processor")
                    else:
                        # For score <= 0.8: add to registry only, NOT to processor dynamic_objects
                        entry = self.cause_registry.get_entry_by_name(vlm_answer)
                        if entry is None:
                            # Encode and add to registry only (skip dynamic_objects)
                            try:
                                import torch
                                with torch.no_grad():
                                    if hasattr(self.naradio_processor, 'radio_encoder') and self.naradio_processor.radio_encoder:
                                        embedding = self.naradio_processor.radio_encoder.encode_labels([vlm_answer])
                                        embedding_np = embedding.detach().cpu().numpy().reshape(-1)
                                        self.cause_registry.upsert_cause(
                                            vlm_answer, embedding_np, source="vlm", type_="dynamic"
                                        )
                            except Exception as e:
                                self.get_logger().warn(f"Failed to add '{vlm_answer}' to registry: {e}")
                        self.cause_registry.record_detection(vlm_answer, score)
                        # Don't add to dynamic_objects - objects with score <= 0.8 won't get predictive similarity
                        self.get_logger().debug(f"'{vlm_answer}' stored in registry only (score: {score:.4f} <= 0.8, no predictive similarity)")
                
                # Use first (highest score) as primary cause for buffer association
                if primary_cause is None:
                    primary_cause = vlm_answer
            
            self.save_cause_registry_snapshot()
            
            # Associate primary cause with buffer and process narration
            if primary_cause:
                self.associate_vlm_answer_with_buffer(primary_cause)
                narration_success = self.process_narration_chain_for_vlm_answer(primary_cause)
                if narration_success:
                    self.get_logger().info(f"Narration processing completed for '{primary_cause}'")
            
        except Exception as e:
            print(f"Error processing VLM answer: {e}")
            import traceback
            traceback.print_exc()

    def process_narration_chain_for_vlm_answer(self, vlm_answer: str) -> bool:
        try:
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                print(f"NARadio processor not ready for narration processing")
                return False
            
            # OPTIMIZATION 1: Early duplicate check - exit before any expensive computation
            entry = self.cause_registry.get_entry_by_name(vlm_answer)
            if entry is not None and entry.vec_id in self.narration_published_vec_ids:
                self.get_logger().info(
                    f"Skipping narration mask for '{vlm_answer}' (vec_id: {entry.vec_id}) - "
                    f"already published narration mask for this or similar cause (similarity >0.8)"
                )
                return True  # Return True since we intentionally skipped (not an error)
            
            # Find the buffer that was just assigned the cause
            target_buffer = None
            with self.lock:
                # Check frozen buffers first (most recent with this cause)
                for buffer in reversed(self.risk_buffer_manager.frozen_buffers):
                    if buffer.cause == vlm_answer:
                        target_buffer = buffer
                        break
                
                # If not found, check active buffers
                if target_buffer is None:
                    for buffer in reversed(self.risk_buffer_manager.active_buffers):
                        if buffer.cause == vlm_answer:
                            target_buffer = buffer
                            break
            
            if not target_buffer:
                print(f"No buffer found with cause '{vlm_answer}' for narration processing")
                return False
            
            buffer_dir = os.path.join(self.current_run_dir, target_buffer.buffer_id)
            
            # Get narration image
            narration_image = None
            if target_buffer.has_narration_image():
                narration_image = target_buffer.get_narration_image()
            else:
                # Fallback: look for narration images on disk
                narration_dir = os.path.join(buffer_dir, 'narration')
                if os.path.exists(narration_dir):
                    narration_files = [f for f in os.listdir(narration_dir) if f.endswith('.png')]
                    if narration_files:
                        # Get the most recent narration image
                        narration_files.sort()
                        latest_narration_file = narration_files[-1]
                        narration_image_path = os.path.join(narration_dir, latest_narration_file)
                        
                        # Load the narration image
                        narration_image = cv2.imread(narration_image_path)
                        if narration_image is not None:
                            # Convert BGR to RGB
                            narration_image = cv2.cvtColor(narration_image, cv2.COLOR_BGR2RGB)
            
            if narration_image is None:
                print(f"No narration image found for buffer {target_buffer.buffer_id}")
                return False
            
            print(f"Processing narration image for '{vlm_answer}' from buffer {target_buffer.buffer_id}")

            # OPTIMIZATION 2: Extract features ONCE and reuse throughout
            feat_map_np, _ = self.naradio_processor.process_features_optimized(
                narration_image, need_visualization=False, reuse_features=False
            )
            if feat_map_np is None:
                print(f"Failed to extract features for narration image")
                return False

            # OPTIMIZATION 3: Compute similarity map ONCE using pre-computed features
            similarity_map = self.naradio_processor.compute_vlm_similarity_map_optimized(
                narration_image, vlm_answer, feat_map_np=feat_map_np, use_softmax=True, chunk_size=4000
            )
            if similarity_map is None:
                print(f"Failed to compute similarity for narration image")
                return False

            # Get threshold from config
            threshold = 0.9
            if hasattr(self.naradio_processor, 'segmentation_config'):
                threshold = self.naradio_processor.segmentation_config.get('segmentation', {}).get('hotspot_threshold', 0.6)
            
            # CRITICAL FIX #2: Use the similarity map already computed (no redundant computation)
            # The similarity_map computed above is already text-based, which is what we want
            similarity_map_final = similarity_map
            
            # Create hotspot mask using the computed similarity map
            hotspot_mask_final = (similarity_map_final > threshold).astype(np.uint8)
            if not np.any(hotspot_mask_final):
                self.get_logger().warn(f"Narration similarity produced no hotspots for '{vlm_answer}'")
                return False

            # Get original image timestamp (CRITICAL for proper synchronization)
            original_image_timestamp = None
            if target_buffer.get_original_image_timestamp() is not None:
                original_image_timestamp = target_buffer.get_original_image_timestamp()
            elif target_buffer.narration_timestamp is not None:
                original_image_timestamp = target_buffer.start_time
            else:
                original_image_timestamp = time.time()
            
            # Update registry (batch saves will be handled by debouncing)
            similarity_score = float(np.max(similarity_map_final))
            self.cause_registry.record_detection(vlm_answer, similarity_score)
            self.cause_registry.set_metadata(vlm_answer, {
                "last_buffer_id": target_buffer.buffer_id,
                "last_original_timestamp": original_image_timestamp,
                "last_similarity_threshold": threshold
            })
            # OPTIMIZATION 5: Debounce registry saves (will be saved periodically, not on every update)
            # Removed immediate save_cause_registry_snapshot() call here
            
            # Mark this cause as having narration mask published (by vec_id for canonical identity)
            if entry is not None:
                self.narration_published_vec_ids.add(entry.vec_id)
                self.get_logger().info(
                    f"Marking '{vlm_answer}' (vec_id: {entry.vec_id}) as having narration mask published"
                )
            else:
                self.get_logger().warn(
                    f"No cause registry entry found for '{vlm_answer}' - cannot track narration publication"
                )
            
            # Publish hotspot mask
            success_pub = self.semantic_bridge.publish_merged_hotspots(
                vlm_hotspots={vlm_answer: hotspot_mask_final},
                timestamp=original_image_timestamp,
                narration=True,
                original_image=narration_image,
                buffer_id=target_buffer.buffer_id
            )
            if not success_pub:
                return False
            
            self.get_logger().info(f"Published narration hotspot mask for '{vlm_answer}'")
            
            # OPTIMIZATION 6: Compute enhanced embedding asynchronously (non-blocking)
            # Uses pre-computed features and similarity map to avoid redundant computation
            try:
                enhanced_embedding = self.naradio_processor.compute_enhanced_cause_embedding(
                    narration_image, vlm_answer, 
                    similarity_map=similarity_map_final,  # Use final similarity map
                    feat_map_np=feat_map_np  # Reuse pre-computed features
                )
                if enhanced_embedding is not None:
                    # Save to buffer and register (async save could be added here)
                    self.naradio_processor._save_enhanced_embedding(vlm_answer, buffer_dir, enhanced_embedding)
                    target_buffer.assign_enhanced_cause_embedding(enhanced_embedding)
                    if self.naradio_processor.is_segmentation_ready():
                        self.naradio_processor.add_enhanced_embedding(vlm_answer, enhanced_embedding)
                    self.get_logger().info(f"Enhanced embedding ready for '{vlm_answer}'")
            except Exception as e:
                self.get_logger().debug(f"Enhanced embedding computation failed (non-critical): {e}")
            
            return True
            
        except Exception as e:
            print(f"Error in narration processing chain: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_cause_registry_snapshot(self, force: bool = False):
        """
        Persist the cause registry to the current run directory for other nodes.
        
        OPTIMIZATION: Debounced to reduce I/O - only saves if enough time has passed
        since last save, unless force=True.
        
        Args:
            force: If True, save immediately regardless of debounce interval
        """
        if not self.cause_registry_snapshot_path:
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_registry_save_time) < self.registry_save_debounce_interval:
            return  # Skip save due to debounce
        
        try:
            snapshot = self.cause_registry.snapshot()
            os.makedirs(os.path.dirname(self.cause_registry_snapshot_path), exist_ok=True)
            with open(self.cause_registry_snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
            self.last_registry_save_time = current_time
        except Exception as e:
            self.get_logger().warn(f"Failed to save cause registry snapshot: {e}")

    def associate_vlm_answer_with_buffer(self, vlm_answer):
        """Associate VLM answer with buffer."""
        
        success = self.risk_buffer_manager.assign_cause(vlm_answer)
        
        if success:
            print(f"Associated '{vlm_answer}' with risk buffer")
        else:
            print(f"No suitable buffer found for '{vlm_answer}'")
                    


    
    def _get_ros_timestamp(self, msg):
        """Extract ROS timestamp as float from message header."""
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return time.time()



def main():
    rclpy.init()
    node = ResilienceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down Resilience Node...")
        node.narration_manager.stop()
        if hasattr(node, 'naradio_running') and node.naradio_running:
            node.naradio_running = False
            if hasattr(node, 'naradio_thread') and node.naradio_thread and node.naradio_thread.is_alive():
                node.naradio_thread.join(timeout=2.0)

        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
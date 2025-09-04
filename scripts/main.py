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


class ResilienceNode(Node):
    """Resilience Node - Clean NARadio pipeline with drift detection and semantic mapping."""
    
    def __init__(self):
        super().__init__('resilience_node')

        # Track recent VLM answers for smart semantic processing
        self.recent_vlm_answers = {}  # vlm_answer -> timestamp
        
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
            ('enable_voxel_mapping', True)
        ])

        param_values = self.get_parameters([
            'flip_y_axis', 'use_tf',
            'radio_model_version', 'radio_lang_model', 'radio_input_resolution',
            'enable_naradio_visualization', 'enable_combined_segmentation',
            'main_config_path', 'mapping_config_path', 'enable_voxel_mapping'
        ])
        
        (self.flip_y_axis, self.use_tf,
         self.radio_model_version, self.radio_lang_model, self.radio_input_resolution,
         self.enable_naradio_visualization, self.enable_combined_segmentation,
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
            
            print(f"Loaded topic configuration from: {config_path}")
            print(f"Path mode: {self.path_config.get('mode', 'json_file')}")
            
        except Exception as e:
            print(f"Error loading topic configuration: {e}")
            # Fallback to default topics
            self.rgb_topic = '/robot_1/sensors/front_stereo/right/image'
            self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
            self.pose_topic = '/robot_1/sensors/front_stereo/pose'
            self.camera_info_topic = '/robot_1/sensors/front_stereo/right/camera_info'
            self.vlm_answer_topic = '/vlm_answer'
            self.drift_narration_topic = '/drift_narration'
            self.narration_text_topic = '/narration_text'
            self.naradio_image_topic = '/naradio_image'
            self.narration_image_topic = '/narration_image'
            self.vlm_similarity_map_topic = '/vlm_similarity_map'
            self.vlm_similarity_colored_topic = '/vlm_similarity_colored'
            self.vlm_objects_legend_topic = '/vlm_objects_legend'
            
            # Default path configuration
            self.path_config = {'mode': 'json_file', 'global_path_topic': '/global_path'}
            print("Using default topic configuration")

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

    def wait_for_path_ready(self):
        """Wait for path to be ready before starting main functionality."""
        print("Waiting for path to be ready...")
        
        # Get timeout from config
        timeout_seconds = 30.0  # Default timeout
        if self.path_manager.get_mode() == 'external_planner':
            timeout_seconds = self.path_config.get('external_planner', {}).get('timeout_seconds', 30.0)
        
        # For external planner mode, check periodically instead of blocking
        if self.path_manager.get_mode() == 'external_planner':
            print(f"External planner mode: Waiting up to {timeout_seconds}s for path...")
            start_time = time.time()
            
            while not self.path_manager.is_ready() and (time.time() - start_time) < timeout_seconds:
                time.sleep(0.5)  # Check every 0.5 seconds
                print(f"Still waiting for external path... ({time.time() - start_time:.1f}s elapsed)")
            
            if self.path_manager.is_ready():
                print("✓ External path received - starting main functionality")
                self.path_ready = True
                
                # Update narration manager with path points
                nominal_points = self.path_manager.get_nominal_points_as_numpy()
                if len(nominal_points) > 0:
                    self.narration_manager.update_intended_trajectory(nominal_points)
                    print("✓ Updated narration manager with external path points")
            else:
                print("✗ External path not received within timeout")
                self.path_ready = False
                self.disable_drift_detection = True
        else:
            # For JSON mode, use the original blocking wait
            if self.path_manager.wait_for_path(timeout_seconds):
                print("✓ Path ready - starting main functionality")
                self.path_ready = True
                
                # Update narration manager with path points if not already set
                nominal_points = self.path_manager.get_nominal_points_as_numpy()
                if len(nominal_points) > 0 and len(self.narration_manager.intended_points) == 0:
                    self.narration_manager.update_intended_trajectory(nominal_points)
                    print("✓ Updated narration manager with path points")
            else:
                print("✗ Path not ready - some functionality may be limited")
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

    def print_initialization_status(self):
        """Print initialization status."""
        print(f"Resilience Node initialized")
        print(f"Path mode: {self.path_manager.get_mode()}")
        print(f"Path topic: {self.path_manager.get_path_topic()}")
        print(f"Path ready: {'YES' if hasattr(self, 'path_ready') and self.path_ready else 'NO'}")
        if hasattr(self, 'disable_drift_detection'):
            print(f"Drift detection: {'DISABLED' if self.disable_drift_detection else 'ENABLED'}")
        print(f"Soft threshold: {self.path_manager.get_thresholds()[0]} ({self.path_manager.get_threshold_source()})")
        print(f"Hard threshold: {self.path_manager.get_thresholds()[1]} ({self.path_manager.get_threshold_source()})")
        print(f"NARadio processing: {'ENABLED' if self.naradio_processor.is_ready() else 'DISABLED'}")
        print(f"Voxel Mapping: {'ENABLED' if self.enable_voxel_mapping else 'DISABLED'}")
        
        vlm_enabled = (self.enable_combined_segmentation and 
                      hasattr(self, 'naradio_processor') and 
                      self.naradio_processor.is_segmentation_ready())
        print(f"VLM Similarity Processing: {'ENABLED' if vlm_enabled else 'DISABLED'}")
        
        if vlm_enabled:
            all_objects = self.naradio_processor.get_all_objects()
            print(f"  - Total objects: {len(all_objects)}")
            
            # Print embedding method configuration
            self.print_embedding_method_config()
            
            self.publish_vlm_objects_legend()
        
        print(f"RGB buffer size: {self.max_rgb_buffer}")

    def print_embedding_method_config(self):
        """Print information about embedding method configuration and availability."""
        try:
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                return
            
            # Get config preference
            config = self.naradio_processor.segmentation_config
            prefer_enhanced = config['segmentation'].get('prefer_enhanced_embeddings', True)
            
            print(f"Embedding method: {'ENHANCED' if prefer_enhanced else 'TEXT'}")
            
        except Exception as e:
            print(f"Error printing embedding method config: {e}")

    def publish_vlm_objects_legend(self):
        """Publish VLM objects legend."""
        if (hasattr(self, 'segmentation_legend_pub') and 
            hasattr(self, 'naradio_processor') and 
            self.naradio_processor.is_segmentation_ready()):
            
            try:
                all_objects = self.naradio_processor.get_all_objects()
                legend_text = "VLM Objects Available for Similarity:\n"
                for i, obj in enumerate(all_objects):
                    legend_text += f"{i}: {obj}\n"
                
                self.segmentation_legend_pub.publish(String(data=legend_text))
            except Exception as e:
                print(f"Error publishing VLM objects legend: {e}")

    def init_components(self):
        """Initialize resilience components."""
        # Initialize path manager with unified interface
        self.path_manager = PathManager(self, self.path_config)
        
        # Get thresholds from path manager
        soft_threshold, hard_threshold = self.path_manager.get_thresholds()
        
        try:
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.main_config_path if self.main_config_path else None
            )
            
            if not self.naradio_processor.is_ready():
                print("Warning: NARadio initialization failed, will retry in processing loop")
            else:
                print("NARadio processor initialized")
                
            if self.enable_combined_segmentation:
                if self.naradio_processor.is_segmentation_ready():
                    print("Combined segmentation initialized")
                else:
                    print("Warning: Combined segmentation initialization failed")
            
            # Read voxel mapping parameter from main config (non-blocking)
            self.enable_voxel_mapping = False  # Default value
            if (self.naradio_processor.is_ready() and 
                hasattr(self.naradio_processor, 'segmentation_config')):
                try:
                    self.enable_voxel_mapping = self.naradio_processor.segmentation_config.get('enable_voxel_mapping', False)
                except Exception as e:
                    print(f"Warning: Could not read voxel mapping parameter from config: {e}")
                    self.enable_voxel_mapping = False
            else:
                print("Warning: NARadio processor not ready, using default voxel mapping: False")
                
        except Exception as e:
            print(f"Error initializing NARadio processor: {e}")
            import traceback
            traceback.print_exc()
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.main_config_path if self.main_config_path else None
            )
        
        self.narration_manager = NarrationManager(soft_threshold, hard_threshold)
        
        # Ensure voxel mapping parameter is always set (final fallback)
        if not hasattr(self, 'enable_voxel_mapping'):
            self.enable_voxel_mapping = False
            print("Voxel mapping parameter not set, using default: False")
        
        # Set nominal trajectory points if available
        nominal_points = self.path_manager.get_nominal_points_as_numpy()
        if len(nominal_points) > 0:
            self.narration_manager.set_intended_trajectory(nominal_points)
        else:
            print("Warning: No nominal points available for narration manager")
    
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
        """Store RGB message with timestamp for hotspot publishing."""
        msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Store RGB image with timestamp for hotspot publishing
        self.rgb_images_with_timestamps.append((msg, msg_timestamp))
        if len(self.rgb_images_with_timestamps) > self.max_rgb_buffer:
            self.rgb_images_with_timestamps.pop(0)
        
        with self.processing_lock:
            self.latest_rgb_msg = msg
            if self.latest_pose is not None:
                self.detection_pose = self.latest_pose.copy()
                self.detection_pose_time = self.latest_pose_time
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        self.image_buffer.append((cv_image, msg_timestamp, msg))
        
        if len(self.image_buffer) > self.max_buffer_size:
            self.image_buffer.pop(0)
        
        current_system_time = time.time()
        self.rolling_image_buffer.append((cv_image, current_system_time, msg))
        
        while self.rolling_image_buffer and (current_system_time - self.rolling_image_buffer[0][1]) > self.rolling_buffer_duration:
            self.rolling_image_buffer.pop(0)
    

        
    def depth_callback(self, msg):
        """Store latest depth message."""
        with self.processing_lock:
            self.latest_depth_msg = msg
        
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
            
            print("BREACH STARTED")
            print(f"Time: {pose_time:.2f}, Drift: {drift:.3f} (threshold: {self.path_manager.get_thresholds()[0]:.3f})")
            
            # Start new buffer when breach begins
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
            
            self.narration_manager.queue_breach_event('start', pose_time)
            
        elif breach_ended:
            self.last_breach_state = False
            self.current_breach_active = False
            
            print("BREACH ENDED")
            print(f"Time: {pose_time:.2f}, Drift: {drift:.3f} (threshold: {self.path_manager.get_thresholds()[0]:.3f})")
            
            # Freeze buffers when breach ends
            if self.risk_buffer_manager:
                frozen_buffers = self.risk_buffer_manager.freeze_active_buffers(pose_time)
            
            self.narration_manager.queue_breach_event('end', pose_time)
            
        elif breach_now and not self.current_breach_active:
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            
            print("BREACH DETECTED (already in progress)")
            print(f"Time: {pose_time:.2f}, Drift: {drift:.3f} (threshold: {self.path_manager.get_thresholds()[0]:.3f})")
            
            # Start new buffer when breach is detected
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
            
            self.narration_manager.queue_breach_event('start', pose_time)
        
        if not breach_started and not breach_ended and not (breach_now and not self.current_breach_active):
            self.last_breach_state = breach_now
        
        with self.lock:
            if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0 and self.current_breach_active:
                self.risk_buffer_manager.add_pose(pose_time, pos, drift)
                
                self.save_lagged_image_for_pose(pose_time, drift)

        if self.current_breach_active and not self.narration_manager.get_narration_sent():
            narration = self.narration_manager.check_for_narration(pose_time, self.breach_idx)
            if narration:
                self.publish_narration_with_image(narration)
                self.narration_pub.publish(String(data=narration))

        with self.processing_lock:
            if (self.latest_rgb_msg is not None and 
                self.latest_depth_msg is not None and 
                self.current_breach_active and
                not self.is_processing):
                self.trigger_detection_processing()

        if self.use_tf:
            self.publish_tf(pos, msg.header.stamp)

    def trigger_detection_processing(self):
        """Trigger detection processing in a separate thread."""
        if not self.is_processing:
            self.is_processing = True
            processing_thread = threading.Thread(target=self.process_detection_async, daemon=True)
            processing_thread.start()

    def process_detection_async(self):
        """Process detection asynchronously to prevent blocking."""
        try:
            with self.processing_lock:
                if (self.latest_rgb_msg is None or 
                    self.latest_depth_msg is None or 
                    self.detection_pose is None):
                    return
                
                rgb_msg = self.latest_rgb_msg
                depth_msg = self.latest_depth_msg
                pose = self.detection_pose
                pose_time = self.detection_pose_time
            
            if not self.naradio_processor.is_ready():
                return
                
            if not self.camera_info_received:
                return
            
            self.process_detection(rgb_msg, depth_msg, pose, pose_time)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False

    def process_detection(self, rgb_msg, depth_msg, pose, pose_time):
        """Process NARadio detection with latest messages."""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            transform_matrix = self.pose_to_transform_matrix(pose)

            # Process features and get similarity maps
            feat_map_np, naradio_vis = self.naradio_processor.process_features_optimized(
                rgb_image, 
                need_visualization=False,  # Disabled for speed
                reuse_features=True
            )
            
            # Get dynamic VLM objects and their similarity maps
            if (self.enable_combined_segmentation and 
                self.naradio_processor.is_segmentation_ready() and
                self.naradio_processor.dynamic_objects and
                feat_map_np is not None):
                
                try:
                    # Create merged hotspot masks for all VLM answers
                    vlm_answers = self.naradio_processor.dynamic_objects
                    vlm_hotspots = self.naradio_processor.create_merged_hotspot_masks(rgb_image, vlm_answers)
                    
                    if vlm_hotspots and len(vlm_hotspots) > 0:
                        # Get RGB timestamp for this image
                        rgb_timestamp = self._get_ros_timestamp(rgb_msg)
                        
                        # Publish merged hotspots with color-based association
                        self.semantic_bridge.publish_merged_hotspots(
                            vlm_hotspots=vlm_hotspots,
                            timestamp=rgb_timestamp,
                            original_image=rgb_image
                        )
                
                except Exception as seg_e:
                    pass  # Silent error handling
            
            # Publish NARadio image if visualization is enabled
            if self.enable_naradio_visualization and naradio_vis is not None:
                naradio_msg = self.bridge.cv2_to_imgmsg(naradio_vis, encoding='rgb8')
                naradio_msg.header = rgb_msg.header
                self.naradio_image_pub.publish(naradio_msg)
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    def pose_to_transform_matrix(self, pose):
        """Create transform matrix from pose to origin."""
        T = np.eye(4)
        initial_pose = self.path_manager.get_initial_pose()
        
        T[0, 3] = pose[0] - initial_pose[0]
        T[1, 3] = pose[1] - initial_pose[1] 
        T[2, 3] = pose[2] - initial_pose[2]
        
        return T

    def publish_tf(self, pos, stamp):
        """Publish transform."""
        if self.use_tf and self.tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = 'map'
            t.child_frame_id = 'robot'
            initial_pose = self.path_manager.get_initial_pose()
            t.transform.translation.x = float(pos[0] - initial_pose[0])
            t.transform.translation.y = float(pos[1] - initial_pose[1])
            t.transform.translation.z = float(pos[2] - initial_pose[2])
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

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

    def save_lagged_image_for_pose(self, pose_time, drift):
        """Extracted lagged image saving logic for pose callback."""
        if not self.image_buffer:
            return
            
        newest_timestamp = self.image_buffer[-1][1]
        current_time = newest_timestamp
        
        target_time_offset = 1.0
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
            current_buffer_id = None
            if len(self.risk_buffer_manager.active_buffers) > 0:
                current_buffer_id = self.risk_buffer_manager.active_buffers[-1].buffer_id
            
            image_timestamp = closest_msg.header.stamp.sec + closest_msg.header.stamp.nanosec * 1e-9
            self.save_lagged_image_to_buffer(closest_image, image_timestamp, closest_msg, current_time, actual_offset, current_buffer_id)

    def save_lagged_image_to_buffer(self, image, image_timestamp, msg, current_time, time_lag, buffer_id):
        """Save a lagged image with reduced verbose logging."""
        try:
            if buffer_id is None:
                return
            
            buffer_dir = os.path.join(self.current_run_dir, buffer_id)
            lagged_dir = os.path.join(buffer_dir, 'lagged_images')
            os.makedirs(lagged_dir, exist_ok=True)
            
            image_filename = f"lagged_image_{current_time:.3f}_{image_timestamp:.3f}.png"
            image_path = os.path.join(lagged_dir, image_filename)
            cv2.imwrite(image_path, image)
            
        except Exception as e:
            print(f"Error saving lagged image to buffer: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Process the narration image to get hotspot mask
            # Use the same similarity processing as regular VLM objects
            similarity_result = self.naradio_processor.process_vlm_similarity_visualization(
                narration_image, vlm_answer
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
                timestamp=original_image_timestamp,  # Use original image timestamp
                original_image=narration_image
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

    # NOTE: Complex hotspot extraction removed - using clean semantic bridge instead





    def _get_ros_timestamp(self, msg):
        """Extract ROS timestamp as float from message header."""
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return time.time()




    def naradio_processing_loop(self):
        """Parallel NARadio processing loop with robust error handling."""
        print("NARadio processing loop started")
        
        last_memory_cleanup = time.time()
        memory_cleanup_interval = 60.0  # SPEED OPTIMIZATION: Increased from 30s
        
        last_device_check = time.time()
        device_check_interval = 15.0  # SPEED OPTIMIZATION: Increased from 5s
        
        while rclpy.ok() and self.naradio_running:
            try:
                current_time = time.time()
                
                # SPEED OPTIMIZATION: Less frequent memory cleanup
                if current_time - last_memory_cleanup > memory_cleanup_interval:
                    self.naradio_processor.cleanup_memory()
                    last_memory_cleanup = current_time
                
                if not self.naradio_processor.is_ready():
                    self.naradio_processor.reinitialize()
                    time.sleep(1.0)  # Wait after reinit
                    continue
                
                # SPEED OPTIMIZATION: Less frequent device checks
                if (self.naradio_processor.is_ready() and 
                    current_time - last_device_check > device_check_interval):
                    if not self.naradio_processor.ensure_device_consistency():
                        self.naradio_processor.reinitialize()
                    last_device_check = current_time
                
                with self.processing_lock:
                    if self.latest_rgb_msg is None:
                        time.sleep(0.01)
                        continue
                    
                    rgb_msg = self.latest_rgb_msg
                    depth_msg = self.latest_depth_msg
                    pose_for_semantic = self.latest_pose.copy() if self.latest_pose is not None else None
                

                
                try:
                    rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
                except Exception as e:
                    time.sleep(0.01)
                    continue
                
                try:
                    # SPEED OPTIMIZATION: Skip visualization to focus on similarity maps
                    feat_map_np, naradio_vis = self.naradio_processor.process_features_optimized(
                        rgb_image, 
                        need_visualization=False,  # Disabled for speed
                        reuse_features=True
                    )
                    
                    # OLD WORKING LOGIC: Process similarity for ALL VLM objects continuously
                    if (self.enable_combined_segmentation and 
                        self.naradio_processor.is_segmentation_ready() and
                        self.naradio_processor.dynamic_objects and
                        feat_map_np is not None):
                        
                        try:
                            # NEW: Create merged hotspot masks for all VLM answers
                            vlm_answers = self.naradio_processor.dynamic_objects
                            vlm_hotspots = self.naradio_processor.create_merged_hotspot_masks(rgb_image, vlm_answers)
                            
                            if vlm_hotspots and len(vlm_hotspots) > 0:
                                # Get RGB timestamp for this image
                                rgb_timestamp = self._get_ros_timestamp(rgb_msg)
                                
                                # Publish merged hotspots with color-based association
                                self.semantic_bridge.publish_merged_hotspots(
                                    vlm_hotspots=vlm_hotspots,
                                    timestamp=rgb_timestamp,
                                    original_image=rgb_image
                                )
                        
                        except Exception as seg_e:
                            pass  # SPEED OPTIMIZATION: Silent error handling
                    
                    # SPEED OPTIMIZATION: Skip NARadio image publishing to reduce overhead
                    # if naradio_vis is not None:
                    #     naradio_msg = self.bridge.cv2_to_imgmsg(naradio_vis, encoding='rgb8')
                    #     naradio_msg.header = rgb_msg.header
                    #     self.naradio_image_pub.publish(naradio_msg)
                    
                    # NOTE: Old continuous similarity processing removed - using smart semantic regions instead
                        
                except Exception as e:
                    time.sleep(0.05)  # SPEED OPTIMIZATION: Reduced sleep on error
                    continue
                
                time.sleep(0.05)  # SPEED OPTIMIZATION: Reduced from 0.15s
                
            except Exception as e:
                time.sleep(0.05)  # SPEED OPTIMIZATION: Reduced sleep on error
    
    def camera_info_callback(self, msg):
        """Handle camera info to get intrinsics."""
        with self.lock:
            if not self.camera_info_received:
                self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
                self.camera_info_received = True

    def vlm_answer_callback(self, msg):
        """Handle VLM answers for cause analysis and buffer association."""
        try:
            vlm_answer = msg.data.strip()
            
            if not vlm_answer or "VLM Error" in vlm_answer or "VLM not available" in vlm_answer:
                return
            
            print(f"VLM ANSWER RECEIVED: '{vlm_answer}'")
            
            # Track this VLM answer as recent for smart semantic processing
            self.recent_vlm_answers[vlm_answer] = time.time()
            
            # Clean up old VLM answers periodically
            self._cleanup_old_vlm_answers(max_age_seconds=300.0)
            
            # Add VLM answer to NARadio processor for continuous monitoring
            if hasattr(self, 'naradio_processor') and self.naradio_processor.is_ready():
                success = self.naradio_processor.add_vlm_object(vlm_answer)
                if success:
                    print(f"VLM object '{vlm_answer}' added for monitoring")
                    
                    if hasattr(self, 'segmentation_legend_pub'):
                        try:
                            all_objects = self.naradio_processor.get_all_objects()
                            legend_text = "VLM Objects Available for Similarity:\n"
                            for i, obj in enumerate(all_objects):
                                legend_text += f"{i}: {obj}\n"
                            
                            self.segmentation_legend_pub.publish(String(data=legend_text))
                        except Exception as e:
                            print(f"Error updating VLM objects legend: {e}")
                    
                else:
                    print(f"Failed to add VLM object '{vlm_answer}' to object list")
            
            self.associate_vlm_answer_with_buffer_reliable(vlm_answer)
            
            if hasattr(self, 'naradio_processor') and self.naradio_processor.is_ready():
                success = self.naradio_processor.add_vlm_object(vlm_answer)
                if success:
                    print(f"✓ VLM object '{vlm_answer}' added - will be continuously monitored")
                    
                    if hasattr(self, 'segmentation_legend_pub'):
                        try:
                            all_objects = self.naradio_processor.get_all_objects()
                            legend_text = "VLM Objects Available for Similarity:\n"
                            for i, obj in enumerate(all_objects):
                                legend_text += f"{i}: {obj}\n"
                            
                            self.segmentation_legend_pub.publish(String(data=legend_text))
                        except Exception as e:
                            print(f"Error updating VLM objects legend: {e}")
                    
                    print(f"✓ VLM object '{vlm_answer}' will be continuously monitored in NARadio processing loop")
                    
                else:
                    print(f"✗ Failed to add VLM object '{vlm_answer}' to object list")
            
            # Process narration image chain IMMEDIATELY when VLM answer is received
            # This ensures hotspot mask is sent to semantic bridge as soon as possible
            narration_success = self.process_narration_chain_for_vlm_answer(vlm_answer)
            
            if narration_success:
                print(f"Narration processing completed for '{vlm_answer}'")
            else:
                print(f"Narration processing failed for '{vlm_answer}'")
            
        except Exception as e:
            print(f"Error processing VLM answer: {e}")

    def process_narration_chain_for_vlm_answer(self, vlm_answer: str) -> bool:
        """
        Complete narration processing chain for a VLM answer.
        This happens immediately when VLM answer is received.
        
        Chain of events:
        1. Find buffer with this cause
        2. Get narration image
        3. Process similarity and get hotspot mask
        4. Send hotspot mask through semantic bridge IMMEDIATELY
        5. Process enhanced embedding (background)
        
        Args:
            vlm_answer: The VLM answer/cause to process
            
        Returns:
            True if narration processing was successful, False otherwise
        """
        try:
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                print(f"NARadio processor not ready for narration processing")
                return False
            
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
            
            # STEP 1: Process similarity and get hotspot mask IMMEDIATELY
            similarity_result = self.naradio_processor.process_vlm_similarity_visualization(
                narration_image, vlm_answer
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
            
            # STEP 2: Get original image timestamp (CRITICAL for proper synchronization)
            original_image_timestamp = None
            if target_buffer.get_original_image_timestamp() is not None:
                # Use the stored original image timestamp
                original_image_timestamp = target_buffer.get_original_image_timestamp()
            elif target_buffer.narration_timestamp is not None:
                # The narration_timestamp is when narration was generated
                # We need to estimate the original image timestamp
                # For now, use the buffer start time as approximation
                original_image_timestamp = target_buffer.start_time
            else:
                # Fallback: use current time
                original_image_timestamp = time.time()
            
            # STEP 3: Send hotspot mask through semantic bridge IMMEDIATELY
            hotspot_success = self.publish_narration_hotspot_mask(
                narration_image=narration_image,
                vlm_answer=vlm_answer,
                original_image_timestamp=original_image_timestamp,
                buffer_id=target_buffer.buffer_id
            )
            
            if hotspot_success:
                print(f"Hotspot mask sent to semantic bridge for '{vlm_answer}'")
            else:
                print(f"Failed to send hotspot mask to semantic bridge")
            
            # STEP 4: Process enhanced embedding in background (non-blocking)
            try:
                # Process the narration image for enhanced embedding
                enhanced_success = self.naradio_processor.process_narration_image_similarity(
                    narration_image=narration_image,
                    vlm_answer=vlm_answer,
                    buffer_id=target_buffer.buffer_id,
                    buffer_dir=buffer_dir
                )
                
                if enhanced_success:
                    # Load and store enhanced embedding in buffer object
                    try:
                        enhanced_embedding = self.load_enhanced_embedding_from_buffer(buffer_dir, vlm_answer)
                        if enhanced_embedding is not None:
                            target_buffer.assign_enhanced_cause_embedding(enhanced_embedding)
                            
                            # Also add to NARadio processor for real-time similarity detection
                            if (hasattr(self, 'naradio_processor') and 
                                self.naradio_processor.is_segmentation_ready()):
                                success = self.naradio_processor.add_enhanced_embedding(vlm_answer, enhanced_embedding)
                                if success:
                                    print(f"Enhanced embedding added to NARadio processor")
                            
                    except Exception as e:
                        print(f"Error loading enhanced embedding: {e}")
            except Exception as e:
                print(f"Error processing enhanced embedding: {e}")
            
            return hotspot_success  # Return success based on hotspot mask publishing
            
        except Exception as e:
            print(f"Error in narration processing chain: {e}")
            import traceback
            traceback.print_exc()
            return False

    def associate_vlm_answer_with_buffer_reliable(self, vlm_answer):
        """Associate VLM answer with buffer."""
        try:
            success = self.risk_buffer_manager.assign_cause(vlm_answer)
            
            if success:
                print(f"Associated '{vlm_answer}' with risk buffer")
            else:
                print(f"No suitable buffer found for '{vlm_answer}'")
                    
        except Exception as e:
            print(f"Error associating VLM answer with buffer: {e}")

    def _cleanup_old_vlm_answers(self, max_age_seconds: float):
        """Clean up old VLM answers to prevent memory buildup."""
        try:
            current_time = time.time()
            self.recent_vlm_answers = {
                vlm_answer: timestamp for vlm_answer, timestamp in self.recent_vlm_answers.items()
                if current_time - timestamp <= max_age_seconds
            }
        except Exception as e:
            print(f"Error cleaning up old VLM answers: {e}")

    def _get_recent_vlm_answers(self, max_age_seconds: float) -> List[str]:
        """
        Get a list of VLM answers that were received recently.
        
        Args:
            max_age_seconds: Maximum age in seconds for VLM answers to be considered recent.
            
        Returns:
            List of VLM answers that were received within the last max_age_seconds.
        """
        current_time = time.time()
        recent_vlm_answers = [
            vlm_answer for vlm_answer, timestamp in self.recent_vlm_answers.items()
            if current_time - timestamp <= max_age_seconds
        ]
        return recent_vlm_answers
    
    
    
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
        
        if hasattr(node, 'naradio_processor') and node.naradio_processor.is_ready():
            try:
                node.naradio_processor.cleanup_memory()
            except Exception as e:
                print(f"Error cleaning up NARadio model: {e}")
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
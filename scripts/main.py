#!/usr/bin/env python3
"""
Resilience Main Node

Integrated node that combines drift detection, YOLO, NARadio, and narration.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import String, Header
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import json
import threading
import time
import torch
import cv2
import warnings
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
warnings.filterwarnings('ignore')

from resilience.drift_calculator import DriftCalculator
from resilience.yolo_sam_detector import YOLODetector
from resilience.naradio_processor import NARadioProcessor
from resilience.narration_manager import NarrationManager
from resilience.pointcloud_manager import PointCloudManager
from resilience.simple_descriptive_narration import XYSpatialDescriptor, TrajectoryPoint
import sensor_msgs_py.point_cloud2 as pc2
from resilience.risk_buffer import RiskBufferManager
from resilience.historical_cause_analysis import HistoricalCauseAnalyzer

# Import semantic bridge
try:
    from resilience.semantic_info_bridge import SemanticHotspotPublisher
    SEMANTIC_BRIDGE_AVAILABLE = True
except ImportError:
    SEMANTIC_BRIDGE_AVAILABLE = False


class ResilienceNode(Node):
    """Resilience Node - Integrated drift detection, YOLO, NARadio, and narration."""
    
    def __init__(self):
        super().__init__('resilience_node')

        self.rgb_topic = '/robot_1/sensors/front_stereo/right/image'
        self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
        self.pose_topic = '/robot_1/sensors/front_stereo/pose'
        self.camera_info_topic = '/robot_1/sensors/front_stereo/right/camera_info'
        
        from ament_index_python.packages import get_package_share_directory
        package_dir = get_package_share_directory('resilience')
        self.nominal_traj_file = os.path.join(package_dir, 'assets', 'adjusted_nominal_spline.json')
        
        # Track recent VLM answers for smart semantic processing
        self.recent_vlm_answers = {}  # vlm_answer -> timestamp
        
        self.declare_parameters('', [
            ('min_confidence', 0.05),
            ('min_detection_distance', 0.5),
            ('max_detection_distance', 2.0),
            ('yolo_model', 'yolov8l-world.pt'),
            ('yolo_imgsz', 480),
            ('detection_color_r', 255),
            ('detection_color_g', 0),
            ('detection_color_b', 0),
            ('flip_y_axis', False),
            ('use_tf', False),
            ('disable_yolo_printing', True),
            ('radio_model_version', 'radio_v2.5-b'),
            ('radio_lang_model', 'siglip'),
            ('radio_input_resolution', 512),
            ('enable_naradio_visualization', True),
            ('enable_combined_segmentation', True),
            ('segmentation_config_path', ''),
            ('publish_original_mask', True),
            ('publish_refined_mask', True)
        ])

        param_values = self.get_parameters([
            'min_confidence', 'min_detection_distance', 'max_detection_distance',
            'yolo_model', 'yolo_imgsz',
            'detection_color_r', 'detection_color_g', 'detection_color_b',
            'flip_y_axis', 'use_tf', 'disable_yolo_printing',
            'radio_model_version', 'radio_lang_model', 'radio_input_resolution',
            'enable_naradio_visualization', 'enable_combined_segmentation',
            'segmentation_config_path', 'publish_original_mask', 'publish_refined_mask'
        ])
        
        (self.min_confidence, self.min_detection_distance, self.max_detection_distance,
         self.yolo_model, self.yolo_imgsz,
         det_color_r, det_color_g, det_color_b,
         self.flip_y_axis, self.use_tf, self.disable_yolo_printing,
         self.radio_model_version, self.radio_lang_model, self.radio_input_resolution,
         self.enable_naradio_visualization, self.enable_combined_segmentation,
         self.segmentation_config_path, self.publish_original_mask, self.publish_refined_mask
        ) = [p.value for p in param_values]
        
        self.detection_color = np.array([det_color_r, det_color_g, det_color_b], dtype=np.uint8)
        
        self.detection_prompts = []
        self.current_detection_prompt = ""
        self.vlm_answer_received = False
        self.last_vlm_answer = ""
        self.detection_enabled = False

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
        self.init_historical_analyzer()
        self.init_parallel_processing()
        self.init_semantic_bridge()
        
        self.start_naradio_thread()
        self.start_historical_analysis_thread()
        
        self.print_initialization_status()

    def init_publishers(self):
        """Initialize publishers."""
        publishers = [
            ('/yolo_bbox_image', Image, 1),
            ('/detection_cloud', PointCloud2, 10),
            ('/drift_narration', String, 10),
            ('/narration_text', String, 10),
            ('/naradio_image', Image, 10),
            ('/narration_image', Image, 10)
        ]
        
        self.yolo_bbox_pub, self.detection_cloud_pub, \
        self.narration_pub, self.narration_text_pub, self.naradio_image_pub, \
        self.narration_image_pub = [self.create_publisher(msg_type, topic, qos) 
                                   for topic, msg_type, qos in publishers]
        
        if self.enable_combined_segmentation:
            vlm_publishers = [
                ('/vlm_similarity_map', Image, 10),
                ('/vlm_similarity_colored', Image, 10),
                ('/vlm_objects_legend', String, 10)
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
            ('/vlm_answer', String, self.vlm_answer_callback, 10)
        ]
        
        for topic, msg_type, callback, qos in subscriptions:
            self.create_subscription(msg_type, topic, callback, qos)

    def print_initialization_status(self):
        """Print initialization status."""
        print(f"Resilience Node initialized")
        print(f"Soft threshold: {self.drift_calculator.soft_threshold}")
        print(f"NARadio processing: {'ENABLED' if self.naradio_processor.is_ready() else 'DISABLED'}")
        
        vlm_enabled = (self.enable_combined_segmentation and 
                      hasattr(self, 'naradio_processor') and 
                      self.naradio_processor.is_segmentation_ready())
        print(f"VLM Similarity Processing: {'ENABLED' if vlm_enabled else 'DISABLED'}")
        
        if vlm_enabled:
            all_objects = self.naradio_processor.get_all_objects()
            dynamic_objects = self.naradio_processor.dynamic_objects
            print(f"  - Base objects: {len(self.naradio_processor.word_list)}")
            print(f"  - Total objects: {len(all_objects)}")
            
            # Print embedding method configuration
            self.print_embedding_method_config()
            
            self.publish_vlm_objects_legend()

    def print_embedding_method_config(self):
        """Print information about embedding method configuration and availability."""
        try:
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                return
            
            # Get config preference
            config = self.naradio_processor.segmentation_config
            prefer_enhanced = config['segmentation'].get('prefer_enhanced_embeddings', True)
            
            print(f"SIMILARITY METHOD: {'ENHANCED EMBEDDINGS' if prefer_enhanced else 'VLM TEXT EMBEDDINGS'}")
            
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
        self.drift_calculator = DriftCalculator(self.nominal_traj_file)
        soft_threshold, hard_threshold = self.drift_calculator.get_thresholds()
        
        self.yolo_detector = YOLODetector(
            yolo_model=self.yolo_model,
            yolo_imgsz=self.yolo_imgsz,
            min_confidence=self.min_confidence,
            disable_yolo_printing=self.disable_yolo_printing
        )
        
        try:
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.segmentation_config_path if self.segmentation_config_path else None
            )
            
            if not self.naradio_processor.is_ready():
                print("Warning: NARadio initialization failed, will retry in processing loop")
            else:
                print("✓ NARadio processor initialized successfully")
                
            if self.enable_combined_segmentation:
                if self.naradio_processor.is_segmentation_ready():
                    print("✓ Combined segmentation initialized successfully")
                else:
                    print("Warning: Combined segmentation initialization failed")
                
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
                segmentation_config_path=self.segmentation_config_path if self.segmentation_config_path else None
            )
        
        self.narration_manager = NarrationManager(soft_threshold, hard_threshold)
        self.pointcloud_manager = PointCloudManager()
        
        nominal_points = self.drift_calculator.get_nominal_points()
        self.narration_manager.set_intended_trajectory(nominal_points)
    
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
            print(f"Buffer save directory: {self.current_run_dir}")
            
            self.node_id = f"resilience_{unique_id}"
            print(f"Node identifier: {self.node_id}")
            
        except Exception as e:
            print(f"Error initializing risk buffer manager: {e}")
            self.risk_buffer_manager = None
    
    def init_historical_analyzer(self):
        """Initialize historical cause analyzer."""
        try:
            if self.risk_buffer_manager:
                historical_yolo_path = self.yolo_model
                
                node_id = getattr(self, 'node_id', 'unknown')
                print(f"[HistoricalAnalysis] Initializing for node: {node_id}")
                
                self.historical_analyzer = HistoricalCauseAnalyzer(
                    yolo_model_path=historical_yolo_path,
                    save_directory=self.current_run_dir
                )
                
                print(f"[HistoricalAnalysis] Initialized historical analyzer with save_dir: {self.current_run_dir}")
                print(f"[HistoricalAnalysis] Using YOLO model: {historical_yolo_path}")
                print(f"[HistoricalAnalysis] Node identifier: {node_id}")
            else:
                self.historical_analyzer = None
                print("Historical analyzer disabled - no risk buffer manager")
                
        except Exception as e:
            print(f"[HistoricalAnalysis] Error initializing historical analyzer: {e}")
            import traceback
            traceback.print_exc()
            self.historical_analyzer = None
    
    def init_parallel_processing(self):
        """Initialize parallel processing infrastructure."""
        self.historical_analysis_queue = []
        self.historical_analysis_condition = threading.Condition()
        self.historical_analysis_lock = threading.Lock()
        self.historical_analysis_running = True
        
        self.parallel_analysis_results = {}
        self.parallel_analysis_status = {}
        
        print("Parallel processing infrastructure initialized")
    
    def init_semantic_bridge(self):
        """Initialize semantic hotspot bridge for communication with octomap."""
        try:
            if SEMANTIC_BRIDGE_AVAILABLE:
                # Load semantic bridge config from segmentation config
                segmentation_config = getattr(self.naradio_processor, 'segmentation_config', {})
                
                from resilience.semantic_info_bridge import SemanticHotspotPublisher
                self.semantic_bridge = SemanticHotspotPublisher(self, segmentation_config)
                print("✓ Semantic hotspot bridge initialized")
            else:
                self.semantic_bridge = None
                print("✗ Semantic bridge not available")
        except Exception as e:
            print(f"Error initializing semantic bridge: {e}")
            self.semantic_bridge = None
    
    def start_naradio_thread(self):
        """Start the parallel NARadio processing thread."""
        if not hasattr(self, 'naradio_thread') or self.naradio_thread is None or not self.naradio_thread.is_alive():
            self.naradio_running = True
            self.naradio_thread = threading.Thread(target=self.naradio_processing_loop, daemon=True)
            self.naradio_thread.start()
            print(f"Parallel NARadio processing thread started")
        else:
            print(f"NARadio thread already running")
    
    def start_historical_analysis_thread(self):
        """Start the parallel historical analysis thread."""
        if not hasattr(self, 'historical_analysis_thread') or self.historical_analysis_thread is None or not self.historical_analysis_thread.is_alive():
            self.historical_analysis_running = True
            self.historical_analysis_thread = threading.Thread(target=self.historical_analysis_worker, daemon=True)
            self.historical_analysis_thread.start()
            print(f"Parallel historical analysis thread started")
        else:
            print(f"Historical analysis thread already running")
    
    def rgb_callback(self, msg):
        """Store latest RGB message with minimal buffer operations to reduce lag."""

        msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
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
        """Store latest depth message with minimal buffer operations to reduce lag."""
        with self.processing_lock:
            self.latest_depth_msg = msg
            
    def pose_callback(self, msg):
        """Process pose and trigger detection with consolidated pose updates."""
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        drift, nearest_idx = self.drift_calculator.compute_drift(pos)
        self.breach_idx = nearest_idx
        
        with self.lock:
            self.latest_pose = pos
            self.latest_pose_time = pose_time
            self.last_pose = pos
            self.last_pose_time = pose_time
            
            self.narration_manager.add_actual_point(pos, pose_time, self.flip_y_axis)

        breach_now = self.drift_calculator.is_breach(drift)
        
        breach_started = not self.last_breach_state and breach_now
        breach_ended = self.last_breach_state and not breach_now
        
        if breach_started:
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            self.detection_enabled = True
            
            print("BREACH STARTED")
            print(f"Time: {pose_time:.2f}, Drift: {drift:.3f} (threshold: {self.drift_calculator.soft_threshold:.3f})")
            
            if not self.detection_prompts:
                self.detection_prompts = []
                self.current_detection_prompt = ""
            
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
                print(f"New buffer started at {pose_time:.3f}")
            
            self.narration_manager.queue_breach_event('start', pose_time)
            
        elif breach_ended:
            self.last_breach_state = False
            self.current_breach_active = False
            
            print("BREACH ENDED")
            print(f"Time: {pose_time:.2f}, Drift: {drift:.3f} (threshold: {self.drift_calculator.soft_threshold:.3f})")
            
            if self.risk_buffer_manager:
                frozen_buffers = self.risk_buffer_manager.freeze_active_buffers(pose_time)
                print(f"Buffers frozen at {pose_time:.3f}")
                
                if frozen_buffers:
                    for buffer in frozen_buffers:
                        self.queue_historical_analysis('buffer_frozen', buffer_id=buffer.buffer_id)
                
                self.save_comprehensive_buffers(pose_time)
            
            self.detection_enabled = False
            
            self.narration_manager.queue_breach_event('end', pose_time)
            
        elif breach_now and not self.current_breach_active:
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            self.detection_enabled = True
            
            print("BREACH DETECTED (already in progress)")
            print(f"Time: {pose_time:.2f}, Drift: {drift:.3f} (threshold: {self.drift_calculator.soft_threshold:.3f})")
            
            if not self.detection_prompts:
                self.detection_prompts = []
                self.current_detection_prompt = ""
            
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
                print(f"New buffer started at {pose_time:.3f}")
            
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
                self.detection_enabled and 
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
            
            if not self.yolo_detector.is_ready():
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
        """Process YOLO detection with latest messages."""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            transform_matrix = self.pose_to_transform_matrix(pose)

            bboxes, labels, confidences = self.yolo_detector.detect_objects(rgb_image)
            
            if not bboxes:
                return

            filtered_bboxes, valid_indices = self.yolo_detector.filter_detections_by_distance(
                bboxes, depth_image, self.min_detection_distance, self.max_detection_distance, self.camera_intrinsics)
            
            if not filtered_bboxes:
                return

            bbox_centers = []
            for bbox in filtered_bboxes:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                bbox_centers.append([center_x, center_y])

            yolo_bbox_img = rgb_image.copy()
            for bbox in filtered_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(yolo_bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(yolo_bbox_img, (center_x, center_y), 5, (255, 0, 0), -1)
            
            yolo_bbox_msg = self.bridge.cv2_to_imgmsg(yolo_bbox_img, encoding='rgb8')
            yolo_bbox_msg.header = rgb_msg.header
            self.yolo_bbox_pub.publish(yolo_bbox_msg)

            if len(bbox_centers) == 0:
                return
                 
            success = self.pointcloud_manager.create_bbox_center_pointclouds(
                filtered_bboxes, depth_image, self.camera_intrinsics, transform_matrix,
                self.min_detection_distance, self.max_detection_distance)
            
            if success:
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'map'
                pointcloud_msg = self.pointcloud_manager.create_pointcloud_message(header)
                self.detection_cloud_pub.publish(pointcloud_msg)
                
                self.pointcloud_manager.increment_frame_count()

        except Exception as e:
            import traceback
            traceback.print_exc()

    def pose_to_transform_matrix(self, pose):
        """Create transform matrix from pose to origin."""
        T = np.eye(4)
        initial_pose = self.drift_calculator.get_initial_pose()
        
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
            initial_pose = self.drift_calculator.get_initial_pose()
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
                self.save_narration_image_to_buffer(closest_image, narration_text, current_time)
                
                # NEW: Store narration data in active buffers for VLM processing
                if self.risk_buffer_manager:
                    self.risk_buffer_manager.store_narration_data(closest_image, narration_text, current_time)
                
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
            print(f"Loaded enhanced embedding from: {embedding_path}")
            print(f"  Shape: {enhanced_embedding.shape}, Norm: {np.linalg.norm(enhanced_embedding):.4f}")
            
            return enhanced_embedding
            
        except Exception as e:
            print(f"Error loading enhanced embedding: {e}")
            return None

    def save_comprehensive_buffers(self, current_time):
        """Save all frozen buffers with comprehensive metadata."""
        try:
            with self.lock:
                frozen_buffers = self.risk_buffer_manager.frozen_buffers.copy()
            
            if not frozen_buffers:
                print("No frozen buffers to save")
                return
            
            print(f"Saving {len(frozen_buffers)} frozen buffers...")
            
            saved_count = 0
            for buffer in frozen_buffers:
                if self.save_single_comprehensive_buffer(buffer):
                    saved_count += 1
            
            self.save_run_summary(frozen_buffers)
            
            print(f"Saved: {saved_count}/{len(frozen_buffers)} buffers to {self.current_run_dir}")
            
        except Exception as e:
            print(f"Error saving comprehensive buffers: {e}")

    def save_single_comprehensive_buffer(self, buffer):
        """Save a single buffer with all metadata, images, narration, and cause data."""
        try:
            print(f"Saving buffer {buffer.buffer_id}: poses={len(buffer.poses)}, images={len(buffer.images)}")
            
            buffer_dir = os.path.join(self.current_run_dir, buffer.buffer_id)
            os.makedirs(buffer_dir, exist_ok=True)
            
            soft_threshold, hard_threshold = self.drift_calculator.get_thresholds()
            
            metadata = {
                'buffer_id': buffer.buffer_id,
                'start_time': buffer.start_time,
                'end_time': buffer.end_time,
                'state': buffer.state.value,
                'cause': buffer.cause,
                'duration': buffer.get_duration(),
                'pose_count': len(buffer.poses)
            }
            
            with open(os.path.join(buffer_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if buffer.poses:
                poses_array = np.array([(t, p[0], p[1], p[2], d) for t, p, d in buffer.poses])
                np.save(os.path.join(buffer_dir, 'poses.npy'), poses_array)
            
            if buffer.images:
                if len(buffer.images) >= 1:
                    first_img = buffer.images[0][1]
                    cv2.imwrite(os.path.join(buffer_dir, 'first_image.png'), first_img)
                
                if len(buffer.images) >= 2:
                    last_img = buffer.images[-1][1]
                    cv2.imwrite(os.path.join(buffer_dir, 'last_image.png'), last_img)
            
            print(f"Saved comprehensive buffer: {buffer.buffer_id}")
            return True
            
        except Exception as e:
            print(f"Error saving buffer {buffer.buffer_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_run_summary(self, frozen_buffers):
        """Save summary of the entire run."""
        try:
            if not frozen_buffers:
                frozen_buffers = []
                
            soft_threshold, hard_threshold = self.drift_calculator.get_thresholds()
            
            run_summary = {
                'run_timestamp': datetime.now().isoformat(),
                'run_directory': self.current_run_dir,
                'total_buffers': len(frozen_buffers),
                'buffers_with_cause': sum(1 for b in frozen_buffers if b.has_cause()),
                'buffers_without_cause': sum(1 for b in frozen_buffers if not b.has_cause()),
                'total_images': sum(len(b.images) for b in frozen_buffers),
                'total_poses': sum(len(b.poses) for b in frozen_buffers),
                'total_duration': sum(b.get_duration() for b in frozen_buffers),
                'soft_threshold': soft_threshold,
                'hard_threshold': hard_threshold,
                'detection_prompts_used': self.detection_prompts,
                'buffer_details': []
            }
            
            for buffer in frozen_buffers:
                buffer_detail = {
                    'buffer_id': buffer.buffer_id,
                    'start_time': buffer.start_time,
                    'end_time': buffer.end_time,
                    'duration': buffer.get_duration(),
                    'cause': buffer.cause,
                    'has_cause': buffer.has_cause(),
                    'data_counts': buffer.get_data_counts()
                }
                run_summary['buffer_details'].append(buffer_detail)
            
            with open(os.path.join(self.current_run_dir, 'run_summary.json'), 'w') as f:
                json.dump(run_summary, f, indent=2)
            
            print(f"Run summary saved: {self.current_run_dir}/run_summary.json")
            
        except Exception as e:
            print(f"Error saving run summary: {e}")
            import traceback
            traceback.print_exc()

    def save_similarity_to_buffer(self, similarity_result, timestamp, vlm_answer):
        """Save VLM similarity results to the current active buffer with reduced I/O."""
        try:
            if not self.risk_buffer_manager or len(self.risk_buffer_manager.active_buffers) == 0:
                return
            
            processing_info = similarity_result.get('processing_info', {})
            if processing_info.get('processing_time', 0) > 2.0:
                print(f"Skipping similarity save - processing took too long: {processing_info.get('processing_time', 0):.2f}s")
                return
            
            current_buffer = self.risk_buffer_manager.active_buffers[-1]
            buffer_dir = os.path.join(self.current_run_dir, current_buffer.buffer_id)
            
            similarity_dir = os.path.join(buffer_dir, 'vlm_similarity')
            os.makedirs(similarity_dir, exist_ok=True)
            
            timestamp_str = f"{timestamp.sec}_{timestamp.nanosec}"
            
            safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            if similarity_result['colored_similarity'] is not None:
                colored_path = os.path.join(similarity_dir, f"colored_similarity_{safe_vlm_name}_{timestamp_str}.png")
                colored_bgr = cv2.cvtColor(similarity_result['colored_similarity'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(colored_path, colored_bgr)
            
            print(f"Saved VLM similarity results for '{vlm_answer}' to buffer {current_buffer.buffer_id} (optimized)")
            
        except Exception as e:
            print(f"Error saving VLM similarity to buffer: {e}")
            import traceback
            traceback.print_exc()

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
            print(f"Saving narration image to buffer: {current_buffer.buffer_id}")
            
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

    # NOTE: Complex hotspot extraction removed - using clean semantic bridge instead

    def naradio_processing_loop(self):
        """Parallel NARadio processing loop with robust error handling."""
        print(f"NARadio processing loop started")
        
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
                    
                    # CLEAN SEMANTIC: Process similarity hotspots for recent VLM answers only
                    if (self.enable_combined_segmentation and 
                        self.naradio_processor.is_segmentation_ready() and
                        self.naradio_processor.dynamic_objects and
                        feat_map_np is not None and
                        self.semantic_bridge):
                        
                        try:
                            # Only process similarity for very recent VLM answers
                            recent_vlm_answers = self._get_recent_vlm_answers(max_age_seconds=5.0)
                            
                            for vlm_answer in recent_vlm_answers:
                                if vlm_answer in self.naradio_processor.dynamic_objects:
                                    # Compute similarity map
                                    similarity_result = self.naradio_processor.process_adaptive_similarity_visualization_optimized(
                                        rgb_image, vlm_answer, feat_map_np)
                                    
                                    if similarity_result and similarity_result.get('similarity_map') is not None:
                                        # Send similarity map + original image to bridge for binary thresholding
                                        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
                                        
                                        if self.latest_pose is not None:
                                            # Get depth image if available
                                            depth_image = None
                                            if self.latest_depth_msg is not None:
                                                try:
                                                    depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')
                                                    # Convert to meters if needed
                                                    if depth_image.dtype == np.uint16:
                                                        depth_image = depth_image.astype(np.float32) / 1000.0
                                                except Exception as e:
                                                    print(f"Could not process depth image: {e}")
                                            
                                            self.semantic_bridge.publish_similarity_hotspots(
                                                vlm_answer=vlm_answer,
                                                similarity_map=similarity_result['similarity_map'],
                                                pose=self.latest_pose,
                                                timestamp=timestamp,
                                                original_image=rgb_image,  # Pass original image for overlay visualization
                                                depth_image=depth_image   # Pass depth image for accurate 3D mapping
                                            )
                                        
                                        # Save to buffer for analysis
                                        if (self.risk_buffer_manager and 
                                            len(self.risk_buffer_manager.active_buffers) > 0 and 
                                            self.current_breach_active):
                                            
                                            self.save_similarity_to_buffer(similarity_result, rgb_msg.header.stamp, vlm_answer)
                                
                        except Exception as seg_e:
                            pass  # Silent error handling
                    
                    # SPEED OPTIMIZATION: Skip NARadio image publishing to reduce overhead
                    # if naradio_vis is not None:
                    #     naradio_msg = self.bridge.cv2_to_imgmsg(naradio_vis, encoding='rgb8')
                    #     naradio_msg.header = rgb_msg.header
                    #     self.naradio_image_pub.publish(naradio_msg)
                    
                    # NOTE: Old continuous similarity processing removed - using smart semantic regions instead
                        
                except torch.cuda.OutOfMemoryError:
                    self.naradio_processor.handle_cuda_out_of_memory()
                    time.sleep(1.0)
                    continue
                except Exception as e:
                    time.sleep(0.05)  # SPEED OPTIMIZATION: Reduced sleep on error
                    continue
                
                time.sleep(0.05)  # SPEED OPTIMIZATION: Reduced from 0.15s
                
            except Exception as e:
                time.sleep(0.05)  # SPEED OPTIMIZATION: Reduced sleep on error
    
    def historical_analysis_worker(self):
        """Parallel historical analysis worker thread that processes analysis requests"""
        print(f"Historical analysis worker started")
        
        while rclpy.ok() and self.historical_analysis_running:
            try:
                with self.historical_analysis_condition:
                    while not self.historical_analysis_queue and self.historical_analysis_running:
                        self.historical_analysis_condition.wait(timeout=1.0)
                    
                    if not self.historical_analysis_running:
                        break
                    
                    queue_size = len(self.historical_analysis_queue)
                    if queue_size > 0:
                        print(f"[HistoricalAnalysis] Processing {queue_size} queued requests")
                    
                    while self.historical_analysis_queue:
                        request = self.historical_analysis_queue.pop(0)
                        self.process_historical_analysis_request(request)
                        
            except Exception as e:
                print(f"Error in historical analysis worker: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print(f"Historical analysis worker ended")

    def process_historical_analysis_request(self, request):
        """Process a historical analysis request in parallel"""
        request_type = request.get('type')
        vlm_answer = request.get('vlm_answer', '')
        buffer_id = request.get('buffer_id', '')
        
        print(f"[ParallelAnalysis] Processing request: {request_type}")
        
        try:
            if request_type == 'vlm_answer_received':
                self.queue_historical_analysis('analyze_vlm_answer', vlm_answer=vlm_answer)
            elif request_type == 'buffer_frozen':
                self.queue_historical_analysis('analyze_frozen_buffer', buffer_id=buffer_id)
            elif request_type == 'analyze_vlm_answer':
                self.perform_parallel_vlm_analysis(vlm_answer)
            elif request_type == 'analyze_frozen_buffer':
                self.perform_parallel_buffer_analysis(buffer_id)
                
        except Exception as e:
            print(f"Error processing historical analysis request: {e}")
            import traceback
            traceback.print_exc()

    def queue_historical_analysis(self, request_type, **kwargs):
        """Queue a historical analysis request for parallel processing"""
        with self.historical_analysis_condition:
            request = {
                'type': request_type,
                'timestamp': time.time(),
                **kwargs
            }
            self.historical_analysis_queue.append(request)
            self.historical_analysis_condition.notify()
            print(f"[ParallelAnalysis] Queued request: {request_type}")

    def perform_parallel_vlm_analysis(self, vlm_answer):
        """Perform VLM answer analysis in parallel without blocking main thread"""
        try:
            print(f"[ParallelAnalysis] Starting VLM analysis for: '{vlm_answer}' (Node: {getattr(self, 'node_id', 'unknown')})")
            
            if not self.historical_analyzer:
                print("[ParallelAnalysis] Historical analyzer not available")
                return
            
            with self.historical_analysis_lock:
                status = self.risk_buffer_manager.get_status()
                frozen_buffers = self.risk_buffer_manager.frozen_buffers.copy()
                active_buffers = self.risk_buffer_manager.active_buffers.copy()
            
            print(f"[ParallelAnalysis] Buffer status: {status['frozen_buffers']} frozen, {status['active_buffers']} active")
            print(f"[ParallelAnalysis] Current run directory: {getattr(self, 'current_run_dir', 'unknown')}")
            
            results = {}
            
            frozen_with_cause = [b for b in frozen_buffers if b.has_cause()]
            if frozen_with_cause:
                print(f"[ParallelAnalysis] Analyzing {len(frozen_with_cause)} frozen buffers with causes...")
                try:
                    frozen_results = self.historical_analyzer.analyze_all_buffers()
                    results.update(frozen_results)
                    print(f"[ParallelAnalysis] Found {len(frozen_results)} locations in frozen buffers")
                except Exception as e:
                    print(f"[ParallelAnalysis] Error analyzing frozen buffers: {e}")
                    import traceback
                    traceback.print_exc()
            
            if not results:
                active_with_cause = [b for b in active_buffers if b.has_cause()]
                if active_with_cause:
                    print(f"[ParallelAnalysis] Analyzing {len(active_with_cause)} active buffers with causes...")
                    try:
                        active_results = self.historical_analyzer.analyze_active_buffers(active_with_cause)
                        results.update(active_results)
                        print(f"[ParallelAnalysis] Found {len(active_results)} locations in active buffers")
                    except Exception as e:
                        print(f"[ParallelAnalysis] Error analyzing active buffers: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"[ParallelAnalysis] VLM analysis completed. Results: {len(results)} locations found")
            
            self.save_parallel_analysis_results(vlm_answer, results)
            
        except Exception as e:
            print(f"[ParallelAnalysis] Error in VLM analysis: {e}")
            import traceback
            traceback.print_exc()

    def perform_parallel_buffer_analysis(self, buffer_id):
        """Perform buffer analysis in parallel without blocking main thread"""
        try:
            print(f"[ParallelAnalysis] Starting buffer analysis for: {buffer_id}")
            
            if not self.historical_analyzer:
                print("[ParallelAnalysis] Historical analyzer not available")
                return
            
            with self.historical_analysis_lock:
                frozen_buffers = self.risk_buffer_manager.frozen_buffers.copy()
                active_buffers = self.risk_buffer_manager.active_buffers.copy()
            
            target_buffer = None
            for buffer in frozen_buffers + active_buffers:
                if buffer.buffer_id == buffer_id:
                    target_buffer = buffer
                    break
            
            if not target_buffer:
                print(f"[ParallelAnalysis] Buffer {buffer_id} not found")
                return
            
            if target_buffer.has_cause():
                print(f"[ParallelAnalysis] Analyzing buffer {buffer_id} with cause: {target_buffer.cause}")
                result = self.historical_analyzer.analyze_single_buffer(target_buffer)
                
                results = [result] if result else []
                print(f"[ParallelAnalysis] Buffer analysis completed. Results: {len(results)} locations found")
                
                self.save_parallel_analysis_results(f"buffer_{buffer_id}", results)
            else:
                print(f"[ParallelAnalysis] Buffer {buffer_id} has no cause assigned")
                
        except Exception as e:
            print(f"[ParallelAnalysis] Error in buffer analysis: {e}")
            import traceback
            traceback.print_exc()

    def save_parallel_analysis_results(self, analysis_id, results):
        """Save parallel analysis results to disk"""
        try:
            analysis_dir = os.path.join(self.current_run_dir, 'parallel_analysis_results')
            os.makedirs(analysis_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            node_id = getattr(self, 'node_id', 'unknown')
            filename = f"analysis_{analysis_id}_{timestamp}_{node_id}.json"
            filepath = os.path.join(analysis_dir, filename)
            
            analysis_data = {
                'analysis_id': analysis_id,
                'node_id': node_id,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'results': results,
                'result_count': len(results) if results else 0,
                'status': 'completed',
                'run_directory': getattr(self, 'current_run_dir', 'unknown')
            }
            
            with open(filepath, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            with self.historical_analysis_lock:
                self.parallel_analysis_results[analysis_id] = {
                    'results': results,
                    'timestamp': time.time(),
                    'status': 'completed',
                    'filepath': filepath,
                    'node_id': node_id
                }
                self.parallel_analysis_status[analysis_id] = 'completed'
            
            print(f"[ParallelAnalysis] Saved results for {analysis_id}: {len(results) if results else 0} locations")
            print(f"[ParallelAnalysis] File: {filepath}")
            
        except Exception as e:
            print(f"Error saving parallel analysis results: {e}")
            import traceback
            traceback.print_exc()

    def get_parallel_analysis_status(self):
        """Get status of parallel analysis"""
        with self.historical_analysis_lock:
            return {
                'queue_size': len(self.historical_analysis_queue),
                'completed_analyses': len(self.parallel_analysis_results),
                'recent_results': list(self.parallel_analysis_results.keys())[-5:]
            }

    def print_parallel_analysis_status(self):
        """Print current parallel analysis status"""
        status = self.get_parallel_analysis_status()
        print(f"[ParallelAnalysis] Status: {status['queue_size']} queued, {status['completed_analyses']} completed")
        if status['recent_results']:
            print(f"[ParallelAnalysis] Recent: {status['recent_results']}")
    
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
                print(f"Skipping VLM answer: '{vlm_answer}' (empty or error)")
                return
            
            print(f"VLM ANSWER RECEIVED: '{vlm_answer}'")
            
            # Track this VLM answer as recent for smart semantic processing
            self.recent_vlm_answers[vlm_answer] = time.time()
            
            # Clean up old VLM answers periodically
            self._cleanup_old_vlm_answers(max_age_seconds=300.0)
            
            if vlm_answer not in self.detection_prompts:
                self.detection_prompts.append(vlm_answer)
                self.current_detection_prompt = vlm_answer
                
                if self.current_breach_active:
                    self.detection_enabled = True
                else:
                    self.detection_enabled = False
                
                if self.yolo_detector.update_prompts(self.detection_prompts):
                    print(f"YOLO updated with new prompts")
                else:
                    print(f"Failed to update YOLO prompts")
                    self.detection_enabled = False
            
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
            
            self.queue_historical_analysis('vlm_answer_received', vlm_answer=vlm_answer)
            
            # Note: Enhanced embedding handling moved to narration processing for efficiency
            
            # NEW: Process narration image from buffer directory AFTER historical analysis queuing
            try:
                if (hasattr(self, 'naradio_processor') and 
                    self.naradio_processor.is_segmentation_ready()):
                    
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
                    
                    if target_buffer:
                        buffer_dir = os.path.join(self.current_run_dir, target_buffer.buffer_id)
                        
                        # First check if narration image is in buffer memory
                        narration_image = None
                        if target_buffer.has_narration_image():
                            narration_image = target_buffer.get_narration_image()
                            print(f"Found narration image in buffer {target_buffer.buffer_id} memory")
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
                                        print(f"Found narration image for buffer {target_buffer.buffer_id} on disk: {latest_narration_file}")
                        
                        if narration_image is not None:
                            print(f"Processing narration image similarity for VLM '{vlm_answer}' from buffer {target_buffer.buffer_id}")
                            
                            # Process the narration image for similarity mapping
                            success = self.naradio_processor.process_narration_image_similarity(
                                narration_image=narration_image,
                                vlm_answer=vlm_answer,
                                buffer_id=target_buffer.buffer_id,
                                buffer_dir=buffer_dir
                            )
                            
                            if success:
                                print(f"✓ Completed narration similarity processing for VLM '{vlm_answer}' in buffer {target_buffer.buffer_id}")
                                
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
                                                print(f"✓ Added enhanced embedding to NARadio processor for real-time detection")
                                            else:
                                                print(f"✗ Failed to add enhanced embedding to NARadio processor")
                                        
                                        print(f"✓ Loaded and stored enhanced embedding in buffer {target_buffer.buffer_id}")
                                    else:
                                        print(f"✗ Failed to load enhanced embedding for buffer {target_buffer.buffer_id}")
                                except Exception as e:
                                    print(f"✗ Error loading enhanced embedding: {e}")
                            else:
                                print(f"✗ Failed to process narration similarity for VLM '{vlm_answer}'")
                        else:
                            print(f"✗ No narration image found for buffer {target_buffer.buffer_id} with VLM '{vlm_answer}'")
                    else:
                        print(f"✗ No buffer found with cause '{vlm_answer}' for narration processing")
                        
            except Exception as e:
                print(f"Error processing narration image similarity: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error processing VLM answer: {e}")

    def associate_vlm_answer_with_buffer_reliable(self, vlm_answer):
        """Associate VLM answer with buffer."""
        try:
            print(f"Associating VLM answer '{vlm_answer}' with buffer...")
            
            success = self.risk_buffer_manager.assign_cause(vlm_answer)
            
            if success:
                print(f"✓ Successfully associated '{vlm_answer}' with risk buffer")
            else:
                print(f"✗ No suitable buffer found for '{vlm_answer}'")
                    
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
        
        if hasattr(node, 'historical_analysis_running') and node.historical_analysis_running:
            node.historical_analysis_running = False
            with node.historical_analysis_condition:
                node.historical_analysis_condition.notify_all()
            if hasattr(node, 'historical_analysis_thread') and node.historical_analysis_thread and node.historical_analysis_thread.is_alive():
                node.historical_analysis_thread.join(timeout=2.0)
        
        if hasattr(node, 'naradio_processor') and node.naradio_processor.is_ready():
            try:
                node.naradio_processor.cleanup_memory()
            except Exception as e:
                print(f"Error cleaning up NARadio model: {e}")
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Resilience Main Node

Integrated node that combines drift detection, YOLO-SAM, NARadio, and narration.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import String, Header
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import json
from pathlib import Path
import threading
import time
import sys
import torch
import cv2
import warnings
import os
import contextlib
import queue
import gc
import torch
import uuid
from datetime import datetime
warnings.filterwarnings('ignore')

# Import resilience components
from resilience.drift_calculator import DriftCalculator
from resilience.yolo_sam_detector import YOLOSAMDetector
from resilience.naradio_processor import NARadioProcessor
from resilience.narration_manager import NarrationManager
from resilience.pointcloud_manager import PointCloudManager

# Import external dependencies
from resilience.simple_descriptive_narration import XYSpatialDescriptor, TrajectoryPoint
import sensor_msgs_py.point_cloud2 as pc2

# Import risk buffer components
from resilience.risk_buffer import RiskBufferManager
from resilience.historical_cause_analysis import HistoricalCauseAnalyzer


class ResilienceNode(Node):
    """
    Resilience Node - Integrated drift detection, YOLO-SAM, NARadio, and narration.
    """
    
    def __init__(self):
        super().__init__('resilience_node')

        # Parameters
        self.rgb_topic = '/robot_1/sensors/front_stereo/right/image'
        self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
        self.pose_topic = '/robot_1/sensors/front_stereo/pose'
        self.camera_info_topic = '/robot_1/sensors/front_stereo/right/camera_info'
        
        # Use proper ROS2 package path resolution for assets
        from ament_index_python.packages import get_package_share_directory
        package_dir = get_package_share_directory('resilience')
        self.nominal_traj_file = os.path.join(package_dir, 'assets', 'adjusted_nominal_spline.json')
        
        # YOLO-SAM parameters
        self.declare_parameter('min_confidence', 0.05)
        self.declare_parameter('min_detection_distance', 0.5)
        self.declare_parameter('max_detection_distance', 2.0)
        self.declare_parameter('sam_checkpoint', os.path.join(package_dir, 'assets', 'sam_vit_b_01ec64.pth'))
        self.declare_parameter('yolo_model', 'yolov8l-world.pt')
        self.declare_parameter('sam_model_type', 'vit_b')
        self.declare_parameter('yolo_imgsz', 480)
        self.declare_parameter('detection_color_r', 255)
        self.declare_parameter('detection_color_g', 0)
        self.declare_parameter('detection_color_b', 0)
        self.declare_parameter('flip_y_axis', False)
        self.declare_parameter('use_tf', False)
        self.declare_parameter('disable_yolo_printing', True)

        # NARadio parameters
        self.declare_parameter('radio_model_version', 'radio_v2.5-b')
        self.declare_parameter('radio_lang_model', 'siglip')
        self.declare_parameter('radio_input_resolution', 512)
        self.declare_parameter('enable_naradio_visualization', True)

        # Combined Segmentation parameters
        self.declare_parameter('enable_combined_segmentation', True)
        self.declare_parameter('segmentation_config_path', '')
        self.declare_parameter('publish_original_mask', True)
        self.declare_parameter('publish_refined_mask', True)

        # Get parameters
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        self.min_detection_distance = self.get_parameter('min_detection_distance').get_parameter_value().double_value
        self.max_detection_distance = self.get_parameter('max_detection_distance').get_parameter_value().double_value
        self.sam_checkpoint = self.get_parameter('sam_checkpoint').get_parameter_value().string_value
        self.yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        self.sam_model_type = self.get_parameter('sam_model_type').get_parameter_value().string_value
        self.yolo_imgsz = self.get_parameter('yolo_imgsz').get_parameter_value().integer_value
        self.flip_y_axis = self.get_parameter('flip_y_axis').get_parameter_value().bool_value
        self.use_tf = self.get_parameter('use_tf').get_parameter_value().bool_value
        self.disable_yolo_printing = self.get_parameter('disable_yolo_printing').get_parameter_value().bool_value
        
        # Get NARadio parameters
        self.radio_model_version = self.get_parameter('radio_model_version').get_parameter_value().string_value
        self.radio_lang_model = self.get_parameter('radio_lang_model').get_parameter_value().string_value
        self.radio_input_resolution = self.get_parameter('radio_input_resolution').get_parameter_value().integer_value
        self.enable_naradio_visualization = self.get_parameter('enable_naradio_visualization').get_parameter_value().bool_value
        
        # Get Combined Segmentation parameters
        self.enable_combined_segmentation = self.get_parameter('enable_combined_segmentation').get_parameter_value().bool_value
        self.segmentation_config_path = self.get_parameter('segmentation_config_path').get_parameter_value().string_value
        self.publish_original_mask = self.get_parameter('publish_original_mask').get_parameter_value().bool_value
        self.publish_refined_mask = self.get_parameter('publish_refined_mask').get_parameter_value().bool_value
        
        # Detection state
        self.detection_prompts = []
        self.current_detection_prompt = ""
        self.vlm_answer_received = False
        self.last_vlm_answer = ""
        self.detection_enabled = False
        
        # Detection color
        self.detection_color = np.array([
            self.get_parameter('detection_color_r').get_parameter_value().integer_value,
            self.get_parameter('detection_color_g').get_parameter_value().integer_value,
            self.get_parameter('detection_color_b').get_parameter_value().integer_value
        ], dtype=np.uint8)

        # Initialize components
        self.init_components()
        

        
        # Breach state
        self.last_breach_state = False
        self.current_breach_active = False
        
        # TF broadcaster (optional)
        if self.use_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        else:
            self.tf_broadcaster = None
            self.tf_buffer = None
            self.tf_listener = None

        # Image bridge
        self.bridge = CvBridge()

        # Camera intrinsics
        self.camera_intrinsics = [186.24478149414062, 186.24478149414062, 238.66322326660156, 141.6264190673828]
        self.camera_info_received = False

        # Publishers
        self.yolo_bbox_pub = self.create_publisher(Image, '/yolo_bbox_image', 1)
        self.sam_mask_pub = self.create_publisher(Image, '/sam_mask_image', 1)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, '/detection_cloud', 10)
        self.narration_pub = self.create_publisher(String, '/drift_narration', 10)
        self.narration_text_pub = self.create_publisher(String, '/narration_text', 10)
        self.naradio_image_pub = self.create_publisher(Image, '/naradio_image', 10)
        self.narration_image_pub = self.create_publisher(Image, '/narration_image', 10)
        
        # Combined Segmentation publishers
        if self.enable_combined_segmentation:
            self.original_mask_pub = self.create_publisher(Image, '/original_segmentation_mask', 10)
            self.refined_mask_pub = self.create_publisher(Image, '/refined_segmentation_mask', 10)
            self.segmentation_legend_pub = self.create_publisher(String, '/segmentation_legend', 10)

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 1)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 1)
        self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 1)
        self.vlm_answer_sub = self.create_subscription(String, '/vlm_answer', self.vlm_answer_callback, 10)

        # State
        self.last_rgb_msg = None
        self.last_depth_msg = None
        self.last_pose = None
        self.last_pose_time = None
        self.lock = threading.Lock()
        self.breach_idx = None
        
        # Processing state
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.latest_pose = None
        self.latest_pose_time = None
        
        # NARadio processing state
        self.naradio_processing_lock = threading.Lock()
        self.naradio_is_processing = False
        
        # Parallel processing infrastructure for NARadio
        self.naradio_queue = queue.Queue(maxsize=5)
        self.naradio_running = True
        
        # Store pose when RGB was received for detection processing
        self.detection_pose = None
        self.detection_pose_time = None
        
        # Image buffer for narration-triggered publishing
        self.image_buffer = []
        self.max_buffer_size = 150
        self.rolling_image_buffer = []
        self.rolling_buffer_duration = 1.0
        
        # Transform caching
        self.transform_matrix_cache = None
        self.last_transform_time = 0
        self.transform_cache_duration = 0.1

        # Initialize risk buffer manager
        self.init_risk_buffer_manager()
        
        # Initialize historical analyzer
        self.init_historical_analyzer()
        
        # Initialize parallel processing infrastructure
        self.init_parallel_processing()
        
        # Initialize narration infrastructure
        self.init_narration_infrastructure()
        
        # Start the parallel NARadio processing thread
        self.start_naradio_thread()
        
        # Start the parallel historical analysis thread
        self.start_historical_analysis_thread()
        
        # Start the narration worker thread
        self.start_narration_thread()

        print(f"Resilience Node initialized")
        print(f"Soft threshold: {self.drift_calculator.soft_threshold}")
        print(f"Monitoring breach transitions: F→T for start, T→F for end")
        print(f"NARadio processing: {'ENABLED' if self.naradio_processor.is_ready() else 'DISABLED'}")
        print(f"Combined Segmentation: {'ENABLED' if self.enable_combined_segmentation and hasattr(self, 'naradio_processor') and self.naradio_processor.is_segmentation_ready() else 'DISABLED'}")
        if self.enable_combined_segmentation and hasattr(self, 'naradio_processor') and self.naradio_processor.is_segmentation_ready():
            all_objects = self.naradio_processor.get_all_objects()
            dynamic_objects = self.naradio_processor.dynamic_objects
            print(f"  - Base objects: {len(self.naradio_processor.word_list)}")
            print(f"  - Dynamic objects: {len(dynamic_objects)}")
            print(f"  - Total objects: {len(all_objects)}")
            if dynamic_objects:
                print(f"  - Dynamic objects: {dynamic_objects}")
        
        # Publish color legend if combined segmentation is enabled and ready
        if (self.enable_combined_segmentation and 
            hasattr(self, 'segmentation_legend_pub') and 
            hasattr(self, 'naradio_processor') and 
            self.naradio_processor.is_segmentation_ready()):
            
            try:
                legend = self.naradio_processor.get_color_legend()
                legend_text = "Segmentation Color Legend:\n"
                for word, color in legend.items():
                    legend_text += f"{word}: RGB{color}\n"
                
                self.segmentation_legend_pub.publish(String(data=legend_text))
                print("✓ Published segmentation color legend")
            except Exception as e:
                print(f"Error publishing color legend: {e}")

    def init_components(self):
        """Initialize all resilience components."""
        # Initialize drift calculator
        self.drift_calculator = DriftCalculator(self.nominal_traj_file)
        soft_threshold, hard_threshold = self.drift_calculator.get_thresholds()
        
        # Initialize YOLO-SAM detector
        self.yolo_sam_detector = YOLOSAMDetector(
            yolo_model=self.yolo_model,
            sam_checkpoint=self.sam_checkpoint,
            sam_model_type=self.sam_model_type,
            yolo_imgsz=self.yolo_imgsz,
            min_confidence=self.min_confidence,
            disable_yolo_printing=self.disable_yolo_printing
        )
        
        # Initialize NARadio processor with robust error handling
        try:
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.segmentation_config_path if self.segmentation_config_path else None
            )
            
            # Verify NARadio initialization
            if not self.naradio_processor.is_ready():
                print("Warning: NARadio initialization failed, will retry in processing loop")
            else:
                print("✓ NARadio processor initialized successfully")
                
            # Check combined segmentation status
            if self.enable_combined_segmentation:
                if self.naradio_processor.is_segmentation_ready():
                    print("✓ Combined segmentation initialized successfully")
                else:
                    print("Warning: Combined segmentation initialization failed")
                
        except Exception as e:
            print(f"Error initializing NARadio processor: {e}")
            import traceback
            traceback.print_exc()
            # Create a dummy processor that will be reinitialized later
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.segmentation_config_path if self.segmentation_config_path else None
            )
        
        # Initialize narration manager
        self.narration_manager = NarrationManager(soft_threshold, hard_threshold)
        
        # Initialize point cloud manager
        self.pointcloud_manager = PointCloudManager()
        
        # Set intended trajectory for narration
        nominal_points = self.drift_calculator.get_nominal_points()
        self.narration_manager.set_intended_trajectory(nominal_points)
    

    
    def init_risk_buffer_manager(self):
        """Initialize risk buffer manager."""
        try:
            # Create run-specific directory with high-precision timestamp and unique identifier
            import uuid
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            unique_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
            self.risk_buffer_save_dir = '/home/navin/ros2_ws/src/buffers'
            os.makedirs(self.risk_buffer_save_dir, exist_ok=True)
            
            self.current_run_dir = os.path.join(self.risk_buffer_save_dir, f"run_{run_timestamp}_{unique_id}")
            os.makedirs(self.current_run_dir, exist_ok=True)
            
            # Initialize buffer manager
            self.risk_buffer_manager = RiskBufferManager(save_directory=self.current_run_dir)
            print(f"Buffer save directory: {self.current_run_dir}")
            
            # Add node identifier to prevent conflicts
            self.node_id = f"resilience_{unique_id}"
            print(f"Node identifier: {self.node_id}")
            
        except Exception as e:
            print(f"Error initializing risk buffer manager: {e}")
            self.risk_buffer_manager = None
    
    def init_historical_analyzer(self):
        """Initialize historical cause analyzer."""
        try:
            if self.risk_buffer_manager:
                # Use the same YOLO model as the main detection system
                historical_yolo_path = self.yolo_model  # Use same model as main system
                
                # Add node identifier to prevent conflicts
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
        # Historical analysis infrastructure
        self.historical_analysis_queue = []
        self.historical_analysis_condition = threading.Condition()
        self.historical_analysis_lock = threading.Lock()
        self.historical_analysis_running = True
        
        # Parallel analysis results storage
        self.parallel_analysis_results = {}
        self.parallel_analysis_status = {}
        
        print("Parallel processing infrastructure initialized")
    
    def init_narration_infrastructure(self):
        """Initialize narration processing infrastructure."""
        # Narration infrastructure
        self.narration_queue = []
        self.narration_condition = threading.Condition()
        self.narration_lock = threading.Lock()
        self.narration_running = True
        
        print("Narration infrastructure initialized")
    
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
    
    def start_narration_thread(self):
        """Start the narration worker thread."""
        if not hasattr(self, 'narration_thread') or self.narration_thread is None or not self.narration_thread.is_alive():
            self.narration_running = True
            self.narration_thread = threading.Thread(target=self.narration_worker, daemon=True)
            self.narration_thread.start()
            print(f"Narration worker thread started")
        else:
            print(f"Narration thread already running")
    
    def narration_worker(self):
        """Narration worker thread that processes narration requests"""
        print(f"Narration worker started")
        
        while rclpy.ok() and self.narration_running:
            try:
                # Wait for narration requests
                with self.narration_condition:
                    while not self.narration_queue and self.narration_running:
                        self.narration_condition.wait(timeout=1.0)
                    
                    if not self.narration_running:
                        break
                    
                    # Process all queued narration requests
                    while self.narration_queue:
                        request = self.narration_queue.pop(0)
                        self.process_narration_request(request)
                        
            except Exception as e:
                print(f"Error in narration worker: {e}")
                time.sleep(0.1)
        
        print(f"Narration worker ended")

    def process_narration_request(self, request):
        """Process a narration request"""
        request_type = request.get('type')
        narration_text = request.get('narration_text', '')
        timestamp = request.get('timestamp', 0.0)
        
        print(f"[Narration] Processing request: {request_type}")
        
        try:
            if request_type == 'publish_narration':
                self.publish_narration_with_image(narration_text)
                print(f"[Narration] Published narration: '{narration_text}'")
            elif request_type == 'save_narration_image':
                self.save_narration_image_to_buffer(request.get('image'), narration_text, timestamp)
                print(f"[Narration] Saved narration image for: '{narration_text}'")
                
        except Exception as e:
            print(f"Error processing narration request: {e}")
            import traceback
            traceback.print_exc()

    def queue_narration(self, request_type, **kwargs):
        """Queue a narration request for processing"""
        with self.narration_condition:
            request = {
                'type': request_type,
                'timestamp': time.time(),
                **kwargs
            }
            self.narration_queue.append(request)
            self.narration_condition.notify()
            print(f"[Narration] Queued request: {request_type}")

    def save_narration_image_to_buffer(self, image, narration_text, current_time):
        """Save narration image to the current buffer directory"""
        try:
            # Check if we have active buffers
            with self.lock:
                if len(self.risk_buffer_manager.active_buffers) == 0:
                    print("No active buffers to save narration image to")
                    return
            
            # Get the most recent active buffer
            current_buffer = self.risk_buffer_manager.active_buffers[-1]
            print(f"Saving narration image to buffer: {current_buffer.buffer_id}")
            
            # Create narration directory in buffer
            buffer_dir = os.path.join(self.current_run_dir, current_buffer.buffer_id)
            narration_dir = os.path.join(buffer_dir, 'narration')
            os.makedirs(narration_dir, exist_ok=True)
            
            # Save the narration image
            timestamp_str = f"{current_time:.3f}"
            image_filename = f"narration_image_{timestamp_str}.png"
            image_path = os.path.join(narration_dir, image_filename)
            cv2.imwrite(image_path, image)
            
            # Save simple metadata
            metadata = {
                'filename': image_filename,
                'timestamp': current_time,
                'narration_text': narration_text,
                'saved_timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(narration_dir, 'narration_image_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved narration image to buffer {current_buffer.buffer_id}: {image_path}")
            
        except Exception as e:
            print(f"Error saving narration image to buffer: {e}")
            import traceback
            traceback.print_exc()

    def save_lagged_image_to_buffer(self, image, image_timestamp, msg, current_time, time_lag, buffer_id):
        """Save a lagged image (from ~1 second ago) to the current active buffer"""
        try:
            print(f"Starting lagged image save: current_time={current_time:.3f}, image_time={image_timestamp:.3f}, lag={time_lag:.3f}s")
            
            # Use the provided buffer_id to avoid lock conflict
            if buffer_id is None:
                print("No buffer ID provided, skipping lagged image save")
                return
            
            print(f"Using buffer: {buffer_id}")
            print("Buffer obtained successfully")
            
            # Create lagged images directory in buffer
            print("Creating directory paths...")
            buffer_dir = os.path.join(self.current_run_dir, buffer_id)
            print(f"Buffer dir: {buffer_dir}")
            lagged_dir = os.path.join(buffer_dir, 'lagged_images')
            print(f"Lagged dir: {lagged_dir}")
            print("Creating lagged images directory...")
            os.makedirs(lagged_dir, exist_ok=True)
            print(f"Created lagged images directory: {lagged_dir}")
            
            # Save the lagged image
            print("Preparing image filename...")
            image_filename = f"lagged_image_{current_time:.3f}_{image_timestamp:.3f}.png"
            print(f"Image filename: {image_filename}")
            image_path = os.path.join(lagged_dir, image_filename)
            print(f"Saving image to: {image_path}")
            print("Calling cv2.imwrite...")
            cv2.imwrite(image_path, image)
            print(f"Image saved successfully")
            
            # Store metadata for the lagged image
            print("Creating image metadata...")
            image_metadata = {
                'image_timestamp': image_timestamp,
                'current_time': current_time,
                'time_lag': time_lag,
                'filename': image_filename,
                'frame_id': msg.header.frame_id,
                'image_width': image.shape[1],
                'image_height': image.shape[0],
                'is_lagged': True,
                'lag_duration': time_lag
            }
            print("Metadata created successfully")
            
            # Save individual image metadata
            print("Preparing metadata filename...")
            metadata_filename = f"lagged_image_{current_time:.3f}_{image_timestamp:.3f}_metadata.json"
            print(f"Metadata filename: {metadata_filename}")
            metadata_path = os.path.join(lagged_dir, metadata_filename)
            print(f"Metadata path: {metadata_path}")
            print("Writing metadata file...")
            with open(metadata_path, 'w') as f:
                json.dump(image_metadata, f, indent=2)
            print("Metadata file written successfully")
            
            print(f"  Saved lagged image: {image_filename} (lag={time_lag:.3f}s)")
            print(f"Lagged image save completed successfully")
            
        except Exception as e:
            print(f"Error saving lagged image to buffer: {e}")
            import traceback
            traceback.print_exc()

    def save_rolling_buffer_images(self, breach_start_system_time):
        """Save rolling buffer images around breach start time"""
        try:
            print(f"Saving rolling buffer images around breach start: {breach_start_system_time:.3f}")
            
            if not self.rolling_image_buffer:
                print("No rolling buffer images available")
                return
            
            # Get current active buffer
            with self.lock:
                if len(self.risk_buffer_manager.active_buffers) == 0:
                    print("No active buffer for rolling images")
                    return
                current_buffer = self.risk_buffer_manager.active_buffers[-1]
            
            # Create rolling buffer directory
            buffer_dir = os.path.join(self.current_run_dir, current_buffer.buffer_id)
            rolling_dir = os.path.join(buffer_dir, 'rolling_buffer')
            os.makedirs(rolling_dir, exist_ok=True)
            
            # Save images around breach time
            saved_count = 0
            for image, timestamp, msg in self.rolling_image_buffer:
                # Calculate time difference from breach start
                time_diff = abs(timestamp - breach_start_system_time)
                
                # Save images within 2 seconds of breach start
                if time_diff <= 2.0:
                    image_filename = f"rolling_{timestamp:.3f}_{time_diff:.3f}.png"
                    image_path = os.path.join(rolling_dir, image_filename)
                    cv2.imwrite(image_path, image)
                    
                    # Save metadata
                    metadata = {
                        'timestamp': timestamp,
                        'time_diff_from_breach': time_diff,
                        'filename': image_filename,
                        'frame_id': msg.header.frame_id
                    }
                    
                    metadata_filename = f"rolling_{timestamp:.3f}_{time_diff:.3f}_metadata.json"
                    metadata_path = os.path.join(rolling_dir, metadata_filename)
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    saved_count += 1
            
            print(f"Saved {saved_count} rolling buffer images around breach start")
            
        except Exception as e:
            print(f"Error saving rolling buffer images: {e}")
            import traceback
            traceback.print_exc()

    def naradio_processing_loop(self):
        """Parallel NARadio processing loop with robust error handling."""
        print(f"NARadio processing loop started")
        
        last_memory_cleanup = time.time()
        memory_cleanup_interval = 30.0  # Clean memory every 30 seconds
        
        while rclpy.ok() and self.naradio_running:
            try:
                # Periodic memory cleanup
                current_time = time.time()
                if current_time - last_memory_cleanup > memory_cleanup_interval:
                    self.naradio_processor.cleanup_memory()
                    last_memory_cleanup = current_time
                
                # Reinitialize radio model if it was reset due to memory issues
                if not self.naradio_processor.is_ready():
                    print("NARadio not ready, attempting reinitialization...")
                    self.naradio_processor.reinitialize()
                
                # Ensure device consistency
                if self.naradio_processor.is_ready():
                    if not self.naradio_processor.ensure_device_consistency():
                        print("Device consistency check failed, reinitializing...")
                        self.naradio_processor.reinitialize()
                
                # Get latest RGB data
                with self.processing_lock:
                    if self.latest_rgb_msg is None:
                        time.sleep(0.01)
                        continue
                    
                    rgb_msg = self.latest_rgb_msg
                
                # Convert RGB image
                try:
                    rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
                except Exception as e:
                    print(f"Error converting RGB image for NARadio: {e}")
                    time.sleep(0.01)
                    continue
                
                # Process NARadio features with error handling
                try:
                    feat_map_np, naradio_vis = self.naradio_processor.process_features(rgb_image)
                    
                    if naradio_vis is not None:
                        # Publish NARadio visualization
                        naradio_msg = self.bridge.cv2_to_imgmsg(naradio_vis, encoding='rgb8')
                        naradio_msg.header = rgb_msg.header
                        self.naradio_image_pub.publish(naradio_msg)
                    
                    # Process combined segmentation if enabled
                    if (self.enable_combined_segmentation and 
                        self.naradio_processor.is_segmentation_ready()):
                        
                        try:
                            # Process segmentation using NARadio processor
                            segmentation_result = self.naradio_processor.process_combined_segmentation(rgb_image)
                            
                            if segmentation_result:
                                # Publish original mask
                                if self.publish_original_mask and segmentation_result['original_mask'] is not None:
                                    original_mask_msg = self.bridge.cv2_to_imgmsg(
                                        segmentation_result['original_mask'], 
                                        encoding='rgb8'
                                    )
                                    original_mask_msg.header = rgb_msg.header
                                    self.original_mask_pub.publish(original_mask_msg)
                                
                                # Publish refined mask
                                if self.publish_refined_mask and segmentation_result['refined_mask'] is not None:
                                    refined_mask_msg = self.bridge.cv2_to_imgmsg(
                                        segmentation_result['refined_mask'], 
                                        encoding='rgb8'
                                    )
                                    refined_mask_msg.header = rgb_msg.header
                                    self.refined_mask_pub.publish(refined_mask_msg)
                                
                                # Save masks to buffer if enabled
                                if (self.risk_buffer_manager and 
                                    len(self.risk_buffer_manager.active_buffers) > 0 and 
                                    self.current_breach_active and
                                    self.naradio_processor.segmentation_config['output']['save_masks_to_buffer']):
                                    
                                    self.save_segmentation_to_buffer(segmentation_result, rgb_msg.header.stamp)
                                
                        except Exception as seg_e:
                            print(f"Error in combined segmentation processing: {seg_e}")
                            # Continue with NARadio processing even if segmentation fails
                        
                except torch.cuda.OutOfMemoryError:
                    print("CUDA out of memory in NARadio processing loop")
                    self.naradio_processor.handle_cuda_out_of_memory()
                    time.sleep(1.0)  # Wait longer after OOM
                    continue
                except Exception as e:
                    print(f"Error processing NARadio features: {e}")
                    time.sleep(0.1)
                    continue
                
                # Rate limiting
                time.sleep(0.1)  # 10 Hz
                
            except Exception as e:
                print(f"Error in NARadio processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def historical_analysis_worker(self):
        """Parallel historical analysis worker thread that processes analysis requests"""
        print(f"Historical analysis worker started")
        
        while rclpy.ok() and self.historical_analysis_running:
            try:
                # Wait for analysis requests
                with self.historical_analysis_condition:
                    while not self.historical_analysis_queue and self.historical_analysis_running:
                        self.historical_analysis_condition.wait(timeout=1.0)
                    
                    if not self.historical_analysis_running:
                        break
                    
                    # Process all queued analysis requests
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
                # Queue the analysis but don't block
                self.queue_historical_analysis('analyze_vlm_answer', vlm_answer=vlm_answer)
            elif request_type == 'buffer_frozen':
                # Queue buffer analysis but don't block
                self.queue_historical_analysis('analyze_frozen_buffer', buffer_id=buffer_id)
            elif request_type == 'analyze_vlm_answer':
                # Actually perform the analysis
                self.perform_parallel_vlm_analysis(vlm_answer)
            elif request_type == 'analyze_frozen_buffer':
                # Actually perform buffer analysis
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
            
            # Get current buffer status without blocking main operations
            with self.historical_analysis_lock:
                status = self.risk_buffer_manager.get_status()
                frozen_buffers = self.risk_buffer_manager.frozen_buffers.copy()
                active_buffers = self.risk_buffer_manager.active_buffers.copy()
            
            print(f"[ParallelAnalysis] Buffer status: {status['frozen_buffers']} frozen, {status['active_buffers']} active")
            print(f"[ParallelAnalysis] Current run directory: {getattr(self, 'current_run_dir', 'unknown')}")
            
            # Perform analysis in parallel
            results = {}
            
            # Analyze frozen buffers with causes first
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
            
            # If no results, try active buffers with causes
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
            
            # Save results to a separate file for later retrieval
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
            
            # Get buffer data without blocking main operations
            with self.historical_analysis_lock:
                frozen_buffers = self.risk_buffer_manager.frozen_buffers.copy()
                active_buffers = self.risk_buffer_manager.active_buffers.copy()
            
            # Find the specific buffer
            target_buffer = None
            for buffer in frozen_buffers + active_buffers:
                if buffer.buffer_id == buffer_id:
                    target_buffer = buffer
                    break
            
            if not target_buffer:
                print(f"[ParallelAnalysis] Buffer {buffer_id} not found")
                return
            
            # Perform analysis in parallel
            if target_buffer.has_cause():
                print(f"[ParallelAnalysis] Analyzing buffer {buffer_id} with cause: {target_buffer.cause}")
                result = self.historical_analyzer.analyze_single_buffer(target_buffer)
                
                # Convert single result to list format for compatibility
                results = [result] if result else []
                print(f"[ParallelAnalysis] Buffer analysis completed. Results: {len(results)} locations found")
                
                # Save results
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
            # Create analysis results directory
            analysis_dir = os.path.join(self.current_run_dir, 'parallel_analysis_results')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Save results with node identifier to prevent conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
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
            
            # Save to memory for quick access
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
                'recent_results': list(self.parallel_analysis_results.keys())[-5:]  # Last 5
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
                self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]  # fx, fy, cx, cy
                self.camera_info_received = True

    def vlm_answer_callback(self, msg):
        """Handle VLM answers for cause analysis and buffer association."""
        try:
            vlm_answer = msg.data.strip()
            
            # Skip if answer is empty or error message
            if not vlm_answer or "VLM Error" in vlm_answer or "VLM not available" in vlm_answer:
                print(f"Skipping VLM answer: '{vlm_answer}' (empty or error)")
                return
            
            print("=" * 50)
            print("VLM ANSWER RECEIVED")
            print("=" * 50)
            print(f"Answer: '{vlm_answer}'")
            print(f"Active breach: {self.current_breach_active}")
            print(f"Current detection prompts: {self.detection_prompts}")
            print(f"Detection enabled: {self.detection_enabled}")
            print("=" * 50)
            
            # Add to detection prompts if not already present
            if vlm_answer not in self.detection_prompts:
                self.detection_prompts.append(vlm_answer)
                print(f"Added '{vlm_answer}' to prompts")
                
                # Update current detection prompt
                self.current_detection_prompt = vlm_answer
                
                # Only enable detection if we're in an active breach
                if self.current_breach_active:
                    self.detection_enabled = True
                    print(f"Detection ENABLED (breach active)")
                else:
                    self.detection_enabled = False
                    print(f"Detection DISABLED (no active breach)")
                
                # Update YOLO model with new prompt
                if self.yolo_sam_detector.update_prompts(self.detection_prompts):
                    if self.current_breach_active:
                        print(f"YOLO updated for active breach")
                    else:
                        print(f"YOLO updated but detection disabled")
                else:
                    print(f"Failed to update YOLO prompts")
                    # Reset detection enabled flag if update failed
                    self.detection_enabled = False
            
            # Associate VLM answer with risk buffer using reliable logic
            self.associate_vlm_answer_with_buffer_reliable(vlm_answer)
            
            # Add VLM answer to object list and encode embedding (simple, no buffer logic)
            if hasattr(self, 'naradio_processor') and self.naradio_processor.is_ready():
                success = self.naradio_processor.add_vlm_object(vlm_answer)
                if success:
                    print(f"✓ VLM object '{vlm_answer}' added to object list")
                    print(f"  Total objects: {len(self.naradio_processor.get_all_objects())}")
                    print(f"  Object list: {self.naradio_processor.get_all_objects()}")
                    print(f"  Object colors: {self.naradio_processor.get_all_colors()}")
                    print(f"  VLM object '{vlm_answer}' will appear in WHITE in combined segmentation")
                else:
                    print(f"✗ Failed to add VLM object '{vlm_answer}' to object list")
            
            # Queue parallel historical analysis (non-blocking)
            self.queue_historical_analysis('vlm_answer_received', vlm_answer=vlm_answer)
            
            # Print current buffer status after VLM answer processing
            print(f"[VLM] Current buffer status after processing:")
            if self.risk_buffer_manager:
                status = self.risk_buffer_manager.get_status()
                print(f"  Active buffers: {status['active_buffers']}")
                print(f"  Frozen buffers: {status['frozen_buffers']}")
                print(f"  Active with cause: {status['active_with_cause']}")
                print(f"  Frozen with cause: {status['frozen_with_cause']}")
                print(f"  Frozen needing cause: {status['frozen_needing_cause']}")

        except Exception as e:
            print(f"Error processing VLM answer: {e}")

    def associate_vlm_answer_with_buffer_reliable(self, vlm_answer):
        """Reliable VLM answer association with buffer management"""
        try:
            print("=" * 60)
            print("VLM ANSWER BUFFER ASSOCIATION")
            print("=" * 60)
            print(f"VLM Answer: '{vlm_answer}'")
            print(f"Current breach active: {self.current_breach_active}")
            
            # Get current buffer status BEFORE assignment
            status = self.risk_buffer_manager.get_status()
            candidates = self.risk_buffer_manager.get_cause_assignment_candidates()
            
            print(f"Buffer status BEFORE assignment:")
            print(f"  Active buffers: {status['active_buffers']}")
            print(f"  Frozen buffers: {status['frozen_buffers']}")
            print(f"  Active with cause: {status['active_with_cause']}")
            print(f"  Frozen with cause: {status['frozen_with_cause']}")
            print(f"  Frozen needing cause: {status['frozen_needing_cause']}")
            
            print(f"Cause assignment candidates:")
            print(f"  Frozen needing cause: {candidates['frozen_needing_cause']}")
            print(f"  Active available: {candidates['active_available']}")
            print(f"  Frozen with cause: {candidates['frozen_with_cause']}")
            print(f"  Active with cause: {candidates['active_with_cause']}")
            
            # Print detailed buffer info
            self.risk_buffer_manager.print_status()
            
            # Try to assign cause using the buffer manager's improved logic
            print(f"Attempting to assign cause '{vlm_answer}'...")
            success = self.risk_buffer_manager.assign_cause(vlm_answer)
            
            if success:
                print("=" * 60)
                print("CAUSE ASSIGNMENT SUCCESSFUL")
                print("=" * 60)
                print(f"Successfully associated '{vlm_answer}' with risk buffer")
                
                # Print detailed buffer status after assignment
                print("Buffer status AFTER assignment:")
                self.risk_buffer_manager.print_status()
                
                # Historical analysis is now handled in parallel (non-blocking)
                print("Historical analysis queued for parallel processing")
            else:
                print("=" * 60)
                print("CAUSE ASSIGNMENT FAILED")
                print("=" * 60)
                print(f"No suitable buffer found for '{vlm_answer}'")
                print(f"Possible reasons:")
                print(f"   - No breaches have occurred yet")
                print(f"   - All buffers already have causes assigned")
                print(f"   - Buffer system is not properly initialized")
                print(f"   - All buffers are in invalid states")
                
                # Print current buffer status for debugging
                print("Final buffer status:")
                self.risk_buffer_manager.print_status()
            
            print("=" * 60)
                    
        except Exception as e:
            print(f"Error associating VLM answer with buffer: {e}")
            import traceback
            traceback.print_exc()

    def rgb_callback(self, msg):
        """Store latest RGB message and add to active buffers."""
        with self.processing_lock:
            self.latest_rgb_msg = msg
            # Store current pose when RGB is received for detection processing
            if self.latest_pose is not None:
                self.detection_pose = self.latest_pose.copy()
                self.detection_pose_time = self.latest_pose_time
            
            # Add to image buffer for narration-triggered publishing
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Add to buffer with proper timestamp
            self.image_buffer.append((cv_image, msg_timestamp, msg))
            
            # Maintain buffer size
            if len(self.image_buffer) > self.max_buffer_size:
                self.image_buffer.pop(0)
            
            # Add to rolling 1-second buffer for breach context
            current_system_time = time.time()
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.rolling_image_buffer.append((rgb_image, current_system_time, msg))
            
            # Clean up old images from rolling buffer
            while self.rolling_image_buffer and (current_system_time - self.rolling_image_buffer[0][1]) > self.rolling_buffer_duration:
                self.rolling_image_buffer.pop(0)
            
            # Add image to active buffers (if any are active AND we're in breach)
            with self.lock:
                if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0 and self.current_breach_active:
                    self.risk_buffer_manager.add_image(msg_timestamp, cv_image, msg)
                    print(f"Added image to buffer at {msg_timestamp:.3f}")
                else:
                    print(f"Not adding image to buffer - active_buffers: {len(self.risk_buffer_manager.active_buffers) if self.risk_buffer_manager else 0}, breach_active: {self.current_breach_active}")

    def depth_callback(self, msg):
        """Store latest depth message and add to active buffers."""
        with self.processing_lock:
            self.latest_depth_msg = msg
            
            # Add depth msg to active buffers (if any are active AND we're in breach)
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.lock:
                if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0 and self.current_breach_active:
                    self.risk_buffer_manager.add_depth_msg(msg_timestamp, msg)
                    print(f"Added depth msg to buffer at {msg_timestamp:.3f}")
                else:
                    print(f"Not adding depth msg to buffer - active_buffers: {len(self.risk_buffer_manager.active_buffers) if self.risk_buffer_manager else 0}, breach_active: {self.current_breach_active}")

    def pose_callback(self, msg):
        """Process pose and trigger detection if needed."""
        # Get pose data first
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Drift calculation
        drift, nearest_idx = self.drift_calculator.compute_drift(pos)
        self.breach_idx = nearest_idx
        
        # Update latest pose data
        with self.lock:
            self.latest_pose = pos
            self.latest_pose_time = pose_time
            self.last_pose = pos
            self.last_pose_time = pose_time
            
            # Store actual trajectory for narration
            self.narration_manager.add_actual_point(pos, pose_time, self.flip_y_axis)

        # Breach detection
        breach_now = self.drift_calculator.is_breach(drift)
        
        # Check for breach state transitions
        breach_started = not self.last_breach_state and breach_now  # F→T
        breach_ended = self.last_breach_state and not breach_now    # T→F
        
        # Print debug info during breach transitions or when in breach
        if breach_started or breach_ended or self.current_breach_active:
            print(f"[BREACH] t={pose_time:.2f}, drift={drift:.3f}, threshold={self.drift_calculator.soft_threshold}")
            print(f"[BREACH] breach_now={breach_now}, last_state={self.last_breach_state}")
            print(f"[BREACH] breach_started={breach_started}, breach_ended={breach_ended}")
            print(f"[BREACH] current_breach_active={self.current_breach_active}")
            print(f"[BREACH] narration_sent={self.narration_manager.get_narration_sent()}")
        
        # Handle breach state transitions
        if breach_started:
            # BREACH START: F→T transition
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            self.detection_enabled = True
            
            print("=" * 60)
            print("BREACH STARTED")
            print("=" * 60)
            print(f"Time: {pose_time:.2f}")
            print(f"Drift: {drift:.3f} (threshold: {self.drift_calculator.soft_threshold:.3f})")
            print(f"Detection enabled")
            
            # Only reset detection prompts if this is the first breach or if we have no prompts
            if not self.detection_prompts:
                self.detection_prompts = []
                self.current_detection_prompt = ""
                print(f"Detection prompts reset (first breach)")
            else:
                print(f"Keeping existing detection prompts: {self.detection_prompts}")
            
            # Start new buffer
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
                print(f"New buffer started at {pose_time:.3f}")
            
            # Save rolling buffer images around breach start
            current_system_time = time.time()
            self.save_rolling_buffer_images(current_system_time)
            
            print("=" * 60)
            
            # Queue the start event for narration processing
            self.narration_manager.queue_breach_event('start', pose_time)
            
        elif breach_ended:
            # BREACH END: T→F transition
            self.last_breach_state = False
            self.current_breach_active = False
            
            print("=" * 60)
            print("BREACH ENDED")
            print("=" * 60)
            print(f"Time: {pose_time:.2f}")
            print(f"Drift: {drift:.3f} (threshold: {self.drift_calculator.soft_threshold:.3f})")
            print(f"State: last_breach_state=True -> False")
            print(f"State: current_breach_active=True -> False")
            
            # Freeze active buffers
            if self.risk_buffer_manager:
                frozen_buffers = self.risk_buffer_manager.freeze_active_buffers(pose_time)
                print(f"Buffers frozen at {pose_time:.3f}")
                
                # Print buffer status after freezing
                with self.lock:
                    frozen_buffers_copy = self.risk_buffer_manager.frozen_buffers.copy()
                    active_buffers = self.risk_buffer_manager.active_buffers.copy()
                
                print(f"After freezing - Active buffers: {len(active_buffers)}, Frozen buffers: {len(frozen_buffers_copy)}")
                for i, buffer in enumerate(frozen_buffers_copy):
                    print(f"  Frozen buffer {i}: {buffer.buffer_id}, poses: {len(buffer.poses)}, images: {len(buffer.images)}")
                    print(f"    Has cause: {buffer.has_cause()}, Cause: '{buffer.cause}'")
                    
                    # Queue parallel analysis for each frozen buffer (non-blocking)
                    # Queue for ALL buffers, not just those with causes
                    self.queue_historical_analysis('buffer_frozen', buffer_id=buffer.buffer_id)
                    print(f"Queued historical analysis for frozen buffer: {buffer.buffer_id}")
                
                # Save comprehensive buffers
                self.save_comprehensive_buffers(pose_time)
            
            self.detection_enabled = False
            print(f"Detection disabled")
            
            if self.narration_manager.get_narration_sent():
                print(f"Narration: SENT")
            else:
                print(f"Narration: NOT SENT")
            
            print("=" * 60)
            
            # Queue the end event for narration processing
            self.narration_manager.queue_breach_event('end', pose_time)
            
        # Handle case where we're already in breach but current_breach_active is False
        elif breach_now and not self.current_breach_active:
            self.last_breach_state = True
            self.current_breach_active = True
            self.narration_manager.reset_narration_state()
            self.detection_enabled = True
            
            print("=" * 60)
            print("BREACH DETECTED (already in progress)")
            print("=" * 60)
            print(f"Time: {pose_time:.2f}")
            print(f"Drift: {drift:.3f} (threshold: {self.drift_calculator.soft_threshold:.3f})")
            print(f"Detection enabled")
            
            # Only reset detection prompts if this is the first breach or if we have no prompts
            if not self.detection_prompts:
                self.detection_prompts = []
                self.current_detection_prompt = ""
                print(f"Detection prompts reset (first breach)")
            else:
                print(f"Keeping existing detection prompts: {self.detection_prompts}")
            
            # Start new buffer for this ongoing breach
            if self.risk_buffer_manager:
                self.risk_buffer_manager.start_buffer(pose_time)
                print(f"New buffer started at {pose_time:.3f}")
            
            print("=" * 60)
            self.narration_manager.queue_breach_event('start', pose_time)
        
        # Update last_breach_state even if no transition occurred
        if not breach_started and not breach_ended and not (breach_now and not self.current_breach_active):
            self.last_breach_state = breach_now
        
        # Add pose to active buffers
        with self.lock:
            if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0 and self.current_breach_active:
                self.risk_buffer_manager.add_pose(pose_time, pos, drift)
                print(f"Added pose to buffer at {pose_time:.3f}, drift: {drift:.3f}")
                
                # CONTINUOUS LOGGING: Save the image that was taken ~1 second ago
                # This gives us lagged image context for each breach frame
                # Use the same logic as narration system - find image from ~1 second ago
                if self.image_buffer:
                    # Use the newest buffer timestamp as current time reference (same as narration)
                    newest_timestamp = self.image_buffer[-1][1]
                    current_time = newest_timestamp
                    
                    # Target time is 1 second ago (same as narration)
                    target_time_offset = 1.0
                    target_time = current_time - target_time_offset
                    
                    # Check if we have enough buffer history for 1 second
                    oldest_timestamp = self.image_buffer[0][1] if self.image_buffer else current_time
                    available_time_back = current_time - oldest_timestamp
                    
                    if available_time_back < target_time_offset:
                        # Not enough history, use the oldest available image
                        target_time = oldest_timestamp
                        actual_offset = available_time_back
                    else:
                        actual_offset = target_time_offset
                    
                    # Find the closest image to the target time (same as narration)
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
                        # Get buffer info while we still have the lock
                        current_buffer_id = None
                        if len(self.risk_buffer_manager.active_buffers) > 0:
                            current_buffer_id = self.risk_buffer_manager.active_buffers[-1].buffer_id
                        
                        # Save this lagged image to the current buffer
                        time_lag = current_time - closest_msg.header.stamp.sec - closest_msg.header.stamp.nanosec * 1e-9
                        self.save_lagged_image_to_buffer(closest_image, closest_msg.header.stamp.sec + closest_msg.header.stamp.nanosec * 1e-9, closest_msg, current_time, actual_offset, current_buffer_id)
                        print(f"Saved lagged image (t={actual_offset:.3f}s ago) to buffer")
                    else:
                        print(f"No suitable lagged image found in image buffer")
                else:
                    print(f"No image buffer available for lagged image logging")
            else:
                print(f"Not adding pose to buffer - active_buffers: {len(self.risk_buffer_manager.active_buffers) if self.risk_buffer_manager else 0}, breach_active: {self.current_breach_active}")

        # Continuous narration checking during active breach
        if self.current_breach_active and not self.narration_manager.get_narration_sent():
            print(f"NARRATION CHECK: drift={drift:.3f}, checking for narration...")
            narration = self.narration_manager.check_for_narration(pose_time, self.breach_idx)
            if narration:
                self.publish_narration_with_image(narration)
                self.narration_pub.publish(String(data=narration))

        # Trigger detection processing (only if in active breach)
        with self.processing_lock:
            if (self.latest_rgb_msg is not None and 
                self.latest_depth_msg is not None and 
                self.detection_enabled and 
                self.current_breach_active and
                not self.is_processing):
                self.trigger_detection_processing()

        # TF publish
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
            # Get latest messages at the start
            with self.processing_lock:
                if (self.latest_rgb_msg is None or 
                    self.latest_depth_msg is None or 
                    self.detection_pose is None):
                    return
                
                rgb_msg = self.latest_rgb_msg
                depth_msg = self.latest_depth_msg
                pose = self.detection_pose
                pose_time = self.detection_pose_time
            
            # Early exit if YOLO model not ready
            if not self.yolo_sam_detector.is_ready():
                return
                
            # Early exit if camera info not received
            if not self.camera_info_received:
                return
            
            # Process detection with the latest messages
            self.process_detection(rgb_msg, depth_msg, pose, pose_time)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False

    def process_detection(self, rgb_msg, depth_msg, pose, pose_time):
        """Process YOLO-SAM detection with latest messages."""
        try:
            # Convert images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Create transform matrix from current pose to origin
            transform_matrix = self.pose_to_transform_matrix(pose)

            # Run YOLO-World detection
            bboxes, labels, confidences = self.yolo_sam_detector.detect_objects(rgb_image)
            
            # Early exit if no YOLO detections
            if not bboxes:
                return

            # Filter detections by distance
            filtered_bboxes, valid_indices = self.yolo_sam_detector.filter_detections_by_distance(
                bboxes, depth_image, self.min_detection_distance, self.max_detection_distance, self.camera_intrinsics)
            
            # Early exit if no detections pass distance filtering
            if not filtered_bboxes:
                return

            # Extract centers from filtered bboxes
            bbox_centers = []
            for bbox in filtered_bboxes:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                bbox_centers.append([center_x, center_y])

            # Draw YOLO bounding boxes
            yolo_bbox_img = rgb_image.copy()
            for bbox in filtered_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(yolo_bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(yolo_bbox_img, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Publish YOLO bbox image
            yolo_bbox_msg = self.bridge.cv2_to_imgmsg(yolo_bbox_img, encoding='rgb8')
            yolo_bbox_msg.header = rgb_msg.header
            self.yolo_bbox_pub.publish(yolo_bbox_msg)

            # Process with SAM
            if len(bbox_centers) == 0:
                return
            
            all_masks, all_scores = self.yolo_sam_detector.segment_objects(rgb_image, bbox_centers)
            
            # Create combined mask visualization
            if len(all_masks) > 0:
                combined_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
                for i, mask in enumerate(all_masks):
                    mask_value = min(255, (i + 1) * 60)
                    combined_mask[mask] = mask_value
                
                # Publish SAM mask image
                mask_img_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
                sam_mask_msg = self.bridge.cv2_to_imgmsg(mask_img_color, encoding='rgb8')
                sam_mask_msg.header = rgb_msg.header
                self.sam_mask_pub.publish(sam_mask_msg)
                 
                # Create 3D points from bbox centers
                success = self.pointcloud_manager.create_bbox_center_pointclouds(
                    filtered_bboxes, depth_image, self.camera_intrinsics, transform_matrix,
                    self.min_detection_distance, self.max_detection_distance)
                
                if success:
                    # Publish accumulated point cloud
                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    header.frame_id = 'map'
                    pointcloud_msg = self.pointcloud_manager.create_pointcloud_message(header)
                    self.detection_cloud_pub.publish(pointcloud_msg)
                    
                    # Increment frame count
                    self.pointcloud_manager.increment_frame_count()

        except Exception as e:
            import traceback
            traceback.print_exc()

    def pose_to_transform_matrix(self, pose):
        """Create transform matrix from pose to origin."""
        T = np.eye(4)
        initial_pose = self.drift_calculator.get_initial_pose()
        
        # Translation: current pose relative to origin
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
        # Use the newest buffer timestamp as current time reference
        if not self.image_buffer:
            # If no image buffer, just publish the text
            self.narration_text_pub.publish(String(data=narration_text))
            return
            
        newest_timestamp = self.image_buffer[-1][1]
        current_time = newest_timestamp  # Use newest buffer timestamp as current time
        
        # For breach narration, we want image from 1 second before the breach
        target_time_offset = 1.0  # Publish image from 1 second ago
        
        if self.image_buffer:
            # Find the closest image to the target time, or go as far back as possible
            target_time = current_time - target_time_offset
            
            # Check if we have enough buffer history for 1 second
            oldest_timestamp = self.image_buffer[0][1] if self.image_buffer else current_time
            available_time_back = current_time - oldest_timestamp
            
            if available_time_back < target_time_offset:
                # Not enough history, use the oldest available image
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
                # Save narration image to current buffer directory
                self.save_narration_image_to_buffer(closest_image, narration_text, current_time)
                
                # Publish the image with the original message's timestamp
                image_msg = self.bridge.cv2_to_imgmsg(closest_image, encoding='rgb8')
                image_msg.header.stamp = closest_msg.header.stamp  # Use original timestamp
                image_msg.header.frame_id = closest_msg.header.frame_id  # Use original frame_id
                self.narration_image_pub.publish(image_msg)
                
                # Publish the narration text immediately after the image
                self.narration_text_pub.publish(String(data=narration_text))
            else:
                # If no suitable image found, just publish the text
                self.narration_text_pub.publish(String(data=narration_text))
        else:
            # If no image buffer, just publish the text
            self.narration_text_pub.publish(String(data=narration_text))

    def save_comprehensive_buffers(self, current_time):
        """Save all frozen buffers with comprehensive metadata, cause images, narration, and other data."""
        try:
            with self.lock:
                frozen_buffers = self.risk_buffer_manager.frozen_buffers.copy()
                active_buffers = self.risk_buffer_manager.active_buffers.copy()
            
            print("=" * 60)
            print("BUFFER STATUS DEBUG")
            print("=" * 60)
            print(f"Active buffers: {len(active_buffers)}")
            print(f"Frozen buffers: {len(frozen_buffers)}")
            
            if active_buffers:
                print("Active buffer details:")
                for i, buffer in enumerate(active_buffers):
                    print(f"  Buffer {i}: {buffer.buffer_id}, poses: {len(buffer.poses)}, images: {len(buffer.images)}")
            
            if frozen_buffers:
                print("Frozen buffer details:")
                for i, buffer in enumerate(frozen_buffers):
                    print(f"  Buffer {i}: {buffer.buffer_id}, poses: {len(buffer.poses)}, images: {len(buffer.images)}")
            print("=" * 60)
            
            if not frozen_buffers:
                print("No frozen buffers to save")
                return
            
            print("=" * 60)
            print("SAVING COMPREHENSIVE BUFFER DATA")
            print("=" * 60)
            print(f"Run directory: {self.current_run_dir}")
            print(f"Frozen buffers to save: {len(frozen_buffers)}")
            print("=" * 60)
            
            saved_count = 0
            for buffer in frozen_buffers:
                if self.save_single_comprehensive_buffer(buffer):
                    saved_count += 1
            
            # Save run summary
            self.save_run_summary(frozen_buffers)
            
            print("=" * 60)
            print(f"COMPREHENSIVE BUFFER SAVE COMPLETED")
            print("=" * 60)
            print(f"Saved: {saved_count}/{len(frozen_buffers)} buffers")
            print(f"Directory: {self.current_run_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error saving comprehensive buffers: {e}")
            import traceback
            traceback.print_exc()

    def save_single_comprehensive_buffer(self, buffer):
        """Save a single buffer with all metadata, images, narration, and cause data."""
        try:
            print(f"Saving buffer {buffer.buffer_id}: poses={len(buffer.poses)}, images={len(buffer.images)}")
            
            # Create buffer-specific directory
            buffer_dir = os.path.join(self.current_run_dir, buffer.buffer_id)
            os.makedirs(buffer_dir, exist_ok=True)
            
            # Get thresholds
            soft_threshold, hard_threshold = self.drift_calculator.get_thresholds()
            
            # Save comprehensive metadata
            metadata = {
                'buffer_id': buffer.buffer_id,
                'start_time': buffer.start_time,
                'end_time': buffer.end_time,
                'state': buffer.state.value,
                'cause': buffer.cause,
                'data_counts': buffer.get_data_counts(),
                'duration': buffer.get_duration(),
                'narration_sent': self.narration_manager.get_narration_sent() if buffer.is_frozen() else False,
                'detection_enabled': self.detection_enabled,
                'detection_prompts': self.detection_prompts,
                'current_detection_prompt': self.current_detection_prompt,
                'soft_threshold': soft_threshold,
                'hard_threshold': hard_threshold,
                'save_timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(buffer_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save pose data
            if buffer.poses:
                poses_data = []
                for timestamp, pose, drift in buffer.poses:
                    poses_data.append({
                        'timestamp': timestamp,
                        'pose_x': float(pose[0]),
                        'pose_y': float(pose[1]), 
                        'pose_z': float(pose[2]),
                        'drift': float(drift)
                    })
                
                with open(os.path.join(buffer_dir, 'poses.json'), 'w') as f:
                    json.dump(poses_data, f, indent=2)
                
                # Also save as numpy array for analysis
                poses_array = np.array([(t, p[0], p[1], p[2], d) for t, p, d in buffer.poses])
                np.save(os.path.join(buffer_dir, 'poses.npy'), poses_array)
            
            # Save images with metadata
            if buffer.images:
                images_dir = os.path.join(buffer_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
                
                images_metadata = []
                for i, (timestamp, cv_image, ros_msg) in enumerate(buffer.images):
                    # Save image
                    image_filename = f"image_{i:04d}_{timestamp:.3f}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, cv_image)
                    
                    # Store image metadata
                    images_metadata.append({
                        'index': i,
                        'timestamp': timestamp,
                        'filename': image_filename,
                        'frame_id': ros_msg.header.frame_id,
                        'image_width': cv_image.shape[1],
                        'image_height': cv_image.shape[0]
                    })
                
                # Save images metadata
                with open(os.path.join(buffer_dir, 'images_metadata.json'), 'w') as f:
                    json.dump(images_metadata, f, indent=2)
                
                # Save first and last images as key frames
                if len(buffer.images) >= 1:
                    first_img = buffer.images[0][1]
                    cv2.imwrite(os.path.join(buffer_dir, 'first_image.png'), first_img)
                
                if len(buffer.images) >= 2:
                    last_img = buffer.images[-1][1]
                    cv2.imwrite(os.path.join(buffer_dir, 'last_image.png'), last_img)
            
            # Save depth messages metadata
            if buffer.depth_msgs:
                depth_metadata = []
                for timestamp, depth_msg in buffer.depth_msgs.items():
                    depth_metadata.append({
                        'timestamp': timestamp,
                        'frame_id': depth_msg.header.frame_id,
                        'width': depth_msg.width,
                        'height': depth_msg.height,
                        'encoding': depth_msg.encoding
                    })
                
                with open(os.path.join(buffer_dir, 'depth_metadata.json'), 'w') as f:
                    json.dump(depth_metadata, f, indent=2)
            
            # Save cause-specific data
            if buffer.cause:
                cause_dir = os.path.join(buffer_dir, 'cause_analysis')
                os.makedirs(cause_dir, exist_ok=True)
                
                cause_data = {
                    'cause': buffer.cause,
                    'assigned_timestamp': datetime.now().isoformat(),
                    'detection_prompts_at_time': self.detection_prompts.copy(),
                    'current_detection_prompt': self.current_detection_prompt,
                    'detection_enabled_at_time': self.detection_enabled
                }
                
                with open(os.path.join(cause_dir, 'cause_data.json'), 'w') as f:
                    json.dump(cause_data, f, indent=2)
            
            # Save narration data if available
            narration_dir = os.path.join(buffer_dir, 'narration')
            os.makedirs(narration_dir, exist_ok=True)
            
            # Try to find narration from image buffer around breach time
            if buffer.images:
                # Find image closest to breach start time
                breach_start_time = buffer.start_time
                closest_image = None
                min_time_diff = float('inf')
                
                for image, timestamp, msg in self.image_buffer:
                    time_diff = abs(timestamp - breach_start_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_image = image
                
                if closest_image is not None:
                    cv2.imwrite(os.path.join(narration_dir, 'breach_context_image.png'), closest_image)
                    
                    narration_context = {
                        'breach_start_time': breach_start_time,
                        'context_image_timestamp': breach_start_time - min_time_diff,
                        'time_difference': min_time_diff,
                        'narration_sent': self.narration_manager.get_narration_sent() if buffer.is_frozen() else False
                    }
                    
                    with open(os.path.join(narration_dir, 'narration_context.json'), 'w') as f:
                        json.dump(narration_context, f, indent=2)
            
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

    def save_segmentation_to_buffer(self, segmentation_result, timestamp):
        """Save segmentation results to the current active buffer."""
        try:
            if not self.risk_buffer_manager or len(self.risk_buffer_manager.active_buffers) == 0:
                return
            
            current_buffer = self.risk_buffer_manager.active_buffers[-1]
            buffer_dir = os.path.join(self.current_run_dir, current_buffer.buffer_id)
            
            # Create segmentation directory
            segmentation_dir = os.path.join(buffer_dir, 'segmentation')
            os.makedirs(segmentation_dir, exist_ok=True)
            
            # Convert timestamp to string
            timestamp_str = f"{timestamp.sec}_{timestamp.nanosec}"
            
            # Save original mask
            if segmentation_result['original_mask'] is not None:
                original_mask_path = os.path.join(segmentation_dir, f"original_mask_{timestamp_str}.png")
                cv2.imwrite(original_mask_path, segmentation_result['original_mask'])
            
            # Save refined mask
            if segmentation_result['refined_mask'] is not None:
                refined_mask_path = os.path.join(segmentation_dir, f"refined_mask_{timestamp_str}.png")
                cv2.imwrite(refined_mask_path, segmentation_result['refined_mask'])
            
            # Save metadata
            metadata = segmentation_result.get('metadata', {})
            metadata['timestamp'] = timestamp.sec + timestamp.nanosec * 1e-9
            metadata['processing_info'] = segmentation_result.get('processing_info', {})
            
            metadata_path = os.path.join(segmentation_dir, f"segmentation_metadata_{timestamp_str}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved segmentation results to buffer {current_buffer.buffer_id}")
            
        except Exception as e:
            print(f"Error saving segmentation to buffer: {e}")
            import traceback
            traceback.print_exc()


def main():
    rclpy.init()
    node = ResilienceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the narration manager
        node.narration_manager.stop()
        
        # Stop the narration worker thread gracefully
        if hasattr(node, 'narration_running') and node.narration_running:
            print("Stopping narration thread...")
            with node.narration_condition:
                node.narration_running = False
                node.narration_condition.notify_all()
            
            # Wait for thread to finish
            if hasattr(node, 'narration_thread') and node.narration_thread and node.narration_thread.is_alive():
                node.narration_thread.join(timeout=2.0)
                print("Narration thread stopped")
        
        # Stop the parallel NARadio thread gracefully
        if node.naradio_running:
            print("Stopping parallel NARadio thread...")
            node.naradio_running = False
            
            if hasattr(node, 'naradio_thread') and node.naradio_thread and node.naradio_thread.is_alive():
                node.naradio_thread.join(timeout=2.0)
                print("Parallel NARadio thread stopped")
        
        # Stop the parallel historical analysis thread gracefully
        if hasattr(node, 'historical_analysis_running') and node.historical_analysis_running:
            print("Stopping parallel historical analysis thread...")
            node.historical_analysis_running = False
            
            # Notify the condition to wake up the thread
            with node.historical_analysis_condition:
                node.historical_analysis_condition.notify_all()
            
            if hasattr(node, 'historical_analysis_thread') and node.historical_analysis_thread and node.historical_analysis_thread.is_alive():
                node.historical_analysis_thread.join(timeout=2.0)
                print("Parallel historical analysis thread stopped")
        
        # Clean up radio model and GPU memory
        if hasattr(node, 'naradio_processor'):
            print("Cleaning up NARadio model...")
            try:
                if node.naradio_processor.is_ready():
                    node.naradio_processor.cleanup_memory()
                    print("NARadio model cleaned up successfully")
                else:
                    print("NARadio model was not ready, skipping cleanup")
            except Exception as e:
                print(f"Error cleaning up NARadio model: {e}")
        

        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
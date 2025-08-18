### Pre Req.
Make sure to have the Yolo World model and the SAM model in the assets directory<br>
Check Rosbag and corresponding topic names, we need pose<br>
Run calibrate_drift to generate a nominal trajectory JSON file and breach thresholds<br>
### We will run the primary node and secondary node ( after exporting the VLM API key ) in two terminals, get them ready and then play the ROS [bag](https://drive.google.com/file/d/1xGwocP0WNUHb9ZkYeE7lBwIxGHjkAZBY/view?usp=sharing) in the third terminal<br>

### Primary Node: `main.py`
Main orchestrator implementing parallel processing across 4 dedicated threads with comprehensive state management and inter-thread communication.

### Secondary Node: `narration_display_node.py`  
VLM integration node handling OpenAI API communication and image-text synchronisation.

### High-Level Overview
Main file imports the necessary modules ->
DriftCalulator -> Handles all trajectory-related queries, loads nominal trajectory, gives current drift based on current pos, gives the breach status -> load nominal trajectory as json file<br>
RiskBufferManager -> All the details for a breach are stored within a buffer, which contains images, lagged images, associated cause, embedding, pose and other data as well. will automatically save the buffer after the breach ends<br>
NarrationManager -> Generates the narration as discussed, all drift values are constantly monitored and added so that we can quickly analyse the nature of the drift to produce narration and publish it<br>
Historical Analysis -> Once we have received the VLM Answer ( this can be after or during the corresponding breach ), we start Historical Analysis, where we analyse the stored images and try to find the <br>location of the cause object using the VLM answer. This location is also stored in the buffer
YoloSamDetector -> Straightforward, all detection and dynamic prompts functions<br>
NaradioProcessor -> All Naradio functionalities, has dynamic embedding + segmentation function, not integrated for re-id yet<br>


```

### Internal Module Structure
```python
from resilience.drift_calculator import DriftCalculator
from resilience.yolo_sam_detector import YOLOSAMDetector  
from resilience.naradio_processor import NARadioProcessor
from resilience.narration_manager import NarrationManager
from resilience.pointcloud_manager import PointCloudManager
from resilience.risk_buffer import RiskBufferManager
from resilience.historical_cause_analysis import HistoricalCauseAnalyzer
from resilience.simple_descriptive_narration import XYSpatialDescriptor, TrajectoryPoint
```

## Threading Architecture

### Thread 1: Main Thread (ResilienceNode)
**Purpose**: ROS2 message processing, breach detection, state management
**Key Methods**:
- `pose_callback()`: Trajectory processing and breach detection logic
- `rgb_callback()`: Image data buffering and active buffer management  
- `depth_callback()`: Depth data collection during breach states
- `vlm_answer_callback()`: VLM response processing and cause association

**State Variables**:
```python
self.current_breach_active: bool          # Primary breach state flag
self.last_breach_state: bool              # Previous breach state for transition detection  
self.detection_enabled: bool              # YOLO detection activation flag
self.detection_prompts: List[str]         # Dynamic YOLO vocabulary
self.latest_rgb_msg: Image               # Thread-safe image storage
self.latest_depth_msg: Image             # Thread-safe depth storage
self.latest_pose: np.ndarray             # Current robot position
```

### Thread 2: Detection Processing Thread
**Purpose**: YOLO-SAM object detection and segmentation
**Trigger**: Activated during breach states via `trigger_detection_processing()`
**Key Operations**:
```python
def process_detection_async(self):
    # Convert ROS messages to OpenCV format
    rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
    depth_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
    
    # YOLO-World detection with dynamic vocabulary
    bboxes, labels, confidences = self.yolo_sam_detector.detect_objects(rgb_image)
    
    # Distance-based filtering using depth data
    filtered_bboxes = self.yolo_sam_detector.filter_detections_by_distance(...)
    
    # SAM segmentation on filtered detections
    all_masks, all_scores = self.yolo_sam_detector.segment_objects(rgb_image, bbox_centers)
```

### Thread 3: NARadio Processing Thread  
**Purpose**: Continuous semantic feature extraction and segmentation
**Execution**: Parallel loop at 10Hz with memory management
**Key Operations**:
```python
def naradio_processing_loop(self):
    while self.naradio_running:
        # Feature extraction using vision transformer
        feat_map_np, naradio_vis = self.naradio_processor.process_features(rgb_image)
        
        # Combined segmentation with dynamic objects
        segmentation_result = self.naradio_processor.process_combined_segmentation(rgb_image)
        
        # Publish visualisation and masks
        self.naradio_image_pub.publish(naradio_msg)
        self.original_mask_pub.publish(original_mask_msg)
        self.refined_mask_pub.publish(refined_mask_msg)
```

### Thread 4: Historical Analysis Thread
**Purpose**: Background historical pattern analysis
**Trigger**: VLM answers and buffer state changes
**Key Operations**:
```python
def historical_analysis_worker(self):
    # Process queued analysis requests
    while self.historical_analysis_queue:
        request = self.historical_analysis_queue.pop(0)
        self.process_historical_analysis_request(request)
        
def perform_parallel_vlm_analysis(self, vlm_answer):
    # Analyze frozen buffers for cause patterns
    frozen_results = self.historical_analyzer.analyze_all_buffers()
    # Cross-reference with active buffers
    active_results = self.historical_analyzer.analyze_active_buffers(active_with_cause)
```

### Thread 5: Narration Processing Thread
**Purpose**: Spatial narration generation and image context management
**Key Operations**:
```python
def narration_worker(self):
    # Process narration requests from the queue
    while self.narration_queue:
        request = self.narration_queue.pop(0)
        self.process_narration_request(request)

def publish_narration_with_image(self, narration_text):
    # Retrieve historical image (1 second offset)
    target_time = current_time - 1.0
    closest_image = self.find_closest_image(target_time)
    # Publish synchronized image and text
```

## Class Architecture and Responsibilities

### ResilienceNode Class
**Primary State Machine**:
```python
# Breach state transitions
breach_started = not self.last_breach_state and breach_now  # F→T
breach_ended = self.last_breach_state and not breach_now    # T→F

# Buffer lifecycle management  
if breach_started:
    self.risk_buffer_manager.start_buffer(pose_time)
elif breach_ended:
    self.risk_buffer_manager.freeze_active_buffers(pose_time)
```

**Thread Synchronization**:
```python
self.processing_lock = threading.Lock()           # Detection thread sync
self.naradio_processing_lock = threading.Lock()   # NARadio thread sync  
self.lock = threading.Lock()                      # Main state sync
self.historical_analysis_condition = threading.Condition()  # Analysis queue
self.narration_condition = threading.Condition()  # Narration queue
```

### DriftCalculator Class
**Core Functionality**: Spline-based trajectory analysis
```python
def compute_drift(self, pos: np.ndarray) -> Tuple[float, int]:
    # Calculate minimum distance to nominal trajectory
    distances = np.linalg.norm(self.nominal_points - pos, axis=1)
    nearest_idx = np.argmin(distances)
    return distances[nearest_idx], nearest_idx

def is_breach(self, drift: float) -> bool:
    return drift > self.soft_threshold
```

### YOLOSAMDetector Class  
**Dynamic Model Management**:
```python
def update_prompts(self, new_prompts: List[str]) -> bool:
    # Update YOLO-World vocabulary dynamically
    if new_prompts != self.current_classes:
        self.world_model.set_classes(new_prompts)
        self.current_classes = new_prompts
        return True
    return False

def detect_objects(self, image: np.ndarray) -> Tuple[List, List, List]:
    # YOLO-World inference with current vocabulary
    results = self.world_model(image, imgsz=self.yolo_imgsz, conf=self.min_confidence)
```

### NARadioProcessor Class
**Feature Processing Pipeline**:
```python
def process_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Vision transformer feature extraction
    with torch.no_grad():
        summary = self.radio_encoder.forward(image_tensor)
        # PCA dimensionality reduction for visualization
        feat_map_rgb = self.pca.transform(features_2d)
        
def add_vlm_object(self, object_name: str) -> bool:
    # Encode new object from VLM response
    object_embedding = self.encode_text_batch([object_name])
    self.dynamic_objects.append(object_name)
    self.dynamic_features = torch.cat([self.dynamic_features, object_embedding])
```

### RiskBufferManager Class
**State-Based Data Collection**:
```python
class BufferState(Enum):
    ACTIVE = "active"    # Data collection during breach
    FROZEN = "frozen"    # Post-breach, awaiting/assigned cause

def start_buffer(self, start_time: float):
    buffer = RiskBuffer(buffer_id=f"buffer_{timestamp}_{unique_id}", start_time=start_time)
    buffer.state = BufferState.ACTIVE
    self.active_buffers.append(buffer)

def freeze_active_buffers(self, end_time: float):
    for buffer in self.active_buffers:
        buffer.end_time = end_time
        buffer.state = BufferState.FROZEN
        self.frozen_buffers.append(buffer)
    self.active_buffers.clear()
```
**Two-State Buffer Lifecycle**: Implements ACTIVE and FROZEN states for comprehensive breach event data collection.

**ACTIVE State Operations**:
- Continuous data ingestion during breach events
- Real-time image, pose, and depth message collection

**FROZEN State Management**:
- Automatic state transition when breach ends

**Cause Association Logic**:
- Priority-based assignment to most recent buffers
- VLM answer correlation with temporal breach events

**Data Persistence Architecture**:
- Per-buffer subdirectories with organised data structure
- JSON metadata files with processing timestamps
- Image storage with original ROS message headers preserved
- Pose trajectory data in both JSON and NumPy formats
  
## ROS2 Communication Interface

### Subscribed Topics
```python
self.rgb_sub = self.create_subscription(Image, '/robot_1/sensors/front_stereo/right/image', self.rgb_callback, 1)
self.depth_sub = self.create_subscription(Image, '/robot_1/sensors/front_stereo/depth/depth_registered', self.depth_callback, 1)  
self.pose_sub = self.create_subscription(PoseStamped, '/robot_1/sensors/front_stereo/pose', self.pose_callback, 10)
self.camera_info_sub = self.create_subscription(CameraInfo, '/robot_1/sensors/front_stereo/right/camera_info', self.camera_info_callback, 1)
self.vlm_answer_sub = self.create_subscription(String, '/vlm_answer', self.vlm_answer_callback, 10)
```

### Published Topics  
```python
self.yolo_bbox_pub = self.create_publisher(Image, '/yolo_bbox_image', 1)
self.sam_mask_pub = self.create_publisher(Image, '/sam_mask_image', 1)
self.detection_cloud_pub = self.create_publisher(PointCloud2, '/detection_cloud', 10)
self.narration_pub = self.create_publisher(String, '/drift_narration', 10)
self.narration_text_pub = self.create_publisher(String, '/narration_text', 10)
self.naradio_image_pub = self.create_publisher(Image, '/naradio_image', 10)
self.narration_image_pub = self.create_publisher(Image, '/narration_image', 10)
```

## Configuration Parameters

### Launch Parameters
```python
self.declare_parameter('min_confidence', 0.05)
self.declare_parameter('min_detection_distance', 0.5)
self.declare_parameter('max_detection_distance', 2.0)
self.declare_parameter('sam_checkpoint', '/path/to/sam_vit_b_01ec64.pth')
self.declare_parameter('yolo_model', 'yolov8l-world.pt')
self.declare_parameter('radio_model_version', 'radio_v2.5-b')
self.declare_parameter('radio_input_resolution', 512)
self.declare_parameter('enable_combined_segmentation', True)
```

### Runtime Configuration
```yaml
# config/combined_segmentation_config.yaml
objects: ["floor", "ceiling", "wall", "person", "chair", "table"]
segmentation:
  apply_softmax: true
  enable_dbscan: true
  dbscan_eps: 0.3
  dbscan_min_samples: 5
processing:
  processing_rate: 10.0
  memory_fraction: 0.8
```

## Data Flow and State Management

### Breach Detection Pipeline
**Continuous Monitoring**: Real-time trajectory analysis with configurable threshold-based breach detection and immediate system state transitions.

**State Transition Logic**: Automatic detection of breach start and end events with comprehensive state management across all system components.

**Buffer Lifecycle Management**: Seamless transition from data collection to preservation states with automatic cause association capabilities.

### VLM Integration Workflow
```python
# In narration_display_node.py
def query_vlm(self, image, narration_text):
    client = OpenAI(api_key=self.api_key)
    image_base64 = self.encode_image_for_api(image)
    full_prompt = f"I am a drone, after 1s {narration_text} What object do you think in this scene is the cause, give one singular word as description"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]}]
    )
    return response.choices[0].message.content.strip()
```


### Runtime Execution
```bash
# Terminal 1: Main resilience system
ros2 run resilience main.py

# Terminal 2: VLM integration node
ros2 run resilience narration_display_node.py

# Terminal 3: Data source
ros2 bag play trajectory_data.bag
```


```bash
export OPENAI_API_KEY=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
```
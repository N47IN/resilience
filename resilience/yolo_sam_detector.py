#!/usr/bin/env python3
"""
YOLO Detector Module

Handles YOLO-World object detection.
"""

import torch
import cv2
import numpy as np
import os
import contextlib
from typing import List, Tuple, Optional
from ultralytics import YOLOWorld


class YOLODetector:
    """YOLO-World detector for object detection."""
    
    def __init__(self, 
                 yolo_model: str = 'yolov8l-world.pt',
                 yolo_imgsz: int = 480,
                 min_confidence: float = 0.05,
                 disable_yolo_printing: bool = True):
        """Initialize YOLO detector."""
        self.yolo_model = yolo_model
        self.yolo_imgsz = yolo_imgsz
        self.min_confidence = min_confidence
        self.disable_yolo_printing = disable_yolo_printing
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.detector_ready = False
        self.world_model = None
        self.current_classes = []
        
        self.init_models()
    
    def init_models(self):
        """Initialize YOLO-World model."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load YOLO-World model
            self.world_model = YOLOWorld(self.yolo_model)
            if torch.cuda.is_available():
                self.world_model.to(self.device)
            
            # Initialize with a dummy class to avoid tensor reshape errors
            self.world_model.set_classes(["dummy"])
            self.current_classes = ["dummy"]
            print(f"YOLO-World model initialized")

            # Verify model is ready
            if hasattr(self, 'world_model') and self.world_model is not None:
                self.detector_ready = True
                print("YOLO-World model loaded successfully")
            else:
                self.detector_ready = False
                print("Warning: YOLO model failed to initialize properly")

        except Exception as e:
            self.detector_ready = False
            print(f"Error initializing models: {e}")
            import traceback
            traceback.print_exc()
    
    def update_prompts(self, detection_prompts: List[str]) -> bool:
        """Update YOLO model with current detection prompts."""
        try:
            # Check if YOLO model is available
            if not hasattr(self, 'world_model') or self.world_model is None:
                print(f"YOLO model not ready")
                return False
            
            # Handle empty prompts case - set to dummy class to avoid tensor reshape error
            if not detection_prompts:
                self.world_model.set_classes(["dummy"])
                self.current_classes = ["dummy"]
                print(f"YOLO prompts set to dummy class (no detection until VLM provides answer)")
                return True
            
            # Set classes to current prompts
            self.world_model.set_classes(detection_prompts)
            self.current_classes = detection_prompts.copy()
            print(f"Updated YOLO prompts: {detection_prompts}")
            return True
            
        except Exception as e:
            print(f"Error updating YOLO prompts: {e}")
            return False
    
    def detect_objects(self, image: np.ndarray) -> Tuple[List, List, List]:
        """Use YOLO-World to detect objects based on prompt."""
        try:
            # Early exit if no detection prompts available
            if not self.current_classes:
                return [], [], []
            
            # Suppress YOLO default printing if enabled
            if self.disable_yolo_printing:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        with torch.no_grad():
                            detections = self.world_model.predict(
                                image,
                                imgsz=self.yolo_imgsz,
                                conf=self.min_confidence,
                                verbose=False
                            )
            else:
                with torch.no_grad():
                    detections = self.world_model.predict(
                        image,
                        imgsz=self.yolo_imgsz,
                        conf=self.min_confidence,
                        verbose=True
                    )
            
            # Early exit if no detections
            if len(detections) == 0 or detections[0].boxes is None:
                return [], [], []
            
            # Extract detection data more efficiently
            detection = detections[0]
            boxes = detection.boxes.xyxy
            confs = detection.boxes.conf
            
            # Convert to numpy arrays efficiently
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            else:
                boxes = np.asarray(boxes)
                
            if isinstance(confs, torch.Tensor):
                confs = confs.cpu().numpy()
            else:
                confs = np.asarray(confs)
            
            # Early exit if no valid boxes
            if len(boxes) == 0:
                return [], [], []
            
            # Process detections efficiently
            bboxes = []
            labels = []
            confidences = []
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.astype(int)
                bboxes.append([x1, y1, x2, y2])
                # Use the first prompt if multiple, or current detection prompt
                label = self.current_classes[0] if self.current_classes else "unknown"
                labels.append(label)
                confidences.append(float(conf))
            
            return bboxes, labels, confidences

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return [], [], []
    
    def filter_detections_by_distance(self, bboxes: List, depth_image: np.ndarray, 
                                    min_distance: float = 0.5, max_distance: float = 2.0,
                                    camera_intrinsics: Optional[List] = None) -> Tuple[List, List]:
        """Filter detections by distance using depth image."""
        if camera_intrinsics is None:
            return bboxes, list(range(len(bboxes)))
        
        # Early exit if no bboxes
        if not bboxes:
            return [], []
        
        filtered_bboxes = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check bounds first for efficiency
            if (center_y < 0 or center_y >= depth_image.shape[0] or 
                center_x < 0 or center_x >= depth_image.shape[1]):
                continue
                
            depth_value = depth_image[center_y, center_x]
            
            # Early exit for invalid depth
            if depth_value <= 0:
                continue
                
            # Convert depth to meters
            depth_m = depth_value / 1000.0 if depth_value > 100 else depth_value
            
            # Check distance threshold
            if min_distance <= depth_m <= max_distance:
                filtered_bboxes.append(bbox)
                valid_indices.append(i)
        
        return filtered_bboxes, valid_indices
    
    def is_ready(self) -> bool:
        """Check if detector is ready for use."""
        return self.detector_ready
    
    def get_current_classes(self) -> List[str]:
        """Get current detection classes."""
        return self.current_classes.copy() 
#!/usr/bin/env python3
"""
Historical Cause Analysis

This module provides functionality to analyze saved risk buffers and determine
the 3D locations of causes using YOLO detection on lagged images.
"""

import os
import json
import numpy as np
import cv2
import torch
import time
from typing import List, Dict, Optional, Tuple, Any
from ultralytics import YOLOWorld
import glob
from pathlib import Path
from datetime import datetime


class HistoricalCauseAnalyzer:
    """Analyzes saved buffers to find 3D locations of causes"""
    
    def __init__(self, yolo_model_path: str = None, save_directory: str = None):
        """
        Initialize the historical cause analyzer
        
        Args:
            yolo_model_path: Path to YOLO model for cause detection
            save_directory: Directory containing saved buffers
        """
        self.save_directory = save_directory or os.path.expanduser("~/risk_buffers")
        self.yolo_model_path = yolo_model_path
        
        # Initialize YOLO model for historical analysis
        self.historical_yolo = None
        self.init_yolo_model()
        
        print(f"[HistoricalAnalyzer] Initialized with save_dir: {self.save_directory}")
        print(f"[HistoricalAnalyzer] YOLO model: {'Loaded' if self.historical_yolo else 'Not available'}")
    
    def init_yolo_model(self):
        """Initialize YOLO model for historical analysis"""
        try:
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if self.yolo_model_path and os.path.exists(self.yolo_model_path):
                self.historical_yolo = YOLOWorld(self.yolo_model_path)
                print(f"[HistoricalAnalyzer] Loaded YOLOWorld model from {self.yolo_model_path}")
            else:
                # Try to use the same default as the main system
                try:
                    self.historical_yolo = YOLOWorld('yolov8l-world.pt')
                    print("[HistoricalAnalyzer] Loaded default YOLOWorld model")
                except Exception as e:
                    print(f"[HistoricalAnalyzer] Failed to load YOLOWorld model: {e}")
                    self.historical_yolo = None
                    return
            
            # Ensure model is on the correct device
            if self.historical_yolo is not None:
                try:
                    # Move model to device
                    self.historical_yolo.to(device)
                    print(f"[HistoricalAnalyzer] YOLOWorld model moved to {device}")
                   
                        
                except Exception as e:
                    print(f"[HistoricalAnalyzer] Error moving model to device: {e}")
                    self.historical_yolo = None
                    
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error initializing YOLOWorld model: {e}")
            self.historical_yolo = None
    
    def get_saved_buffers(self) -> List[str]:
        """Get list of saved buffer directories"""
        if not os.path.exists(self.save_directory):
            print(f"[HistoricalAnalyzer] Save directory does not exist: {self.save_directory}")
            return []
        
        buffer_dirs = []
        for item in os.listdir(self.save_directory):
            item_path = os.path.join(self.save_directory, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    buffer_dirs.append(item)
        
        print(f"[HistoricalAnalyzer] Found {len(buffer_dirs)} saved buffers")
        return buffer_dirs
    
    def load_buffer_metadata(self, buffer_id: str) -> Optional[Dict]:
        """Load metadata for a specific buffer"""
        metadata_path = os.path.join(self.save_directory, buffer_id, 'metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error loading metadata for {buffer_id}: {e}")
            return None
    
    def get_buffer_lagged_images(self, buffer_id: str) -> List[Tuple[str, np.ndarray]]:
        """Get all lagged images for a buffer"""
        lagged_dir = os.path.join(self.save_directory, buffer_id, 'lagged_images')
        if not os.path.exists(lagged_dir):
            print(f"[HistoricalAnalyzer] No lagged_images directory for {buffer_id}")
            return []
        
        images = []
        for img_file in sorted(os.listdir(lagged_dir)):
            if img_file.endswith('.png'):
                img_path = os.path.join(lagged_dir, img_file)
                try:
                    cv_image = cv2.imread(img_path)
                    if cv_image is not None:
                        images.append((img_file, cv_image))
                except Exception as e:
                    print(f"[HistoricalAnalyzer] Error loading image {img_path}: {e}")
        
        print(f"[HistoricalAnalyzer] Loaded {len(images)} lagged images for {buffer_id}")
        return images
    
    def detect_cause_in_image(self, image: np.ndarray, cause_prompt: str) -> Optional[Tuple[List[int], float]]:
        """
        Detect cause object in image using YOLOWorld
        
        Returns:
            Tuple of (bbox, confidence) or None if no detection
        """
        if self.historical_yolo is None:
            print("[HistoricalAnalyzer] YOLOWorld model not available for detection")
            return None
        
        try:
            # Convert BGR to RGB for YOLOWorld
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure model is on the correct device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.historical_yolo.to(device)
            except Exception as e:
                print(f"[HistoricalAnalyzer] Error ensuring model device: {e}")
            
            # Set the cause as the class to detect
            self.historical_yolo.set_classes([cause_prompt])
            
            # Run YOLOWorld detection with the cause as prompt
            with torch.no_grad():
                try:
                    results = self.historical_yolo.predict(
                        rgb_image,
                        conf=0.3,  # Lower confidence for historical analysis
                        verbose=False
                    )
                except RuntimeError as e:
                    if "same device" in str(e).lower() or "cuda:0 and cpu" in str(e).lower():
                        # Try to move model to CPU as fallback
                        try:
                            self.historical_yolo.to('cpu')
                            results = self.historical_yolo.predict(
                                rgb_image,
                                conf=0.3,
                                verbose=False
                            )
                        except Exception as cpu_e:
                            print(f"[HistoricalAnalyzer] Failed to process on CPU: {cpu_e}")
                            return None
                    else:
                        raise e
            
            if len(results) == 0 or results[0].boxes is None:
                return None
            
            # Get the best detection
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            if len(boxes) == 0:
                return None
            
            # Return the highest confidence detection
            best_idx = np.argmax(confidences)
            bbox = boxes[best_idx].astype(int).tolist()
            confidence = float(confidences[best_idx])
            
            return bbox, confidence
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error in cause detection: {e}")
            return None

    
    def analyze_single_buffer(self, buffer) -> Dict:
        """
        Analyze a single buffer object for cause location
        
        Args:
            buffer: Buffer object with buffer_id, cause, and other attributes
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not hasattr(buffer, 'buffer_id'):
                print("[HistoricalAnalyzer] Buffer object missing buffer_id")
                return {}
                
            buffer_id = buffer.buffer_id
            
            # Check if buffer has a cause
            if not hasattr(buffer, 'cause') or not buffer.cause:
                print(f"[HistoricalAnalyzer] Buffer {buffer_id} has no cause assigned")
                return {}
                
            cause = buffer.cause
            
            # Check if already has location
            if hasattr(buffer, 'cause_location') and buffer.cause_location:
                print(f"[HistoricalAnalyzer] Buffer {buffer_id} already has cause location: {buffer.cause_location}")
                return {
                    'buffer_id': buffer_id,
                    'cause': cause,
                    'location': buffer.cause_location,
                    'status': 'already_analyzed'
                }
            
            # Analyze based on buffer type
            if hasattr(buffer, 'is_frozen') and buffer.is_frozen:
                # Frozen buffer - analyze using saved images
                location = self.analyze_buffer_cause_location(buffer_id)
            else:
                # Active buffer - analyze using buffer object
                location = self.analyze_active_buffer_cause_location(buffer)
            
            if location:
                return {
                    'buffer_id': buffer_id,
                    'cause': cause,
                    'location': location,
                    'status': 'analyzed'
                }
            else:
                return {
                    'buffer_id': buffer_id,
                    'cause': cause,
                    'location': None,
                    'status': 'no_location_found'
                }
                
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error analyzing single buffer: {e}")
            return {
                'buffer_id': getattr(buffer, 'buffer_id', 'unknown'),
                'cause': getattr(buffer, 'cause', 'unknown'),
                'location': None,
                'status': 'error',
                'error': str(e)
            }
    
    def extract_3d_location(self, bbox: List[int], image: np.ndarray, 
                          depth_data: np.ndarray, pose_data: np.ndarray,
                          camera_intrinsics: List[float]) -> Optional[List[float]]:
        """
        Extract 3D location from 2D bbox using depth and pose data
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            image: RGB image
            depth_data: Depth image data
            pose_data: Robot pose data [timestamp, x, y, z, drift]
            camera_intrinsics: [fx, fy, cx, cy]
        
        Returns:
            3D location [x, y, z] in world coordinates or None
        """
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check bounds
            if (center_y < 0 or center_y >= depth_data.shape[0] or 
                center_x < 0 or center_x >= depth_data.shape[1]):
                return None
            
            # Get depth at center
            depth_value = depth_data[center_y, center_x]
            if depth_value <= 0:
                return None
            
            # Convert depth to meters
            depth_m = depth_value / 1000.0 if depth_value > 100 else depth_value
            
            # Convert to 3D camera coordinates
            fx, fy, cx, cy = camera_intrinsics
            camera_x = (center_x - cx) * depth_m / fx
            camera_y = (center_y - cy) * depth_m / fy
            camera_z = depth_m
    
            # Transform to world coordinates using robot pose
            # This is a simplified transformation - you may need more sophisticated pose handling
            robot_x, robot_y, robot_z = pose_data[1:4]  # Extract x, y, z from pose
            
            # Simple addition (assumes camera is at robot origin)
            world_x = robot_x + camera_x
            world_y = robot_y + camera_y
            world_z = robot_z + camera_z
            
            return [world_x, world_y, world_z]
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error extracting 3D location: {e}")
            return None
    
    def analyze_buffer_cause_location(self, buffer_id: str) -> Optional[List[float]]:
        """
        Analyze a single buffer to find the 3D location of its cause
        
        Returns:
            3D location [x, y, z] or None if not found
        """
        print(f"[HistoricalAnalyzer] Analyzing cause location for {buffer_id}")
        
        # Load metadata
        metadata = self.load_buffer_metadata(buffer_id)
        if not metadata:
            return None
        
        cause = metadata.get('cause')
        if not cause:
            print(f"[HistoricalAnalyzer] No cause found for {buffer_id}")
            return None
        
        # Check if already has location
        if metadata.get('cause_location'):
            print(f"[HistoricalAnalyzer] {buffer_id} already has cause location: {metadata['cause_location']}")
            return metadata['cause_location']
        
        # Get lagged images
        lagged_images = self.get_buffer_lagged_images(buffer_id)
        if not lagged_images:
            print(f"[HistoricalAnalyzer] No lagged images found for {buffer_id}")
            return None
        
        # Load pose and depth data
        poses_path = os.path.join(self.save_directory, buffer_id, 'poses.npy')
        poses_data = None
        if os.path.exists(poses_path):
            poses_data = np.load(poses_path)
        
        # For now, use default camera intrinsics (you may want to load from camera_info)
        camera_intrinsics = [525.0, 525.0, 319.5, 239.5]  # Default values
        
        # Process images to find cause
        for img_file, cv_image in lagged_images:
            print(f"[HistoricalAnalyzer] Processing {img_file} for cause '{cause}'")
            
            # Detect cause in image
            detection = self.detect_cause_in_image(cv_image, cause)
            if detection is None:
                continue
            
            bbox, confidence = detection
            print(f"[HistoricalAnalyzer] Found cause '{cause}' in {img_file} with confidence {confidence:.3f}")
            
            # Extract 3D location
            if poses_data is not None and len(poses_data) > 0:
                # Use the first pose for now (you may want to match timestamps)
                pose_data = poses_data[0]
                
                # Create dummy depth data (you may want to load actual depth data)
                depth_data = np.ones((cv_image.shape[0], cv_image.shape[1])) * 1000  # 1m depth
                
                location = self.extract_3d_location(bbox, cv_image, depth_data, pose_data, camera_intrinsics)
                if location:
                    print(f"[HistoricalAnalyzer] Extracted cause location: {location}")
                    
                    # Save image with bbox center overlaid
                    annotated_image_path = self.save_annotated_image(buffer_id, cv_image, bbox, cause, confidence)
                    
                    # Save cause location to separate file
                    location_file_path = self.save_cause_location_file(buffer_id, location, cause, confidence)
                    
                    # Update metadata with both location and image path
                    self.update_buffer_with_location_and_image(buffer_id, location, annotated_image_path, location_file_path)
                    
                    return location
            
            # Stop after first detection
            break
        
        print(f"[HistoricalAnalyzer] No cause location found for {buffer_id}")
        return None
    
    def save_annotated_image(self, buffer_id: str, image: np.ndarray, bbox: List[int], 
                           cause: str, confidence: float) -> Optional[str]:
        """
        Save image with bbox center overlaid
        
        Args:
            buffer_id: Buffer identifier
            image: Original image
            bbox: Bounding box [x1, y1, x2, y2]
            cause: Cause description
            confidence: Detection confidence
            
        Returns:
            Path to saved annotated image or None if failed
        """
        try:
            # Create annotated image copy
            annotated_image = image.copy()
            
            # Calculate bbox center
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(annotated_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, (center_x, center_y), 7, (255, 255, 255), 2)
            
            # Add text labels
            label = f"{cause} ({confidence:.2f})"
            cv2.putText(annotated_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add center coordinates
            center_label = f"Center: ({center_x}, {center_y})"
            cv2.putText(annotated_image, center_label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save annotated image
            buffer_dir = os.path.join(self.save_directory, buffer_id)
            annotated_image_path = os.path.join(buffer_dir, 'cause_detection_annotated.png')
            cv2.imwrite(annotated_image_path, annotated_image)
            
            print(f"[HistoricalAnalyzer] Saved annotated image: {annotated_image_path}")
            return annotated_image_path
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error saving annotated image: {e}")
            return None
    
    def save_cause_location_file(self, buffer_id: str, location: List[float], 
                               cause: str, confidence: float) -> Optional[str]:
        """
        Save cause location to a separate JSON file
        
        Args:
            buffer_id: Buffer identifier
            location: 3D location [x, y, z]
            cause: Cause description
            confidence: Detection confidence
            
        Returns:
            Path to saved location file or None if failed
        """
        try:
            location_data = {
                'cause': cause,
                'location_3d': location,
                'confidence': confidence,
                'timestamp': time.time(),
                'units': 'meters',
                'coordinate_system': 'world'
            }
            
            buffer_dir = os.path.join(self.save_directory, buffer_id)
            location_file_path = os.path.join(buffer_dir, 'cause_location.json')
            
            with open(location_file_path, 'w') as f:
                json.dump(location_data, f, indent=2)
            
            print(f"[HistoricalAnalyzer] Saved cause location file: {location_file_path}")
            return location_file_path
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error saving cause location file: {e}")
            return None
    
    def update_buffer_with_location_and_image(self, buffer_id: str, location: List[float], 
                                            annotated_image_path: Optional[str], 
                                            location_file_path: Optional[str]) -> bool:
        """Update buffer metadata with cause location and image path"""
        metadata_path = os.path.join(self.save_directory, buffer_id, 'metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['cause_location'] = location
            
            if annotated_image_path:
                metadata['cause_detection_image'] = os.path.basename(annotated_image_path)
            
            if location_file_path:
                metadata['cause_location_file'] = os.path.basename(location_file_path)
            
            # Add analysis timestamp
            metadata['cause_analysis_timestamp'] = time.time()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[HistoricalAnalyzer] Updated {buffer_id} with cause location and image paths")
            return True
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error updating {buffer_id}: {e}")
            return False
    
    def update_buffer_with_location(self, buffer_id: str, location: List[float]) -> bool:
        """Update buffer metadata with cause location"""
        metadata_path = os.path.join(self.save_directory, buffer_id, 'metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['cause_location'] = location
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[HistoricalAnalyzer] Updated {buffer_id} with cause location: {location}")
            return True
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error updating {buffer_id}: {e}")
            return False
    
    def analyze_all_buffers(self) -> Dict[str, List[float]]:
        """
        Analyze all saved buffers to find cause locations
        
        Returns:
            Dictionary mapping buffer_id to cause location
        """
        print("[HistoricalAnalyzer] Starting analysis of all buffers")
        
        buffer_dirs = self.get_saved_buffers()
        if not buffer_dirs:
            print("[HistoricalAnalyzer] No saved buffers to analyze")
            return {}
        
        results = {}
        total_buffers = len(buffer_dirs)
        
        for i, buffer_id in enumerate(buffer_dirs, 1):
            print(f"[HistoricalAnalyzer] Progress: {i}/{total_buffers} - Analyzing {buffer_id}")
            
            location = self.analyze_buffer_cause_location(buffer_id)
            if location:
                results[buffer_id] = location
                # Note: Metadata is already updated in analyze_buffer_cause_location method
        
        print(f"[HistoricalAnalyzer] Analysis complete. Found locations for {len(results)}/{total_buffers} saved buffers")
        return results

    def analyze_active_buffers(self, active_buffers: List) -> Dict[str, List[float]]:
        """
        Analyze active buffers in memory to find cause locations
        
        Args:
            active_buffers: List of active RiskBuffer objects
            
        Returns:
            Dictionary mapping buffer_id to cause location
        """
        print("[HistoricalAnalyzer] Starting analysis of active buffers")
        
        if not active_buffers:
            print("[HistoricalAnalyzer] No active buffers to analyze")
            return {}
        
        results = {}
        total_buffers = len(active_buffers)
        
        for i, buffer in enumerate(active_buffers, 1):
            print(f"[HistoricalAnalyzer] Progress: {i}/{total_buffers} - Analyzing active buffer {buffer.buffer_id}")
            
            # Only analyze buffers that have causes
            if not buffer.has_cause():
                print(f"[HistoricalAnalyzer] Skipping {buffer.buffer_id} - no cause assigned")
                continue
            
            location = self.analyze_active_buffer_cause_location(buffer)
            if location:
                results[buffer.buffer_id] = location
        
        print(f"[HistoricalAnalyzer] Active buffer analysis complete. Found locations for {len(results)}/{total_buffers} active buffers")
        return results

    def analyze_frozen_buffers_needing_causes(self, frozen_buffers: List) -> Dict[str, List[float]]:
        """
        Analyze frozen buffers that have assigned causes to find cause locations
        
        Args:
            frozen_buffers: List of frozen RiskBuffer objects that have causes assigned
            
        Returns:
            Dictionary mapping buffer_id to cause location
        """
        print("[HistoricalAnalyzer] Starting analysis of frozen buffers with causes")
        
        if not frozen_buffers:
            print("[HistoricalAnalyzer] No frozen buffers with causes to analyze")
            return {}
        
        results = {}
        total_buffers = len(frozen_buffers)
        
        for i, buffer in enumerate(frozen_buffers, 1):
            print(f"[HistoricalAnalyzer] Progress: {i}/{total_buffers} - Analyzing frozen buffer {buffer.buffer_id}")
            
            # Only analyze buffers that have assigned causes
            if buffer.has_cause():
                location = self.analyze_frozen_buffer_for_potential_causes(buffer)
                if location:
                    results[buffer.buffer_id] = location
            else:
                print(f"[HistoricalAnalyzer] Skipping {buffer.buffer_id} - no cause assigned")
        
        print(f"[HistoricalAnalyzer] Frozen buffers analysis complete. Found locations for {len(results)}/{total_buffers} buffers")
        return results

    def analyze_active_buffers_for_causes(self, active_buffers: List) -> Dict[str, List[float]]:
        """
        Analyze active buffers that have assigned causes to find cause locations
        
        Args:
            active_buffers: List of active RiskBuffer objects that have causes assigned
            
        Returns:
            Dictionary mapping buffer_id to cause location
        """
        print("[HistoricalAnalyzer] Starting analysis of active buffers with causes")
        
        if not active_buffers:
            print("[HistoricalAnalyzer] No active buffers with causes to analyze")
            return {}
        
        results = {}
        total_buffers = len(active_buffers)
        
        for i, buffer in enumerate(active_buffers, 1):
            print(f"[HistoricalAnalyzer] Progress: {i}/{total_buffers} - Analyzing active buffer {buffer.buffer_id}")
            
            # Only analyze buffers that have assigned causes
            if buffer.has_cause():
                location = self.analyze_active_buffer_for_potential_causes(buffer)
                if location:
                    results[buffer.buffer_id] = location
            else:
                print(f"[HistoricalAnalyzer] Skipping {buffer.buffer_id} - no cause assigned")
        
        print(f"[HistoricalAnalyzer] Active buffers analysis complete. Found locations for {len(results)}/{total_buffers} buffers")
        return results

    def analyze_active_buffer_cause_location(self, buffer) -> Optional[List[float]]:
        """
        Analyze an active buffer in memory to find the 3D location of its cause
        
        Args:
            buffer: RiskBuffer object in memory
            
        Returns:
            3D location [x, y, z] or None if not found
        """
        print(f"[HistoricalAnalyzer] Analyzing cause location for active buffer {buffer.buffer_id}")
        
        cause = buffer.cause
        if not cause:
            print(f"[HistoricalAnalyzer] No cause found for active buffer {buffer.buffer_id}")
            return None
        
        # Check if already has location
        if hasattr(buffer, 'cause_location') and buffer.cause_location:
            print(f"[HistoricalAnalyzer] {buffer.buffer_id} already has cause location: {buffer.cause_location}")
            return buffer.cause_location
        
        # Get lagged images from active buffer
        lagged_images = self.get_active_buffer_lagged_images(buffer)
        if not lagged_images:
            print(f"[HistoricalAnalyzer] No lagged images found for active buffer {buffer.buffer_id}")
            return None
        
        # Get pose data from active buffer
        poses_data = None
        if buffer.poses:
            # Convert poses to numpy array format
            poses_data = np.array([(t, p[0], p[1], p[2], d) for t, p, d in buffer.poses])
        
        # For now, use default camera intrinsics (you may want to load from camera_info)
        camera_intrinsics = [525.0, 525.0, 319.5, 239.5]  # Default values
        
        # Process images to find cause
        for img_file, cv_image in lagged_images:
            print(f"[HistoricalAnalyzer] Processing {img_file} for cause '{cause}' in active buffer")
            
            # Detect cause in image
            detection = self.detect_cause_in_image(cv_image, cause)
            if detection is None:
                continue
            
            bbox, confidence = detection
            print(f"[HistoricalAnalyzer] Found cause '{cause}' in {img_file} with confidence {confidence:.3f}")
            
            # Extract 3D location
            if poses_data is not None and len(poses_data) > 0:
                # Use the first pose for now (you may want to match timestamps)
                pose_data = poses_data[0]
                
                # Create dummy depth data (you may want to load actual depth data)
                depth_data = np.ones((cv_image.shape[0], cv_image.shape[1])) * 1000  # 1m depth
                
                location = self.extract_3d_location(bbox, cv_image, depth_data, pose_data, camera_intrinsics)
                if location:
                    print(f"[HistoricalAnalyzer] Extracted cause location for active buffer: {location}")
                    
                    # Save image with bbox center overlaid to active buffer directory
                    annotated_image_path = self.save_active_buffer_annotated_image(buffer, cv_image, bbox, cause, confidence)
                    
                    # Save cause location to active buffer directory
                    location_file_path = self.save_active_buffer_cause_location_file(buffer, location, cause, confidence)
                    
                    # Update active buffer with location
                    buffer.assign_cause_location(location)
                    
                    return location
            
            # Stop after first detection
            break
        
        print(f"[HistoricalAnalyzer] No cause location found for active buffer {buffer.buffer_id}")
        return None

    def analyze_frozen_buffer_for_potential_causes(self, buffer) -> Optional[List[float]]:
        """
        Analyze a frozen buffer that has an assigned cause to find cause location
        
        Args:
            buffer: Frozen RiskBuffer object that has a cause assigned
            
        Returns:
            3D location [x, y, z] or None if not found
        """
        print(f"[HistoricalAnalyzer] Analyzing potential causes for frozen buffer {buffer.buffer_id}")
        
        # Check if already has location
        if hasattr(buffer, 'cause_location') and buffer.cause_location:
            print(f"[HistoricalAnalyzer] {buffer.buffer_id} already has cause location: {buffer.cause_location}")
            return buffer.cause_location
        
        # Check if buffer has an assigned cause - if so, use it directly
        if buffer.has_cause():
            print(f"[HistoricalAnalyzer] Buffer {buffer.buffer_id} has assigned cause: {buffer.cause}")
            # Use the existing method for buffers with causes
            return self.analyze_active_buffer_cause_location(buffer)
        
        # If no cause is assigned, we cannot proceed with analysis
        print(f"[HistoricalAnalyzer] Buffer {buffer.buffer_id} has no assigned cause - cannot analyze for potential causes")
        return None

    def analyze_active_buffer_for_potential_causes(self, buffer) -> Optional[List[float]]:
        """
        Analyze an active buffer that has an assigned cause to find cause location
        
        Args:
            buffer: Active RiskBuffer object that has a cause assigned
            
        Returns:
            3D location [x, y, z] or None if not found
        """
        print(f"[HistoricalAnalyzer] Analyzing potential causes for active buffer {buffer.buffer_id}")
        
        # Check if already has location
        if hasattr(buffer, 'cause_location') and buffer.cause_location:
            print(f"[HistoricalAnalyzer] {buffer.buffer_id} already has cause location: {buffer.cause_location}")
            return buffer.cause_location
        
        # Check if buffer has an assigned cause - if so, use it directly
        if buffer.has_cause():
            print(f"[HistoricalAnalyzer] Buffer {buffer.buffer_id} has assigned cause: {buffer.cause}")
            # Use the existing method for buffers with causes
            return self.analyze_active_buffer_cause_location(buffer)
        
        # If no cause is assigned, we cannot proceed with analysis
        print(f"[HistoricalAnalyzer] Buffer {buffer.buffer_id} has no assigned cause - cannot analyze for potential causes")
        return None

    def get_active_buffer_lagged_images(self, buffer) -> List[Tuple[str, np.ndarray]]:
        """Get all lagged images for an active buffer in memory"""
        # For active buffers, we need to access the lagged images that were saved during the breach
        # These would be in the buffer's lagged_images directory if it was saved
        lagged_dir = os.path.join(self.save_directory, buffer.buffer_id, 'lagged_images')
        if not os.path.exists(lagged_dir):
            print(f"[HistoricalAnalyzer] No lagged_images directory for active buffer {buffer.buffer_id}")
            return []
        
        images = []
        for img_file in sorted(os.listdir(lagged_dir)):
            if img_file.endswith('.png'):
                img_path = os.path.join(lagged_dir, img_file)
                try:
                    cv_image = cv2.imread(img_path)
                    if cv_image is not None:
                        images.append((img_file, cv_image))
                except Exception as e:
                    print(f"[HistoricalAnalyzer] Error loading image {img_path}: {e}")
        
        print(f"[HistoricalAnalyzer] Loaded {len(images)} lagged images for active buffer {buffer.buffer_id}")
        return images

    def save_active_buffer_annotated_image(self, buffer, image: np.ndarray, bbox: List[int], 
                                         cause: str, confidence: float) -> Optional[str]:
        """
        Save annotated image for active buffer
        
        Args:
            buffer: Active RiskBuffer object
            image: Original image
            bbox: Bounding box [x1, y1, x2, y2]
            cause: Cause description
            confidence: Detection confidence
            
        Returns:
            Path to saved annotated image or None if failed
        """
        try:
            # Create annotated image copy
            annotated_image = image.copy()
            
            # Calculate bbox center
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw bbox
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(annotated_image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Add text
            text = f"{cause} ({confidence:.2f})"
            cv2.putText(annotated_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save to active buffer directory
            buffer_dir = os.path.join(self.save_directory, buffer.buffer_id)
            os.makedirs(buffer_dir, exist_ok=True)
            
            image_path = os.path.join(buffer_dir, 'cause_detection_annotated.png')
            cv2.imwrite(image_path, annotated_image)
            
            print(f"[HistoricalAnalyzer] Saved annotated image for active buffer: {image_path}")
            return image_path
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error saving annotated image for active buffer: {e}")
            return None

    def save_active_buffer_cause_location_file(self, buffer, location: List[float], 
                                             cause: str, confidence: float) -> Optional[str]:
        """
        Save cause location data for active buffer
        
        Args:
            buffer: Active RiskBuffer object
            location: 3D location [x, y, z]
            cause: Cause description
            confidence: Detection confidence
            
        Returns:
            Path to saved location file or None if failed
        """
        try:
            # Create buffer directory
            buffer_dir = os.path.join(self.save_directory, buffer.buffer_id)
            os.makedirs(buffer_dir, exist_ok=True)
            
            # Save cause location data
            location_data = {
                'buffer_id': buffer.buffer_id,
                'cause': cause,
                'location': location,
                'confidence': confidence,
                'analysis_timestamp': datetime.now().isoformat(),
                'buffer_state': 'active'
            }
            
            location_file_path = os.path.join(buffer_dir, 'cause_location.json')
            with open(location_file_path, 'w') as f:
                json.dump(location_data, f, indent=2)
            
            print(f"[HistoricalAnalyzer] Saved cause location for active buffer: {location_file_path}")
            return location_file_path
            
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error saving cause location for active buffer: {e}")
            return None
    
    def load_cause_location_data(self, buffer_id: str) -> Optional[Dict]:
        """
        Load cause location data from the separate JSON file
        
        Args:
            buffer_id: Buffer identifier
            
        Returns:
            Cause location data dictionary or None if not found
        """
        location_file_path = os.path.join(self.save_directory, buffer_id, 'cause_location.json')
        try:
            with open(location_file_path, 'r') as f:
                location_data = json.load(f)
            return location_data
        except Exception as e:
            print(f"[HistoricalAnalyzer] Error loading cause location data for {buffer_id}: {e}")
            return None
    
    def get_cause_detection_image_path(self, buffer_id: str) -> Optional[str]:
        """
        Get path to the cause detection annotated image
        
        Args:
            buffer_id: Buffer identifier
            
        Returns:
            Path to annotated image or None if not found
        """
        image_path = os.path.join(self.save_directory, buffer_id, 'cause_detection_annotated.png')
        if os.path.exists(image_path):
            return image_path
        return None


def main():
    """Main function for standalone historical analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze historical cause locations')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory containing saved buffers')
    parser.add_argument('--yolo-model', type=str, default=None, help='Path to YOLO model')
    
    args = parser.parse_args()
    
    analyzer = HistoricalCauseAnalyzer(
        yolo_model_path=args.yolo_model,
        save_directory=args.save_dir
    )
    
    results = analyzer.analyze_all_buffers()
    
    print("\n=== Historical Analysis Results ===")
    for buffer_id, location in results.items():
        print(f"{buffer_id}: {location}")
        
        # Show additional saved data
        location_data = analyzer.load_cause_location_data(buffer_id)
        if location_data:
            print(f"  Cause: {location_data.get('cause')}")
            print(f"  Confidence: {location_data.get('confidence', 'N/A')}")
            print(f"  Location file: cause_location.json")
        
        image_path = analyzer.get_cause_detection_image_path(buffer_id)
        if image_path:
            print(f"  Annotated image: cause_detection_annotated.png")
        
        print()


if __name__ == "__main__":
    main() 
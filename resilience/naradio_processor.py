#!/usr/bin/env python3
"""
NARadio Processor Module

Handles NARadio feature extraction and visualization with robust error handling
and GPU memory management to prevent crashes.

OPTIMIZATIONS IMPLEMENTED:
==========================
1. Feature Caching: Avoid redundant feature extraction for the same image
   - Image hash-based caching with TTL (5 seconds)
   - Automatic cache cleanup and size limits
   
2. Reduced Redundant Operations:
   - process_features_optimized() extracts features once, reuses for both visualization and similarity
   - compute_vlm_similarity_map_optimized() can reuse pre-computed features
   - Cached image preprocessing to avoid repeated tensor conversion
   
3. Memory Management Optimizations:
   - Reduced CUDA cache clearing frequency (only when needed)
   - Reduced memory cleanup interval (10s instead of continuous)
   - Better tensor cleanup and garbage collection
   
4. Processing Optimizations:
   - Optional visualization computation (skip when not needed)
   - Chunked similarity computation for memory efficiency
   - Early exit conditions to avoid unnecessary work
   
5. I/O Optimizations:
   - Skip saving raw similarity maps (only save colored visualizations)
   - Lightweight metadata saving
   - Skip saves for failed/slow computations
   
6. Device Consistency Optimizations:
   - Reduced device consistency check frequency (5s instead of continuous)
   - Better error handling to avoid repeated reinitializations

These optimizations should significantly reduce the processing time for combined
segmentation while maintaining the same functionality and output quality.
"""

import torch
import numpy as np
import cv2
import gc
import sys
import time
import warnings
import yaml
import os
import json
from typing import Optional, Tuple, Dict, List
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hashlib
import threading

# Suppress warnings
warnings.filterwarnings('ignore')


class NARadioProcessor:
    """NARadio feature extraction and processing with robust error handling and optimizations."""
    
    def __init__(self, 
                 radio_model_version: str = 'radio_v2.5-b',
                 radio_lang_model: str = 'siglip',
                 radio_input_resolution: int = 512,
                 enable_visualization: bool = True,
                 enable_combined_segmentation: bool = False,
                 segmentation_config_path: str = None):
        """Initialize NARadio processor."""
        self.radio_model_version = radio_model_version
        self.radio_lang_model = radio_lang_model
        self.radio_input_resolution = radio_input_resolution
        self.enable_visualization = enable_visualization
        self.enable_combined_segmentation = enable_combined_segmentation
        
        # Initialize NARadio components
        self.naradio_ready = False
        self.radio_encoder = None
        self.pca = PCA(n_components=3)  # For RGB visualization
        self.pca_fitted = False
        
        # NARadio availability
        self.NARADIO_AVAILABLE = False
        self.NARadioEncoder = None
        
        # Combined segmentation components
        self.segmentation_config = None
        self.word_features = None
        self.word_list = []
        self.colors = []
        self.segmentation_ready = False
        
        # Dynamic object management
        self.dynamic_objects = []  # Objects added from VLM answers
        self.dynamic_features = None  # Features for dynamic objects
        self.dynamic_colors = []  # Colors for dynamic objects
        
        # NEW: Enhanced embedding management for better similarity detection
        self.enhanced_embeddings = {}  # Dict: {vlm_answer: enhanced_embedding_np}
        self.enhanced_embedding_colors = {}  # Dict: {vlm_answer: color}
        
        # OPTIMIZATION: Feature caching to avoid redundant computation
        self.feature_cache = {}
        self.max_cache_size = 10
        self.cache_ttl = 5.0  # Cache TTL in seconds
        
        # OPTIMIZATION: Reusable tensors
        self._cached_tensor_image = None
        self._cached_image_hash = None
        self._cached_feat_map = None
        self._cached_aligned_feat_map = None
        
        # OPTIMIZATION: Memory cleanup tracking
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30.0  # SPEED OPTIMIZATION: Increased cleanup frequency
        
        # Initialize NARadio
        self.init_naradio()
        
        # Initialize combined segmentation if enabled
        if self.enable_combined_segmentation:
            self.init_combined_segmentation(segmentation_config_path)
        
        # Risk library for tracking VLM answers with enhanced embeddings
        self.risk_library = {}  # vlm_answer -> {enhanced_embedding, metadata, timestamp}
        self.risk_library_lock = threading.Lock()
    
    def _compute_image_hash(self, rgb_image: np.ndarray) -> str:
        """Compute hash for image caching."""
        return hashlib.md5(rgb_image.tobytes()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        for key, entry in self.feature_cache.items():
            if current_time - entry['timestamp'] > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.feature_cache[key]
        
        # Limit cache size
        if len(self.feature_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.feature_cache.items(), key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_items[:len(sorted_items) - self.max_cache_size]:
                del self.feature_cache[key]

    def init_naradio(self):
        """Initialize NARadio model for feature extraction with robust error handling."""
        try:
            # Add the radio_enhanced directory to the path
            radio_path = '/home/navin/ros2_ws/src/resilience/resilience'
            if radio_path not in sys.path:
                sys.path.append(radio_path)
                print(f"Added {radio_path} to Python path")
            
            # Try to import NARadio
            try:
                print("Attempting to import NARadioEncoder...")
                from naradio import NARadioEncoder
                print("✓ NARadioEncoder imported successfully")
                self.NARadioEncoder = NARadioEncoder
                self.NARADIO_AVAILABLE = True
            except ImportError as e:
                print(f"✗ ImportError: {e}")
                self.NARadioEncoder = None
                self.NARADIO_AVAILABLE = False
                print("Warning: NARadioEncoder not available")
                return

            if not self.NARADIO_AVAILABLE:
                print("NARadio not available, skipping initialization")
                self.naradio_ready = False
                return

            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            # Initialize radio_encoder to None first
            self.radio_encoder = None
            
            # Load NARadio model with proper error handling
            if self.NARadioEncoder is not None:
                try:
                    print(f"Initializing NARadio model with version={self.radio_model_version}, lang_model={self.radio_lang_model}, resolution={self.radio_input_resolution}")
                    self.radio_encoder = self.NARadioEncoder(
                        model_version=self.radio_model_version,
                        lang_model=self.radio_lang_model,
                        input_resolution=(self.radio_input_resolution, self.radio_input_resolution),
                        device=str(device)
                    )
                    print("✓ NARadio model created successfully")
                    
                    # Set NARadio to evaluation mode if it has eval method
                    if hasattr(self.radio_encoder, 'eval'):
                        self.radio_encoder.eval()
                        print("✓ Set model to evaluation mode")
                    
                    # Verify the model is working by checking its attributes
                    if hasattr(self.radio_encoder, 'encode_image_to_feat_map'):
                        print("✓ Model has encode_image_to_feat_map method")
                        # Test the model with a dummy input
                        if self.test_naradio_model():
                            print("✓ Model test passed")
                            self.naradio_ready = True
                        else:
                            print("✗ Model test failed")
                            self.radio_encoder = None
                            self.naradio_ready = False
                            return
                    else:
                        print("✗ Model missing encode_image_to_feat_map method")
                        self.radio_encoder = None
                        self.naradio_ready = False
                        return
                    
                except Exception as e:
                    print(f"Failed to load NARadio model: {e}")
                    import traceback
                    traceback.print_exc()
                    self.radio_encoder = None
                    self.naradio_ready = False
                    return
            else:
                print("NARadioEncoder not available")
                self.naradio_ready = False
                return
            
            print(f"NARadio initialization completed. Ready: {self.naradio_ready}")
            
        except Exception as e:
            self.naradio_ready = False
            print(f"Error initializing NARadio: {e}")
            import traceback
            traceback.print_exc()

    def test_naradio_model(self) -> bool:
        """Test if NARadio model is working properly."""
        try:
            if self.radio_encoder is None:
                return False
                
            # Create a dummy input tensor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dummy_input = torch.randn(1, 3, self.radio_input_resolution, self.radio_input_resolution).to(device)
            
            # Try to encode the dummy input
            with torch.no_grad():
                feat_map = self.radio_encoder.encode_image_to_feat_map(dummy_input)
                return True
                
        except Exception as e:
            print(f"Model test failed: {e}")
            return False

    def _preprocess_image_cached(self, rgb_image: np.ndarray) -> Tuple[torch.Tensor, str]:
        """
        Preprocess image with caching to avoid redundant operations.
        
        Returns:
            Tuple of (preprocessed_tensor, image_hash)
        """
        # Compute image hash for caching
        image_hash = self._compute_image_hash(rgb_image)
        
        # Check if we have cached tensor for this image
        if (self._cached_tensor_image is not None and 
            self._cached_image_hash == image_hash):
            return self._cached_tensor_image, image_hash
        
        # Preprocess image
        if len(rgb_image.shape) == 3:
            # Convert BGR to RGB if needed
            if rgb_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model input resolution
        resized_image = cv2.resize(rgb_image, (self.radio_input_resolution, self.radio_input_resolution))
        
        # Convert to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = torch.from_numpy(resized_image).float().to(device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to CHW and add batch dimension
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Cache the result
        self._cached_tensor_image = image_tensor
        self._cached_image_hash = image_hash
        
        return image_tensor, image_hash

    def extract_features_cached(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract NARadio features from RGB image with caching."""
        try:
            if not self.is_ready():
                return None
            
            # SPEED OPTIMIZATION: Compute hash only when cache is likely to be useful
            image_hash = None
            cached_result = None
            
            # Only use cache if we have reasonable cache size
            if len(self.feature_cache) > 0:
                image_hash = self._compute_image_hash(rgb_image)
                
                # Check cache first
                if image_hash in self.feature_cache:
                    cache_entry = self.feature_cache[image_hash]
                    if self._is_cache_valid(cache_entry):
                        return cache_entry['feat_map']
            
            # Ensure device consistency (less frequent check)
            if not self.ensure_device_consistency():
                self.reinitialize()
                if not self.is_ready():
                    return None
            
            # Preprocess image (with caching)
            image_tensor, computed_hash = self._preprocess_image_cached(rgb_image)
            
            # Extract features
            with torch.no_grad():
                feat_map = self.radio_encoder.encode_image_to_feat_map(image_tensor)
                
                # Convert to numpy
                if isinstance(feat_map, torch.Tensor):
                    feat_map_np = feat_map.cpu().numpy()
                else:
                    feat_map_np = feat_map
                
                # SPEED OPTIMIZATION: Cache only if hash was computed and cache isn't full
                if image_hash is None:
                    image_hash = computed_hash
                    
                if len(self.feature_cache) < self.max_cache_size:
                    self.feature_cache[image_hash] = {
                        'feat_map': feat_map_np,
                        'timestamp': time.time()
                    }
                
                # SPEED OPTIMIZATION: Cleanup only occasionally
                current_time = time.time()
                if (current_time - self.last_cleanup_time > self.cleanup_interval and 
                    len(self.feature_cache) > self.max_cache_size * 0.8):
                    self._cleanup_cache()
                    self.last_cleanup_time = current_time
                
                return feat_map_np
                
        except torch.cuda.OutOfMemoryError:
            self.handle_cuda_out_of_memory()
            return None
        except Exception as e:
            return None

    def extract_features(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Legacy method - redirects to cached version."""
        return self.extract_features_cached(rgb_image)

    def create_visualization(self, feat_map_np: np.ndarray) -> Optional[np.ndarray]:
        """Create visualization of NARadio features."""
        try:
            if not self.enable_visualization:
                return None
                
            if feat_map_np is None:
                return None
            
            # Reshape feature map if needed
            if len(feat_map_np.shape) == 4:  # Batch, channels, height, width
                feat_map_np = feat_map_np[0]  # Remove batch dimension
            
            # Flatten spatial dimensions
            h, w = feat_map_np.shape[1], feat_map_np.shape[2]
            feat_flat = feat_map_np.reshape(feat_map_np.shape[0], -1).T  # (H*W, C)
            
            # Fit PCA if not already fitted
            if not self.pca_fitted:
                self.pca.fit(feat_flat)
                self.pca_fitted = True
            
            # Transform features to 3D
            feat_3d = self.pca.transform(feat_flat)
            
            # Normalize to [0, 255] for visualization
            feat_3d = (feat_3d - feat_3d.min()) / (feat_3d.max() - feat_3d.min() + 1e-8)
            feat_3d = (feat_3d * 255).astype(np.uint8)
            
            # Reshape back to image dimensions
            feat_vis = feat_3d.reshape(h, w, 3)
            
            # Resize to original image size for better visualization
            feat_vis = cv2.resize(feat_vis, (512, 512))
            
            return feat_vis
            
        except Exception as e:
            print(f"Error creating NARadio visualization: {e}")
            return None

    def process_features_optimized(self, rgb_image: np.ndarray, 
                                 need_visualization: bool = True,
                                 reuse_features: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        OPTIMIZED: Process NARadio features with optional visualization and feature reuse.
        
        Args:
            rgb_image: Input RGB image
            need_visualization: Whether to compute visualization (expensive)
            reuse_features: Whether to use cached features if available
        """
        try:
            # Extract features (with caching if enabled)
            if reuse_features:
                feat_map_np = self.extract_features_cached(rgb_image)
            else:
                feat_map_np = self.extract_features(rgb_image)
            
            # Create visualization only if needed
            naradio_vis = None
            if feat_map_np is not None and need_visualization:
                naradio_vis = self.create_visualization(feat_map_np)
            
            return feat_map_np, naradio_vis
            
        except Exception as e:
            print(f"Error processing NARadio features: {e}")
            return None, None

    def process_features(self, rgb_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Legacy method - redirects to optimized version."""
        return self.process_features_optimized(rgb_image, need_visualization=True, reuse_features=True)

    def cleanup_memory(self):
        """OPTIMIZED: Periodic cleanup of radio model memory with reduced frequency."""
        try:
            current_time = time.time()
            # SPEED OPTIMIZATION: Increase cleanup interval to reduce overhead
            if current_time - self.last_cleanup_time < self.cleanup_interval:
                return
                
            if torch.cuda.is_available():
                # SPEED OPTIMIZATION: Only cleanup if memory usage is high
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                # Only cleanup if using more than 70% of GPU memory
                if allocated > total * 0.7:
                    torch.cuda.empty_cache()
                
            # SPEED OPTIMIZATION: Reduce cache cleaning frequency
            if current_time - self.last_cleanup_time > self.cleanup_interval * 2:
                self._cleanup_cache()
                
            self.last_cleanup_time = current_time
                
        except Exception as e:
            pass  # Silently handle cleanup errors

    def handle_cuda_out_of_memory(self):
        """Handle CUDA out of memory errors."""
        try:
            if torch.cuda.is_available():
                # Clear all CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Clear our feature cache
                self.feature_cache.clear()
                self._cached_tensor_image = None
                self._cached_image_hash = None
                self._cached_feat_map = None
                self._cached_aligned_feat_map = None
                
                # Get memory info
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                print(f"CUDA OOM handled - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
                
                # If still high memory usage, try to reset the radio model
                if allocated > total * 0.8:  # If more than 80% of GPU memory is used
                    if self.radio_encoder is not None:
                        print("High memory usage detected, resetting radio model...")
                        del self.radio_encoder
                        self.radio_encoder = None
                        self.naradio_ready = False
                        torch.cuda.empty_cache()
                        print("Radio model reset due to high memory usage")
                
        except Exception as e:
            print(f"Error handling CUDA out of memory: {e}")

    def ensure_device_consistency(self) -> bool:
        """Ensure radio model is on the correct device."""
        try:
            if self.radio_encoder is None:
                return False
                
            # Check if model is valid
            return hasattr(self.radio_encoder, 'model')
            
        except Exception as e:
            print(f"Device consistency check failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if NARadio is ready for processing."""
        return self.naradio_ready and self.radio_encoder is not None

    def reinitialize(self) -> bool:
        """Reinitialize NARadio model."""
        try:
            print("Reinitializing NARadio model...")
            
            # Clean up existing model
            if self.radio_encoder is not None:
                del self.radio_encoder
                self.radio_encoder = None
            
            # Clear all caches
            self.feature_cache.clear()
            self._cached_tensor_image = None
            self._cached_image_hash = None
            
            # Clear memory
            self.cleanup_memory()
            
            # Reinitialize
            self.init_naradio()
            
            # Reinitialize combined segmentation if enabled
            if self.enable_combined_segmentation:
                self.init_combined_segmentation()
            
            success = self.is_ready()
            print(f"NARadio reinitialization {'successful' if success else 'failed'}")
            return success
            
        except Exception as e:
            print(f"Error reinitializing NARadio: {e}")
            return False
    
    def init_combined_segmentation(self, config_path: str = None):
        """Initialize combined segmentation with configuration."""
        try:
            print("Initializing Combined Segmentation...")
            
            # Load configuration
            self.segmentation_config = self.load_segmentation_config(config_path)
            if not self.segmentation_config:
                print("Failed to load segmentation configuration")
                return
            
            # Encode word embeddings
            self.encode_word_embeddings()
            
            # Set up colors
            self.setup_colors()
            
            self.segmentation_ready = True
            print("✓ Combined Segmentation initialized successfully")
            
        except Exception as e:
            print(f"Error initializing combined segmentation: {e}")
            import traceback
            traceback.print_exc()
            self.segmentation_ready = False
    
    def load_segmentation_config(self, config_path: str = None) -> Optional[Dict]:
        """Load segmentation configuration from YAML file."""
        if config_path is None:
            # Use default config path
            try:
                from ament_index_python.packages import get_package_share_directory
                package_dir = get_package_share_directory('resilience')
                config_path = os.path.join(package_dir, 'config', 'main_config.yaml')
            except ImportError:
                # Fallback for non-ROS environments
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'main_config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded segmentation config from: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading segmentation config from {config_path}: {e}")
            return self.get_default_segmentation_config()
    
    def get_default_segmentation_config(self) -> Dict:
        """Get default segmentation configuration."""
        return {
            'objects': ['floor', 'ceiling', 'wall', 'person', 'chair', 'table'],
            'segmentation': {
                'apply_softmax': True,
                'normalize_features': True,
                'prefer_enhanced_embeddings': True,
                'enhanced_similarity_threshold': 0.5,
                'enable_dbscan': True,
                'dbscan_eps': 0.3,
                'dbscan_min_samples': 5,
                'colors': [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
            }
        }
    
    def encode_word_embeddings(self):
        """Encode word embeddings for all objects."""
        try:
            if not self.naradio_ready or not self.radio_encoder:
                print("NARadio not ready for word encoding")
                return
            
            self.word_list = self.segmentation_config['objects']
            batch_size = 4  # Process words in batches
            
            print(f"Encoding {len(self.word_list)} base objects into embeddings...")
            
            # Process words in batches
            word_features_list = []
            
            for i in range(0, len(self.word_list), batch_size):
                batch_words = self.word_list[i:i+batch_size]
                
                with torch.no_grad():
                    batch_features = self.radio_encoder.encode_labels(batch_words)
                    word_features_list.append(batch_features)
            
            # Concatenate all batch features
            self.word_features = torch.cat(word_features_list, dim=0)
            
            print(f"✓ Encoded {len(self.word_list)} base objects, features shape: {self.word_features.shape}")
            
        except Exception as e:
            print(f"Error encoding word embeddings: {e}")
            import traceback
            traceback.print_exc()
    
    def add_vlm_object(self, vlm_answer: str) -> bool:
        """
        Add a VLM answer as a new object for segmentation.
        
        Args:
            vlm_answer: The VLM answer to add as a segmentation object
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            if not self.naradio_ready or not self.radio_encoder:
                print(f"Cannot add VLM object '{vlm_answer}': NARadio not ready")
                return False
            
            # Check if object already exists
            if vlm_answer in self.word_list or vlm_answer in self.dynamic_objects:
                print(f"VLM object '{vlm_answer}' already exists in segmentation list")
                return True
            
            print(f"Adding VLM object '{vlm_answer}' to segmentation list...")
            
            # Encode the new object
            with torch.no_grad():
                new_features = self.radio_encoder.encode_labels([vlm_answer])
            
            # Ensure device consistency with base features
            if self.word_features is not None:
                target_device = self.word_features.device
                if new_features.device != target_device:
                    print(f"Moving VLM features from {new_features.device} to {target_device}")
                    new_features = new_features.to(target_device)
            
            # Add to dynamic objects
            self.dynamic_objects.append(vlm_answer)
            
            # Update dynamic features
            if self.dynamic_features is None:
                self.dynamic_features = new_features
            else:
                # Ensure device consistency with existing dynamic features
                if self.dynamic_features.device != new_features.device:
                    print(f"Moving new features to match dynamic features device: {self.dynamic_features.device}")
                    new_features = new_features.to(self.dynamic_features.device)
                self.dynamic_features = torch.cat([self.dynamic_features, new_features], dim=0)
            
            # Add color for the new object (white for new VLM objects)
            new_color = [255, 255, 255]  # White color for new VLM objects
            self.dynamic_colors.append(new_color)
            
            print(f"✓ Added VLM object '{vlm_answer}' with white color (total dynamic objects: {len(self.dynamic_objects)})")
            print(f"  VLM features shape: {new_features.shape}, device: {new_features.device}")
            return True
            
        except Exception as e:
            print(f"Error adding VLM object '{vlm_answer}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_enhanced_embedding(self, vlm_answer: str, enhanced_embedding: np.ndarray) -> bool:
        """
        Add an enhanced embedding for better similarity detection.
        
        Args:
            vlm_answer: The VLM answer associated with this embedding
            enhanced_embedding: The enhanced embedding computed from weighted spatial features
            
        Returns:
            bool: True if successfully added
        """
        try:
            if not isinstance(enhanced_embedding, np.ndarray):
                print(f"Enhanced embedding must be numpy array, got {type(enhanced_embedding)}")
                return False
            
            # Store the enhanced embedding
            self.enhanced_embeddings[vlm_answer] = enhanced_embedding.copy()
            
            # Assign a distinctive color for enhanced embeddings (bright cyan for visibility)
            self.enhanced_embedding_colors[vlm_answer] = [0, 255, 255]  # Bright cyan
            
            print(f"✓ Added enhanced embedding for '{vlm_answer}' (shape: {enhanced_embedding.shape})")
            print(f"  Total enhanced embeddings: {len(self.enhanced_embeddings)}")
            return True
            
        except Exception as e:
            print(f"Error adding enhanced embedding for '{vlm_answer}': {e}")
            return False
    
    def has_enhanced_embedding(self, vlm_answer: str) -> bool:
        """Check if we have an enhanced embedding for this VLM answer."""
        return vlm_answer in self.enhanced_embeddings
    
    def get_enhanced_embedding(self, vlm_answer: str) -> Optional[np.ndarray]:
        """Get enhanced embedding for a VLM answer."""
        return self.enhanced_embeddings.get(vlm_answer)
    
    def get_all_enhanced_objects(self) -> List[str]:
        """Get list of VLM answers that have enhanced embeddings."""
        return list(self.enhanced_embeddings.keys())
    
    def remove_enhanced_embedding(self, vlm_answer: str) -> bool:
        """Remove enhanced embedding for a VLM answer."""
        if vlm_answer in self.enhanced_embeddings:
            del self.enhanced_embeddings[vlm_answer]
            if vlm_answer in self.enhanced_embedding_colors:
                del self.enhanced_embedding_colors[vlm_answer]
            print(f"Removed enhanced embedding for '{vlm_answer}'")
            return True
        return False
    
    def generate_color_for_object(self, object_index: int) -> List[int]:
        """Generate a color for an object based on its index."""
        # Use a predefined color palette
        color_palette = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 128, 0],    # Orange
            [128, 0, 255],    # Purple
            [128, 128, 0],    # Olive
            [0, 128, 128],    # Teal
            [128, 0, 128],    # Maroon
            [255, 165, 0],    # Orange Red
            [75, 0, 130],     # Indigo
            [240, 230, 140],  # Khaki
            [255, 20, 147],   # Deep Pink
        ]
        
        # Cycle through the palette
        return color_palette[object_index % len(color_palette)]
    
    def get_all_objects(self) -> List[str]:
        """Get all objects (base + dynamic) for segmentation."""
        return self.word_list + self.dynamic_objects
    
    def get_all_features(self):
        """Get all features (base + dynamic) for segmentation."""
        if self.dynamic_features is None:
            return self.word_features
        else:
            # Ensure device consistency before concatenation
            if self.word_features.device != self.dynamic_features.device:
                print(f"Device mismatch in get_all_features: base={self.word_features.device}, dynamic={self.dynamic_features.device}")
                # Move dynamic features to base device
                self.dynamic_features = self.dynamic_features.to(self.word_features.device)
                print(f"Moved dynamic features to {self.word_features.device}")
            
            return torch.cat([self.word_features, self.dynamic_features], dim=0)
    
    def get_all_colors(self) -> List[List[int]]:
        """Get all colors (base + dynamic) for segmentation."""
        return self.colors + self.dynamic_colors
    
    def save_dynamic_objects_to_buffer(self, buffer_dir: str, vlm_answer: str):
        """Save dynamic object embeddings to buffer directory."""
        try:
            if not self.naradio_ready or not self.radio_encoder:
                return
            
            # Create embeddings directory
            embeddings_dir = os.path.join(buffer_dir, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Encode the VLM answer
            with torch.no_grad():
                vlm_features = self.radio_encoder.encode_labels([vlm_answer])
            
            # Save the embedding
            embedding_filename = f"vlm_object_{vlm_answer.replace(' ', '_').replace('/', '_')}.pt"
            embedding_path = os.path.join(embeddings_dir, embedding_filename)
            torch.save(vlm_features.cpu(), embedding_path)
            
            # Save metadata
            metadata = {
                'vlm_answer': vlm_answer,
                'embedding_filename': embedding_filename,
                'embedding_shape': list(vlm_features.shape),
                'timestamp': float(time.time()),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'object_type': 'vlm_dynamic',
                'total_dynamic_objects': len(self.dynamic_objects)
            }
            
            metadata_filename = f"vlm_object_{vlm_answer.replace(' ', '_').replace('/', '_')}_metadata.json"
            metadata_path = os.path.join(embeddings_dir, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved VLM object embedding: {embedding_path}")
            
        except Exception as e:
            print(f"Error saving VLM object embedding: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_colors(self):
        """Set up color mapping for visualization."""
        self.colors = self.segmentation_config['segmentation']['colors']
        
        # Keep original colors for base objects, use white for additional objects
        original_color_count = len(self.colors)
        original_object_count = len(self.word_list)
        
        # Trim colors to match the number of base objects
        if original_color_count > original_object_count:
            self.colors = self.colors[:original_object_count]
            print(f"  Trimmed colors from {original_color_count} to {len(self.colors)} to match {original_object_count} objects")
        
        # If we have more objects than original colors, add white colors for the extra objects
        if original_object_count > len(self.colors):
            extra_colors_needed = original_object_count - len(self.colors)
            for _ in range(extra_colors_needed):
                self.colors.append([255, 255, 255])  # White for additional objects
        
        print(f"✓ Set up {len(self.colors)} colors for {len(self.word_list)} objects")
        print(f"  Final color count: {len(self.colors)}")
    
    def process_combined_segmentation(self, rgb_image: np.ndarray) -> Optional[Dict]:
        """
        Process combined segmentation using NARadio embeddings.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            
        Returns:
            dict: Contains original_mask, refined_mask, metadata, and processing_info
        """
        if not self.segmentation_ready or not self.naradio_ready:
            print("Combined segmentation not ready")
            return None
        
        try:
            # Start timing
            start_time = time.time()
            
            # Get current object list for debugging
            all_objects = self.get_all_objects()
            all_colors = self.get_all_colors()
            # if self.dynamic_objects:
            #     print(f"[CombinedSeg] Processing with {len(all_objects)} objects (including {len(self.dynamic_objects)} VLM objects)")
            #     print(f"[CombinedSeg] VLM objects: {self.dynamic_objects}")
            #     print(f"[CombinedSeg] VLM colors: {self.dynamic_colors}")
            
            # Process with NARadio to get feature map
            feat_map, naradio_vis = self.process_features(rgb_image)
            
            if feat_map is None:
                print("Failed to get feature map from NARadio")
                return None
            
            # Compute similarity maps
            similarity_maps = self.compute_similarity_maps(feat_map)
            
            if similarity_maps is None:
                print("Failed to compute similarity maps")
                return None
            
            # Create segmentation masks
            original_mask = self.create_colored_mask(similarity_maps)
            
            # Apply DBSCAN refinement if enabled
            if self.segmentation_config['segmentation']['enable_dbscan']:
                refined_mask = self.apply_dbscan_refinement(similarity_maps, original_mask)
            else:
                refined_mask = original_mask.copy()
            
            # Prepare metadata
            metadata = self.create_segmentation_metadata(similarity_maps, start_time)
            
            # Prepare processing info
            all_objects = self.get_all_objects()
            
            # Ensure all values are JSON serializable
            feat_map_shape = None
            if feat_map is not None:
                if isinstance(feat_map, np.ndarray):
                    feat_map_shape = list(feat_map.shape)
                elif isinstance(feat_map, torch.Tensor):
                    feat_map_shape = list(feat_map.shape)
                else:
                    feat_map_shape = str(type(feat_map))
            
            processing_info = {
                'processing_time': float(time.time() - start_time),
                'num_base_objects': len(self.word_list),
                'num_dynamic_objects': len(self.dynamic_objects),
                'num_total_objects': len(all_objects),
                'image_shape': list(rgb_image.shape),
                'feature_map_shape': feat_map_shape,
                'naradio_visualization': naradio_vis is not None
            }
            
            result = {
                'original_mask': original_mask,
                'refined_mask': refined_mask,
                'metadata': metadata,
                'processing_info': processing_info
            }
            
            return result
            
        except Exception as e:
            print(f"Error in combined segmentation processing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_similarity_maps(self, feat_map: np.ndarray) -> Optional[np.ndarray]:
        """Compute similarity maps between word embeddings and image features."""
        try:
            with torch.no_grad():
                # Convert feat_map to tensor if it's a numpy array
                if isinstance(feat_map, np.ndarray):
                    feat_map_tensor = torch.from_numpy(feat_map).to(self.word_features.device)
                else:
                    feat_map_tensor = feat_map
                
                # Get language-aligned feature map
                lang_aligned_feat_map = self.radio_encoder.align_spatial_features_with_language(feat_map_tensor)
                
                if lang_aligned_feat_map is None:
                    return None
                
                # Get all features (base + dynamic)
                all_features = self.get_all_features()
                
                # Get dimensions
                B, C, H, W = lang_aligned_feat_map.shape
                num_words = all_features.shape[0]
                
                # Debug: Check if VLM objects are included in similarity computation
                # if self.dynamic_objects:
                #     print(f"[Similarity] Computing similarity for {num_words} objects")
                #     print(f"[Similarity] VLM objects included: {self.dynamic_objects}")
                #     print(f"[Similarity] Feature tensor shape: {all_features.shape}")
                
                # Reshape spatial features: (B, C, H*W) -> (H*W, C)
                spatial_features = lang_aligned_feat_map.squeeze(0).permute(1, 2, 0).reshape(H * W, C)
                
                # Normalize features for cosine similarity
                if self.segmentation_config['segmentation']['normalize_features']:
                    word_features_norm = all_features.float() / (all_features.float().norm(dim=-1, keepdim=True) + 1e-8)
                    spatial_features_norm = spatial_features.float() / (spatial_features.float().norm(dim=-1, keepdim=True) + 1e-8)
                else:
                    word_features_norm = all_features.float()
                    spatial_features_norm = spatial_features.float()
                
                # Compute similarity: (num_words, H*W)
                similarity = torch.mm(word_features_norm, spatial_features_norm.t())
                
                # Reshape back to spatial dimensions: (num_words, H, W)
                similarity_maps = similarity.view(num_words, H, W)
                
                # Apply softmax across words if enabled
                if self.segmentation_config['segmentation']['apply_softmax']:
                    similarity_reshaped = similarity_maps.view(num_words, H * W)
                    softmax_similarity = torch.softmax(similarity_reshaped, dim=0)
                    similarity_maps = softmax_similarity.view(num_words, H, W)
                
                return similarity_maps
                
        except Exception as e:
            print(f"Error computing similarity maps: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_colored_mask(self, similarity_maps) -> Optional[np.ndarray]:
        """Create colored segmentation mask from similarity maps."""
        try:
            # Ensure similarity_maps is a tensor
            if isinstance(similarity_maps, np.ndarray):
                similarity_maps = torch.from_numpy(similarity_maps)
            
            # Get all objects and colors
            all_objects = self.get_all_objects()
            all_colors = self.get_all_colors()
            
            # Debug: Check if VLM objects are present
            # SPEED OPTIMIZATION: Disable verbose debugging for performance
            # if self.dynamic_objects:
            #     print(f"[ColoredMask] Creating mask with {len(all_objects)} objects")
            #     print(f"[ColoredMask] VLM objects indices: {[all_objects.index(obj) for obj in self.dynamic_objects if obj in all_objects]}")
            #     print(f"[ColoredMask] VLM colors: {self.dynamic_colors}")
            
            # Find best matching word for each pixel
            best_word_indices = torch.argmax(similarity_maps, dim=0)
            best_word_indices_np = best_word_indices.cpu().numpy()
            
            # Get dimensions
            H, W = best_word_indices_np.shape
            
            # Create RGB mask
            colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
            
            # SPEED OPTIMIZATION: Skip pixel counting for debugging
            for i in range(len(all_objects)):
                mask = (best_word_indices_np == i)
                colored_mask[mask] = all_colors[i]
            
            # SPEED OPTIMIZATION: Disable verbose debugging for VLM objects
            # if self.dynamic_objects:
            #     print(f"[ColoredMask] All colors length: {len(all_colors)}")
            #     print(f"[ColoredMask] All objects length: {len(all_objects)}")
            #     for vlm_obj in self.dynamic_objects:
            #         if vlm_obj in pixel_counts:
            #             obj_index = all_objects.index(vlm_obj)
            #             obj_similarity = similarity_maps[obj_index]
            #             max_similarity = torch.max(obj_similarity).item()
            #             mean_similarity = torch.mean(obj_similarity).item()
            #             actual_color = all_colors[obj_index] if obj_index < len(all_colors) else "OUT_OF_RANGE"
            #             print(f"[ColoredMask] VLM object '{vlm_obj}' (index {obj_index}): {pixel_counts[vlm_obj]} pixels, max_sim={max_similarity:.3f}, mean_sim={mean_similarity:.3f} (color: {actual_color})")
            
            return colored_mask
            
        except Exception as e:
            print(f"Error creating colored mask: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def apply_dbscan_refinement(self, similarity_maps, original_mask: np.ndarray) -> np.ndarray:
        """Apply DBSCAN clustering to refine segmentation."""
        try:
            # Ensure similarity_maps is a tensor
            if isinstance(similarity_maps, np.ndarray):
                similarity_maps = torch.from_numpy(similarity_maps)
            
            # Get dimensions
            num_words, H, W = similarity_maps.shape
            
            # Extract features for clustering
            spatial_positions = np.mgrid[0:H, 0:W].transpose(1, 2, 0)
            similarity_scores = similarity_maps.cpu().numpy()
            
            # Normalize spatial positions
            spatial_positions_norm = spatial_positions.astype(np.float32)
            spatial_positions_norm[:, :, 0] = spatial_positions_norm[:, :, 0] / H
            spatial_positions_norm[:, :, 1] = spatial_positions_norm[:, :, 1] / W
            
            # Create feature matrix for clustering
            features_for_clustering = np.zeros((H * W, 4))
            
            for i in range(H):
                for j in range(W):
                    pixel_idx = i * W + j
                    features_for_clustering[pixel_idx, 0] = spatial_positions_norm[i, j, 0]
                    features_for_clustering[pixel_idx, 1] = spatial_positions_norm[i, j, 1]
                    features_for_clustering[pixel_idx, 2] = np.max(similarity_scores[:, i, j])
                    features_for_clustering[pixel_idx, 3] = np.std(similarity_scores[:, i, j])
            
            # Apply DBSCAN
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_for_clustering)
            
            dbscan = DBSCAN(
                eps=self.segmentation_config['segmentation']['dbscan_eps'],
                min_samples=self.segmentation_config['segmentation']['dbscan_min_samples']
            )
            cluster_labels = dbscan.fit_predict(features_scaled)
            
            # Reshape cluster labels back to spatial dimensions
            cluster_mask = cluster_labels.reshape(H, W)
            
            # Refine segmentation based on clusters
            refined_mask = np.zeros((H, W, 3), dtype=np.uint8)
            best_word_indices = torch.argmax(similarity_maps, dim=0).cpu().numpy()
            
            # Get all colors
            all_colors = self.get_all_colors()
            
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Noise points
                    noise_mask = (cluster_mask == cluster_id)
                    refined_mask[noise_mask] = original_mask[noise_mask]
                    continue
                
                # Get pixels in this cluster
                cluster_pixels = (cluster_mask == cluster_id)
                
                if np.sum(cluster_pixels) == 0:
                    continue
                
                # Find the most common word assignment in this cluster
                cluster_word_assignments = best_word_indices[cluster_pixels]
                unique_assignments, counts = np.unique(cluster_word_assignments, return_counts=True)
                most_common_word = unique_assignments[np.argmax(counts)]
                
                # Assign the most common word to all pixels in this cluster
                refined_mask[cluster_pixels] = all_colors[most_common_word]
            
            return refined_mask
            
        except Exception as e:
            print(f"Error applying DBSCAN refinement: {e}")
            import traceback
            traceback.print_exc()
            return original_mask
    
    def create_segmentation_metadata(self, similarity_maps: np.ndarray, start_time: float) -> Dict:
        """Create metadata for the segmentation results."""
        try:
            all_objects = self.get_all_objects()
            
            # Ensure similarity_maps_shape is JSON serializable
            similarity_shape = None
            if similarity_maps is not None:
                if isinstance(similarity_maps, np.ndarray):
                    similarity_shape = list(similarity_maps.shape)
                elif isinstance(similarity_maps, torch.Tensor):
                    similarity_shape = list(similarity_maps.shape)
                else:
                    similarity_shape = str(type(similarity_maps))
            
            metadata = {
                'timestamp': float(start_time),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
                'base_objects': self.word_list,
                'dynamic_objects': self.dynamic_objects,
                'all_objects': all_objects,
                'num_base_objects': len(self.word_list),
                'num_dynamic_objects': len(self.dynamic_objects),
                'num_total_objects': len(all_objects),
                'similarity_maps_shape': similarity_shape,
                'config': {
                    'apply_softmax': self.segmentation_config['segmentation']['apply_softmax'],
                    'enable_dbscan': self.segmentation_config['segmentation']['enable_dbscan'],
                    'dbscan_eps': float(self.segmentation_config['segmentation']['dbscan_eps']),
                    'dbscan_min_samples': int(self.segmentation_config['segmentation']['dbscan_min_samples'])
                }
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error creating segmentation metadata: {e}")
            return {}
    
    def get_color_legend(self) -> Dict:
        """Get color legend for the segmentation masks."""
        legend = {}
        all_objects = self.get_all_objects()
        all_colors = self.get_all_colors()
        
        for i, word in enumerate(all_objects):
            if i < len(all_colors):
                legend[word] = all_colors[i]
        return legend

    @torch.no_grad()
    def norm_img_01(self, x):
        """Normalize image tensor to [0, 1] range like RayFronts utils."""
        B, C, H, W = x.shape
        x = x - torch.min(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1)
        x = x / (torch.max(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1) + 1e-8)
        return x

    @torch.no_grad()
    def compute_cos_sim(self, vec1: torch.FloatTensor, vec2: torch.FloatTensor, softmax: bool = False) -> torch.FloatTensor:
        """Compute cosine similarity between two batches of vectors - RayFronts implementation."""
        N, C1 = vec1.shape
        M, C2 = vec2.shape
        if C1 != C2:
            raise ValueError(f"vec1 feature dimension '{C1}' does not match vec2 feature dimension '{C2}'")
        
        vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)
        vec1 = vec1.reshape(1, N, 1, C1)
        
        vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)
        vec2 = vec2.reshape(M, 1, C1, 1)
        
        sim = (vec1 @ vec2).reshape(M, N)
        if softmax:
            return torch.softmax(100 * sim, dim=-1)
        else:
            return sim

    @torch.inference_mode()
    def compute_enhanced_similarity_map_optimized(self, rgb_image: np.ndarray, vlm_answer: str, 
                                                 feat_map_np: Optional[np.ndarray] = None,
                                                 chunk_size: int = 4000) -> Optional[np.ndarray]:
        """Compute similarity map using enhanced embedding only."""
        try:
            if not self.has_enhanced_embedding(vlm_answer):
                return None
            
            enhanced_embedding = self.get_enhanced_embedding(vlm_answer)
            if enhanced_embedding is None:
                return None
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            enhanced_embedding_tensor = torch.from_numpy(enhanced_embedding).to(device).float()
            if len(enhanced_embedding_tensor.shape) == 1:
                enhanced_embedding_tensor = enhanced_embedding_tensor.unsqueeze(0)
            
            # Reuse features if provided
            if feat_map_np is not None:
                feat_map_tensor = torch.from_numpy(feat_map_np).to(device) if isinstance(feat_map_np, np.ndarray) else feat_map_np
            else:
                resolution = (self.radio_input_resolution, self.radio_input_resolution)
                tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1).to(device).float() / 255.0
                tensor_image = torch.nn.functional.interpolate(
                    tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)
                feat_map_tensor = self.radio_encoder.encode_image_to_feat_map(tensor_image)
                del tensor_image
            
            # Align features and compute similarity
            feat_map_aligned = self.radio_encoder.align_spatial_features_with_language(feat_map_tensor)
            feat_map_aligned = torch.nn.functional.interpolate(
                feat_map_aligned, (self.radio_input_resolution, self.radio_input_resolution), mode="bilinear", antialias=True)
            feat_map_aligned = feat_map_aligned.squeeze(0).permute(1, 2, 0)
            
            H, W, C = feat_map_aligned.shape
            feat_map_flat = feat_map_aligned.reshape(-1, C)
            del feat_map_aligned
            
            # Compute cosine similarity in chunks
            num_chunks = max(1, int(np.ceil(feat_map_flat.shape[0] / chunk_size)))
            similarity_chunks = []
            
            for c in range(num_chunks):
                chunk_features = feat_map_flat[c*chunk_size:(c+1)*chunk_size]
                enhanced_norm = enhanced_embedding_tensor / (torch.norm(enhanced_embedding_tensor, dim=-1, keepdim=True) + 1e-8)
                chunk_norm = chunk_features / (torch.norm(chunk_features, dim=-1, keepdim=True) + 1e-8)
                chunk_similarity = torch.mm(enhanced_norm, chunk_norm.t()).squeeze(0)
                similarity_chunks.append(chunk_similarity)
                del chunk_features, chunk_norm
            
            similarity_flat = torch.cat(similarity_chunks, dim=0)
            enhanced_similarity_map = similarity_flat.reshape(H, W)
            
            # Normalize to [0,1] range
            enhanced_similarity_map = self.norm_img_01(enhanced_similarity_map.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            
            return enhanced_similarity_map.cpu().numpy()
            
        except Exception as e:
            return None

    @torch.inference_mode()
    def compute_vlm_similarity_map_optimized(self, rgb_image: np.ndarray, vlm_answer: str, 
                                            feat_map_np: Optional[np.ndarray] = None,
                                            use_softmax: bool = True, chunk_size: int = 4000) -> Optional[np.ndarray]:
        """Legacy method - redirects to enhanced similarity only."""
        return self.compute_enhanced_similarity_map_optimized(rgb_image, vlm_answer, feat_map_np, chunk_size)

    @torch.inference_mode()
    def compute_vlm_similarity_map(self, rgb_image: np.ndarray, vlm_answer: str, use_softmax: bool = True, chunk_size: int = 2000) -> Optional[np.ndarray]:
        """Legacy method - redirects to enhanced similarity only."""
        return self.compute_enhanced_similarity_map_optimized(rgb_image, vlm_answer, None, chunk_size)

    def apply_colormap(self, image: np.ndarray, cmap_name='viridis') -> np.ndarray:
        """Apply a colormap to a grayscale image and return an RGB uint8 image."""
        from matplotlib import cm
        
        # Ensure image is normalized to [0, 1]
        if image.dtype != np.float16 and image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32) / 255.0
        image = np.clip(image, 0, 1)
        cmap = cm.get_cmap(cmap_name)
        colored = cmap(image)[:, :, :3]  # Drop alpha channel
        return (colored * 255).astype(np.uint8)

    def process_enhanced_similarity_visualization_optimized(self, rgb_image: np.ndarray, vlm_answer: str,
                                                           feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Process similarity map using enhanced embedding only."""
        if not self.segmentation_ready or not self.naradio_ready:
            return None
        
        try:
            start_time = time.time()
            
            similarity_map = self.compute_enhanced_similarity_map_optimized(rgb_image, vlm_answer, feat_map_np)
            if similarity_map is None:
                return None
            
            # Resize to original image dimensions
            original_height, original_width = rgb_image.shape[:2]
            similarity_resized = cv2.resize(similarity_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            
            # Apply colormap
            colored_similarity = self.apply_colormap(similarity_resized, cmap_name='viridis')
            
            processing_time = time.time() - start_time
            
            return {
                'similarity_map': similarity_resized,
                'colored_similarity': colored_similarity,
                'metadata': {
                    'vlm_answer': vlm_answer,
                    'processing_time': processing_time,
                    'method': 'enhanced_embedding'
                },
                'processing_info': {'processing_time': processing_time}
            }
            
        except Exception as e:
            return None

    def process_vlm_similarity_visualization_optimized(self, rgb_image: np.ndarray, vlm_answer: str,
                                                     feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Legacy method - redirects to enhanced similarity only."""
        return self.process_similarity_visualization(rgb_image, vlm_answer, feat_map_np)

    def get_similarity_method_info(self, vlm_answer: str) -> Dict[str, any]:
        """Get information about similarity method (enhanced only)."""
        has_enhanced = self.has_enhanced_embedding(vlm_answer)
        return {
            'vlm_answer': vlm_answer,
            'method': "enhanced_embedding" if has_enhanced else "none",
            'has_enhanced_embedding': has_enhanced
        }

    def process_vlm_similarity_visualization(self, rgb_image: np.ndarray, vlm_answer: str) -> Optional[Dict]:
        """Legacy method - redirects to enhanced similarity only."""
        return self.process_similarity_visualization(rgb_image, vlm_answer, None)

    def is_segmentation_ready(self) -> bool:
        """Check if combined segmentation is ready."""
        return self.segmentation_ready and self.naradio_ready

    @torch.inference_mode()
    def compute_enhanced_cause_embedding(self, rgb_image: np.ndarray, vlm_answer: str, 
                                       similarity_map: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute enhanced cause embedding by weighted averaging spatial features using similarity as weights.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            vlm_answer: The VLM answer to compute enhanced embedding for
            similarity_map: Pre-computed similarity map (optional, will compute if not provided)
            
        Returns:
            np.ndarray: Enhanced cause embedding vector, or None if failed
        """
        try:
            # Check if VLM answer is in our objects
            all_objects = self.get_all_objects()
            if vlm_answer not in all_objects:
                print(f"VLM answer '{vlm_answer}' not in object list for enhanced embedding")
                return None
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Extract spatial features from the image
            resolution = (self.radio_input_resolution, self.radio_input_resolution)
            if hasattr(self.radio_encoder, "input_resolution"):
                self.radio_encoder.input_resolution = resolution
            
            # Convert image to tensor
            tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1)
            tensor_image = tensor_image.to(device).float() / 255.0
            tensor_image = torch.nn.functional.interpolate(
                tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)
            
            # Extract features
            feat_map_tensor = self.radio_encoder.encode_image_to_feat_map(tensor_image)
            
            # Align features with language
            feat_map_aligned = self.radio_encoder.align_spatial_features_with_language(feat_map_tensor)
            
            # Resize to final resolution
            feat_map_aligned = torch.nn.functional.interpolate(
                feat_map_aligned, resolution, mode="bilinear", antialias=True)
            feat_map_aligned = feat_map_aligned.squeeze(0).permute(1, 2, 0)  # H x W x C
            
            H, W, C = feat_map_aligned.shape
            
            # Get or compute similarity map using enhanced embeddings
            if similarity_map is None:
                similarity_map = self.compute_enhanced_similarity_map_optimized(rgb_image, vlm_answer)
                if similarity_map is None:
                    print(f"Failed to compute similarity map for enhanced embedding")
                    return None
            
            # Ensure similarity map matches feature map dimensions
            if similarity_map.shape != (H, W):
                similarity_map = cv2.resize(similarity_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Convert similarity map to tensor weights
            weights = torch.from_numpy(similarity_map).to(device).float()  # H x W
            
            # ENHANCED FIX: Only consider pixels with high similarity to avoid garbage features
            similarity_threshold = self.segmentation_config['segmentation'].get('enhanced_similarity_threshold', 0.5)
            high_sim_mask = weights > similarity_threshold
            
            # Zero out low-similarity pixels to focus only on the actual cause/hotspots
            weights = weights * high_sim_mask.float()
            
            # Check if we have enough high-similarity pixels
            num_high_sim_pixels = torch.sum(high_sim_mask).item()
            total_pixels = H * W
            
            if num_high_sim_pixels == 0:
                print(f"Warning: No pixels above similarity threshold {similarity_threshold}, using all pixels")
                weights = torch.from_numpy(similarity_map).to(device).float()
            else:
                print(f"Enhanced embedding using {num_high_sim_pixels}/{total_pixels} high-similarity pixels (>{similarity_threshold})")
            
            # Normalize weights to sum to 1 for proper weighted averaging
            total_weight = torch.sum(weights)
            if total_weight > 1e-8:
                weights = weights / total_weight
            else:
                print(f"Warning: similarity weights sum to near zero after thresholding")
                return None
            
            # Flatten spatial dimensions for weighted averaging
            feat_map_flat = feat_map_aligned.reshape(-1, C)  # (H*W) x C
            weights_flat = weights.reshape(-1)  # (H*W)
            
            # Compute weighted average of features using only high-similarity pixels
            # This gives us a much cleaner enhanced embedding focused on the actual cause
            enhanced_embedding = torch.sum(feat_map_flat * weights_flat.unsqueeze(1), dim=0)  # C
            
            # Normalize the enhanced embedding
            enhanced_embedding = enhanced_embedding / (torch.norm(enhanced_embedding) + 1e-8)
            
            # Convert to numpy and return
            enhanced_embedding_np = enhanced_embedding.cpu().numpy()
            
            print(f"✓ Computed enhanced cause embedding for '{vlm_answer}' (shape: {enhanced_embedding_np.shape})")
            print(f"  Used {total_weight.item():.3f} total similarity weight from {H*W} pixels")
            
            # Cleanup
            del tensor_image, feat_map_tensor, feat_map_aligned, weights, feat_map_flat, weights_flat
            
            return enhanced_embedding_np
            
        except Exception as e:
            print(f"Error computing enhanced cause embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_narration_image_similarity(self, narration_image: np.ndarray, vlm_answer: str, 
                                         buffer_id: str, buffer_dir: str) -> bool:
        """
        Process narration image for VLM similarity mapping and enhanced cause embedding.
        
        Args:
            narration_image: Narration image as numpy array (H, W, 3)
            vlm_answer: The VLM answer to compute similarity for
            buffer_id: Buffer identifier
            buffer_dir: Buffer directory path
            
        Returns:
            bool: True if successfully processed
        """
        try:
            if not self.is_segmentation_ready():
                print(f"Cannot process narration image - segmentation not ready")
                return False
            
            if vlm_answer not in self.get_all_objects():
                print(f"VLM answer '{vlm_answer}' not in object list, cannot process narration image")
                return False
            
            print(f"Processing narration similarity and enhanced embedding for VLM '{vlm_answer}' in buffer {buffer_id}")
            
            # Process similarity for the narration image using enhanced embeddings
            similarity_result = self.process_similarity_visualization(
                narration_image, vlm_answer, feat_map_np=None)
            
            if similarity_result is None:
                print(f"Failed to compute similarity for narration image")
                return False
            
            # Extract similarity map for enhanced embedding computation
            similarity_map = similarity_result.get('similarity_map')
            if similarity_map is not None:
                # Compute enhanced cause embedding using similarity as weights
                enhanced_embedding = self.compute_enhanced_cause_embedding(
                    narration_image, vlm_answer, similarity_map)
                
                if enhanced_embedding is not None:
                    # Save enhanced embedding to buffer directory
                    self._save_enhanced_embedding(vlm_answer, buffer_dir, enhanced_embedding)
                    
                    # Add enhanced embedding to similarity result for completeness
                    similarity_result['enhanced_embedding'] = enhanced_embedding
                    
                    # NEW: Add to risk library for semantic mapping
                    metadata = {
                        'buffer_id': buffer_id,
                        'buffer_dir': buffer_dir,
                        'similarity_map_shape': list(similarity_map.shape),
                        'similarity_map_max': float(np.max(similarity_map)),
                        'similarity_map_mean': float(np.mean(similarity_map)),
                        'computation_method': 'similarity_weighted_spatial_features'
                    }
                    self.add_to_risk_library(vlm_answer, enhanced_embedding, metadata)
                    
                    print(f"✓ Computed and saved enhanced embedding for '{vlm_answer}' (shape: {enhanced_embedding.shape})")
                else:
                    print(f"✗ Failed to compute enhanced embedding for '{vlm_answer}'")
            else:
                print(f"✗ No similarity map available for enhanced embedding computation")
            
            # NEW: Extract and save hotspot mask
            if similarity_map is not None:
                self._save_hotspot_mask(vlm_answer, buffer_dir, similarity_map, narration_image)
            
            # Save the similarity result to the buffer
            self._save_narration_similarity_result_simple(
                vlm_answer, buffer_id, buffer_dir, similarity_result)
            
            print(f"✓ Completed narration similarity processing for VLM '{vlm_answer}' in buffer {buffer_id}")
            return True
            
        except Exception as e:
            print(f"Error processing narration similarity: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_enhanced_embedding(self, vlm_answer: str, buffer_dir: str, enhanced_embedding: np.ndarray):
        """Save enhanced cause embedding to buffer directory."""
        try:
            # Create embeddings directory
            embeddings_dir = os.path.join(buffer_dir, 'enhanced_embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Create safe filename
            safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
            timestamp_str = f"{time.time():.3f}"
            
            # Save enhanced embedding as numpy array
            embedding_path = os.path.join(embeddings_dir, f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}.npy")
            np.save(embedding_path, enhanced_embedding)
            
            # Save metadata
            metadata = {
                'vlm_answer': vlm_answer,
                'embedding_filename': f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}.npy",
                'embedding_shape': list(enhanced_embedding.shape),
                'embedding_norm': float(np.linalg.norm(enhanced_embedding)),
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Enhanced cause embedding computed from weighted averaging of spatial features using similarity as weights',
                'computation_method': 'similarity_weighted_spatial_features'
            }
            
            metadata_path = os.path.join(embeddings_dir, f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved enhanced embedding: {embedding_path}")
            print(f"  Shape: {enhanced_embedding.shape}, Norm: {np.linalg.norm(enhanced_embedding):.4f}")
            
        except Exception as e:
            print(f"Error saving enhanced embedding: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_narration_similarity_result_simple(self, vlm_answer: str, buffer_id: str, 
                                                buffer_dir: str, similarity_result: Dict):
        """Save narration similarity results to buffer directory."""
        try:
            # Create narration similarity directory
            narration_sim_dir = os.path.join(buffer_dir, 'narration_similarity')
            os.makedirs(narration_sim_dir, exist_ok=True)
            
            # Save colored similarity map
            if similarity_result.get('colored_similarity') is not None:
                safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
                timestamp_str = f"{time.time():.3f}"
                
                colored_path = os.path.join(narration_sim_dir, f"narration_similarity_{safe_vlm_name}_{timestamp_str}.png")
                colored_bgr = cv2.cvtColor(similarity_result['colored_similarity'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(colored_path, colored_bgr)
                
                print(f"Saved narration similarity map: {colored_path}")
            
            # Save metadata
            metadata_path = os.path.join(narration_sim_dir, f"narration_similarity_{vlm_answer.replace(' ', '_')}_{timestamp_str}_metadata.json")
            
            combined_metadata = {
                'vlm_answer': vlm_answer,
                'buffer_id': buffer_id,
                'similarity_metadata': similarity_result.get('metadata', {}),
                'processing_info': similarity_result.get('processing_info', {}),
                'saved_timestamp': time.time(),
                'saved_datetime': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(combined_metadata, f, indent=2)
            
        except Exception as e:
            print(f"Error saving narration similarity result: {e}")
            import traceback
            traceback.print_exc() 

    def process_similarity_visualization(self, rgb_image: np.ndarray, vlm_answer: str,
                                        feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Process similarity visualization using enhanced embeddings only."""
        return self.process_enhanced_similarity_visualization_optimized(rgb_image, vlm_answer, feat_map_np)

    def _save_hotspot_mask(self, vlm_answer: str, buffer_dir: str, similarity_map: np.ndarray, 
                          original_image: np.ndarray):
        """Save hotspot mask extracted from similarity map to buffer directory."""
        try:
            # Create hotspot masks directory
            hotspot_dir = os.path.join(buffer_dir, 'hotspot_masks')
            os.makedirs(hotspot_dir, exist_ok=True)
            
            # Get hotspot threshold from config (same as semantic bridge)
            hotspot_threshold = self.segmentation_config['segmentation'].get('hotspot_threshold', 0.6)
            min_area = self.segmentation_config['segmentation'].get('min_hotspot_area', 50)
            
            # Apply binary threshold to similarity map
            hotspot_mask = similarity_map > hotspot_threshold
            
            if not np.any(hotspot_mask):
                print(f"No hotspots found for '{vlm_answer}' with threshold {hotspot_threshold}")
                return
            
            # Filter small regions using connected components
            hotspot_mask_uint8 = (hotspot_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(hotspot_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create filtered mask with only significant regions
            filtered_mask = np.zeros_like(hotspot_mask, dtype=np.uint8)
            hotspot_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    cv2.fillPoly(filtered_mask, [contour], 255)
                    hotspot_count += 1
            
            if hotspot_count == 0:
                print(f"No significant hotspots found for '{vlm_answer}' (min area: {min_area})")
                return
            
            # Create safe filename
            safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
            timestamp_str = f"{time.time():.3f}"
            
            # Save binary hotspot mask
            mask_path = os.path.join(hotspot_dir, f"hotspot_mask_{safe_vlm_name}_{timestamp_str}.png")
            cv2.imwrite(mask_path, filtered_mask)
            
            # Save colored hotspot mask (green hotspots on black background)
            colored_mask = np.zeros((filtered_mask.shape[0], filtered_mask.shape[1], 3), dtype=np.uint8)
            colored_mask[filtered_mask > 0] = [0, 255, 0]  # Green hotspots
            colored_mask_path = os.path.join(hotspot_dir, f"hotspot_mask_colored_{safe_vlm_name}_{timestamp_str}.png")
            cv2.imwrite(colored_mask_path, colored_mask)
            
            # Save hotspot overlay on original image
            overlay = original_image.copy()
            overlay[filtered_mask > 0] = [0, 255, 0]  # Green hotspots on original image
            overlay_path = os.path.join(hotspot_dir, f"hotspot_overlay_{safe_vlm_name}_{timestamp_str}.png")
            cv2.imwrite(overlay_path, overlay)
            
            # Compute statistics
            hotspot_pixels = int(np.sum(filtered_mask > 0))
            hotspot_similarity_values = similarity_map[hotspot_mask]
            
            # Save metadata
            metadata = {
                'vlm_answer': vlm_answer,
                'hotspot_threshold': float(hotspot_threshold),
                'min_area': int(min_area),
                'hotspot_count': int(hotspot_count),
                'hotspot_pixels': int(hotspot_pixels),
                'max_similarity': float(np.max(hotspot_similarity_values)) if len(hotspot_similarity_values) > 0 else 0.0,
                'avg_similarity': float(np.mean(hotspot_similarity_values)) if len(hotspot_similarity_values) > 0 else 0.0,
                'mask_shape': list(filtered_mask.shape),
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'files': {
                    'binary_mask': f"hotspot_mask_{safe_vlm_name}_{timestamp_str}.png",
                    'colored_mask': f"hotspot_mask_colored_{safe_vlm_name}_{timestamp_str}.png",
                    'overlay': f"hotspot_overlay_{safe_vlm_name}_{timestamp_str}.png"
                }
            }
            
            metadata_path = os.path.join(hotspot_dir, f"hotspot_mask_{safe_vlm_name}_{timestamp_str}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved hotspot mask for '{vlm_answer}' to {hotspot_dir}")
            print(f"  Hotspots: {hotspot_count}, Pixels: {hotspot_pixels}")
            print(f"  Files: {mask_path}, {colored_mask_path}, {overlay_path}")
            
        except Exception as e:
            print(f"Error saving hotspot mask: {e}")
            import traceback
            traceback.print_exc()

    def add_to_risk_library(self, vlm_answer: str, enhanced_embedding: np.ndarray, metadata: dict = None):
        """Add VLM answer with enhanced embedding to risk library."""
        try:
            with self.risk_library_lock:
                risk_entry = {
                    'vlm_answer': vlm_answer,
                    'enhanced_embedding': enhanced_embedding,
                    'embedding_shape': list(enhanced_embedding.shape),
                    'embedding_norm': float(np.linalg.norm(enhanced_embedding)),
                    'timestamp': time.time(),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'metadata': metadata or {}
                }
                
                self.risk_library[vlm_answer] = risk_entry
                print(f"✓ Added '{vlm_answer}' to risk library (embedding shape: {enhanced_embedding.shape})")
                
        except Exception as e:
            print(f"Error adding to risk library: {e}")
            import traceback
            traceback.print_exc()

    def get_risk_library(self) -> dict:
        """Get current risk library."""
        with self.risk_library_lock:
            return self.risk_library.copy()

    def save_risk_library(self, buffer_dir: str):
        """Save risk library to buffer directory."""
        try:
            with self.risk_library_lock:
                if not self.risk_library:
                    print("No risk library entries to save")
                    return
                
                # Create risk library directory
                risk_lib_dir = os.path.join(buffer_dir, 'risk_library')
                os.makedirs(risk_lib_dir, exist_ok=True)
                
                timestamp_str = f"{time.time():.3f}"
                
                # Save each risk entry
                for vlm_answer, entry in self.risk_library.items():
                    safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    
                    # Save enhanced embedding
                    embedding_path = os.path.join(risk_lib_dir, f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}.npy")
                    np.save(embedding_path, entry['enhanced_embedding'])
                    
                    # Create JSON-serializable metadata (without numpy arrays)
                    metadata = {
                        'vlm_answer': vlm_answer,
                        'enhanced_embedding_path': f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}.npy",
                        'enhanced_embedding_shape': list(entry['enhanced_embedding'].shape) if entry['enhanced_embedding'] is not None else None,
                        'enhanced_embedding_norm': float(np.linalg.norm(entry['enhanced_embedding'])) if entry['enhanced_embedding'] is not None else None,
                        'metadata': self._make_json_serializable(entry.get('metadata', {})),
                        'timestamp': entry.get('timestamp', time.time()),
                        'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'description': 'Risk library entry with enhanced embedding for semantic mapping'
                    }
                    
                    # Save metadata
                    metadata_path = os.path.join(risk_lib_dir, f"risk_entry_{safe_vlm_name}_{timestamp_str}.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                # Save complete risk library summary
                summary = {
                    'total_entries': len(self.risk_library),
                    'entries': list(self.risk_library.keys()),
                    'timestamp': time.time(),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'description': 'Risk library containing VLM answers with enhanced embeddings for semantic mapping'
                }
                
                summary_path = os.path.join(risk_lib_dir, f"risk_library_summary_{timestamp_str}.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"✓ Saved risk library to {risk_lib_dir}")
                print(f"  Total entries: {len(self.risk_library)}")
                print(f"  Entries: {list(self.risk_library.keys())}")
                
        except Exception as e:
            print(f"Error saving risk library: {e}")
            import traceback
            traceback.print_exc()

    def create_merged_hotspot_masks(self, rgb_image: np.ndarray, vlm_answers: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """Create hotspot masks using enhanced embeddings only."""
        try:
            if not self.is_segmentation_ready():
                return None
            
            hotspot_threshold = self.segmentation_config['segmentation'].get('hotspot_threshold', 0.6)
            min_area = self.segmentation_config['segmentation'].get('min_hotspot_area', 50)
            
            vlm_hotspots = {}
            
            for vlm_answer in vlm_answers:
                if not self.has_enhanced_embedding(vlm_answer):
                    continue
                
                # Use enhanced embedding similarity
                similarity_map = self.compute_enhanced_similarity_map_optimized(rgb_image, vlm_answer)
                if similarity_map is None:
                    continue
                
                # Apply threshold and filter
                hotspot_mask = similarity_map > hotspot_threshold
                if not np.any(hotspot_mask):
                    continue
                
                # Filter small regions
                hotspot_mask_uint8 = (hotspot_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(hotspot_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                filtered_mask = np.zeros_like(hotspot_mask, dtype=np.uint8)
                for contour in contours:
                    if cv2.contourArea(contour) >= min_area:
                        cv2.fillPoly(filtered_mask, [contour], 255)
                
                if np.any(filtered_mask):
                    vlm_hotspots[vlm_answer] = filtered_mask
            
            return vlm_hotspots if vlm_hotspots else None
            
        except Exception as e:
            return None

    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format, handling numpy arrays and other non-serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
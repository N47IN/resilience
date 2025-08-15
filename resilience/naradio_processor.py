#!/usr/bin/env python3
"""
NARadio Processor Module

Handles NARadio feature extraction and visualization with robust error handling
and GPU memory management to prevent crashes.
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

# Suppress warnings
warnings.filterwarnings('ignore')


class NARadioProcessor:
    """NARadio feature extraction and processing with robust error handling."""
    
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
        
        # Initialize NARadio
        self.init_naradio()
        
        # Initialize combined segmentation if enabled
        if self.enable_combined_segmentation:
            self.init_combined_segmentation(segmentation_config_path)
    
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

    def extract_features(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract NARadio features from RGB image."""
        try:
            if not self.is_ready():
                print("NARadio not ready for feature extraction")
                return None
            
            # Ensure device consistency
            if not self.ensure_device_consistency():
                print("Device consistency check failed, reinitializing...")
                self.reinitialize()
                if not self.is_ready():
                    return None
            
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
            
            # Extract features
            with torch.no_grad():
                feat_map = self.radio_encoder.encode_image_to_feat_map(image_tensor)
                
                # Convert to numpy
                if isinstance(feat_map, torch.Tensor):
                    feat_map_np = feat_map.cpu().numpy()
                else:
                    feat_map_np = feat_map
                
                return feat_map_np
                
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory during feature extraction")
            self.handle_cuda_out_of_memory()
            return None
        except Exception as e:
            print(f"Error extracting NARadio features: {e}")
            return None

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

    def process_features(self, rgb_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process NARadio features and create visualization."""
        try:
            # Extract features
            feat_map_np = self.extract_features(rgb_image)
            
            # Create visualization
            naradio_vis = None
            if feat_map_np is not None:
                naradio_vis = self.create_visualization(feat_map_np)
            
            return feat_map_np, naradio_vis
            
        except Exception as e:
            print(f"Error processing NARadio features: {e}")
            return None, None

    def cleanup_memory(self):
        """Periodic cleanup of radio model memory."""
        try:
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Get memory info for monitoring
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                
                print(f"NARadio memory cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
        except Exception as e:
            print(f"Error during radio memory cleanup: {e}")

    def handle_cuda_out_of_memory(self):
        """Handle CUDA out of memory errors."""
        try:
            if torch.cuda.is_available():
                # Clear all CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
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
                config_path = os.path.join(package_dir, 'config', 'combined_segmentation_config.yaml')
            except ImportError:
                # Fallback for non-ROS environments
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'combined_segmentation_config.yaml')
        
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
            if self.dynamic_objects:
                print(f"[ColoredMask] Creating mask with {len(all_objects)} objects")
                print(f"[ColoredMask] VLM objects indices: {[all_objects.index(obj) for obj in self.dynamic_objects if obj in all_objects]}")
                print(f"[ColoredMask] VLM colors: {self.dynamic_colors}")
            
            # Find best matching word for each pixel
            best_word_indices = torch.argmax(similarity_maps, dim=0)
            best_word_indices_np = best_word_indices.cpu().numpy()
            
            # Get dimensions
            H, W = best_word_indices_np.shape
            
            # Create RGB mask
            colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
            
            # Count pixels for each object for debugging
            pixel_counts = {}
            
            for i in range(len(all_objects)):
                mask = (best_word_indices_np == i)
                colored_mask[mask] = all_colors[i]
                pixel_counts[all_objects[i]] = np.sum(mask)
            
            # Debug: Print pixel counts and similarity scores for VLM objects
            if self.dynamic_objects:
                print(f"[ColoredMask] All colors length: {len(all_colors)}")
                print(f"[ColoredMask] All objects length: {len(all_objects)}")
                for vlm_obj in self.dynamic_objects:
                    if vlm_obj in pixel_counts:
                        obj_index = all_objects.index(vlm_obj)
                        obj_similarity = similarity_maps[obj_index]
                        max_similarity = torch.max(obj_similarity).item()
                        mean_similarity = torch.mean(obj_similarity).item()
                        actual_color = all_colors[obj_index] if obj_index < len(all_colors) else "OUT_OF_RANGE"
                        print(f"[ColoredMask] VLM object '{vlm_obj}' (index {obj_index}): {pixel_counts[vlm_obj]} pixels, max_sim={max_similarity:.3f}, mean_sim={mean_similarity:.3f} (color: {actual_color})")
            
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
    
    def is_segmentation_ready(self) -> bool:
        """Check if combined segmentation is ready."""
        return self.segmentation_ready and self.naradio_ready 
#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import sys
import os
import time
import gc
from matplotlib import cm

# Add resilience module to path
sys.path.append('/home/navin/ros2_ws/src/resilience')

from resilience.naradio_processor import NARadioProcessor

def apply_colormap(image: np.ndarray, cmap_name='viridis') -> np.ndarray:
    """Apply a colormap to a grayscale image and return an RGB uint8 image."""
    # Ensure image is normalized to [0, 1]
    if image.dtype != np.float16 and image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0, 1)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(image)[:, :, :3]  # Drop alpha channel
    return (colored * 255).astype(np.uint8)

@torch.no_grad()
def norm_img_01(x):
    """Normalize image tensor to [0, 1] range like RayFronts utils."""
    B, C, H, W = x.shape
    x = x - torch.min(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1)
    x = x / (torch.max(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1) + 1e-8)
    return x

@torch.no_grad()
def compute_cos_sim(vec1: torch.FloatTensor, vec2: torch.FloatTensor, softmax: bool = False) -> torch.FloatTensor:
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
def compute_per_object_similarity(processor, rgb_image: np.ndarray, use_softmax: bool = True, chunk_size: int = 2000):
    """Compute per-object similarity maps exactly like RayFronts encoder_semseg_app.py"""
    try:
        # Clear CUDA cache before computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get all features (base + dynamic objects) - this is our "prompt_embeddings"
        all_features = processor.get_all_features()
        all_objects = processor.get_all_objects()
        
        if all_features is None or len(all_objects) == 0:
            print("No objects available for similarity computation")
            return None
        
        N = len(all_objects)
        if use_softmax and N == 1:
            print("With softmax enabled, you need at least two objects")
            return None
        
        print(f"Computing similarity for {N} objects: {all_objects}")
        
        # Set model resolution
        resolution = (processor.radio_input_resolution, processor.radio_input_resolution)
        if hasattr(processor.radio_encoder, "input_resolution"):
            processor.radio_encoder.input_resolution = resolution
        
        print("Computing feature map...")
        # Convert image to tensor exactly like RayFronts
        tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1)
        tensor_image = tensor_image.to(device).float() / 255.0
        tensor_image = torch.nn.functional.interpolate(
            tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)
        
        # Extract features exactly like RayFronts
        feat_map = processor.radio_encoder.encode_image_to_feat_map(tensor_image)
        # Clear cache after feature extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        feat_map = processor.radio_encoder.align_spatial_features_with_language(feat_map)
        # Clear cache after alignment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        feat_map = torch.nn.functional.interpolate(
            feat_map, resolution, mode="bilinear", antialias=True)
        feat_map = feat_map.squeeze(0).permute(1, 2, 0)
        
        # Delete tensor_image to free memory
        del tensor_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Computing cosine similarity...")
        # Compute similarity exactly like RayFronts with chunking
        H, W, C = feat_map.shape
        feat_map_flat = feat_map.reshape(-1, C)  # H*W x C
        
        # Delete original feat_map to save memory
        del feat_map
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use chunking for memory efficiency like RayFronts
        num_chunks = int(np.ceil(feat_map_flat.shape[0] / chunk_size))
        cos_sim = []
        
        print(f"Processing {num_chunks} chunks of size {chunk_size}")
        for c in range(num_chunks):
            # Clear cache before each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            chunk_features = feat_map_flat[c*chunk_size:(c+1)*chunk_size]
            
            # Use RayFronts compute_cos_sim function
            with torch.no_grad():
                chunk_sim = compute_cos_sim(all_features, chunk_features, softmax=use_softmax)
                cos_sim.append(chunk_sim.cpu())  # Move to CPU immediately
            
            # Delete chunk to free memory
            del chunk_features, chunk_sim
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks on CPU
        cos_sim = torch.cat(cos_sim, dim=0)  # H*W x N
        cos_sim = cos_sim.to(device)  # Move back to device for final operations
        cos_sim = cos_sim.reshape(H, W, N)   # H x W x N
        
        print("Visualizing...")
        # Apply normalization if not using softmax (exactly like RayFronts)
        if not use_softmax:
            cos_sim = norm_img_01(cos_sim.permute(2, 0, 1).unsqueeze(0))  # 1 x N x H x W
            cos_sim = cos_sim.squeeze(0).permute(1, 2, 0)  # H x W x N
        
        # Clear cache after computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return as numpy array with objects dimension first (N x H x W)
        result = cos_sim.permute(2, 0, 1).cpu().numpy()
        return result, all_objects
        
    except Exception as e:
        print(f"Error computing per-object similarity: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Clear CUDA cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleared CUDA cache at startup")
    
    # Initialize NARadio processor with combined segmentation
    processor = NARadioProcessor(
        radio_model_version='radio_v2.5-b',
        radio_lang_model='siglip',
        radio_input_resolution=512,
        enable_visualization=True,
        enable_combined_segmentation=True
    )
    
    # Check if ready
    if not processor.is_ready():
        print("NARadio not ready")
        return
    
    if not processor.is_segmentation_ready():
        print("Segmentation not ready")
        return
    
    # Load image
    image_path = '/home/navin/ros2_ws_latest/src/map_anything/radio_enhanced/assets/fan2.png'
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {rgb_image.shape}")
    
    # Add VLM objects
    vlm_objects = ["fan", "blade", "metal", "white", "ceiling"]
    for vlm_obj in vlm_objects:
        processor.add_vlm_object(vlm_obj)
    
    print(f"Objects: {processor.get_all_objects()}")
    
    # Compute per-object similarity maps with small chunk size for memory efficiency
    result = compute_per_object_similarity(processor, rgb_image, use_softmax=True, chunk_size=1000)
    
    if result is not None:
        similarity_maps, object_labels = result
        print(f"Generated similarity maps for {len(object_labels)} objects")
        
        # Get original image dimensions
        original_height, original_width = rgb_image.shape[:2]
        print(f"Original image size: {original_width}x{original_height}")
        print(f"Similarity map size: {similarity_maps.shape[2]}x{similarity_maps.shape[1]}")
        
        # Display original image
        cv2.imshow('Original Image', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # Find and display only the "fan" similarity map
        fan_found = False
        for i, obj_label in enumerate(object_labels):
            if obj_label.lower() == "fan":
                similarity_map = similarity_maps[i]
                
                # Resize to original image dimensions
                similarity_resized = cv2.resize(similarity_map, (original_width, original_height), 
                                              interpolation=cv2.INTER_LINEAR)
                
                # Apply colormap for visualization
                colored_similarity = apply_colormap(similarity_resized, cmap_name='viridis')
                
                # Convert to BGR for OpenCV display
                colored_similarity_bgr = cv2.cvtColor(colored_similarity, cv2.COLOR_RGB2BGR)
                
                # Display the similarity map
                window_name = f'Similarity: {obj_label}'
                cv2.imshow(window_name, colored_similarity_bgr)
                
                print(f"Displaying similarity map for '{obj_label}'")
                fan_found = True
                break
        
        if not fan_found:
            print("Warning: 'fan' object not found in similarity maps")
            print(f"Available objects: {object_labels}")
        
        print("\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("Failed to compute similarity maps")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import sys
import os
import time
import gc

# Add resilience module to path
sys.path.append('/home/navin/ros2_ws/src/resilience')

from resilience.naradio_processor import NARadioProcessor

def main():
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
    
    # Print initial objects and colors
    print("=" * 60)
    print("INITIAL OBJECTS AND COLORS")
    print("=" * 60)
    print(f"Objects: {processor.get_all_objects()}")
    print(f"Colors: {processor.get_all_colors()}")
    print("=" * 60)
    
    # Load image
    image_path = '/home/navin/ros2_ws/src/map_anything/radio_enhanced/assets/fan2.png'
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {rgb_image.shape}")
    
    # Add VLM object
    vlm_answer = "fan"
    processor.add_vlm_object(vlm_answer)
    
    # Print objects and colors after adding VLM object
    print("=" * 60)
    print("AFTER ADDING VLM OBJECT")
    print("=" * 60)
    print(f"Objects: {processor.get_all_objects()}")
    print(f"Colors: {processor.get_all_colors()}")
    print("=" * 60)
    
    # Process combined segmentation (exact workflow from main node)
    result = processor.process_combined_segmentation(rgb_image)
    
    if result is not None:
        original_mask = result['original_mask']
        refined_mask = result['refined_mask']
        
        # Get original image dimensions
        original_height, original_width = rgb_image.shape[:2]
        print(f"Original image size: {original_width}x{original_height}")
        print(f"Mask size: {original_mask.shape[1]}x{original_mask.shape[0]}")
        
        # Better resizing method for segmentation masks
        def resize_segmentation_mask(mask, target_width, target_height):
            """Resize segmentation mask while preserving color boundaries."""
            # Convert to LAB color space for better interpolation
            mask_lab = cv2.cvtColor(mask, cv2.COLOR_RGB2LAB)
            
            # Resize each channel separately with nearest neighbor to preserve colors
            resized_lab = np.zeros((target_height, target_width, 3), dtype=np.float32)
            for i in range(3):
                resized_lab[:, :, i] = cv2.resize(mask_lab[:, :, i], (target_width, target_height), 
                                                interpolation=cv2.INTER_NEAREST)
            
            # Convert back to RGB
            resized_mask = cv2.cvtColor(resized_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            # Apply slight Gaussian blur to smooth edges
            resized_mask = cv2.GaussianBlur(resized_mask, (3, 3), 0.5)
            
            return resized_mask
        
        # Resize masks using the improved method
        original_mask_resized = resize_segmentation_mask(original_mask, original_width, original_height)
        refined_mask_resized = resize_segmentation_mask(refined_mask, original_width, original_height)
        
        # Save original size masks
        cv2.imwrite('original_mask.png', original_mask)
        cv2.imwrite('refined_mask.png', refined_mask)
        
        # Save resized masks
        cv2.imwrite('original_mask_resized.png', original_mask_resized)
        cv2.imwrite('refined_mask_resized.png', refined_mask_resized)
        
        print(f"Original mask saved as 'original_mask.png'")
        print(f"Refined mask saved as 'refined_mask.png'")
        print(f"Resized original mask saved as 'original_mask_resized.png'")
        print(f"Resized refined mask saved as 'refined_mask_resized.png'")
        print(f"Objects: {processor.get_all_objects()}")
        print(f"Colors: {processor.get_all_colors()}")
    else:
        print("Segmentation failed")

if __name__ == "__main__":
    main() 
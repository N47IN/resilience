#!/usr/bin/env python3
"""
Semantic Voxel Mapper

Helper class for semantic voxel mapping with VLM similarity-based labeling.
Integrates with OctoMap nodes to provide semantic annotations based on
enhanced VLM embeddings and cosine similarity thresholding.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from std_msgs.msg import ColorRGBA
import os
import json
import threading
import time
from cv_bridge import CvBridge


class SemanticVoxelMapper:
    """
    Handles semantic voxel mapping with VLM similarity-based semantic labeling.
    Voxels receive semantic labels only when cosine similarity > threshold with enhanced VLM embeddings.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 semantic_color_r: float = 0.0,
                 semantic_color_g: float = 1.0, 
                 semantic_color_b: float = 0.0,
                 default_color_r: float = 0.5,
                 default_color_g: float = 0.5,
                 default_color_b: float = 0.5,
                 embedding_dim: int = 1152,
                 verbose: bool = False):
        """
        Initialize semantic voxel mapper.
        
        Args:
            similarity_threshold: Cosine similarity threshold for semantic labeling (default: 0.6)
            semantic_color_*: RGB color for semantic hotspots (default: green)
            default_color_*: RGB color for non-semantic voxels (default: grey)
            embedding_dim: Expected embedding dimension
            verbose: If True, prints informational messages
        """
        self.similarity_threshold = float(similarity_threshold)
        self.semantic_color = np.array([semantic_color_r, semantic_color_g, semantic_color_b], dtype=np.float32)
        self.default_color = np.array([default_color_r, default_color_g, default_color_b], dtype=np.float32)
        self.embedding_dim = int(embedding_dim)
        self.verbose = bool(verbose)
        
        # VLM embeddings storage: vlm_answer -> enhanced_embedding
        self.vlm_embeddings: Dict[str, np.ndarray] = {}
        self.vlm_embeddings_lock = threading.Lock()
        
        # Voxel semantic labels: (vx,vy,vz) -> {'vlm_answer': str, 'similarity': float}
        self.voxel_semantics: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.voxel_semantics_lock = threading.Lock()
        
        self.bridge = CvBridge()
        
        if self.verbose:
            print(f"SemanticVoxelMapper initialized:")
            print(f"  - Similarity threshold: {self.similarity_threshold}")
            print(f"  - Semantic color (RGB): {self.semantic_color}")
            print(f"  - Default color (RGB): {self.default_color}")
            print(f"  - Embedding dimension: {self.embedding_dim}")
    
    def add_vlm_embedding(self, vlm_answer: str, enhanced_embedding: np.ndarray) -> bool:
        """
        Add or update enhanced VLM embedding for semantic matching.
        
        Args:
            vlm_answer: VLM answer text
            enhanced_embedding: Enhanced embedding vector (should be normalized)
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            if enhanced_embedding is None:
                return False
                
            # Ensure correct dimension
            if enhanced_embedding.shape[-1] != self.embedding_dim:
                if enhanced_embedding.shape[-1] > self.embedding_dim:
                    enhanced_embedding = enhanced_embedding[:self.embedding_dim]
                else:
                    pad = np.zeros(self.embedding_dim - enhanced_embedding.shape[-1], dtype=np.float32)
                    enhanced_embedding = np.concatenate([enhanced_embedding, pad])
            
            # Normalize embedding
            norm = np.linalg.norm(enhanced_embedding)
            if norm > 1e-8:
                enhanced_embedding = enhanced_embedding / norm
            else:
                return False
            
            with self.vlm_embeddings_lock:
                self.vlm_embeddings[vlm_answer] = enhanced_embedding.astype(np.float32)
                
            if self.verbose:
                print(f"✓ Added VLM embedding for '{vlm_answer}' (dim: {enhanced_embedding.shape}, norm: {np.linalg.norm(enhanced_embedding):.4f})")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"✗ Error adding VLM embedding for '{vlm_answer}': {e}")
            return False
    
    def remove_vlm_embedding(self, vlm_answer: str) -> bool:
        """Remove VLM embedding and clear associated voxel semantics."""
        try:
            with self.vlm_embeddings_lock:
                if vlm_answer in self.vlm_embeddings:
                    del self.vlm_embeddings[vlm_answer]
                else:
                    return False
            
            # Clear voxel semantics for this VLM answer
            with self.voxel_semantics_lock:
                to_remove = []
                for voxel_key, semantic_data in self.voxel_semantics.items():
                    if semantic_data.get('vlm_answer') == vlm_answer:
                        to_remove.append(voxel_key)
                
                for key in to_remove:
                    del self.voxel_semantics[key]
            
            if self.verbose:
                print(f"✓ Removed VLM embedding for '{vlm_answer}' and cleared {len(to_remove)} voxel semantics")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"✗ Error removing VLM embedding for '{vlm_answer}': {e}")
            return False
    
    def compute_voxel_similarity(self, voxel_embedding: np.ndarray, voxel_key: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        """
        Compute cosine similarity between voxel embedding and all VLM embeddings.
        
        Args:
            voxel_embedding: Voxel's aggregated embedding
            voxel_key: Voxel coordinate key (vx, vy, vz)
            
        Returns:
            Dict with best match info if similarity > threshold, None otherwise
        """
        try:
            if voxel_embedding is None or np.linalg.norm(voxel_embedding) < 1e-8:
                return None
            
            # Normalize voxel embedding
            voxel_norm = np.linalg.norm(voxel_embedding)
            if voxel_norm < 1e-8:
                return None
            voxel_embedding_normalized = voxel_embedding / voxel_norm
            
            best_similarity = -1.0
            best_vlm_answer = None
            
            with self.vlm_embeddings_lock:
                vlm_embeddings_copy = self.vlm_embeddings.copy()
            
            # Compute similarity with all VLM embeddings
            for vlm_answer, vlm_embedding in vlm_embeddings_copy.items():
                try:
                    # Ensure compatible dimensions
                    min_dim = min(len(voxel_embedding_normalized), len(vlm_embedding))
                    voxel_emb_trunc = voxel_embedding_normalized[:min_dim]
                    vlm_emb_trunc = vlm_embedding[:min_dim]
                    
                    # Cosine similarity
                    similarity = float(np.dot(voxel_emb_trunc, vlm_emb_trunc))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_vlm_answer = vlm_answer
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error computing similarity with '{vlm_answer}': {e}")
                    continue
            
            # Check if best similarity exceeds threshold
            if best_similarity >= self.similarity_threshold and best_vlm_answer is not None:
                semantic_info = {
                    'vlm_answer': best_vlm_answer,
                    'similarity': best_similarity,
                    'timestamp': time.time()
                }
                
                # Store in voxel semantics
                with self.voxel_semantics_lock:
                    self.voxel_semantics[voxel_key] = semantic_info
                
                return semantic_info
            else:
                # Remove any existing semantic label if below threshold
                with self.voxel_semantics_lock:
                    if voxel_key in self.voxel_semantics:
                        del self.voxel_semantics[voxel_key]
                
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"Error computing voxel similarity for {voxel_key}: {e}")
            return None
    
    def get_voxel_color(self, voxel_key: Tuple[int, int, int], base_probability: float = 0.7) -> ColorRGBA:
        """
        Get color for voxel based on semantic labeling.
        
        Args:
            voxel_key: Voxel coordinate key (vx, vy, vz)
            base_probability: Base occupancy probability for alpha blending
            
        Returns:
            ColorRGBA message for visualization
        """
        try:
            with self.voxel_semantics_lock:
                semantic_info = self.voxel_semantics.get(voxel_key)
            
            if semantic_info is not None:
                # Semantic hotspot - use semantic color (green by default)
                similarity = semantic_info.get('similarity', 0.0)
                alpha = min(0.9, 0.6 + similarity * 0.3)  # Brighter for higher similarity
                return ColorRGBA(
                    r=float(self.semantic_color[0]), 
                    g=float(self.semantic_color[1]), 
                    b=float(self.semantic_color[2]), 
                    a=float(alpha)
                )
            else:
                # Regular voxel - use default color (grey by default)
                alpha = min(0.8, 0.5 + base_probability * 0.3)
                return ColorRGBA(
                    r=float(self.default_color[0]), 
                    g=float(self.default_color[1]), 
                    b=float(self.default_color[2]), 
                    a=float(alpha)
                )
                
        except Exception as e:
            if self.verbose:
                print(f"Error getting voxel color for {voxel_key}: {e}")
            # Fallback to default grey
            return ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)
    
    def update_voxel_semantics(self, voxel_metadata: Dict[Tuple[int, int, int], Dict[str, Any]]) -> int:
        """
        Update semantic labels for all voxels based on their embeddings.
        
        Args:
            voxel_metadata: Dict mapping voxel keys to metadata (must contain 'emb_mean')
            
        Returns:
            Number of voxels that received semantic labels
        """
        try:
            if not voxel_metadata:
                return 0
            
            semantic_count = 0
            
            for voxel_key, metadata in voxel_metadata.items():
                try:
                    voxel_embedding = metadata.get('emb_mean')
                    if voxel_embedding is not None:
                        semantic_info = self.compute_voxel_similarity(voxel_embedding, voxel_key)
                        if semantic_info is not None:
                            semantic_count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing voxel {voxel_key}: {e}")
                    continue
            
            return semantic_count
            
        except Exception as e:
            if self.verbose:
                print(f"Error updating voxel semantics: {e}")
            return 0
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get statistics about current semantic labeling."""
        try:
            with self.voxel_semantics_lock:
                semantic_voxels = len(self.voxel_semantics)
                
                # Count by VLM answer
                vlm_counts = {}
                similarities = []
                
                for semantic_info in self.voxel_semantics.values():
                    vlm_answer = semantic_info.get('vlm_answer', 'unknown')
                    vlm_counts[vlm_answer] = vlm_counts.get(vlm_answer, 0) + 1
                    similarities.append(semantic_info.get('similarity', 0.0))
            
            with self.vlm_embeddings_lock:
                available_embeddings = len(self.vlm_embeddings)
                vlm_answers = list(self.vlm_embeddings.keys())
            
            stats = {
                'semantic_voxels': semantic_voxels,
                'available_vlm_embeddings': available_embeddings,
                'vlm_answers': vlm_answers,
                'vlm_distribution': vlm_counts,
                'similarity_threshold': self.similarity_threshold,
                'avg_similarity': float(np.mean(similarities)) if similarities else 0.0,
                'max_similarity': float(np.max(similarities)) if similarities else 0.0,
                'min_similarity': float(np.min(similarities)) if similarities else 0.0
            }
            
            return stats
            
        except Exception as e:
            if self.verbose:
                print(f"Error getting semantic statistics: {e}")
            return {'error': str(e)}
    
    def load_vlm_embeddings_from_buffers(self, buffers_directory: str) -> int:
        """
        Load enhanced VLM embeddings from saved buffer directories.
        
        Args:
            buffers_directory: Path to buffers directory containing run folders
            
        Returns:
            Number of embeddings loaded
        """
        try:
            if not os.path.exists(buffers_directory):
                if self.verbose:
                    print(f"Buffers directory not found: {buffers_directory}")
                return 0
            
            embeddings_loaded = 0
            
            # Scan all run directories
            for run_dir in os.listdir(buffers_directory):
                run_path = os.path.join(buffers_directory, run_dir)
                if not os.path.isdir(run_path) or not run_dir.startswith('run_'):
                    continue
                
                # Scan buffer directories in this run
                for buffer_dir in os.listdir(run_path):
                    buffer_path = os.path.join(run_path, buffer_dir)
                    if not os.path.isdir(buffer_path) or buffer_dir == 'parallel_analysis_results':
                        continue
                    
                    # Look for enhanced embeddings
                    embeddings_dir = os.path.join(buffer_path, 'enhanced_embeddings')
                    if os.path.exists(embeddings_dir):
                        for emb_file in os.listdir(embeddings_dir):
                            if emb_file.startswith('enhanced_embedding_') and emb_file.endswith('.npy'):
                                try:
                                    # Extract VLM answer from filename
                                    vlm_name = emb_file.replace('enhanced_embedding_', '').replace('.npy', '')
                                    # Remove timestamp suffix if present
                                    if '_' in vlm_name and vlm_name.split('_')[-1].replace('.', '').isdigit():
                                        vlm_name = '_'.join(vlm_name.split('_')[:-1])
                                    
                                    vlm_answer = vlm_name.replace('_', ' ')
                                    
                                    # Load embedding
                                    emb_path = os.path.join(embeddings_dir, emb_file)
                                    enhanced_embedding = np.load(emb_path)
                                    
                                    if self.add_vlm_embedding(vlm_answer, enhanced_embedding):
                                        embeddings_loaded += 1
                                        if self.verbose:
                                            print(f"✓ Loaded embedding for '{vlm_answer}' from {emb_file}")
                                    
                                except Exception as e:
                                    if self.verbose:
                                        print(f"Error loading embedding from {emb_file}: {e}")
                                    continue
            
            if self.verbose:
                print(f"✓ Loaded {embeddings_loaded} VLM embeddings from buffer directories")
            return embeddings_loaded
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading VLM embeddings from buffers: {e}")
            return 0
    
    def clear_all_semantics(self):
        """Clear all semantic data."""
        try:
            with self.vlm_embeddings_lock:
                self.vlm_embeddings.clear()
            
            with self.voxel_semantics_lock:
                self.voxel_semantics.clear()
            
            if self.verbose:
                print("✓ Cleared all semantic data")
            
        except Exception as e:
            if self.verbose:
                print(f"Error clearing semantic data: {e}")
    
    def export_semantic_data(self, filepath: str) -> bool:
        """Export semantic mapping data to file."""
        try:
            semantic_data = {
                'timestamp': time.time(),
                'config': {
                    'similarity_threshold': self.similarity_threshold,
                    'semantic_color': self.semantic_color.tolist(),
                    'default_color': self.default_color.tolist(),
                    'embedding_dim': self.embedding_dim
                },
                'vlm_embeddings': {},
                'voxel_semantics': {},
                'statistics': self.get_semantic_statistics()
            }
            
            # Export VLM embeddings (just metadata, not the actual embeddings)
            with self.vlm_embeddings_lock:
                for vlm_answer, embedding in self.vlm_embeddings.items():
                    semantic_data['vlm_embeddings'][vlm_answer] = {
                        'shape': embedding.shape,
                        'norm': float(np.linalg.norm(embedding))
                    }
            
            # Export voxel semantics
            with self.voxel_semantics_lock:
                for voxel_key, semantic_info in self.voxel_semantics.items():
                    key_str = f"{voxel_key[0]}_{voxel_key[1]}_{voxel_key[2]}"
                    semantic_data['voxel_semantics'][key_str] = semantic_info
            
            with open(filepath, 'w') as f:
                json.dump(semantic_data, f, indent=2)
            
            if self.verbose:
                print(f"✓ Exported semantic data to {filepath}")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error exporting semantic data: {e}")
            return False 
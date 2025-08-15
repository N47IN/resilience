#!/usr/bin/env python3
"""
Drift Calculator Module

Handles trajectory drift calculations and breach detection.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any


class DriftCalculator:
    """Calculate drift between actual and nominal trajectories."""
    
    def __init__(self, nominal_traj_file: str):
        """Initialize drift calculator with nominal trajectory."""
        self.nominal_points, self.soft_threshold, self.hard_threshold = self.load_nominal_trajectory(nominal_traj_file)
        self.nominal_np = np.array([np.array([p['position']['x'], p['position']['y'], p['position']['z']]) for p in self.nominal_points])
        self.initial_pose = self.nominal_np[0] if len(self.nominal_np) > 0 else np.array([0, 0, 0])
        self.n_points = len(self.nominal_points)
        
    def load_nominal_trajectory(self, traj_file: str) -> Tuple[List[Dict], float, float]:
        """Load nominal trajectory from JSON file."""
        try:
            with open(traj_file, 'r') as f:
                data = json.load(f)
            points = data['points']
            soft_threshold = data['calibration']['soft_threshold']
            hard_threshold = data['calibration']['hard_threshold']
            return points, soft_threshold, hard_threshold
        except Exception as e:
            print(f"Warning: Could not load nominal trajectory from {traj_file}: {e}")
            # Return default values if file not found
            return [], 0.1, 0.5
    
    def compute_drift(self, pos: np.ndarray) -> Tuple[float, int]:
        """Compute drift between current position and nearest nominal point."""
        if len(self.nominal_np) == 0:
            return 0.0, 0
        dists = np.linalg.norm(self.nominal_np - pos, axis=1)
        nearest_idx = int(np.argmin(dists))
        drift = dists[nearest_idx]
        return drift, nearest_idx
    
    def is_breach(self, drift: float) -> bool:
        """Check if drift exceeds soft threshold (breach condition)."""
        return drift > self.soft_threshold
    
    def is_hard_breach(self, drift: float) -> bool:
        """Check if drift exceeds hard threshold (severe breach condition)."""
        return drift > self.hard_threshold
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get soft and hard thresholds."""
        return self.soft_threshold, self.hard_threshold
    
    def get_nominal_points(self) -> np.ndarray:
        """Get nominal trajectory points."""
        return self.nominal_np.copy()
    
    def get_initial_pose(self) -> np.ndarray:
        """Get initial pose (first nominal point)."""
        return self.initial_pose.copy() 
"""
Marker-controlled watershed segmentation for nuclei separation
"""

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Tuple, Union
import torch


class MarkerControlledWatershed:
    """Marker-controlled watershed for nuclei separation."""
    
    def __init__(self, min_distance: int = 5, min_size: int = 3, threshold: float = 0.1):
        self.min_distance = min_distance
        self.min_size = min_size
        self.threshold = threshold
    
    def create_markers(self, binary_mask: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
        """Create markers from distance map local maxima."""
        
        # Find local maxima in distance map
        local_maxima = peak_local_max(
            distance_map,
            min_distance=self.min_distance,
            threshold_abs=self.threshold
        )
        
        # Create marker image
        markers = np.zeros_like(binary_mask, dtype=int)
        height, width = binary_mask.shape
        for i, (y, x) in enumerate(local_maxima):
            # Ensure coordinates are within bounds
            if 0 <= y < height and 0 <= x < width:
                markers[y, x] = i + 1
        
        return markers
    
    def apply_watershed(self, binary_mask: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
        """Apply marker-controlled watershed segmentation."""
        
        # Create markers
        markers = self.create_markers(binary_mask, distance_map)
        
        # Apply watershed
        labels = watershed(-distance_map, markers, mask=binary_mask)
        
        # Filter small regions
        filtered_labels = self.filter_small_regions(labels)
        
        return filtered_labels
    
    def filter_small_regions(self, labels: np.ndarray) -> np.ndarray:
        """Filter out small regions."""
        filtered_labels = labels.copy()
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            mask = labels == label
            region_size = np.sum(mask)
            
            if region_size < self.min_size:
                filtered_labels[mask] = 0
        
        return filtered_labels

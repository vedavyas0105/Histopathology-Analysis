"""
Post-processing utilities for segmentation refinement
"""

import numpy as np
import torch
from typing import Union, Tuple
from scipy import ndimage


def remove_small_objects(labels: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove small objects from segmentation labels."""
    labeled, _ = ndimage.label(ndimage.binary_opening(labels > 0, structure=np.ones((3,3))))  # type: ignore
    return labeled


def fill_holes(labels: np.ndarray) -> np.ndarray:
    """Fill holes in segmentation masks."""
    filled_labels = labels.copy()
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        mask = labels == label
        filled_mask = ndimage.binary_fill_holes(mask)
        filled_labels[filled_mask] = label
    
    return filled_labels


def smooth_boundaries(labels: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Smooth segmentation boundaries."""
    smoothed_labels = labels.copy()
    
    for _ in range(iterations):
        smoothed_labels = ndimage.median_filter(smoothed_labels, size=3)
    
    return smoothed_labels


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Relabel segmentation to have sequential labels starting from 1."""
    unique_labels = np.unique(labels)
    relabeled = np.zeros_like(labels)
    
    for new_label, old_label in enumerate(unique_labels, 1):
        if old_label == 0:  # Keep background as 0
            continue
        relabeled[labels == old_label] = new_label
    
    return relabeled


def compute_nuclei_statistics(labels: np.ndarray) -> dict:
    """Compute statistics for segmented nuclei."""
    unique_labels = np.unique(labels)
    nuclei_count = len(unique_labels) - 1  # Exclude background
    
    if nuclei_count == 0:
        return {
            'count': 0,
            'areas': [],
            'mean_area': 0,
            'std_area': 0
        }
    
    areas = []
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        area = np.sum(labels == label)
        areas.append(area)
    
    areas = np.array(areas)
    
    return {
        'count': nuclei_count,
        'areas': areas,
        'mean_area': float(np.mean(areas)),
        'std_area': float(np.std(areas))
    }

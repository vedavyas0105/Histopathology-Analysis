"""
Simple visualization tools for segmentation results and debugging.
Essential for Phase 2 completion - no over-engineering.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import cv2


class SegmentationVisualizer:
    """Simple visualizer for segmentation results."""
    
    def __init__(self, figsize: tuple = (15, 5)):
        self.figsize = figsize
    
    def plot_segmentation_result(self, 
                                image: np.ndarray, 
                                binary_mask: np.ndarray,
                                watershed_labels: np.ndarray,
                                distance_map: np.ndarray,
                                title: str = "Segmentation Result",
                                save_path: str = None) -> None:
        """Plot complete segmentation pipeline result."""
        
        fig, axes = plt.subplots(1, 4, figsize=self.figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Binary mask
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title(f"Binary Mask\n({np.sum(binary_mask > 0)} pixels)")
        axes[1].axis('off')
        
        # Distance map
        axes[2].imshow(distance_map, cmap='viridis')
        axes[2].set_title(f"Distance Map\n(max: {distance_map.max():.2f})")
        axes[2].axis('off')
        
        # Watershed result
        axes[3].imshow(watershed_labels, cmap='nipy_spectral')
        axes[3].set_title(f"Watershed Labels\n({len(np.unique(watershed_labels))-1} nuclei)")
        axes[3].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Plot saved to: {save_path}")
        
        plt.show(block=True)
        plt.pause(0.1)
    
    def plot_nuclei_overlay(self, 
                           image: np.ndarray, 
                           watershed_labels: np.ndarray,
                           title: str = "Nuclei Detection") -> None:
        """Plot nuclei detection overlay on original image."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Overlay
        overlay = image.copy()
        unique_labels = np.unique(watershed_labels)
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == 0:  # Skip background
                continue
            mask = watershed_labels == label
            overlay[mask] = overlay[mask] * 0.7 + colors[i][:3] * 255 * 0.3
        
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(f"Nuclei Overlay ({len(unique_labels)-1} detected)")
        axes[1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show(block=True)
        plt.pause(0.1)
    
    def plot_debug_info(self, 
                       binary_mask: np.ndarray,
                       distance_map: np.ndarray,
                       markers: np.ndarray,
                       title: str = "Debug Info") -> None:
        """Plot debugging information for watershed."""
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Binary mask
        axes[0].imshow(binary_mask, cmap='gray')
        axes[0].set_title(f"Binary Mask\n({np.sum(binary_mask > 0)} pixels)")
        axes[0].axis('off')
        
        # Distance map with peaks
        axes[1].imshow(distance_map, cmap='viridis')
        axes[1].set_title(f"Distance Map\n(max: {distance_map.max():.2f})")
        axes[1].axis('off')
        
        # Markers
        axes[2].imshow(markers, cmap='tab10')
        axes[2].set_title(f"Markers\n({len(np.unique(markers))-1} markers)")
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show(block=True)
        plt.pause(0.1)
    
    def print_nuclei_stats(self, watershed_labels: np.ndarray) -> None:
        """Print nuclei statistics."""
        
        unique_labels = np.unique(watershed_labels)
        num_nuclei = len(unique_labels) - 1  # Exclude background (0)
        
        if num_nuclei == 0:
            print("❌ No nuclei detected")
            return
        
        areas = []
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            area = np.sum(watershed_labels == label)
            areas.append(area)
        
        areas = np.array(areas)
        
        print(f"📊 Nuclei Statistics:")
        print(f"   Count: {num_nuclei}")
        print(f"   Mean area: {areas.mean():.1f} pixels")
        print(f"   Min area: {areas.min():.1f} pixels")
        print(f"   Max area: {areas.max():.1f} pixels")
        print(f"   Std area: {areas.std():.1f} pixels")


def quick_visualize(image: np.ndarray, 
                   binary_mask: np.ndarray, 
                   watershed_labels: np.ndarray,
                   distance_map: np.ndarray,
                   save_path: str = None) -> None:
    """Quick visualization function for testing."""
    
    visualizer = SegmentationVisualizer()
    visualizer.plot_segmentation_result(image, binary_mask, watershed_labels, distance_map, save_path=save_path)
    visualizer.print_nuclei_stats(watershed_labels)

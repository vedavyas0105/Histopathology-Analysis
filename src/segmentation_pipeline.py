"""
Complete histopathology segmentation pipeline
"""

import torch
import numpy as np
from typing import Dict, Union, Optional

from .enhanced_unet import EnhancedUNet
from .watershed_segmentation import MarkerControlledWatershed
from .postprocessing_utils import relabel_sequential, compute_nuclei_statistics


class HistopathologySegmentationPipeline:
    """Complete segmentation pipeline for histopathology analysis."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        
        # Initialize components
        self.unet_model = EnhancedUNet()
        if model_path:
            self.unet_model.load_state_dict(torch.load(model_path, map_location=device))
        self.unet_model.to(device).eval()
        
        # Watershed segmentation method
        self.watershed = MarkerControlledWatershed()
    
    def process_patch(self, 
                     image: torch.Tensor, 
                     tissue_type: str = 'breast') -> Dict:
        """Process a single image patch through the complete pipeline."""
        
        with torch.no_grad():
            # Stage 1: U-Net prediction
            semantic_pred, instance_pred = self.unet_model(image)
            
            # Extract channels
            binary_mask = torch.sigmoid(semantic_pred[:, 0:1]).cpu().numpy()
            distance_map = instance_pred[:, 1:2].cpu().numpy()
            direction_map = instance_pred[:, 2:3].cpu().numpy()
            
            # Convert to numpy arrays
            binary_mask = binary_mask[0, 0]  # Remove batch and channel dims
            distance_map = distance_map[0, 0]
            direction_map = direction_map[0, 0]
            
            # Stage 2: Watershed segmentation
            final_labels = self.watershed.apply_watershed(binary_mask, distance_map)
            
            # Stage 3: Post-processing
            final_labels = relabel_sequential(final_labels)
            nuclei_stats = compute_nuclei_statistics(final_labels)
            
            return {
                'binary_mask': binary_mask,
                'distance_map': distance_map,
                'direction_map': direction_map,
                'final_labels': final_labels,
                'nuclei_count': nuclei_stats['count'],
                'nuclei_stats': nuclei_stats
            }
    
    def process_batch(self, 
                     images: torch.Tensor, 
                     tissue_types: Optional[list] = None) -> list:
        """Process a batch of images."""
        
        if tissue_types is None:
            tissue_types = ['breast'] * len(images)
        
        results = []
        for i, image in enumerate(images):
            image_batch = image.unsqueeze(0)  # Add batch dimension
            result = self.process_patch(image_batch, tissue_types[i])
            results.append(result)
        
        return results

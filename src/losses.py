"""
Loss functions for histopathology segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Compute edges
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        
        return F.mse_loss(pred_edges, target_edges)

    def _compute_edges(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        sobel_x = self.sobel_x.to(x.device).view(1, 1, 3, 3)
        sobel_y = self.sobel_y.to(x.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(x, sobel_x, padding=1)
        edges_y = F.conv2d(x, sobel_y, padding=1)
        
        return torch.sqrt(edges_x**2 + edges_y**2)


class DualLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.4, edge_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss()
        
    def forward(self, predictions, targets, edges=None):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        if edges is not None and self.edge_weight > 0:
            edge = self.edge_loss(predictions, edges)
            total_loss += self.edge_weight * edge
            return total_loss, {'bce': bce, 'dice': dice, 'edge': edge}
        
        return total_loss, {'bce': bce, 'dice': dice}

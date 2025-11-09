"""
Evaluation metrics for instance segmentation
"""

import numpy as np
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment


def compute_aji(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Compute Aggregated Jaccard Index (AJI)."""
    
    pred_unique = np.unique(pred_labels)
    gt_unique = np.unique(gt_labels)
    
    # Remove background
    pred_unique = pred_unique[pred_unique > 0]
    gt_unique = gt_unique[gt_unique > 0]
    
    if len(pred_unique) == 0 and len(gt_unique) == 0:
        return 1.0
    if len(pred_unique) == 0 or len(gt_unique) == 0:
        return 0.0
    
    # Compute intersection matrix
    intersection_matrix = np.zeros((len(pred_unique), len(gt_unique)))
    
    for i, pred_label in enumerate(pred_unique):
        pred_mask = pred_labels == pred_label
        for j, gt_label in enumerate(gt_unique):
            gt_mask = gt_labels == gt_label
            intersection_matrix[i, j] = np.sum(pred_mask & gt_mask)
    
    # Hungarian algorithm for optimal matching
    row_indices, col_indices = linear_sum_assignment(-intersection_matrix)
    
    # Compute AJI
    total_intersection = 0
    total_union = 0
    
    for i, j in zip(row_indices, col_indices):
        pred_label = pred_unique[i]
        gt_label = gt_unique[j]
        
        pred_mask = pred_labels == pred_label
        gt_mask = gt_labels == gt_label
        
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)
        
        total_intersection += intersection
        total_union += union
    
    # Add unmatched predictions and ground truth
    matched_pred = set(row_indices)
    matched_gt = set(col_indices)
    
    for i in range(len(pred_unique)):
        if i not in matched_pred:
            pred_mask = pred_labels == pred_unique[i]
            total_union += np.sum(pred_mask)
    
    for j in range(len(gt_unique)):
        if j not in matched_gt:
            gt_mask = gt_labels == gt_unique[j]
            total_union += np.sum(gt_mask)
    
    if total_union == 0:
        return 0.0
    
    return float(total_intersection / total_union)  # type: ignore


def compute_dice_coefficient(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Compute Dice coefficient for instance segmentation."""
    
    pred_binary = pred_labels > 0
    gt_binary = gt_labels > 0
    
    intersection = np.sum(pred_binary & gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union  # pyright: ignore[reportOperatorIssue, reportReturnType]


def compute_precision_recall(pred_labels: np.ndarray, gt_labels: np.ndarray, 
                           iou_threshold: float = 0.5) -> Tuple[float, float]:
    """Compute precision and recall for instance segmentation."""
    
    pred_unique = np.unique(pred_labels)
    gt_unique = np.unique(gt_labels)
    
    # Remove background
    pred_unique = pred_unique[pred_unique > 0]
    gt_unique = gt_unique[gt_unique > 0]
    
    if len(pred_unique) == 0 and len(gt_unique) == 0:
        return 1.0, 1.0
    if len(pred_unique) == 0:
        return 0.0, 0.0
    if len(gt_unique) == 0:
        return 0.0, 1.0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_unique), len(gt_unique)))
    
    for i, pred_label in enumerate(pred_unique):
        pred_mask = pred_labels == pred_label
        for j, gt_label in enumerate(gt_unique):
            gt_mask = gt_labels == gt_label
            
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            
            if union > 0:
                iou_matrix[i, j] = intersection / union
    
    # Count matches above threshold
    matches = iou_matrix >= iou_threshold
    matched_pred = np.any(matches, axis=1)
    matched_gt = np.any(matches, axis=0)
    
    precision = np.sum(matched_pred) / len(pred_unique)
    recall = np.sum(matched_gt) / len(gt_unique)
    
    return precision, recall


def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_per_class_metrics(pred_labels: np.ndarray, gt_labels: np.ndarray) -> Dict[str, float]:
    """Compute per-class metrics (like the repo implementation)."""
    
    # Convert to binary for per-class calculation
    pred_binary = pred_labels > 0
    gt_binary = gt_labels > 0
    
    # Intersection and Union
    intersection = np.sum(pred_binary & gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection  # type: ignore
    
    # Dice coefficient
    if union == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = 2.0 * intersection / union
    
    # IoU (Intersection over Union)
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
    
    # Precision and Recall
    if np.sum(pred_binary) == 0:
        precision = 0.0
    else:
        precision = intersection / np.sum(pred_binary)
    
    if np.sum(gt_binary) == 0:
        recall = 0.0
    else:
        recall = intersection / np.sum(gt_binary)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }


def evaluate_segmentation(pred_labels: np.ndarray, gt_labels: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics using binary ground truth."""
    
    # Use binary metrics since ground truth is binary mask
    dice = compute_dice_coefficient(pred_labels, gt_labels)
    per_class = compute_per_class_metrics(pred_labels, gt_labels)
    
    # For binary ground truth, use per-class metrics as main metrics
    precision = per_class['precision']
    recall = per_class['recall']
    f1 = compute_f1_score(precision, recall)
    
    # AJI is not meaningful with binary ground truth, so we'll use a proxy
    # Use IoU as a proxy for instance-level quality
    aji_proxy = per_class['iou']
    
    return {
        'aji': aji_proxy,  # Using IoU as proxy since GT is binary
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_dice': per_class['dice'],
        'per_class_iou': per_class['iou'],
        'per_class_precision': per_class['precision'],
        'per_class_recall': per_class['recall']
    }

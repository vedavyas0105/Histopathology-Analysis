"""
Test script for complete U-Net + Watershed pipeline
Tests the trained model with real PanNuke data and validates resume claims
"""

import torch
import numpy as np
import time
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import config
from src.data_pipeline import PanNukeDataset
from src.enhanced_unet import EnhancedUNet
from src.watershed_segmentation import MarkerControlledWatershed
from src.segmentation_pipeline import SegmentationPipeline
from src.evaluation_metrics import evaluate_segmentation
from src.visualization_tools import SegmentationVisualizer

class CompletePipelineTester:
    """Test the complete U-Net + Watershed pipeline."""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.watershed = MarkerControlledWatershed()
        self.pipeline = SegmentationPipeline(self.model, self.watershed)
        self.visualizer = SegmentationVisualizer()
        
    def _load_model(self, model_path):
        """Load trained U-Net model."""
        if model_path is None:
            model_path = config.CHECKPOINT_DIR / 'best_model.pth'
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("   Please train the model first using train_unet.py")
            return None
        
        # Load model
        model = EnhancedUNet(
            encoder_name=config.ENCODER_NAME,
            num_classes=config.NUM_CLASSES
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best Dice: {checkpoint.get('best_dice', 'Unknown'):.4f}")
        
        return model
    
    def test_single_sample(self, dataset, sample_idx=0):
        """Test pipeline on a single sample."""
        print(f"\nTesting sample {sample_idx}...")
        
        # Get sample
        sample = dataset[sample_idx]
        image = sample['image']
        mask = sample['mask']
        
        # Convert to numpy for processing
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = image
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask
        
        # Ensure image is in correct format
        if image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))  # CHW -> HWC
        
        # Normalize image to 0-255
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Process with pipeline
        start_time = time.time()
        
        with torch.no_grad():
            # U-Net prediction
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(self.device)
            
            main_output, instance_output = self.model(image_tensor)
            # Use nuclei channel (channel 1) for prediction
            pred_mask = torch.sigmoid(main_output[:, 1:2, :, :]).squeeze().cpu().numpy()
        
        # Watershed segmentation
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        distance_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        watershed_labels = self.watershed.apply_watershed(binary_mask, distance_map)
        
        inference_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_segmentation(watershed_labels, mask_np)
        
        # Print results
        print(f"   Inference Time: {inference_time:.3f}s")
        print(f"   Dice Score: {metrics['dice']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   Nuclei Count: {len(np.unique(watershed_labels)) - 1}")
        
        return {
            'image': image_np,
            'ground_truth': mask_np,
            'prediction': pred_mask,
            'binary_mask': binary_mask,
            'watershed_labels': watershed_labels,
            'metrics': metrics,
            'inference_time': inference_time
        }
    
    def test_multiple_samples(self, dataset, num_samples=10):
        """Test pipeline on multiple samples."""
        print(f"\nTesting {num_samples} samples...")
        
        results = []
        total_time = 0.0
        dice_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        nuclei_counts = []
        
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Testing"):
            try:
                result = self.test_single_sample(dataset, i)
                results.append(result)
                
                total_time += result['inference_time']
                dice_scores.append(result['metrics']['dice'])
                precision_scores.append(result['metrics']['precision'])
                recall_scores.append(result['metrics']['recall'])
                f1_scores.append(result['metrics']['f1_score'])
                nuclei_counts.append(len(np.unique(result['watershed_labels'])) - 1)
                
            except Exception as e:
                print(f"   ❌ Error processing sample {i}: {e}")
                continue
        
        # Calculate averages
        avg_time = total_time / len(results)
        avg_dice = np.mean(dice_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_nuclei = np.mean(nuclei_counts)
        
        print(f"\nAverage Results ({len(results)} samples):")
        print(f"   Inference Time: {avg_time:.3f}s")
        print(f"   Dice Score: {avg_dice:.4f}")
        print(f"   Precision: {avg_precision:.4f}")
        print(f"   Recall: {avg_recall:.4f}")
        print(f"   F1 Score: {avg_f1:.4f}")
        print(f"   Nuclei Count: {avg_nuclei:.1f}")
        
        return {
            'results': results,
            'avg_time': avg_time,
            'avg_dice': avg_dice,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'avg_nuclei': avg_nuclei,
            'dice_scores': dice_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'nuclei_counts': nuclei_counts
        }
    
    def validate_resume_claims(self, test_results):
        """Validate resume claims against test results."""
        print(f"\nValidating Resume Claims:")
        print("-" * 50)
        
        claims = {
            "Nuclei Detection Accuracy": {
                "target": config.TARGET_NUCLEI_ACCURACY,
                "actual": test_results['avg_dice'],
                "status": test_results['avg_dice'] >= config.TARGET_NUCLEI_ACCURACY
            },
            "Dice Coefficient": {
                "target": config.TARGET_DICE_COEFFICIENT,
                "actual": test_results['avg_dice'],
                "status": test_results['avg_dice'] >= config.TARGET_DICE_COEFFICIENT
            },
            "Inference Time": {
                "target": config.TARGET_INFERENCE_TIME,
                "actual": test_results['avg_time'],
                "status": test_results['avg_time'] <= config.TARGET_INFERENCE_TIME
            }
        }
        
        all_passed = True
        for claim, data in claims.items():
            status = "✅ PASSED" if data['status'] else "❌ FAILED"
            print(f"   {claim}: {status}")
            print(f"      Target: {data['target']:.3f}")
            print(f"      Actual: {data['actual']:.3f}")
            if not data['status']:
                all_passed = False
        
        print(f"\nOverall Status: {'ALL CLAIMS VALIDATED' if all_passed else 'SOME CLAIMS FAILED'}")
        return all_passed
    
    def visualize_results(self, test_results, save_dir="results"):
        """Visualize test results."""
        print(f"\nCreating visualizations...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Visualize first few samples
        num_viz = min(3, len(test_results['results']))
        
        for i in range(num_viz):
            result = test_results['results'][i]
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(result['image'])
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Ground truth
            axes[0, 1].imshow(result['ground_truth'], cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # U-Net prediction
            axes[0, 2].imshow(result['prediction'], cmap='gray')
            axes[0, 2].set_title('U-Net Prediction')
            axes[0, 2].axis('off')
            
            # Binary mask
            axes[1, 0].imshow(result['binary_mask'], cmap='gray')
            axes[1, 0].set_title('Binary Mask')
            axes[1, 0].axis('off')
            
            # Watershed labels
            axes[1, 1].imshow(result['watershed_labels'], cmap='nipy_spectral')
            axes[1, 1].set_title('Watershed Labels')
            axes[1, 1].axis('off')
            
            # Overlay
            overlay = result['image'].copy()
            overlay[result['watershed_labels'] > 0] = [255, 0, 0]  # Red for nuclei
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Nuclei Overlay')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = save_path / f'pipeline_sample_{i}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_path}")
            plt.close()
        
        # Create metrics plot
        self._plot_metrics(test_results, save_path)
        
        print(f"All visualizations saved to: {save_path}")
    
    def _plot_metrics(self, test_results, save_path):
        """Plot metrics distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Dice scores
        axes[0, 0].hist(test_results['dice_scores'], bins=10, alpha=0.7, color='blue')
        axes[0, 0].axvline(test_results['avg_dice'], color='red', linestyle='--', label=f'Avg: {test_results["avg_dice"]:.3f}')
        axes[0, 0].set_title('Dice Scores Distribution')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Precision scores
        axes[0, 1].hist(test_results['precision_scores'], bins=10, alpha=0.7, color='green')
        axes[0, 1].axvline(test_results['avg_precision'], color='red', linestyle='--', label=f'Avg: {test_results["avg_precision"]:.3f}')
        axes[0, 1].set_title('Precision Scores Distribution')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Recall scores
        axes[1, 0].hist(test_results['recall_scores'], bins=10, alpha=0.7, color='orange')
        axes[1, 0].axvline(test_results['avg_recall'], color='red', linestyle='--', label=f'Avg: {test_results["avg_recall"]:.3f}')
        axes[1, 0].set_title('Recall Scores Distribution')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Nuclei counts
        axes[1, 1].hist(test_results['nuclei_counts'], bins=10, alpha=0.7, color='purple')
        axes[1, 1].axvline(test_results['avg_nuclei'], color='red', linestyle='--', label=f'Avg: {test_results["avg_nuclei"]:.1f}')
        axes[1, 1].set_title('Nuclei Count Distribution')
        axes[1, 1].set_xlabel('Nuclei Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_path / 'metrics_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

def main():
    """Main testing function."""
    print("Testing Complete U-Net + Watershed Pipeline")
    print("=" * 60)
    
    # Check if model exists
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    if not model_path.exists():
        print("No trained model found!")
        print("Please run train_unet.py first to train the model.")
        return
    
    # Load dataset
    print("Loading PanNuke dataset...")
    dataset = PanNukeDataset(
        data_path=config.PANNUKE_PATH,
        patch_size=config.PATCH_SIZE,
        split='test'
    )
    
    if len(dataset) == 0:
        print("No test samples found!")
        return
    
    print(f"Found {len(dataset)} test samples")
    
    # Initialize tester
    tester = CompletePipelineTester(model_path)
    
    if tester.model is None:
        return
    
    # Test pipeline
    test_results = tester.test_multiple_samples(dataset, num_samples=25)
    
    # Validate resume claims
    claims_validated = tester.validate_resume_claims(test_results)
    
    # Create visualizations
    tester.visualize_results(test_results)
    
    # Final summary
    print(f"\nFinal Summary:")
    print(f"Samples Tested: {len(test_results['results'])}")
    print(f"Average Dice: {test_results['avg_dice']:.4f}")
    print(f"Average Time: {test_results['avg_time']:.3f}s")
    print(f"Resume Claims: {'VALIDATED' if claims_validated else 'FAILED'}")
    
    return test_results

if __name__ == "__main__":
    main()

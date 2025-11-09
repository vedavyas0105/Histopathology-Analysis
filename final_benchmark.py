"""
Final comprehensive benchmark for the complete histopathology analysis pipeline
Tests U-Net + Watershed on PanNuke dataset and validates all resume claims
"""

import torch
import numpy as np
import time
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import config
from src.data_pipeline import PanNukeDataset
from src.enhanced_unet import EnhancedUNet
from src.watershed_segmentation import MarkerControlledWatershed
# from src.segmentation_pipeline import SegmentationPipeline
from src.evaluation_metrics import evaluate_segmentation
from src.visualization_tools import SegmentationVisualizer

class FinalBenchmark:
    """Comprehensive benchmark for the complete pipeline."""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize results FIRST
        self.results = {
            'model_info': {},
            'performance_metrics': {},
            'resume_claims': {},
            'detailed_results': [],
            'summary': {}
        }
        
        # Then load model
        self.model = self._load_model(model_path)
        self.watershed = MarkerControlledWatershed()
        # self.pipeline = SegmentationPipeline(self.model, self.watershed)
        self.visualizer = SegmentationVisualizer()
        
    def _load_model(self, model_path):
        """Load trained U-Net model."""
        if model_path is None:
            model_path = config.CHECKPOINT_DIR / 'best_model.pth'
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
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
        
        # Store model info
        self.results['model_info'] = {
            'path': str(model_path),
            'encoder': config.ENCODER_NAME,
            'num_classes': config.NUM_CLASSES,
            'best_dice': checkpoint.get('best_dice', 0.0),
            'epoch': checkpoint.get('epoch', 0),
            'device': str(self.device)
        }
        
        print(f"Model loaded: {config.ENCODER_NAME}")
        print(f"Best Dice: {checkpoint.get('best_dice', 0.0):.4f}")
        print(f"Device: {self.device}")
        
        return model
    
    def benchmark_performance(self, dataset, num_samples=50):
        """Benchmark performance on multiple samples."""
        print(f"\nBenchmarking performance on {num_samples} samples...")
        
        # Performance tracking
        inference_times = []
        dice_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        nuclei_counts = []
        memory_usage = []
        
        # Test samples
        test_samples = min(num_samples, len(dataset))
        
        for i in tqdm(range(test_samples), desc="Benchmarking"):
            try:
                # Get sample
                sample = dataset[i]
                image = sample['image']
                mask = sample['mask']
                
                # Convert to numpy
                if isinstance(image, torch.Tensor):
                    image_np = image.squeeze().cpu().numpy()
                else:
                    image_np = image
                
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.squeeze().cpu().numpy()
                else:
                    mask_np = mask
                
                # Ensure correct format
                if image_np.ndim == 3:
                    image_np = np.transpose(image_np, (1, 2, 0))
                
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                # Measure inference time
                start_time = time.time()
                
                with torch.no_grad():
                    # U-Net prediction
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    image_tensor = image_tensor.to(self.device)
                    
                    if self.model is not None:
                        main_output, instance_output = self.model(image_tensor)
                        # Use nuclei channel (channel 1) for prediction
                        pred_mask = torch.sigmoid(main_output[:, 1:2, :, :]).squeeze().cpu().numpy()
                    else:
                        continue
                
                # Watershed segmentation
                binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                distance_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
                watershed_labels = self.watershed.apply_watershed(binary_mask, distance_map)
                
                inference_time = time.time() - start_time
                
                # Evaluate metrics
                metrics = evaluate_segmentation(watershed_labels, mask_np)
                
                # Track memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1e9)
                
                # Store results
                inference_times.append(inference_time)
                dice_scores.append(metrics['dice'])
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
                f1_scores.append(metrics['f1_score'])
                nuclei_counts.append(len(np.unique(watershed_labels)) - 1)
                
                # Store detailed result
                self.results['detailed_results'].append({
                    'sample_id': i,
                    'inference_time': inference_time,
                    'dice': metrics['dice'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'nuclei_count': len(np.unique(watershed_labels)) - 1,
                    'memory_gb': memory_usage[-1] if memory_usage else 0.0
                })
                
            except Exception as e:
                print(f"   ❌ Error processing sample {i}: {e}")
                continue
        
        # Calculate performance metrics
        self.results['performance_metrics'] = {
            'num_samples': len(inference_times),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'avg_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'avg_precision': np.mean(precision_scores),
            'std_precision': np.std(precision_scores),
            'avg_recall': np.mean(recall_scores),
            'std_recall': np.std(recall_scores),
            'avg_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'avg_nuclei_count': np.mean(nuclei_counts),
            'std_nuclei_count': np.std(nuclei_counts),
            'avg_memory_gb': np.mean(memory_usage) if memory_usage else 0.0,
            'max_memory_gb': np.max(memory_usage) if memory_usage else 0.0
        }
        
        # Print results
        print(f"\nPerformance Results:")
        print(f"   Samples Processed: {len(inference_times)}")
        print(f"   Average Inference Time: {np.mean(inference_times):.3f}s")
        print(f"   Standard Deviation: {np.std(inference_times):.3f}s")
        print(f"   Min/Max Time: {np.min(inference_times):.3f}s / {np.max(inference_times):.3f}s")
        print(f"   Average Dice Score: {np.mean(dice_scores):.4f}")
        print(f"   Average Precision: {np.mean(precision_scores):.4f}")
        print(f"   Average Recall: {np.mean(recall_scores):.4f}")
        print(f"   Average F1 Score: {np.mean(f1_scores):.4f}")
        print(f"   Average Nuclei Count: {np.mean(nuclei_counts):.1f}")
        
        if memory_usage:
            print(f"   Average Memory Usage: {np.mean(memory_usage):.2f} GB")
            print(f"   Peak Memory Usage: {np.max(memory_usage):.2f} GB")
        
        return self.results['performance_metrics']
    
    def validate_resume_claims(self):
        """Validate all resume claims against benchmark results."""
        print(f"\nValidating Resume Claims:")
        print("-" * 50)
        
        metrics = self.results['performance_metrics']
        
        # Resume claims validation
        claims = {
            "Nuclei Detection Accuracy >= 89%": {
                "target": config.TARGET_NUCLEI_ACCURACY,
                "actual": metrics['avg_dice'],
                "status": metrics['avg_dice'] >= config.TARGET_NUCLEI_ACCURACY,
                "description": "Dice coefficient for nuclei detection"
            },
            "Dice Coefficient >= 0.87": {
                "target": config.TARGET_DICE_COEFFICIENT,
                "actual": metrics['avg_dice'],
                "status": metrics['avg_dice'] >= config.TARGET_DICE_COEFFICIENT,
                "description": "Overall segmentation quality"
            },
            "Inference Time <= 1.0s": {
                "target": config.TARGET_INFERENCE_TIME,
                "actual": metrics['avg_inference_time'],
                "status": metrics['avg_inference_time'] <= config.TARGET_INFERENCE_TIME,
                "description": "Per-patch processing time"
            },
            "Sub-second Inference": {
                "target": 1.0,
                "actual": metrics['avg_inference_time'],
                "status": metrics['avg_inference_time'] < 1.0,
                "description": "Fast inference for real-time applications"
            }
        }
        
        # Print claim validation
        all_passed = True
        for claim, data in claims.items():
            status = "✅ PASSED" if data['status'] else "❌ FAILED"
            print(f"   {claim}: {status}")
            print(f"      Target: {data['target']:.3f}")
            print(f"      Actual: {data['actual']:.3f}")
            print(f"      Description: {data['description']}")
            if not data['status']:
                all_passed = False
            print()
        
        # Store results
        self.results['resume_claims'] = claims
        
        # Overall status
        print(f"Overall Resume Validation: {'ALL CLAIMS VALIDATED' if all_passed else 'SOME CLAIMS FAILED'}")
        
        return all_passed
    
    def generate_summary(self):
        """Generate comprehensive summary."""
        print(f"\nComprehensive Summary:")
        print("=" * 60)
        
        # Model info
        model_info = self.results['model_info']
        print(f"Model Information:")
        print(f"   Architecture: {model_info['encoder']}")
        print(f"   Classes: {model_info['num_classes']}")
        print(f"   Training Dice: {model_info['best_dice']:.4f}")
        print(f"   Device: {model_info['device']}")
        
        # Performance metrics
        metrics = self.results['performance_metrics']
        print(f"\nPerformance Metrics:")
        print(f"   Samples Tested: {metrics['num_samples']}")
        print(f"   Average Inference Time: {metrics['avg_inference_time']:.3f}s ± {metrics['std_inference_time']:.3f}s")
        print(f"   Average Dice Score: {metrics['avg_dice']:.4f} ± {metrics['std_dice']:.4f}")
        print(f"   Average Precision: {metrics['avg_precision']:.4f} ± {metrics['std_precision']:.4f}")
        print(f"   Average Recall: {metrics['avg_recall']:.4f} ± {metrics['std_recall']:.4f}")
        print(f"   Average F1 Score: {metrics['avg_f1']:.4f} ± {metrics['std_f1']:.4f}")
        print(f"   Average Nuclei Count: {metrics['avg_nuclei_count']:.1f} ± {metrics['std_nuclei_count']:.1f}")
        
        if metrics['avg_memory_gb'] > 0:
            print(f"   Memory Usage: {metrics['avg_memory_gb']:.2f} GB (peak: {metrics['max_memory_gb']:.2f} GB)")
        
        # Resume claims
        claims = self.results['resume_claims']
        passed_claims = sum(1 for claim in claims.values() if claim['status'])
        total_claims = len(claims)
        
        print(f"\nResume Claims Validation:")
        print(f"   Passed: {passed_claims}/{total_claims}")
        print(f"   Success Rate: {passed_claims/total_claims*100:.1f}%")
        
        # Overall assessment
        overall_success = passed_claims == total_claims
        print(f"\nOverall Assessment: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
        
        # Store summary
        self.results['summary'] = {
            'overall_success': overall_success,
            'claims_passed': passed_claims,
            'total_claims': total_claims,
            'success_rate': passed_claims/total_claims*100,
            'avg_dice': metrics['avg_dice'],
            'avg_inference_time': metrics['avg_inference_time'],
            'num_samples_tested': metrics['num_samples']
        }
        
        return self.results['summary']
    
    def save_results(self, output_path="results"):
        """Save benchmark results to file."""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary as text
        summary_file = output_dir / "benchmark_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("HISTOPATHOLOGY ANALYSIS PIPELINE - BENCHMARK RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Model info
            model_info = self.results['model_info']
            f.write(f"Model: {model_info['encoder']}\n")
            f.write(f"Device: {model_info['device']}\n")
            f.write(f"Training Dice: {model_info['best_dice']:.4f}\n\n")
            
            # Performance metrics
            metrics = self.results['performance_metrics']
            f.write(f"Samples Tested: {metrics['num_samples']}\n")
            f.write(f"Average Inference Time: {metrics['avg_inference_time']:.3f}s\n")
            f.write(f"Average Dice Score: {metrics['avg_dice']:.4f}\n")
            f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n")
            f.write(f"Average Recall: {metrics['avg_recall']:.4f}\n")
            f.write(f"Average F1 Score: {metrics['avg_f1']:.4f}\n\n")
            
            # Resume claims
            claims = self.results['resume_claims']
            f.write("Resume Claims Validation:\n")
            for claim, data in claims.items():
                status = "PASSED" if data['status'] else "FAILED"
                f.write(f"  {claim}: {status}\n")
            
            # Overall assessment
            summary = self.results['summary']
            f.write(f"\nOverall Success: {'YES' if summary['overall_success'] else 'NO'}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
        
        print(f"Results saved to: {output_dir}")
        print(f"Detailed: {results_file}")
        print(f"Summary: {summary_file}")

def main():
    """Main benchmark function."""
    print("Final Comprehensive Benchmark")
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
        data_path=str(config.PANNUKE_PATH),
        patch_size=config.PATCH_SIZE,
        split='test'
    )
    
    if len(dataset) == 0:
        print("No test samples found!")
        return
    
    print(f"Found {len(dataset)} test samples")
    
    # Initialize benchmark
    benchmark = FinalBenchmark(model_path)
    
    if benchmark.model is None:
        return
    
    # Run benchmark
    print(f"\nStarting comprehensive benchmark...")
    benchmark.benchmark_performance(dataset, num_samples=50)
    
    # Validate resume claims
    claims_validated = benchmark.validate_resume_claims()
    
    # Generate summary
    summary = benchmark.generate_summary()
    
    # Save results
    benchmark.save_results()
    
    # Final status
    print(f"\nFinal Status:")
    print(f"Resume Claims: {'VALIDATED' if claims_validated else 'FAILED'}")
    print(f"Overall Success: {'YES' if summary['overall_success'] else 'NO'}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    return benchmark.results

if __name__ == "__main__":
    main()

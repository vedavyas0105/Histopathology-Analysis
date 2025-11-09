"""
Configuration file for Histopathology Analysis Project
Contains all hyperparameters, paths, and training settings
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for the histopathology project."""
    
    # ==================== PROJECT PATHS ====================
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / "data"
    PATCHCAMELYON_PATH = DATA_ROOT / "patchcamelyon"
    PANNUKE_PATH = DATA_ROOT / "raw" / "folds"
    
    # Output directories (will be created automatically)
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"
    SRC_DIR = PROJECT_ROOT / "src"
    
    # ==================== DATA PARAMETERS ====================
    PATCH_SIZE = 96
    BATCH_SIZE = 16
    NUM_WORKERS = 0  # Set to 0 for Windows multiprocessing compatibility
    PIN_MEMORY = True
    
    # Dataset splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # ==================== MODEL PARAMETERS ====================
    # U-Net Architecture
    ENCODER_NAME = 'efficientnet-b7'
    ENCODER_WEIGHTS = 'imagenet'
    NUM_CLASSES = 2  # Background, Tumor (for PatchCamelyon binary classification)
    
    # SE Block parameters
    SE_REDUCTION = 16
    
    # Instance segmentation head
    INSTANCE_HEAD_CHANNELS = 64
    
    # ==================== TRAINING PARAMETERS ====================
    # Learning rates - CPU optimized
    INITIAL_LR = 2e-4  # Slightly higher for better convergence with fewer epochs
    MIN_LR = 1e-7
    LR_SCHEDULER_PATIENCE = 8  # Reduced for faster adaptation with 10 epochs
    
    # Training settings - CPU optimized
    EPOCHS = 10
    PATIENCE = 5
    WEIGHT_DECAY = 1e-4  # Slightly higher for better regularization
    GRADIENT_CLIP_VAL = 1.0
    
    # Mixed precision training - disabled for CPU
    USE_AMP = False  # Disabled for CPU training (not supported)
    
    # ==================== LOSS FUNCTION WEIGHTS ====================
    BCE_WEIGHT = 0.4
    DICE_WEIGHT = 0.4
    EDGE_WEIGHT = 0.2
    
    # Class weights for handling imbalance
    CLASS_WEIGHTS = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0]  # Background gets lower weight
    
    # ==================== DATA AUGMENTATION ====================
    # Geometric transforms
    ROTATION_LIMIT = 90
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
    
    # Intensity transforms
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    HUE_LIMIT = 0.1
    SATURATION_LIMIT = 0.1
    
    # Noise and blur
    GAUSSIAN_NOISE_VAR = 0.01
    GAUSSIAN_BLUR_PROB = 0.1
    
    # ==================== DBSCAN PARAMETERS ====================
    # Tissue-specific DBSCAN parameters
    DBSCAN_PARAMS = {
        'breast': {'eps': 7, 'min_samples': 4},
        'colon': {'eps': 6, 'min_samples': 5},
        'lung': {'eps': 8, 'min_samples': 3},
        'prostate': {'eps': 6, 'min_samples': 4},
        'stomach': {'eps': 7, 'min_samples': 5}
    }
    
    # Density thresholds for adaptive parameters
    DENSE_TISSUE_THRESHOLD = 0.4
    SPARSE_TISSUE_THRESHOLD = 0.2
    
    # ==================== WATERSHED PARAMETERS ====================
    MIN_DISTANCE = 5  # Optimized for better nuclei detection
    MIN_NUCLEI_SIZE = 3  # Reduced to detect smaller nuclei
    WATERSHED_THRESHOLD = 0.1  # Optimized threshold
    
    # ==================== EVALUATION METRICS ====================
    # Target metrics for validation
    TARGET_ACCURACY = 0.90
    TARGET_NUCLEI_ACCURACY = 0.89
    TARGET_DICE_COEFFICIENT = 0.87
    TARGET_INFERENCE_TIME = 1.0  # seconds per patch
    
    # Evaluation thresholds
    BINARY_THRESHOLD = 0.5
    MIN_NUCLEI_AREA = 10
    
    # ==================== INFERENCE OPTIMIZATION ====================
    # Batch processing
    INFERENCE_BATCH_SIZE = 32
    SLIDING_WINDOW_OVERLAP = 0.5
    
    # Model optimization
    USE_FP16 = True
    PRUNING_SPARSITY = 0.25
    
    # ==================== LOGGING & MONITORING ====================
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    VALIDATION_INTERVAL = 1
    
    # TensorBoard logging
    USE_TENSORBOARD = True
    TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"
    
    # Weights & Biases (optional)
    USE_WANDB = False
    WANDB_PROJECT = "histopathology-analysis"
    
    # ==================== RANDOM SEEDS ====================
    RANDOM_SEED = 42
    TORCH_SEED = 42
    NUMPY_SEED = 42
    
    # ==================== DEVICE CONFIGURATION ====================
    DEVICE = "cpu"  # Force CPU training
    MULTI_GPU = False  # Set to True if using multiple GPUs
    
    # ==================== CLASS DEFINITIONS ====================
    CLASSES = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    CLASS_COLORS = [(0, 0, 0), (255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # ==================== UTILITY METHODS ====================
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
            cls.SRC_DIR,
            cls.TENSORBOARD_LOG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_dbscan_params(cls, tissue_type, nuclei_density):
        """Get DBSCAN parameters based on tissue type and density."""
        base_params = cls.DBSCAN_PARAMS.get(tissue_type, {'eps': 7, 'min_samples': 4})
        
        # Adjust based on density
        if nuclei_density > cls.DENSE_TISSUE_THRESHOLD:
            base_params['min_samples'] += 2
        elif nuclei_density < cls.SPARSE_TISSUE_THRESHOLD:
            base_params['min_samples'] = max(2, base_params['min_samples'] - 1)
        
        return base_params
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 50)
        print("HISTOPATHOLOGY ANALYSIS CONFIGURATION")
        print("=" * 50)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Data Root: {cls.DATA_ROOT}")
        print(f"Patch Size: {cls.PATCH_SIZE}x{cls.PATCH_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Encoder: {cls.ENCODER_NAME}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Learning Rate: {cls.INITIAL_LR}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 50)

# Create global config instance
config = Config()

# Initialize configuration
if __name__ == "__main__":
    config.create_directories()

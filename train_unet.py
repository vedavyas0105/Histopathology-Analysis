"""
Training script for Enhanced U-Net on PanNuke dataset
GPU-compatible with mixed precision training and comprehensive logging
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import config
from src.data_pipeline import PanNukeDataset, UnifiedDataLoader
from src.enhanced_unet import EnhancedUNet
from src.losses import DualLoss
from src.evaluation_metrics import compute_dice_coefficient, compute_per_class_metrics
from src.visualization_tools import SegmentationVisualizer

class UNetTrainer:
    """Enhanced U-Net trainer with GPU support and mixed precision training."""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if config.USE_AMP else None
        self.writer = self._setup_logging()
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
    def _setup_device(self):
        """Setup device for CPU training."""
        device = torch.device('cpu')
        print("Using CPU for training")
        print("Note: Mixed precision training disabled for CPU")
        
        return device
    
    def _setup_model(self):
        """Setup Enhanced U-Net model."""
        model = EnhancedUNet(
            encoder_name=config.ENCODER_NAME,
            num_classes=config.NUM_CLASSES
        )
        
        model = model.to(self.device)
        return model
    
    def _setup_criterion(self):
        """Setup loss function."""
        return DualLoss(
            bce_weight=config.BCE_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            edge_weight=config.EDGE_WEIGHT
        )
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer optimized for CPU training."""
        # For CPU training, use single learning rate for better stability
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.INITIAL_LR,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),  # Standard Adam betas
            eps=1e-8
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler optimized for CPU training."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR
        )
    
    def _setup_logging(self):
        """Setup TensorBoard logging."""
        if config.USE_TENSORBOARD:
            writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)
            return writer
        return None
    
    def _create_data_loaders(self):
        """Create training and validation data loaders."""
        dataset = PanNukeDataset(
            data_path=str(config.PANNUKE_PATH),
            patch_size=config.PATCH_SIZE,
            split='train'
        )
        
        total_size = len(dataset)
        train_size = int(config.TRAIN_SPLIT * total_size)
        val_size = int(config.VAL_SPLIT * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=False
        )
        
        return train_loader, val_loader, test_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_edge = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Handle mask dimensions - keep as integer labels for multi-class
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            
            # Convert to binary for binary segmentation (background vs nuclei)
            # This preserves the multi-class information in the original data
            masks_binary = (masks > 0).float()  # Any non-zero pixel is nuclei
            
            # Convert binary mask to 2-channel for loss calculation
            masks_2ch = torch.stack([1 - masks_binary, masks_binary], dim=1)  # [B, 2, H, W]
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    main_output, instance_output = self.model(images)
                    loss, loss_dict = self.criterion(main_output, masks_2ch)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                main_output, instance_output = self.model(images)
                loss, loss_dict = self.criterion(main_output, masks_2ch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_VAL)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_bce += loss_dict['bce'].item()
            total_dice += loss_dict['dice'].item()
            if 'edge' in loss_dict:
                total_edge += loss_dict['edge'].item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BCE': f'{loss_dict["bce"].item():.4f}',
                'Dice': f'{loss_dict["dice"].item():.4f}'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'bce': total_bce / len(train_loader),
            'dice': total_dice / len(train_loader),
            'edge': total_edge / len(train_loader) if total_edge > 0 else 0.0
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_bce = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Handle mask dimensions - keep as integer labels for multi-class
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                
                # Convert to binary for binary segmentation (background vs nuclei)
                masks_binary = (masks > 0).float()  # Any non-zero pixel is nuclei
                
                # Convert binary mask to 2-channel for loss calculation
                masks_2ch = torch.stack([1 - masks_binary, masks_binary], dim=1)  # [B, 2, H, W]
                
                if self.scaler:
                    with autocast():
                        main_output, instance_output = self.model(images)
                        loss, loss_dict = self.criterion(main_output, masks_2ch)
                else:
                    main_output, instance_output = self.model(images)
                    loss, loss_dict = self.criterion(main_output, masks_2ch)
                
                # Use soft Dice calculation on probabilities instead of hard binary
                pred_probs = torch.sigmoid(main_output[:, 1:2, :, :])  # [B, 1, H, W]
                pred_binary = pred_probs > 0.5
                
                # Calculate soft Dice using probabilities
                dice = self._compute_soft_dice(pred_probs, masks_binary.unsqueeze(1))
                
                total_loss += loss.item()
                total_dice += dice
                total_bce += loss_dict['bce'].item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        return {
            'loss': total_loss / len(val_loader),
            'dice': total_dice / len(val_loader),
            'bce': total_bce / len(val_loader)
        }
    
    def _compute_soft_dice(self, pred_probs, target, smooth=1e-7):
        """Compute soft Dice coefficient using probabilities."""
        intersection = (pred_probs * target).sum()
        dice = (2. * intersection + smooth) / (pred_probs.sum() + target.sum() + smooth)
        return dice.item()
    
    def load_best_model(self):
        """Load the best saved model if it exists."""
        model_path = config.CHECKPOINT_DIR / 'best_model.pth'
        if model_path.exists():
            print(f"🔄 Loading best model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state (optional - for exact resume)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state restored")
            
            # Load scheduler state (optional - for exact resume)
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Scheduler state restored")
            
            # Load best dice score
            self.best_dice = checkpoint.get('best_dice', 0.0)
            
            # Load training history if available
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.val_losses = checkpoint['val_losses']
                self.val_dice_scores = checkpoint['val_dice_scores']
                print(f"✅ Training history restored ({len(self.train_losses)} epochs)")
            
            print(f"✅ Loaded best model with Dice: {self.best_dice:.4f}")
            print(f"📅 From epoch: {checkpoint.get('epoch', 'unknown')}")
            return True
        else:
            print("ℹ️  No best model found, starting fresh")
            return False
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with timestamped subfolders."""
        import time
        timestamp = int(time.time())
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
            'config': self.config.__dict__
        }
        
        # Create timestamped subfolder for this training run
        run_dir = config.CHECKPOINT_DIR / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = run_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = run_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            # Also save to main directory for easy access
            main_best_path = config.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, main_best_path)
            print(f"New best model saved! Dice: {self.best_dice:.4f}")
    
    def test_epoch(self, test_loader):
        """Test for one epoch on test dataset."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_bce = 0.0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing", leave=False)
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Handle mask dimensions - keep as integer labels for multi-class
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                
                # Convert to binary for binary segmentation (background vs nuclei)
                masks_binary = (masks > 0).float()  # Any non-zero pixel is nuclei
                
                # Convert binary mask to 2-channel for loss calculation
                masks_2ch = torch.stack([1 - masks_binary, masks_binary], dim=1)  # [B, 2, H, W]
                
                if self.scaler:
                    with autocast():
                        main_output, instance_output = self.model(images)
                        loss, loss_dict = self.criterion(main_output, masks_2ch)
                else:
                    main_output, instance_output = self.model(images)
                    loss, loss_dict = self.criterion(main_output, masks_2ch)
                
                # Use soft Dice calculation on probabilities
                pred_probs = torch.sigmoid(main_output[:, 1:2, :, :])  # [B, 1, H, W]
                dice = self._compute_soft_dice(pred_probs, masks_binary.unsqueeze(1))
                
                total_loss += loss.item()
                total_dice += dice
                total_bce += loss_dict['bce'].item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        return {
            'loss': total_loss / len(test_loader),
            'dice': total_dice / len(test_loader),
            'bce': total_bce / len(test_loader)
        }
    
    def train(self):
        """Main training loop."""
        print("Starting U-Net training...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {config.USE_AMP}")
        print(f"Batch Size: {config.BATCH_SIZE}")
        print(f"Epochs: {config.EPOCHS}")
        print(f"Learning Rate: {config.INITIAL_LR}")
        print(f"Optimizer: AdamW (CPU optimized)")
        print(f"LR Scheduler: ReduceLROnPlateau (patience={config.LR_SCHEDULER_PATIENCE})")
        print("-" * 60)
        
        # Try to load best model if exists
        model_loaded = self.load_best_model()
        if model_loaded:
            print("🔄 Resuming training from best model")
        else:
            print("🆕 Starting fresh training")
        
        train_loader, val_loader, test_dataset = self._create_data_loaders()
        
        # Create test loader from test dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=False
        )
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            self.scheduler.step(val_metrics['dice'])
            
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_dice_scores.append(val_metrics['dice'])
            
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
                self.writer.add_scalar('BCE/Train', train_metrics['bce'], epoch)
                self.writer.add_scalar('BCE/Val', val_metrics['bce'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            is_best = val_metrics['dice'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.test_epoch(test_loader)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best Validation Dice Score: {self.best_dice:.4f}")
        print(f"Test Dice Score: {test_metrics['dice']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        
        self.save_checkpoint(epoch, is_best=True)
        
        if self.writer:
            self.writer.close()
        
        return self.best_dice, test_metrics['dice']

def main():
    """Main training function."""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.NUMPY_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.TORCH_SEED)
    
    config.create_directories()
    config.print_config()
    
    trainer = UNetTrainer(config)
    best_val_dice, test_dice = trainer.train()
    
    print(f"\nFinal Results:")
    print(f"Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")
    print(f"Target Dice: {config.TARGET_DICE_COEFFICIENT}")
    print(f"Validation Status: {'PASSED' if best_val_dice >= config.TARGET_DICE_COEFFICIENT else 'FAILED'}")
    print(f"Test Status: {'PASSED' if test_dice >= config.TARGET_DICE_COEFFICIENT else 'FAILED'}")
    
    return best_val_dice, test_dice

if __name__ == "__main__":
    main()

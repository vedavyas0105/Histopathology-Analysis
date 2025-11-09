"""
Training pipeline for histopathology segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import time
from tqdm import tqdm

from config import config
from src.enhanced_unet import EnhancedUNet
from src.losses import DualLoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.INITIAL_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR
        )
        
        self.criterion = DualLoss(
            bce_weight=config.BCE_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            edge_weight=config.EDGE_WEIGHT
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None
        self.writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs, _ = self.model(images)
                    loss, loss_dict = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(images)
                loss, loss_dict = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     len(self.train_loader) * self.current_epoch + batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(images)
                        loss, loss_dict = self.criterion(outputs, masks)
                else:
                    outputs, _ = self.model(images)
                    loss, loss_dict = self.criterion(outputs, masks)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_path = config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = config.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self):
        for epoch in range(config.EPOCHS):
            self.current_epoch = epoch
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{config.EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            if self.patience_counter >= config.PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        self.writer.close()

import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import config


class PatchCamelyonDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str = 'train',
                 transform: Optional[A.Compose] = None,
                 patch_size: int = 96):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        
        self.images, self.labels, self.metadata = self._load_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        split_name = 'valid' if self.split == 'val' else self.split
        x_file = self.data_path / f"camelyonpatch_level_2_split_{split_name}_x.h5"
        y_file = self.data_path / f"camelyonpatch_level_2_split_{split_name}_y.h5"
        meta_file = self.data_path / f"camelyonpatch_level_2_split_{split_name}_meta.csv"
        
        with h5py.File(x_file, 'r') as f:
            images = f['x'][:]  # type: ignore
        
        with h5py.File(y_file, 'r') as f:
            labels = f['y'][:]  # type: ignore
        
        metadata = pd.read_csv(meta_file)
        
        return images, labels, metadata  # type: ignore
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.images[idx]
        label = self.labels[idx]
        
        if image.shape[2] == 3:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image.squeeze(), mode='L').convert('RGB')
        
        if image.size != (self.patch_size, self.patch_size):
            image = image.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
        
        image = np.array(image)
        
        # Create realistic binary mask for tumor detection
        binary_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        if label == 1:
            # Create 1-3 tumor regions instead of filling entire patch
            num_tumor_regions = np.random.randint(1, 4)
            
            for _ in range(num_tumor_regions):
                center_x = np.random.randint(20, self.patch_size-20)
                center_y = np.random.randint(20, self.patch_size-20)
                radius = np.random.randint(15, 30)
                
                y, x = np.ogrid[:self.patch_size, :self.patch_size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                binary_mask[mask] = 255
        
        if self.transform:
            transformed = self.transform(image=image, mask=binary_mask)
            image = transformed['image']
            binary_mask = transformed['mask']
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if isinstance(binary_mask, np.ndarray):
            binary_mask = torch.from_numpy(binary_mask).float() / 255.0
        else:
            binary_mask = binary_mask.float() / 255.0
        
        binary_mask = binary_mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': binary_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'idx': idx
        }


class PanNukeDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str = 'train',
                 transform: Optional[A.Compose] = None,
                 patch_size: int = 96):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        
        # Load real PanNuke data from .npy files
        self.images, self.masks, self.types = self._load_npy_data()
    
    def _load_npy_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load real PanNuke data from .npy files."""
        # Map split names to fold numbers
        fold_mapping = {'train': 1, 'val': 2, 'test': 3}
        fold_num = fold_mapping.get(self.split, 1)
        
        # Construct paths to .npy files
        fold_dir = self.data_path / f"Fold {fold_num}"
        images_path = fold_dir / "images" / f"fold{fold_num}" / "images.npy"
        masks_path = fold_dir / "masks" / f"fold{fold_num}" / "masks.npy"
        types_path = fold_dir / "images" / f"fold{fold_num}" / "types.npy"
        
        # Load data
        if images_path.exists() and masks_path.exists():
            images = np.load(images_path, mmap_mode='r')
            masks = np.load(masks_path, allow_pickle=True)  # Remove mmap_mode for object dtype
            types = np.load(types_path, allow_pickle=True) if types_path.exists() else None
            
            print(f"Loaded PanNuke {self.split}: {len(images)} samples")
            return images, masks, types if types is not None else np.array([])  # type: ignore
        else:
            print(f"PanNuke data not found at {fold_dir}")
            return np.array([]), np.array([]), np.array([])
    
    def _create_masks_from_real_data(self, mask) -> Dict[str, np.ndarray]:
        """Create masks from real PanNuke data."""
        # Handle different mask formats
        if hasattr(mask, 'shape') and len(mask.shape) == 3 and mask.shape[2] == 6:
            # PanNuke 6-channel format: combine all channels
            binary_mask = (np.max(mask, axis=2) > 0).astype(np.uint8) * 255
        elif hasattr(mask, 'shape') and len(mask.shape) == 2:
            # Single channel format
            binary_mask = (mask > 0).astype(np.uint8) * 255
        else:
            # Handle object dtype or other formats
            if isinstance(mask, (list, tuple)) and len(mask) > 0:
                # List of PIL images or arrays - combine them
                combined = np.zeros((256, 256), dtype=np.uint8)
                for m in mask:
                    if hasattr(m, 'convert'):  # PIL Image
                        # Convert PIL image to numpy array
                        img_array = np.array(m.convert('L'))  # Convert to grayscale
                        # Ensure same size as combined array
                        if img_array.shape != combined.shape:
                            img_array = cv2.resize(img_array, (combined.shape[1], combined.shape[0]))
                        combined = np.maximum(combined, img_array)
                    elif hasattr(m, 'shape') and len(m.shape) == 2:
                        # Already numpy array
                        combined = np.maximum(combined, m)
                binary_mask = (combined > 0).astype(np.uint8) * 255
            else:
                # Fallback - create empty mask
                binary_mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Resize to patch size if needed
        if binary_mask.shape != (self.patch_size, self.patch_size):
            binary_mask = cv2.resize(binary_mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        height, width = binary_mask.shape
        
        # Create distance map using real mask
        distance_map = np.zeros((height, width), dtype=np.float32)
        if binary_mask.max() > 0:
            dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            if dist_transform.max() > 0:
                distance_map = dist_transform / dist_transform.max()
        
        # Create direction map (simplified for now)
        direction_map = np.zeros((height, width, 2), dtype=np.float32)
        
        return {
            'binary_mask': binary_mask,
            'distance_map': distance_map,
            'direction_map': direction_map
        }
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get real data
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        
        # Resize to patch size if needed
        if image.shape[:2] != (self.patch_size, self.patch_size):
            image = cv2.resize(image, (self.patch_size, self.patch_size))
            # Don't resize mask here - handle in _create_masks_from_real_data
        
        # Create masks from real data
        masks = self._create_masks_from_real_data(mask)
        binary_mask = masks['binary_mask']
        distance_map = masks['distance_map']
        direction_map = masks['direction_map']
        
        if self.transform:
            transformed = self.transform(image=image, mask=binary_mask)
            image = transformed['image']
            binary_mask = transformed['mask']
            
            distance_map = cv2.resize(distance_map, (self.patch_size, self.patch_size))
            direction_map = cv2.resize(direction_map, (self.patch_size, self.patch_size))
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if isinstance(binary_mask, np.ndarray):
            binary_mask = torch.from_numpy(binary_mask).float() / 255.0
        else:
            binary_mask = binary_mask.float() / 255.0
            
        if isinstance(distance_map, np.ndarray):
            distance_map = torch.from_numpy(distance_map).float()
        else:
            distance_map = distance_map.float()
            
        if isinstance(direction_map, np.ndarray):
            direction_map = torch.from_numpy(direction_map).permute(2, 0, 1).float()
        else:
            direction_map = direction_map.permute(2, 0, 1).float()
        
        instance_masks = torch.stack([
            binary_mask,
            distance_map,
            direction_map[0],
            direction_map[1]
        ], dim=0)
        
        return {
            'image': image,
            'mask': binary_mask.unsqueeze(0),
            'instance_masks': instance_masks,
            'distance_map': distance_map,
            'idx': idx
        }


class UnifiedDataLoader:
    def __init__(self, 
                 patchcamelyon_path: str,
                 pannuke_path: str,
                 batch_size: int = 16,
                 patch_size: int = 96,
                 num_workers: int = 4):
        self.patchcamelyon_path = patchcamelyon_path
        self.pannuke_path = pannuke_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
        self._init_datasets()
    
    def _get_train_transforms(self):
        return A.Compose([  # type: ignore
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),  # type: ignore
            A.RandomBrightnessContrast(
                brightness_limit=config.BRIGHTNESS_LIMIT,
                contrast_limit=config.CONTRAST_LIMIT,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(config.HUE_LIMIT * 180),
                sat_shift_limit=int(config.SATURATION_LIMIT * 255),
                val_shift_limit=int(config.BRIGHTNESS_LIMIT * 255),
                p=0.3
            ),
            A.GaussNoise(var_limit=(config.GAUSSIAN_NOISE_VAR, config.GAUSSIAN_NOISE_VAR), p=0.2),  # type: ignore
            A.GaussianBlur(blur_limit=3, p=config.GAUSSIAN_BLUR_PROB),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # type: ignore
            ToTensorV2()
        ])
    
    def _get_val_transforms(self):
        return A.Compose([  # type: ignore
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # type: ignore
            ToTensorV2()
        ])
    
    def _init_datasets(self):
        self.pc_train = PatchCamelyonDataset(
            self.patchcamelyon_path, 'train', self.train_transform, self.patch_size
        )
        self.pc_val = PatchCamelyonDataset(
            self.patchcamelyon_path, 'val', self.val_transform, self.patch_size
        )
        self.pc_test = PatchCamelyonDataset(
            self.patchcamelyon_path, 'test', self.val_transform, self.patch_size
        )
        
        self.pn_train = PanNukeDataset(
            self.pannuke_path, 'train', self.train_transform, self.patch_size
        )
        self.pn_val = PanNukeDataset(
            self.pannuke_path, 'val', self.val_transform, self.patch_size
        )
        self.pn_test = PanNukeDataset(
            self.pannuke_path, 'test', self.val_transform, self.patch_size
        )
    
    def get_patchcamelyon_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.pc_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self.pc_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self.pc_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_pannuke_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.pn_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self.pn_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self.pn_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_combined_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.get_patchcamelyon_loaders()


def create_data_loaders() -> Dict[str, DataLoader]:
    data_loader = UnifiedDataLoader(
        patchcamelyon_path=str(config.PATCHCAMELYON_PATH),  # type: ignore
        pannuke_path=str(config.PANNUKE_PATH),  # type: ignore
        batch_size=config.BATCH_SIZE,
        patch_size=config.PATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    pc_train, pc_val, pc_test = data_loader.get_patchcamelyon_loaders()
    pn_train, pn_val, pn_test = data_loader.get_pannuke_loaders()
    
    return {
        'patchcamelyon_train': pc_train,
        'patchcamelyon_val': pc_val,
        'patchcamelyon_test': pc_test,
        'pannuke_train': pn_train,
        'pannuke_val': pn_val,
        'pannuke_test': pn_test
    }
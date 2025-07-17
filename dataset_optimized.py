import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, List, Tuple, Optional
import config_optimized
import json

class ParasiteDataset(Dataset):
    def __init__(self, data_path: str, transform=None, is_train: bool = True, class_to_idx: Optional[Dict[str, int]] = None, label_json_path: Optional[str] = None):
        self.data_path = data_path
        self.transform = transform
        self.is_train = is_train
        self.class_to_idx = class_to_idx
        self.label_json_path = label_json_path
        
        # Get all image files and their labels
        self.images, self.labels, self.class_to_idx = self._load_dataset()
        
    def _load_dataset(self) -> Tuple[List[str], List[int], Optional[Dict[str, int]]]:
        """Load dataset and create class mapping"""
        images = []
        labels = []
        
        # If using COCO-style label file (for test set)
        if self.label_json_path is not None:
            # Load label file
            with open(self.label_json_path, 'r') as f:
                label_data = json.load(f)
            # Build file_name -> image_id
            file_to_id = {img['file_name']: img['id'] for img in label_data['images']}
            # Build image_id -> category_id (use first annotation per image)
            imageid_to_catid = {}
            for ann in label_data['annotations']:
                if ann['image_id'] not in imageid_to_catid:
                    imageid_to_catid[ann['image_id']] = ann['category_id']
            # Build category_id -> class name
            catid_to_name = {cat['id']: cat['name'] for cat in label_data['categories']}
            # Use provided class_to_idx mapping
            class_to_idx = self.class_to_idx
            # For each file in data_path, assign label if possible
            for filename in os.listdir(self.data_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.data_path, filename)
                    # Find image_id
                    image_id = file_to_id.get(filename)
                    if image_id is None:
                        continue
                    # Find category_id
                    category_id = imageid_to_catid.get(image_id)
                    if category_id is None:
                        continue
                    # Find class name
                    class_name = catid_to_name.get(category_id)
                    if class_name is None:
                        continue
                    # Map to index
                    if class_name in class_to_idx:
                        images.append(img_path)
                        labels.append(class_to_idx[class_name])
            return images, labels, class_to_idx
        
        # Standard behavior (train/val set)
        extracted_class_names = []
        if self.class_to_idx is None:
            # Build mapping from this dataset (for training set)
            class_names = set()
            for filename in os.listdir(self.data_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_name = filename.split('_')[0]
                    class_names.add(class_name)
                    if len(extracted_class_names) < 10:
                        extracted_class_names.append(class_name)
            class_names = sorted(list(class_names))
            class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
            print("[DEBUG] Built class_to_idx:", class_to_idx)
            print("[DEBUG] First 10 extracted class names from filenames:", extracted_class_names)
        else:
            # Use provided mapping (for test set)
            class_to_idx = self.class_to_idx
        
        # Load images and assign labels using mapping
        for filename in os.listdir(self.data_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.data_path, filename)
                class_name = filename.split('_')[0]
                if class_to_idx is not None:
                    idx = class_to_idx.get(class_name)
                    if idx is None:
                        continue  # skip if class not found
                    images.append(img_path)
                    labels.append(idx)
                else:
                    continue  # skip if class_to_idx is None
        print(f"[DEBUG] Loaded {len(images)} images for {'training' if self.is_train else 'test/val'} set from {self.data_path}")
        return images, labels, class_to_idx
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

def get_transforms_optimized(image_size: int = 384, is_train: bool = True, config=None):
    """Get optimized data augmentation transforms based on CoAtNet techniques"""
    if is_train:
        return A.Compose([
            # Resize to target size (baseado no CoAtNet)
            A.Resize(image_size, image_size),
            
            # Augmentações básicas (baseadas no CoAtNet)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            
            # Transformações geométricas moderadas
            A.Affine(
                translate_percent=0.1,  # Reduzido
                scale=(0.9, 1.1),      # Reduzido
                rotate=(-15, 15),       # Reduzido baseado no CoAtNet
                p=0.6
            ),
            
            # Blur e noise (específicos do CoAtNet)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.4),
            
            # Noise (baseado no CoAtNet)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),
            ], p=0.4),
            
            # Ajustes de cor moderados
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # Reduzido
                    contrast_limit=0.2,    # Reduzido
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,    # Reduzido
                    sat_shift_limit=30,    # Reduzido
                    val_shift_limit=20,    # Reduzido
                    p=0.3
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            ], p=0.4),
            
            # Dropout espacial (reduzido)
            A.CoarseDropout(
                max_holes=4, max_height=16, max_width=16,
                min_holes=1, min_height=4, min_width=4,
                p=0.2
            ),
            
            # Normalização ImageNet (baseada no CoAtNet)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def get_dataloaders_optimized(config):
    """Create optimized train and validation dataloaders"""
    # Transforms
    train_transform = get_transforms_optimized(config.image_size, is_train=True, config=config)
    val_transform = get_transforms_optimized(config.image_size, is_train=False, config=config)
    
    # Split train data into train/val (80/20)
    full_dataset = ParasiteDataset(config.train_data_path, transform=train_transform, is_train=True)
    
    # Calculate split indices
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update transforms for validation
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Test dataset
    test_label_json = os.path.join("..", "Chula-ParasiteEgg-11_test", "test_labels_200.json")
    test_dataset = ParasiteDataset(
        config.test_data_path,
        transform=val_transform,
        is_train=False,
        class_to_idx=full_dataset.class_to_idx,
        label_json_path=test_label_json
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader, full_dataset.class_to_idx 
import os
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    # Dataset paths - same as before
    train_data_path: str = "/home/edvl/TCC/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data"
    test_data_path: str = "/home/edvl/TCC/Chula-ParasiteEgg-11_test/test/data"
    
    # Model configurations (improved for better generalization)
    num_classes: int = 11
    image_size: int = 224
    batch_size: int = 16  # Reduced for better generalization
    num_epochs: int = 30  # Increased for better convergence
    learning_rate: float = 5e-5  # Reduced for more stable training
    weight_decay: float = 1e-3  # Increased for better regularization
    
    # Training settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    pin_memory: bool = True
    
    # Data augmentation
    train_transform: bool = True
    test_transform: bool = False
    
    # Model specific settings (improved regularization)
    # CNN (EfficientNetV2-S)
    cnn_model_name: str = "tf_efficientnetv2_s"
    cnn_dropout: float = 0.5  # Increased dropout
    
    # Vision Transformer (Tiny ViT - much lighter)
    vit_model_name: str = "vit_tiny_patch16_224"
    vit_patch_size: int = 16
    vit_embed_dim: int = 192
    vit_depths: Tuple[int, ...] = (3, 3, 3)
    vit_num_heads: Tuple[int, ...] = (3, 6, 12)
    
    # Hybrid Model
    hybrid_cnn_backbone: str = "tf_efficientnetv2_s"
    hybrid_vit_model: str = "vit_tiny_patch16_224"
    hybrid_fusion_dim: int = 64
    
    # Paths
    model_save_dir: str = "models_improved"
    results_dir: str = "results_improved"
    logs_dir: str = "logs_improved"
    
    # Early stopping (more patient)
    patience: int = 10  # Increased patience
    min_delta: float = 0.001
    
    # Additional regularization settings
    label_smoothing: float = 0.1  # Label smoothing for better generalization
    gradient_clip: float = 1.0  # Gradient clipping
    
    def __post_init__(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

config = Config() 
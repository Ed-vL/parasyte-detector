import os
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    # Dataset paths
    train_data_path: str = "/home/edvl/TCC/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data"
    test_data_path: str = "/home/edvl/TCC/Chula-ParasiteEgg-11_test/test/data"
    
    # Model configurations (otimizado baseado no CoAtNet)
    num_classes: int = 11
    image_size: int = 384  # Aumentado baseado no CoAtNet (melhor acurácia)
    batch_size: int = 8  # Otimizado para RTX 2070 Super
    num_epochs: int = 50  # Mais épocas para convergência
    learning_rate: float = 1e-4  # Learning rate moderado
    weight_decay: float = 5e-4  # Regularização moderada
    
    # Training settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # Otimizado para Ryzen 3600X
    pin_memory: bool = True
    
    # Data augmentation (baseado no CoAtNet)
    train_transform: bool = True
    test_transform: bool = False
    
    # Model specific settings (otimizado)
    # CNN (EfficientNetV2-S)
    cnn_model_name: str = "tf_efficientnetv2_s"
    cnn_dropout: float = 0.5  # Baseado no CoAtNet
    
    # Vision Transformer (Tiny ViT)
    vit_model_name: str = "vit_tiny_patch16_384"
    vit_patch_size: int = 16
    vit_embed_dim: int = 192
    vit_depths: Tuple[int, ...] = (3, 3, 3)
    vit_num_heads: Tuple[int, ...] = (3, 6, 12)
    
    # Hybrid Model
    hybrid_cnn_backbone: str = "tf_efficientnetv2_s"
    hybrid_vit_model: str = "vit_tiny_patch16_384"
    hybrid_fusion_dim: int = 128  # Aumentado para melhor fusão
    
    # Paths
    model_save_dir: str = "models_optimized"
    results_dir: str = "results_optimized"
    logs_dir: str = "logs_optimized"
    
    # Early stopping (mais paciente)
    patience: int = 15
    min_delta: float = 0.001
    
    # Regularização otimizada (baseada no CoAtNet)
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # Learning rate scheduler (baseado no CoAtNet)
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    
    # Data augmentation específico (baseado no CoAtNet)
    augmentation_blur: bool = True
    augmentation_noise: bool = True
    augmentation_rotation: int = 15  # Rotação moderada
    augmentation_brightness: float = 0.2
    augmentation_contrast: float = 0.2
    
    def __post_init__(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

config = Config() 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
from tqdm import tqdm
import config
from models import get_model, get_model_summary
from dataset import get_dataloaders

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

# Mapping for translation to Brazilian Portuguese
RESULTS_KEY_TRANSLATION = {
    'accuracy': 'acuracia',
    'loss': 'perda',
    'classification_report': 'relatorio_classificacao',
    'confusion_matrix': 'matriz_confusao',
    'predictions': 'predicoes',
    'targets': 'rotulos_reais',
    'macro avg': 'media_macro',
    'weighted avg': 'media_ponderada',
    'precision': 'precisao',
    'recall': 'revocacao',
    'f1-score': 'pontuacao_f1',
    'support': 'suporte',
}

def translate_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = RESULTS_KEY_TRANSLATION.get(k, k)
            new_dict[new_key] = translate_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [translate_keys(v) for v in obj]
    else:
        return obj

class ParasiteTrainer:
    def __init__(self, model_type: str, config):
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = get_model(model_type, config, num_classes=config.num_classes).to(self.device)
        
        # Get model summary
        total_params, trainable_params = get_model_summary(self.model)
        
        # Initialize dataloaders
        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = get_dataloaders(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Loss function with label smoothing
        if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f"{config.logs_dir}/{model_type}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training {self.model_type}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config, 'gradient_clip') and self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=f"Validating {self.model_type}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.model_type} model...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.save_model(f"best_{self.model_type}.pth")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        print(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.2f}% at epoch {self.best_epoch+1}")
    
    def evaluate(self, dataloader=None):
        """Evaluate model on test set"""
        if dataloader is None:
            dataloader = self.test_loader
        
        if len(dataloader) == 0:
            print(f"[WARNING] The dataloader is empty. No samples to evaluate.")
            return {
                'accuracy': float('nan'),
                'loss': float('nan'),
                'classification_report': {},
                'confusion_matrix': [],
                'predictions': [],
                'targets': []
            }
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc=f"Evaluating {self.model_type}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_loss = total_loss / len(dataloader)
        
        # Classification report
        class_names = list(self.class_to_idx.keys())
        report = classification_report(all_targets, all_predictions, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_model(self, filename):
        """Save model checkpoint"""
        save_path = os.path.join(self.config.model_save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'class_to_idx': self.class_to_idx
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename):
        """Load model checkpoint"""
        load_path = os.path.join(self.config.model_save_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_path}")
    
    def plot_training_history(self):
        """Plot training history with professional formatting"""
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color scheme
        train_color = '#2E86AB'  # Blue
        val_color = '#A23B72'    # Purple
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, color=train_color, linewidth=2.5, 
                marker='o', markersize=4, label='Treinamento', alpha=0.8)
        ax1.plot(epochs, self.val_losses, color=val_color, linewidth=2.5, 
                marker='s', markersize=4, label='Validação', alpha=0.8)
        
        ax1.set_title('Histórico de Perda', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Época', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Perda', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12, framealpha=0.9, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accuracies, color=train_color, linewidth=2.5, 
                marker='o', markersize=4, label='Treinamento', alpha=0.8)
        ax2.plot(epochs, self.val_accuracies, color=val_color, linewidth=2.5, 
                marker='s', markersize=4, label='Validação', alpha=0.8)
        
        ax2.set_title('Histórico de Acurácia', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Época', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Acurácia (%)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12, framealpha=0.9, loc='lower right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor('#f8f9fa')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Set y-axis limits for accuracy to show meaningful range
        if self.val_accuracies:
            min_acc = min(min(self.train_accuracies), min(self.val_accuracies))
            max_acc = max(max(self.train_accuracies), max(self.val_accuracies))
            ax2.set_ylim(max(0, min_acc - 5), min(100, max_acc + 5))
        
        plt.tight_layout()
        save_path = f"{self.config.results_dir}/{self.model_type}_training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Training history plot saved to {save_path}")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix with professional formatting"""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap with better styling
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Número de Amostras', 'shrink': 0.8},
                   ax=ax, square=True, linewidths=0.5, linecolor='white')
        
        # Customize the plot
        ax.set_title(f'Matriz de Confusão - Modelo {self.model_type.upper()}', 
                    fontsize=18, fontweight='bold', pad=30)
        ax.set_xlabel('Predições', fontsize=16, fontweight='bold', labelpad=20)
        ax.set_ylabel('Valores Reais', fontsize=16, fontweight='bold', labelpad=20)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(class_names, rotation=0, fontsize=10)
        
        # Add grid lines
        ax.grid(False)
        
        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel('Número de Amostras', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = f"{self.config.results_dir}/{self.model_type}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Confusion matrix plot saved to {save_path}")
    
    def save_results(self, results):
        """Save evaluation results"""
        results_path = f"{self.config.results_dir}/{self.model_type}_results.json"
        
        # Convert all numpy types to native Python types for JSON serialization
        results_copy = to_python_type(results.copy())
        # Translate keys to Brazilian Portuguese
        results_copy = translate_keys(results_copy)
        
        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {results_path}") 
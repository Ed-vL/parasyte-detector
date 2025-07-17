import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from tqdm import tqdm
import config_optimized
from dataset_optimized import get_dataloaders_optimized
from models import get_model, get_model_summary

class ParasiteTrainerOptimized:
    def __init__(self, model_type: str, config):
        self.model_type = model_type
        self.config = config
        self.device = torch.device(config.device)
        
        # Get dataloaders
        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = get_dataloaders_optimized(config)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (baseado no CoAtNet)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
            # verbose=True  # Removido pois não é aceito nesta versão
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
    def _create_model(self):
        """Create model based on type"""
        from models import get_model, get_model_summary
        model = get_model(self.model_type, self.config)
        get_model_summary(model, input_size=(3, self.config.image_size, self.config.image_size))
        return model
    
    def get_model_name_pt(self):
        """Get Portuguese model name"""
        model_names = {
            'cnn': 'CNN',
            'vit': 'ViT',
            'hybrid': 'Híbrido'
        }
        return model_names.get(self.model_type, self.model_type.upper())
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training {self.model_type.upper()}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Gradient clipping (baseado no CoAtNet)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=f"Validating {self.model_type.upper()}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def train(self):
        """Train the model with early stopping"""
        print(f"\n{'='*60}")
        print(f"TREINANDO MODELO {self.model_type.upper()} (OTIMIZADO)")
        print(f"{'='*60}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nÉpoca {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_acc)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          f"{self.config.model_save_dir}/{self.model_type}_best.pth")
                print(f"✅ Novo melhor modelo salvo! Acurácia: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ Early stopping: {self.patience_counter}/{self.config.patience}")
                
                if self.patience_counter >= self.config.patience:
                    print(f"🛑 Early stopping ativado após {epoch+1} épocas")
                    break
        
        print(f"\n🎯 Melhor acurácia de validação: {self.best_val_accuracy:.4f}")
    
    def evaluate(self):
        """Evaluate on test set"""
        print(f"\nAvaliando modelo {self.model_type.upper()} no conjunto de teste...")
        
        # Load best model
        best_model_path = f"{self.config.model_save_dir}/{self.model_type}_best.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print("✅ Modelo melhor carregado para avaliação")
        
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        test_loss /= len(self.test_loader)
        test_accuracy = sum(1 for x, y in zip(all_predictions, all_targets) if x == y) / len(all_targets)
        
        # Classification report
        class_names = list(self.class_to_idx.keys())
        report = classification_report(all_targets, all_predictions, 
                                    target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_results(self, results):
        """Save results to JSON"""
        results_path = f"{self.config.results_dir}/{self.model_type}_results.json"
        
        # Helper function to convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                if obj.size == 1:  # numpy scalar
                    return obj.item()
                else:  # numpy array
                    return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all results to JSON-serializable format
        results_to_save = {
            'acuracia': convert_numpy_types(results['accuracy']),
            'perda': convert_numpy_types(results['loss']),
            'relatorio_classificacao': convert_numpy_types(results['classification_report']),
            'matriz_confusao': convert_numpy_types(results['confusion_matrix']),
            'predicoes': convert_numpy_types(results['predictions'])
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        
        print(f"Resultados salvos em {results_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        num_epochs = len(self.train_losses)
        epochs = list(range(1, num_epochs + 1))
        
        # Função para definir os ticks do eixo x
        def get_epoch_ticks(num_epochs):
            if num_epochs <= 20:
                return list(range(1, num_epochs + 1))
            elif num_epochs <= 50:
                ticks = list(range(1, num_epochs + 1, 5))
            else:
                ticks = list(range(1, num_epochs + 1, 10))
            if ticks[-1] != num_epochs:
                ticks.append(num_epochs)
            return ticks
        epoch_ticks = get_epoch_ticks(num_epochs)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, label='Treino', color='blue', marker='o')
        ax1.plot(epochs, self.val_losses, label='Validação', color='red', marker='o')
        ax1.set_title(f'Histórico de Perda - {self.get_model_name_pt()} (Otimizado)')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Perda')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        max_loss = max(self.train_losses + self.val_losses) if (self.train_losses and self.val_losses) else 1
        ax1.set_ylim(0, max_loss * 1.1)
        ax1.set_xticks(epoch_ticks)
        
        # Accuracy plot (convert to percentage)
        train_acc_pct = [acc * 100 for acc in self.train_accuracies]
        val_acc_pct = [acc * 100 for acc in self.val_accuracies]
        ax2.plot(epochs, train_acc_pct, label='Treino', color='blue', marker='o')
        ax2.plot(epochs, val_acc_pct, label='Validação', color='red', marker='o')
        ax2.set_title(f'Histórico de Acurácia - {self.get_model_name_pt()} (Otimizado)')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax2.set_xticks(epoch_ticks)
        
        plt.tight_layout()
        save_path = f"{self.config.results_dir}/{self.model_type}_training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de histórico salvo em {save_path}")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusão - {self.get_model_name_pt()} (Otimizado)')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        save_path = f"{self.config.results_dir}/{self.model_type}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Matriz de confusão salva em {save_path}")
    
    def plot_test_accuracy_analysis(self, test_results):
        """Plot detailed test accuracy analysis"""
        class_names = list(self.class_to_idx.keys())
        report = test_results['classification_report']
        
        # Prepare data for plotting
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                classes.append(class_name)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])
        
        # Debug prints
        print('DEBUG - Classes:', classes)
        print('DEBUG - Precisões:', precisions)
        print('DEBUG - Revocações:', recalls)
        print('DEBUG - F1-scores:', f1_scores)
        if not classes or not precisions or not recalls or not f1_scores:
            print('AVISO: Alguma das listas para plotagem está vazia! O gráfico pode sair em branco.')
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall test accuracy
        test_acc = test_results['accuracy']
        test_loss = test_results['loss']
        
        ax1.bar(['Acurácia', 'Perda'], [test_acc, test_loss], 
                color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax1.set_title(f'Métricas Gerais de Teste - {self.get_model_name_pt()} (Otimizado)')
        ax1.set_ylabel('Valor')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (acc, loss) in enumerate([(test_acc, test_loss)]):
            ax1.text(0, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            ax1.text(1, loss + 0.01, f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precision by class
        bars1 = ax2.bar(range(len(classes)), precisions, alpha=0.8, color='#2E86AB')
        ax2.set_title(f'Precisão por Classe - {self.get_model_name_pt()} (Otimizado)')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Precisão')
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, prec in zip(bars1, precisions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 3. Recall by class
        bars2 = ax3.bar(range(len(classes)), recalls, alpha=0.8, color='#A23B72')
        ax3.set_title(f'Revocação por Classe - {self.get_model_name_pt()} (Otimizado)')
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Revocação')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rec in zip(bars2, recalls):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. F1-Score by class
        bars3 = ax4.bar(range(len(classes)), f1_scores, alpha=0.8, color='#F18F01')
        ax4.set_title(f'F1-Score por Classe - {self.get_model_name_pt()} (Otimizado)')
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('F1-Score')
        ax4.set_xticks(range(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, f1 in zip(bars3, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        save_path = f"{self.config.results_dir}/{self.model_type}_test_accuracy_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Análise detalhada de acurácia de teste salva em {save_path}") 
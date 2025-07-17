#!/usr/bin/env python3
"""
Script para executar experimentos melhorados com técnicas anti-overfitting
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import json
import os
import sys
import argparse

from trainer import ParasiteTrainer
from models import get_model_summary
import config_improved

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_single_model(model_type: str):
    """Train a single model with improved configuration"""
    set_random_seed()
    
    print(f"\n{'='*50}")
    print(f"Treinando Modelo {model_type.upper()} (Configuração Melhorada)")
    print(f"{'='*50}")
    
    # Initialize trainer with improved config
    trainer = ParasiteTrainer(model_type, config_improved.config)
    
    # Train model
    trainer.train()
    
    # Evaluate on test set
    print(f"\nAvaliando modelo {model_type.upper()} no conjunto de teste...")
    test_results = trainer.evaluate()
    
    # Save results
    trainer.save_results(test_results)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Plot confusion matrix
    class_names = list(trainer.class_to_idx.keys())
    trainer.plot_confusion_matrix(test_results['confusion_matrix'], class_names)
    
    print(f"Resultados do Modelo {model_type.upper()}:")
    print(f"Acurácia de Teste: {test_results['accuracy']:.4f}")
    print(f"Perda de Teste: {test_results['loss']:.4f}")
    print(f"Melhor Acurácia de Validação: {trainer.best_val_accuracy:.4f}")
    
    return {
        'accuracy': test_results['accuracy'],
        'loss': test_results['loss'],
        'classification_report': test_results['classification_report'],
        'best_val_accuracy': trainer.best_val_accuracy
    }

def train_all_models():
    """Train all three models with improved configuration"""
    set_random_seed()
    
    models = ['cnn', 'vit', 'hybrid']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Treinando Modelo {model_type.upper()} (Configuração Melhorada)")
        print(f"{'='*50}")
        
        # Initialize trainer with improved config
        trainer = ParasiteTrainer(model_type, config_improved.config)
        
        # Train model
        trainer.train()
        
        # Evaluate on test set
        print(f"\nAvaliando modelo {model_type.upper()} no conjunto de teste...")
        test_results = trainer.evaluate()
        
        # Save results
        trainer.save_results(test_results)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Plot confusion matrix
        class_names = list(trainer.class_to_idx.keys())
        trainer.plot_confusion_matrix(test_results['confusion_matrix'], class_names)
        
        # Store results
        results[model_type] = {
            'accuracy': test_results['accuracy'],
            'loss': test_results['loss'],
            'classification_report': test_results['classification_report'],
            'best_val_accuracy': trainer.best_val_accuracy
        }
        
        print(f"Resultados do Modelo {model_type.upper()}:")
        print(f"Acurácia de Teste: {test_results['accuracy']:.4f}")
        print(f"Perda de Teste: {test_results['loss']:.4f}")
        print(f"Melhor Acurácia de Validação: {trainer.best_val_accuracy:.4f}")
    
    return results

def compare_models(results):
    """Compare and visualize results from all models"""
    print(f"\n{'='*60}")
    print("RESULTADOS DA COMPARAÇÃO DE MODELOS (MELHORADOS)")
    print(f"{'='*60}")
    print("NOTA: Os valores de 'Test Accuracy' são da avaliação final no conjunto de teste.")
    print("Os valores de 'Best Val Accuracy' são da melhor época durante o treinamento.")
    print("Diferenças menores indicam melhor generalização.")
    print(f"{'='*60}")
    
    # Create comparison dataframe
    comparison_data = []
    for model_type, result in results.items():
        comparison_data.append({
            'Model': model_type.upper(),
            'Test Accuracy': result['accuracy'],
            'Test Loss': result['loss'],
            'Best Val Accuracy': result['best_val_accuracy']
        })
    
    df = pd.DataFrame(comparison_data)
    print("\nComparação de Performance dos Modelos:")
    print(df.to_string(index=False))
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    models = [data['Model'] for data in comparison_data]
    test_accuracies = [data['Test Accuracy'] for data in comparison_data]
    val_accuracies = [data['Best Val Accuracy'] for data in comparison_data]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, test_accuracies, width, label='Acurácia de Teste', alpha=0.8)
    ax1.bar(x + width/2, val_accuracies, width, label='Acurácia de Validação', alpha=0.8)
    ax1.set_xlabel('Modelos')
    ax1.set_ylabel('Acurácia')
    ax1.set_title('Comparação de Acurácia dos Modelos (Melhorados)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss comparison
    test_losses = [data['Test Loss'] for data in comparison_data]
    ax2.bar(models, test_losses, alpha=0.8, color='orange')
    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Perda')
    ax2.set_title('Comparação de Perda dos Modelos (Melhorados)')
    ax2.grid(True, alpha=0.3)
    
    # Detailed accuracy breakdown
    accuracy_data = []
    for model_type, result in results.items():
        report = result['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                accuracy_data.append({
                    'Model': model_type.upper(),
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score']
                })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # Precision comparison by class
    pivot_precision = accuracy_df.pivot(index='Class', columns='Model', values='Precision')
    pivot_precision.plot(kind='bar', ax=ax3, alpha=0.8)
    ax3.set_title('Precisão por Classe e Modelo (Melhorados)')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Precisão')
    ax3.legend(title='Modelo')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # F1-Score comparison by class
    pivot_f1 = accuracy_df.pivot(index='Class', columns='Model', values='F1-Score')
    pivot_f1.plot(kind='bar', ax=ax4, alpha=0.8)
    ax4.set_title('F1-Score por Classe e Modelo (Melhorados)')
    ax4.set_xlabel('Classes')
    ax4.set_ylabel('F1-Score')
    ax4.legend(title='Modelo')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config_improved.config.model_save_dir}/model_comparison_improved.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comparison results
    comparison_path = f"{config_improved.config.results_dir}/model_comparison_improved.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResultados da comparação salvos em {comparison_path}")
    
    # Print best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 MELHOR MODELO: {best_model[0].upper()}")
    print(f"   Acurácia de Teste: {best_model[1]['accuracy']:.4f}")
    print(f"   Perda de Teste: {best_model[1]['loss']:.4f}")
    
    # Analyze overfitting
    print(f"\n📊 ANÁLISE DE OVERFITTING (MELHORADO):")
    for model_type, result in results.items():
        val_acc = result['best_val_accuracy'] / 100.0  # Convert from percentage
        test_acc = result['accuracy']
        overfitting_gap = val_acc - test_acc
        print(f"   {model_type.upper()}: Validação {val_acc:.3f} → Teste {test_acc:.3f} (Gap: {overfitting_gap:.3f})")

def generate_test_plots(results):
    """Generate test-specific visualization plots"""
    print(f"\n{'='*60}")
    print("GERANDO GRÁFICOS ESPECÍFICOS DE TESTE (MELHORADOS)")
    print(f"{'='*60}")
    
    # Create test performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Test accuracy comparison
    models = list(results.keys())
    test_accuracies = [results[model]['accuracy'] for model in models]
    test_losses = [results[model]['loss'] for model in models]
    
    # Bar plot for test accuracy
    bars1 = ax1.bar(models, test_accuracies, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_title('Acurácia de Teste por Modelo (Melhorado)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Modelos', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Acurácia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, test_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Bar plot for test loss
    bars2 = ax2.bar(models, test_losses, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_title('Perda de Teste por Modelo (Melhorado)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Modelos', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Perda', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, loss in zip(bars2, test_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Detailed per-class performance
    accuracy_data = []
    for model_type, result in results.items():
        report = result['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                accuracy_data.append({
                    'Model': model_type.upper(),
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score']
                })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # F1-Score by class
    pivot_f1 = accuracy_df.pivot(index='Class', columns='Model', values='F1-Score')
    pivot_f1.plot(kind='bar', ax=ax3, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax3.set_title('F1-Score por Classe (Teste - Melhorado)', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Classes', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax3.legend(title='Modelo', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor('#f8f9fa')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Precision vs Recall scatter
    for i, model in enumerate(models):
        model_data = accuracy_df[accuracy_df['Model'] == model.upper()]
        ax4.scatter(model_data['Precision'], model_data['Recall'], 
                   s=100, alpha=0.7, label=model.upper(), 
                   color=['#2E86AB', '#A23B72', '#F18F01'][i])
    
    ax4.set_title('Precisão vs Revocação por Classe (Teste - Melhorado)', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Precisão', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Revocação', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor('#f8f9fa')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add diagonal line for reference
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Linha de Referência')
    
    plt.tight_layout()
    save_path = f"{config_improved.config.results_dir}/test_performance_analysis_improved.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Gráficos de teste melhorados salvos em {save_path}")
    
    # Create overfitting analysis plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for overfitting analysis
    model_names = []
    val_accuracies = []
    test_accuracies = []
    overfitting_gaps = []
    
    for model_type, result in results.items():
        model_names.append(model_type.upper())
        val_acc = result['best_val_accuracy'] / 100.0  # Convert from percentage
        test_acc = result['accuracy']
        gap = val_acc - test_acc
        
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        overfitting_gaps.append(gap)
    
    # Create grouped bar chart
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_accuracies, width, label='Validação', 
                   alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, test_accuracies, width, label='Teste', 
                   alpha=0.8, color='#A23B72')
    
    # Add gap lines
    for i, (val_acc, test_acc) in enumerate(zip(val_accuracies, test_accuracies)):
        ax.plot([i - width/2, i + width/2], [val_acc, test_acc], 'k-', linewidth=2, alpha=0.7)
        ax.text(i, (val_acc + test_acc) / 2, f'Gap: {overfitting_gaps[i]:.3f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_title('Análise de Overfitting: Validação vs Teste (Melhorado)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Modelos', fontsize=14, fontweight='bold')
    ax.set_ylabel('Acurácia', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, acc in zip(bars1, val_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    for bar, acc in zip(bars2, test_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    save_path = f"{config_improved.config.results_dir}/overfitting_analysis_improved.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Análise de overfitting melhorada salva em {save_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train parasite detection models with improved configuration')
    parser.add_argument('--model', type=str, choices=['cnn', 'vit', 'hybrid', 'all'], 
                       default='all', help='Model to train (default: all)')
    
    args = parser.parse_args()
    
    print("Classificação de Ovos de Parasitas - Experimento Melhorado")
    print("=" * 60)
    print("Este script irá treinar e comparar três modelos com técnicas anti-overfitting:")
    print("1. CNN (EfficientNetV2-S) - Dropout aumentado")
    print("2. Vision Transformer (Tiny ViT) - Regularização melhorada")
    print("3. Modelo Híbrido (EfficientNetV2-S + Tiny ViT) - Fusão otimizada")
    print("=" * 60)
    print("Melhorias implementadas:")
    print("- Data augmentation mais agressivo")
    print("- Dropout aumentado (0.2 → 0.5)")
    print("- Learning rate reduzido (2e-4 → 5e-5)")
    print("- Weight decay aumentado (1e-4 → 1e-3)")
    print("- Label smoothing (0.1)")
    print("- Gradient clipping (1.0)")
    print("- Early stopping mais paciente (5 → 10 épocas)")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA não disponível, usando CPU")
    
    # Train models based on argument
    if args.model == 'all':
        results = train_all_models()
    else:
        single_result = train_single_model(args.model)
        results = {args.model: single_result}
    
    # Print final comparison if training all models
    if args.model == 'all':
        print("\n" + "="*60)
        print("COMPARAÇÃO FINAL (MELHORADA)")
        print("="*60)
        for model_type, result in results.items():
            print(f"{model_type.upper()}:")
            print(f"  Acurácia de Teste: {result['accuracy']:.4f}")
            print(f"  Perda de Teste: {result['loss']:.4f}")
            print(f"  Melhor Acurácia de Validação: {result['best_val_accuracy']:.4f}")
            print()

    # Compare models
    compare_models(results)
    
    # Generate test-specific plots
    generate_test_plots(results)
    
    print(f"\n{'='*60}")
    print("EXPERIMENTO MELHORADO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}")
    print("Todos os resultados foram salvos no diretório 'results_improved'.")
    print("Você pode encontrar:")
    print("- Checkpoints dos modelos no diretório 'models_improved'")
    print("- Logs de treinamento no diretório 'logs_improved'") 
    print("- Resultados de avaliação e gráficos no diretório 'results_improved'")
    print("- Gráficos específicos de teste e análise de overfitting")
    print("- TensorBoard logs para monitoramento detalhado do treinamento")

if __name__ == "__main__":
    main() 
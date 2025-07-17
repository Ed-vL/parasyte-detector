#!/usr/bin/env python3
"""
Script para gerar análise detalhada de acurácia de teste
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def carregar_resultados():
    """Carrega os resultados dos modelos"""
    resultados = {}
    
    # Carregar resultados da pasta results_light
    for modelo in ['cnn', 'vit', 'hybrid']:
        arquivo = f'results_light/{modelo}_results.json'
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                resultados[modelo] = {
                    'acuracia': dados['acuracia'],
                    'perda': dados['perda'],
                    'relatorio_classificacao': dados['relatorio_classificacao']
                }
        except FileNotFoundError:
            print(f"⚠️ Arquivo {arquivo} não encontrado")
    
    return resultados

def gerar_analise_detalhada_teste(resultados):
    """Gera análise detalhada de acurácia de teste"""
    if not resultados:
        print("❌ Nenhum resultado encontrado para análise")
        return
    
    print(f"\n{'='*60}")
    print("GERANDO ANÁLISE DETALHADA DE ACURÁCIA DE TESTE")
    print(f"{'='*60}")
    
    # Configurar estilo dos gráficos
    plt.style.use('default')
    sns.set_palette("husl")
    
    model_names_pt = {
        'cnn': 'CNN',
        'vit': 'ViT',
        'hybrid': 'Híbrido'
    }
    
    for modelo, dados in resultados.items():
        model_name_pt = model_names_pt.get(modelo, modelo.upper())
        print(f"\n📊 Gerando análise para modelo {model_name_pt}...")
        
        # Preparar dados
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for classe, metricas in dados['relatorio_classificacao'].items():
            if isinstance(metricas, dict) and 'precisao' in metricas:
                classes.append(classe)
                precisions.append(metricas['precisao'])
                recalls.append(metricas['revocacao'])
                f1_scores.append(metricas['pontuacao_f1'])
        
        # Criar figura com subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Métricas gerais de teste
        test_acc = dados['acuracia']
        test_loss = dados['perda']
        
        bars1 = ax1.bar(['Acurácia', 'Perda'], [test_acc, test_loss], 
                        color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax1.set_title(f'Métricas Gerais de Teste - {model_name_pt}', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Valor', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Adicionar valores nas barras
        ax1.text(0, test_acc + 0.01, f'{test_acc:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(1, test_loss + 0.01, f'{test_loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precisão por classe
        bars2 = ax2.bar(range(len(classes)), precisions, alpha=0.8, color='#2E86AB')
        ax2.set_title(f'Precisão por Classe - {model_name_pt}', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Classes', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Precisão', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor('#f8f9fa')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Adicionar valores nas barras
        for bar, prec in zip(bars2, precisions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 3. Revocação por classe
        bars3 = ax3.bar(range(len(classes)), recalls, alpha=0.8, color='#A23B72')
        ax3.set_title(f'Revocação por Classe - {model_name_pt}', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Classes', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Revocação', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_facecolor('#f8f9fa')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Adicionar valores nas barras
        for bar, rec in zip(bars3, recalls):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. F1-Score por classe
        bars4 = ax4.bar(range(len(classes)), f1_scores, alpha=0.8, color='#F18F01')
        ax4.set_title(f'F1-Score por Classe - {model_name_pt}', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Classes', fontsize=14, fontweight='bold')
        ax4.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor('#f8f9fa')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # Adicionar valores nas barras
        for bar, f1 in zip(bars4, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        
        # Salvar gráfico
        save_path = f"results_light/{modelo}_test_accuracy_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Análise detalhada salva em: {save_path}")
    
    # Criar gráfico comparativo de acurácia de teste
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    modelos = list(resultados.keys())
    acuracias = [resultados[m]['acuracia'] for m in modelos]
    perdas = [resultados[m]['perda'] for m in modelos]
    
    # Gráfico de acurácia de teste
    cores = ['#2E86AB', '#A23B72', '#F18F01']
    model_names_pt_list = [model_names_pt.get(m, m.upper()) for m in modelos]
    bars1 = ax1.bar(model_names_pt_list, acuracias, color=cores, alpha=0.8)
    ax1.set_title('Acurácia de Teste por Modelo', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Modelos', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Acurácia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars1, acuracias):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico de perda de teste
    bars2 = ax2.bar(model_names_pt_list, perdas, color=cores, alpha=0.8)
    ax2.set_title('Perda de Teste por Modelo', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Modelos', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Perda', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Adicionar valores nas barras
    for bar, loss in zip(bars2, perdas):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = "results_light/test_accuracy_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Comparação de acurácia de teste salva em: {save_path}")

def main():
    """Função principal"""
    print("🔍 Carregando resultados dos modelos...")
    resultados = carregar_resultados()
    
    if resultados:
        gerar_analise_detalhada_teste(resultados)
        print("\n✅ Análise detalhada de acurácia de teste concluída!")
    else:
        print("❌ Nenhum resultado encontrado para análise")

if __name__ == "__main__":
    main() 
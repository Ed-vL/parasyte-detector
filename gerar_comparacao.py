#!/usr/bin/env python3
"""
Script para gerar gráficos de comparação dos modelos usando dados em português
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
    
    # Carregar resultados da pasta results_light (que tem todos os modelos)
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

def gerar_graficos_comparacao(resultados):
    """Gera gráficos de comparação dos modelos"""
    if not resultados:
        print("❌ Nenhum resultado encontrado para comparação")
        return
    
    print(f"\n{'='*60}")
    print("GERANDO GRÁFICOS DE COMPARAÇÃO DOS MODELOS")
    print(f"{'='*60}")
    
    # Configurar estilo dos gráficos
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Criar figura com subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dados para os gráficos
    modelos = list(resultados.keys())
    acuracias = [resultados[m]['acuracia'] for m in modelos]
    perdas = [resultados[m]['perda'] for m in modelos]
    
    # 1. Gráfico de barras - Acurácia
    cores = ['#2E86AB', '#A23B72', '#F18F01']
    model_names_pt = {
        'cnn': 'CNN',
        'vit': 'ViT',
        'hybrid': 'Híbrido'
    }
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
    
    # 2. Gráfico de barras - Perda
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
    
    # 3. F1-Score por classe
    dados_f1 = []
    for modelo, dados in resultados.items():
        for classe, metricas in dados['relatorio_classificacao'].items():
            if isinstance(metricas, dict) and 'pontuacao_f1' in metricas:
                dados_f1.append({
                    'Modelo': model_names_pt.get(modelo, modelo.upper()),
                    'Classe': classe,
                    'F1-Score': metricas['pontuacao_f1']
                })
    
    if dados_f1:
        df_f1 = pd.DataFrame(dados_f1)
        pivot_f1 = df_f1.pivot(index='Classe', columns='Modelo', values='F1-Score')
        pivot_f1.plot(kind='bar', ax=ax3, alpha=0.8, color=cores)
        ax3.set_title('F1-Score por Classe e Modelo', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Classes', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
        ax3.legend(title='Modelo', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_facecolor('#f8f9fa')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    
    # 4. Precisão vs Revocação
    dados_pr = []
    for modelo, dados in resultados.items():
        for classe, metricas in dados['relatorio_classificacao'].items():
            if isinstance(metricas, dict) and 'precisao' in metricas and 'revocacao' in metricas:
                dados_pr.append({
                    'Modelo': model_names_pt.get(modelo, modelo.upper()),
                    'Classe': classe,
                    'Precisão': metricas['precisao'],
                    'Revocação': metricas['revocacao']
                })
    
    if dados_pr:
        df_pr = pd.DataFrame(dados_pr)
        for i, modelo in enumerate(modelos):
            model_name_pt = model_names_pt.get(modelo, modelo.upper())
            dados_modelo = df_pr[df_pr['Modelo'] == model_name_pt]
            ax4.scatter(dados_modelo['Precisão'], dados_modelo['Revocação'], 
                       s=100, alpha=0.7, label=model_name_pt, color=cores[i])
        
        ax4.set_title('Precisão vs Revocação por Classe', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Precisão', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Revocação', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor('#f8f9fa')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # Linha de referência
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Linha de Referência')
    
    plt.tight_layout()
    
    # Salvar gráfico
    save_path = "results_improved/comparacao_modelos_melhorada.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Gráfico de comparação salvo em: {save_path}")
    
    # Imprimir tabela de resultados
    print(f"\n{'='*60}")
    print("RESULTADOS DA COMPARAÇÃO DE MODELOS")
    print(f"{'='*60}")
    
    df_resultados = pd.DataFrame([
        {
            'Modelo': model_names_pt.get(modelo, modelo.upper()),
            'Acurácia': f"{dados['acuracia']:.4f}",
            'Perda': f"{dados['perda']:.4f}"
        }
        for modelo, dados in resultados.items()
    ])
    
    print(df_resultados.to_string(index=False))
    
    # Melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['acuracia'])
    melhor_nome = model_names_pt.get(melhor_modelo[0], melhor_modelo[0].upper())
    print(f"\n🏆 MELHOR MODELO: {melhor_nome}")
    print(f"   Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
    print(f"   Perda: {melhor_modelo[1]['perda']:.4f}")

def main():
    """Função principal"""
    print("🔍 Carregando resultados dos modelos...")
    resultados = carregar_resultados()
    
    if resultados:
        gerar_graficos_comparacao(resultados)
        print("\n✅ Análise de comparação concluída!")
    else:
        print("❌ Nenhum resultado encontrado para análise")

if __name__ == "__main__":
    main() 
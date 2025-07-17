# Classificação de Ovos de Parasitas - Estudo Comparativo de Modelos

Este projeto implementa e compara três modelos de deep learning state-of-the-art para classificação de ovos de parasitas utilizando o dataset Chula-ParasiteEgg-11. O estudo foi projetado para uma tese de bacharelado comparando arquiteturas CNN, Vision Transformer e Híbrida.

## 🎯 Visão Geral do Projeto

O objetivo é classificar 11 tipos diferentes de ovos de parasitas a partir de imagens médicas:
- Ascaris lumbricoides
- Capillaria philippinensis  
- Enterobius vermicularis
- Fasciolopsis buski
- Ovo de Ancilóstomo
- Hymenolepis diminuta
- Hymenolepis nana
- Opisthorchis viverrine
- Paragonimus spp
- Ovo de Taenia spp.
- Trichuris trichiura

## 🏗️ Arquiteturas dos Modelos

### 1. Modelo CNN - EfficientNetV2-S
- **Arquitetura**: EfficientNetV2-S com classificador customizado
- **Vantagens**: 
  - Performance comprovada em tarefas de imagens médicas
  - Uso eficiente de parâmetros
  - Excelentes capacidades de transfer learning
- **Caso de Uso**: Performance baseline CNN para comparação

### 2. Vision Transformer - Tiny ViT
- **Arquitetura**: Vision Transformer Tiny com design compacto
- **Vantagens**:
  - Performance state-of-the-art em análise de imagens médicas
  - Captura características locais e globais
  - Mecanismo de atenção eficiente
- **Caso de Uso**: Abordagem moderna baseada em transformer

### 3. Modelo Híbrido - EfficientNetV2-S + Tiny ViT
- **Arquitetura**: Combina backbone CNN EfficientNetV2-S com Vision Transformer Tiny
- **Vantagens**:
  - Melhor dos dois mundos: extração de características locais da CNN + atenção global do ViT
  - Fusão de características baseada em atenção
  - Performance superior através de forças complementares
- **Caso de Uso**: Abordagem híbrida avançada para máxima performance

## 📁 Estrutura do Projeto

```
parasyte-detector/
├── config.py                # Configurações do modelo lightweight
├── dataset.py               # Carregamento e pré-processamento do dataset
├── models.py                # Arquiteturas dos modelos
├── trainer.py               # Lógica de treinamento e avaliação
├── main.py                  # Script principal de execução
├── requirements.txt         # Dependências Python
├── README.md               # Este arquivo
├── MODEL_ANALYSIS_LIGHT.md # Análise técnica dos modelos lightweight
├── Chula-ParasiteEgg-11/   # Dataset de treinamento
├── Chula-ParasiteEgg-11_test/ # Dataset de teste
├── models_light/           # Checkpoints dos modelos salvos
├── results_light/          # Resultados de avaliação e gráficos
└── logs_light/            # Logs do TensorBoard
```

## 🚀 Instalação

1. **Clone o repositório**:
```bash
git clone <url-do-repositorio>
cd parasyte-detector
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Verifique a estrutura do dataset**:
Certifique-se de que seu dataset está organizado como:
```
Chula-ParasiteEgg-11/
└── data/
    ├── Ascaris lumbricoides/
    ├── Capillaria philippinensis/
    ├── Enterobius vermicularis/
    └── ... (outras classes)
```

## 🎮 Uso

### Início Rápido
Execute o experimento completo comparando todos os três modelos:

```bash
python main.py
```

### Experimento Melhorado (Anti-Overfitting)
Execute o experimento com técnicas anti-overfitting:

```bash
python run_improved.py
```

**Melhorias implementadas**:
- Data augmentation mais agressivo
- Dropout aumentado (0.2 → 0.5)
- Learning rate reduzido (2e-4 → 5e-5)
- Weight decay aumentado (1e-4 → 1e-3)
- Label smoothing (0.1)
- Gradient clipping (1.0)
- Early stopping mais paciente (5 → 10 épocas)

Isso irá:
- Treinar todos os três modelos (CNN, ViT, Híbrido)
- Gerar gráficos de comparação e métricas
- Salvar resultados para análise da tese
- Criar relatórios abrangentes

### Treinamento Individual de Modelos
Treine um modelo específico:

```python
from trainer import ParasiteTrainer
import config

# Treinar modelo CNN
trainer = ParasiteTrainer('cnn', config.config)
trainer.train()
trainer.evaluate()

# Treinar Vision Transformer
trainer = ParasiteTrainer('vit', config.config)
trainer.train()
trainer.evaluate()

# Treinar modelo Híbrido
trainer = ParasiteTrainer('hybrid', config.config)
trainer.train()
trainer.evaluate()
```

### Configuração
Modifique `config.py` para ajustar hiperparâmetros:

```python
# Exemplo de mudanças de configuração
config.num_epochs = 100
config.batch_size = 16
config.learning_rate = 5e-5
config.image_size = 384
```

## 📊 Resultados e Análise

O experimento gera resultados abrangentes incluindo:

### Métricas de Performance
- **Acurácia**: Acurácia geral de classificação
- **Precisão/Revocação/F1-Score**: Performance por classe
- **Matriz de Confusão**: Análise detalhada de erros
- **Curvas de Treinamento**: Perda e acurácia ao longo do tempo

### Visualizações
- Gráficos de comparação de modelos
- Histórico de treinamento
- Matrizes de confusão
- Breakdown de performance por classe

### Arquivos de Saída
- `results_light/comparacao_modelos.json`: Dados detalhados de comparação
- `results_light/relatorio_tese.json`: Relatório abrangente da tese
- `results_light/*_historico_treinamento.png`: Curvas de treinamento
- `results_light/*_matriz_confusao.png`: Matrizes de confusão
- `models_light/melhor_*.pth`: Melhores checkpoints dos modelos

## 🔬 Design Experimental

### Divisão do Dataset
- **Treinamento**: 80% dos dados de treinamento (modelo aprende com esses dados)
- **Validação**: 20% dos dados de treinamento (monitoramento durante treinamento)
- **Teste**: Dataset de teste separado (avaliação final real)

**⚠️ IMPORTANTE**: Os valores de validação (95-98%) são diferentes dos valores de teste (56-73%) porque:
- **Validação**: Vem do mesmo dataset de treinamento (dados similares)
- **Teste**: Dataset completamente separado (dados novos)
- **Diferença**: Indica overfitting - o modelo memoriza os dados de treinamento

### Aumento de Dados
- Flips horizontais/verticais
- Rotações aleatórias
- Ajustes de brilho/contraste
- Efeitos de ruído e blur
- Jittering de cor

### Estratégia de Treinamento
- **Otimizador**: AdamW com weight decay
- **Scheduler**: ReduceLROnPlateau
- **Loss**: Cross-entropy loss
- **Early Stopping**: Baseado na acurácia de validação
- **Transfer Learning**: Modelos pré-treinados no ImageNet

## 🎯 Modelos Lightweight

Este projeto utiliza versões lightweight dos modelos para treinamento eficiente em GPUs locais:

### EfficientNetV2-S (CNN)
- **Parâmetros**: ~22M
- **Tamanho de entrada**: 224x224x3
- **Ideal para**: GPUs com 8GB VRAM ou menos

### Tiny ViT (Vision Transformer)
- **Parâmetros**: ~5M
- **Tamanho de entrada**: 224x224x3
- **Patch size**: 16x16
- **Dimensão de embedding**: 192

### Modelo Híbrido (EfficientNetV2-S + Tiny ViT)
- **Backbone CNN**: EfficientNetV2-S (~22M parâmetros)
- **Backbone ViT**: Tiny ViT (~5M parâmetros)
- **Fusão**: Mecanismo de atenção para combinar características

## 📈 Performance Esperada
- **Acurácia**: 85–92% (ligeiramente menor que modelos grandes, mas muito mais rápido)
- **Tempo de Treinamento**: 3–5x mais rápido que modelos grandes
- **Melhor para**: Prototipagem, experimentos locais e iteração rápida

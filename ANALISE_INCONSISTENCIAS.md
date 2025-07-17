# Análise das Inconsistências nos Resultados

## Problema Identificado

Foram encontradas inconsistências entre os valores mostrados no terminal durante o treinamento e os valores finais no arquivo `model_comparison.json`. Esta análise explica as diferenças e suas causas.

## Diferença entre Validação e Teste

### Valores do Terminal (Validação)
Os valores mostrados no terminal são de **validação** durante o treinamento:
- **CNN**: 98.73% (melhor validação)
- **ViT**: 95.41% (melhor validação)  
- **Híbrido**: 98.73% (melhor validação)

### Valores do JSON (Teste)
Os valores no `model_comparison.json` são de **teste** (avaliação final):
- **CNN**: 67.59% (teste)
- **ViT**: 56.32% (teste)
- **Híbrido**: 73.73% (teste)

## Análise de Overfitting

### Definição
**Overfitting** ocorre quando o modelo "memoriza" os dados de treinamento em vez de aprender padrões generalizáveis.

### Evidências de Overfitting Severo

| Modelo | Validação | Teste | Gap | Severidade |
|--------|-----------|-------|-----|------------|
| CNN | 98.73% | 67.59% | 31.14% | **Muito Alto** |
| ViT | 95.41% | 56.32% | 39.09% | **Muito Alto** |
| Híbrido | 98.73% | 73.73% | 24.96% | **Alto** |

### Interpretação
- **Gap > 20%**: Overfitting severo
- **Gap > 30%**: Overfitting muito severo
- Todos os modelos estão sofrendo overfitting significativo

## Causas Prováveis

### 1. Dataset Pequeno
- O dataset pode ser insuficiente para treinar modelos complexos
- Poucos exemplos por classe levam à memorização

### 2. Configuração Lightweight
- Modelos muito simples para a complexidade da tarefa
- Falta de regularização adequada

### 3. Hiperparâmetros
- Learning rate pode estar muito alto
- Early stopping pode estar parando muito cedo
- Dropout pode estar insuficiente

## Recomendações

### 1. Aumentar Regularização
```python
# Aumentar dropout
cnn_dropout: float = 0.5  # Atual: 0.2
# Adicionar weight decay mais forte
weight_decay: float = 1e-3  # Atual: 1e-4
```

### 2. Data Augmentation Mais Agressivo
```python
# Adicionar mais transformações
A.RandomErasing(p=0.3),
A.CoarseDropout(p=0.2),
A.Cutout(p=0.2)
```

### 3. Learning Rate Menor
```python
learning_rate: float = 5e-5  # Atual: 2e-4
```

### 4. Early Stopping Mais Paciente
```python
patience: int = 10  # Atual: 5
```

### 5. Validação Cruzada
Implementar k-fold cross-validation para melhor avaliação.

## Conclusão

As inconsistências são **normais** e indicam um problema real de overfitting. Os valores de teste são os **corretos** para avaliação final, enquanto os valores de validação mostram o potencial máximo dos modelos.

**Ação necessária**: Implementar as recomendações acima para reduzir o overfitting e melhorar a generalização dos modelos. 
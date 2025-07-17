# Otimização Baseada no CoAtNet (93% Acurácia)

## 📊 Análise do Paper CoAtNet

O paper "Parasitic egg recognition using convolution and attention network" demonstrou que o CoAtNet atingiu **93% de acurácia** no dataset Chula-ParasiteEgg-11. Analisamos as técnicas utilizadas e implementamos otimizações para modelos leves.

## 🔍 Técnicas-Chave Identificadas

### 1. **Input Size Otimizado**
- **CoAtNet**: 384x384 pixels
- **Nossa implementação**: 384x384 (aumentado de 224x224)
- **Justificativa**: Maior resolução melhora a detecção de detalhes microscópicos

### 2. **Batch Size Otimizado**
- **CoAtNet**: 8-16 por GPU
- **Nossa implementação**: 8 (otimizado para RTX 2070 Super)
- **Justificativa**: Balanceia memória e estabilidade do gradiente

### 3. **Learning Rate com Scheduler**
- **CoAtNet**: ReduceLROnPlateau
- **Nossa implementação**: ReduceLROnPlateau com factor=0.1, patience=5
- **Justificativa**: Reduz LR automaticamente quando performance para de melhorar

### 4. **Data Augmentation Específico**
- **CoAtNet**: Blur e noise para simular condições microscópicas
- **Nossa implementação**: 
  - GaussianBlur, MotionBlur, MedianBlur
  - GaussNoise, ISONoise, MultiplicativeNoise
  - Rotação moderada (-15° a +15°)
  - Ajustes de cor suaves

### 5. **Regularização Moderada**
- **CoAtNet**: Dropout 0.5
- **Nossa implementação**: 
  - Dropout 0.5
  - Weight decay 5e-4 (reduzido)
  - Label smoothing 0.1
  - Gradient clipping 1.0

### 6. **Fine-tuning Completo**
- **CoAtNet**: Todas as camadas treinadas
- **Nossa implementação**: Fine-tuning completo dos modelos leves

## 🚀 Configurações Implementadas

### `config_optimized.py`
```python
# Configurações baseadas no CoAtNet
image_size: int = 384          # Aumentado de 224
batch_size: int = 8            # Otimizado para RTX 2070 Super
learning_rate: float = 1e-4    # Learning rate moderado
weight_decay: float = 5e-4     # Regularização moderada
num_epochs: int = 50           # Mais épocas para convergência

# Learning rate scheduler
scheduler_factor: float = 0.1
scheduler_patience: int = 5
scheduler_min_lr: float = 1e-6

# Augmentation específico
augmentation_blur: bool = True
augmentation_noise: bool = True
augmentation_rotation: int = 15
```

### `dataset_optimized.py`
- **Augmentation baseado no CoAtNet**:
  - Blur: GaussianBlur, MotionBlur, MedianBlur
  - Noise: GaussNoise, ISONoise, MultiplicativeNoise
  - Transformações geométricas moderadas
  - Ajustes de cor suaves

### `trainer_optimized.py`
- **Learning rate scheduler**: ReduceLROnPlateau
- **Gradient clipping**: 1.0
- **Label smoothing**: 0.1
- **Early stopping**: Mais paciente (patience=15)

## 📈 Comparação de Configurações

| Aspecto | Light | Improved | Optimized (CoAtNet) |
|---------|-------|----------|---------------------|
| Image Size | 224x224 | 224x224 | **384x384** |
| Batch Size | 32 | 16 | **8** |
| Learning Rate | 1e-3 | 5e-5 | **1e-4** |
| Weight Decay | 1e-4 | 1e-3 | **5e-4** |
| Dropout | 0.3 | 0.5 | **0.5** |
| LR Scheduler | ❌ | ❌ | **✅ ReduceLROnPlateau** |
| Augmentation | Básico | Agressivo | **Blur + Noise** |
| Gradient Clipping | ❌ | ✅ | **✅** |
| Label Smoothing | ❌ | ✅ | **✅** |

## 🎯 Expectativas de Melhoria

### Baseado no CoAtNet (93% acurácia):
1. **Input size maior**: +5-10% acurácia
2. **Augmentation específico**: +3-5% acurácia
3. **LR scheduler**: +2-3% acurácia
4. **Regularização otimizada**: Melhor generalização

### Meta Realista:
- **Acurácia de teste**: 75-80% (vs 68-74% atual)
- **Redução do overfitting**: Gap validação-teste < 10%
- **Melhor F1-score**: Especialmente para classes difíceis

## 🔧 Como Executar

### Treinar todos os modelos:
```bash
cd parasyte-detector
source parasite_env/bin/activate
python run_optimized.py --model all
```

### Treinar modelo específico:
```bash
python run_optimized.py --model hybrid
```

### Arquivos gerados:
- `results_optimized/`: Resultados e gráficos
- `models_optimized/`: Modelos treinados
- `logs_optimized/`: Logs de treinamento

## 📊 Monitoramento

### Métricas importantes:
1. **Acurácia de teste**: Meta > 75%
2. **Gap validação-teste**: Meta < 10%
3. **F1-score por classe**: Especialmente classes difíceis
4. **Tempo de treinamento**: ~2-3 horas por modelo

### Gráficos gerados:
- Histórico de treinamento
- Matriz de confusão
- Comparação de modelos
- Análise de overfitting

## 🎯 Próximos Passos

1. **Executar experimento otimizado**
2. **Comparar com resultados anteriores**
3. **Analisar métricas por classe**
4. **Ajustar hiperparâmetros se necessário**
5. **Testar ensemble dos melhores modelos**

## 📚 Referências

- **Paper CoAtNet**: "Parasitic egg recognition using convolution and attention network"
- **Dataset**: Chula-ParasiteEgg-11 (ICIP 2022 Challenge)
- **Técnicas**: ReduceLROnPlateau, Blur+Noise augmentation, Gradient clipping 
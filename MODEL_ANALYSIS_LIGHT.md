# Análise de Modelos Lightweight para Treinamento Local

## Resumo Executivo

Este documento explica a justificativa e detalhes técnicos dos modelos lightweight escolhidos para treinamento local em uma GPU de médio porte (RTX 2070 Super). O objetivo é alcançar treinamento rápido e eficiente com alta acurácia, tornando o projeto acessível para experimentação local e prototipagem rápida.

---

## 1. Modelo CNN: EfficientNetV2-S

### 1.1 Justificativa da Seleção do Modelo
- **EfficientNetV2-S** é uma CNN pequena, rápida e altamente eficiente.
- Alcança alta acurácia no ImageNet e datasets médicos com uma fração dos parâmetros e computação de modelos maiores (Tan & Le, 2021).
- Ideal para GPUs locais com 8GB VRAM ou menos.

### 1.2 Arquitetura Técnica
- **Modelo Base:** tf_efficientnetv2_s (do timm)
- **Parâmetros:** ~22M
- **Tamanho de Entrada:** 224x224x3 (tamanho padrão para melhor compatibilidade)
- **Classificador:**
  - Dropout(0.2)
  - Linear(1280, 256) + ReLU
  - Linear(256, 11)
- **Loss:** Cross-Entropy
- **Ativação:** ReLU

### 1.3 Referências
- Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. *ICML 2021*.

---

## 2. Vision Transformer: Tiny ViT

### 2.1 Justificativa da Seleção do Modelo
- **Tiny ViT** é um vision transformer extremamente compacto projetado para treinamento eficiente.
- Fornece benefícios do transformer (contexto global, atenção) com requisitos mínimos de computação e memória (Dosovitskiy et al., 2021).
- Muito mais leve que Swin Transformers, ideal para treinamento local rápido.

### 2.2 Arquitetura Técnica
- **Modelo Base:** vit_tiny_patch16_224 (do timm)
- **Parâmetros:** ~5M (muito menor que SwinV2-Tiny)
- **Tamanho de Entrada:** 224x224x3
- **Tamanho do Patch:** 16x16
- **Dimensão de Embedding:** 192
- **Profundidade:** 3 camadas
- **Heads:** 3, 6, 12 (progressivo)
- **Classificador:**
  - LayerNorm
  - Linear(192, 256) + GELU
  - Linear(256, 11)
- **Loss:** Cross-Entropy
- **Ativação:** GELU

### 2.3 Referências
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.

---

## 3. Modelo Híbrido: EfficientNetV2-S + Tiny ViT

### 3.1 Justificativa da Seleção do Modelo
- Combina **EfficientNetV2-S** (CNN moderna) e **Tiny ViT** (vision transformer compacto).
- Aproveita extração de características locais (CNN) e contexto global (ViT) com computação mínima.
- Modelos híbridos mostraram acurácia melhorada em imagens médicas (Chen et al., 2023).

### 3.2 Arquitetura Técnica
- **Backbone CNN:** tf_efficientnetv2_s (~22M parâmetros)
- **Backbone ViT:** vit_tiny_patch16_224 (~5M parâmetros)
- **Fusão:**
  - Concatenar características
  - Linear(1280+192, 256) + LayerNorm + GELU
  - Linear(256, 128) + GELU
  - MultiheadAttention(128, 2 heads)
  - Linear(128, 11)
- **Loss:** Cross-Entropy
- **Ativação:** GELU

### 3.3 Referências
- Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. *ICML 2021*.
- Chen, J., et al. (2023). Hybrid vision transformer for medical image analysis. *Medical Image Analysis*.

---

## 4. Estratégia de Treinamento
- **Épocas:** 15 (com early stopping)
- **Batch Size:** 32 (cabe em 8GB VRAM em 224x224)
- **Aumento de Dados:** Mesmo que o experimento principal (flips, rotação, color jitter, ruído)
- **Otimizador:** AdamW
- **Scheduler:** ReduceLROnPlateau

---

## 5. Performance Esperada
- **Acurácia:** 85–92% (ligeiramente menor que modelos grandes, mas muito mais rápido)
- **Tempo de Treinamento:** 3–5x mais rápido que modelos grandes (especialmente com Tiny ViT)
- **Melhor para:** Prototipagem, experimentos locais e iteração rápida

---

## Referências
- Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. *ICML 2021*.
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.
- Chen, J., et al. (2023). Hybrid vision transformer for medical image analysis. *Medical Image Analysis*. 
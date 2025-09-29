# Projeto: Treinamento e Modificação de CNN

## Objetivo
Desenvolver um projeto de aprendizado de máquina usando arquiteturas CNN para classificação de imagens nos datasets CIFAR-10 e CIFAR-100, com o objetivo de otimizar o desempenho para superar 90% de acurácia.

## Estrutura do Projeto

```
├── main.py                 # Script principal com todas as implementações
├── requirements.txt        # Dependências do projeto
├── README.md              # Documentação do projeto
├── data/                  # Datasets CIFAR (baixados automaticamente)
├── models/                # Modelos treinados salvos
├── plots/                 # Visualizações e gráficos
└── results/               # Resultados e métricas
```

## Etapas do Projeto

### Etapa 1: Preparação do Modelo Base
- **Arquitetura**: CNN base com 3 camadas convolucionais
- **Dataset**: CIFAR-10 (10 classes, imagens 32x32)
- **Características**:
  - Batch Normalization
  - Dropout (0.5)
  - MaxPooling
  - 2 camadas fully connected

### Etapa 2: Modificação da Arquitetura
- **Melhorias implementadas**:
  - 4 camadas convolucionais (vs 3 no modelo base)
  - Mais filtros por camada (64, 128, 256, 512)
  - Global Average Pooling
  - 3 camadas fully connected
  - Dropout diferenciado (0.3 e 0.5)
  - Regularização L2

### Etapa 3: Teste com Segundo Dataset
- **Dataset**: CIFAR-100 (100 classes, imagens 32x32)
- **Teste**: Ambos os modelos (base e modificado) no CIFAR-100
- **Comparação**: Performance entre os dois modelos

### Etapa 4: Técnica de Estado da Arte
- **Arquitetura**: ResNet-18
- **Modificações**:
  - Conv1 adaptado para imagens 32x32
  - MaxPooling removido
  - Treinamento em ambos os datasets

## Instalação e Execução

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o projeto

#### **Versão Otimizada (Recomendada):**
```bash
cd "/Users/macbookair/Desktop/Treinamento e Modificação de CNN"
./venv/bin/python main_optimized.py
```

#### **Versão Base:**
```bash
cd "/Users/macbookair/Desktop/Treinamento e Modificação de CNN"
./venv/bin/python main.py
```

#### **Alternativa com ativação manual do venv:**
```bash
cd "/Users/macbookair/Desktop/Treinamento e Modificação de CNN"
source venv/bin/activate
python main_optimized.py
```

### ⚠️ **Nota Importante:**
Se encontrar erro `ModuleNotFoundError: No module named 'tqdm'`, use o comando direto com o Python do venv:
```bash
./venv/bin/python main_optimized.py
```

## Versões Disponíveis

### **main.py - Versão Base**
- Treinamento completo com 50 épocas
- Batch size: 128
- Implementação padrão de todas as arquiteturas
- Tempo de execução: ~2-3 horas

### **main_optimized.py - Versão Otimizada (Recomendada)**
- **Treinamento mais rápido** (30 épocas vs 50)
- **Batch size maior** (256 vs 128)
- **Otimizações de performance**:
  - AdamW optimizer com weight decay
  - CosineAnnealingLR scheduler
  - Pin memory habilitado
  - Non-blocking transfers
  - Mais workers (4 vs 2)
- **ResNet customizado** para CIFAR
- **Diagramas de arquitetura** automáticos
- Tempo de execução: ~1-1.5 horas

## Resultados Esperados

O projeto gera automaticamente:
- **Modelos treinados** na pasta `models/`
- **Gráficos de treinamento** na pasta `plots/`
- **Matrizes de confusão** para cada modelo
- **Relatório de classificação** detalhado
- **Comparação final** de todos os modelos
- **Diagramas de arquitetura** comparativos

## Arquiteturas Implementadas

### 1. CNN Base
```python
Conv2d(3→32) → BatchNorm → ReLU → MaxPool
Conv2d(32→64) → BatchNorm → ReLU → MaxPool  
Conv2d(64→128) → BatchNorm → ReLU → MaxPool
Flatten → FC(512) → Dropout → FC(10/100)
```

### 2. CNN Modificada
```python
Conv2d(3→64) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
Conv2d(64→128) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
Conv2d(128→256) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
Conv2d(256→512) → BatchNorm → ReLU → MaxPool → Dropout(0.5)
GlobalAvgPool → FC(1024) → Dropout → FC(512) → Dropout → FC(10/100)
```

### 3. ResNet-18 (SOTA)
```python
ResNet-18 adaptado para CIFAR
- Conv1 modificado para 32x32
- MaxPooling removido
- FC final adaptado para número de classes
```

## Hiperparâmetros

- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Optimizer**: Adam
- **Scheduler**: StepLR (step_size=20, gamma=0.5)
- **Weight Decay**: 1e-4
- **Data Augmentation**: RandomCrop, RandomHorizontalFlip

## Métricas de Avaliação

- **Acurácia**: Percentual de classificação correta
- **Loss**: CrossEntropyLoss
- **Matriz de Confusão**: Visualização das predições
- **Relatório de Classificação**: Precision, Recall, F1-score

## Objetivo de Performance

- **Meta**: >90% de acurácia no CIFAR-10
- **Verificação**: Automática no final da execução
- **Relatório**: Status de cada modelo em relação ao objetivo

## Tecnologias Utilizadas

- **PyTorch**: Framework de deep learning
- **TorchVision**: Datasets e modelos pré-treinados
- **Matplotlib/Seaborn**: Visualizações
- **Scikit-learn**: Métricas de avaliação
- **NumPy**: Computação numérica

## Estrutura de Saída

### Arquivos Gerados
- `models/base_cnn_cifar10.pth`
- `models/modified_cnn_cifar10.pth`
- `models/resnet_cifar10.pth`
- `models/resnet_cifar100.pth`

### Visualizações
- `plots/*_training_history.png`
- `plots/*_confusion_matrix_*.png`

## Execução Automática

O script `main.py` executa automaticamente todas as etapas:
1. Carrega e prepara os datasets
2. Treina o modelo base
3. Treina o modelo modificado
4. Testa ambos no CIFAR-100
5. Implementa e treina ResNet
6. Gera comparação final
7. Verifica objetivo de 90% de acurácia

## Solução de Problemas

### **Projeto trava após criar diagramas de arquitetura**
**Solução:** O problema foi corrigido automaticamente. O código agora usa backend sem interface gráfica (`matplotlib.use('Agg')`) e não trava mais.

### **Erro: ModuleNotFoundError: No module named 'tqdm'**
**Solução:** Use o comando direto com o Python do venv:
```bash
./venv/bin/python main_optimized.py
```

### **Erro: CUDA out of memory**
**Solução:** Reduza o batch size no código ou use CPU:
```python
DEVICE = torch.device('cpu')  # Forçar uso de CPU
```

### **Ambiente virtual não ativa corretamente**
**Solução:** Use o caminho completo do Python:
```bash
./venv/bin/python main_optimized.py
```

## Requisitos do Sistema

- Python 3.8+
- CUDA (opcional, para aceleração GPU)
- 8GB+ RAM recomendado
- Espaço em disco: ~2GB para datasets e modelos
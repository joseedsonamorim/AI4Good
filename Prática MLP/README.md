# 🧠 MLP from Scratch - Classificação Médica

Implementação completa de uma Rede Neural Multi-Layer Perceptron (MLP) do zero para classificação médica usando datasets reais do Kaggle e UCI.

## ✨ Características

- **Implementação do zero**: Sem uso de TensorFlow, PyTorch ou similares
- **Estrutura de grafo**: Representação visual da rede como grafo de nós e conexões
- **Forward & Backpropagation**: Implementação manual completa
- **Visualização em tempo real**: Acompanhe o treinamento com gráficos dinâmicos
- **Dois datasets reais**: Heart Disease (Kaggle) e Diabetes (UCI)
- **Menu interativo**: Escolha entre os datasets disponíveis
- **Interface moderna**: Design inspirado no aesthetic Apple liquid glass

## 🏗️ Arquitetura

A MLP implementada possui:
- **Camada de entrada**: Variável (13 features para Heart Disease, 11 para Diabetes)
- **Camadas ocultas**: 8 e 4 neurônios respectivamente
- **Camada de saída**: 1 neurônio (classificação binária)
- **Função de ativação**: Sigmoid
- **Otimização**: Gradiente descendente

## 🚀 Como executar

1. **Instalar dependências**:
```bash
pip install -r requirements.txt
```

2. **Executar o programa**:
```bash
python mlp_from_scratch.py
```

## 📊 Visualizações

O programa oferece 4 visualizações em tempo real com explicações detalhadas:

1. **🏗️ Arquitetura da Rede**: Grafo interativo com neurônios e conexões coloridas
2. **📈 Progresso do Treinamento**: Curvas de loss e accuracy com valores atuais
3. **⚖️ Distribuição dos Pesos**: Histograma separando pesos positivos e negativos
4. **🔥 Ativações das Camadas**: Barras coloridas mostrando atividade por camada

**📚 Explicações Incluídas**: Cada gráfico possui legendas e textos explicativos para que qualquer pessoa possa entender o que está acontecendo durante o treinamento.

## 🔬 Funcionalidades Técnicas

### Forward Propagation
- Propagação dos valores através das camadas
- Aplicação de funções de ativação
- Cálculo de saídas finais

### Backward Propagation
- Cálculo de gradientes
- Atualização de pesos e bias
- Otimização por gradiente descendente

### Datasets Disponíveis

#### ❤️ Heart Disease (Kaggle)
- **Fonte**: Kaggle (johnsmith88/heart-disease-dataset)
- **Amostras**: 1,025 casos reais
- **Features**: 13 características clínicas
- **Classes**: 499 casos sem doença, 526 com doença cardíaca
- **Problema**: Classificação de doença cardíaca

#### 🩺 Diabetes (UCI)
- **Fonte**: UCI ML Repository (ID: 296)
- **Amostras**: 101,766 casos reais
- **Features**: 11 características numéricas selecionadas
- **Classes**: 54,864 sem readmissão, 46,902 com readmissão
- **Problema**: Classificação de readmissão hospitalar

**Ambos datasets**: Normalização automática e divisão treino/teste (80/20)

## 📈 Métricas

- **Loss**: Erro quadrático médio
- **Accuracy**: Taxa de acerto
- **Confiança**: Probabilidade da predição

## 🎯 Resultados Esperados

### ❤️ Heart Disease
- **Accuracy de teste**: ~80-85% (testado: 81.46%)
- **Convergência**: ~300-500 épocas
- **Tempo**: ~58 segundos (500 épocas)

### 🩺 Diabetes
- **Accuracy de teste**: ~60-65% (testado: 63.36%)
- **Convergência**: ~200-400 épocas
- **Tempo**: ~2-3 minutos (500 épocas)

## 🛠️ Estrutura do Código

```
mlp_from_scratch.py
├── Node: Representação de neurônios
├── Connection: Conexões entre neurônios
├── MLP: Classe principal da rede neural
├── Visualizações: Gráficos em tempo real
└── Dataset: Carregamento do dataset real do Kaggle
```

## 🔧 Personalização

Você pode facilmente modificar:
- **Arquitetura**: Alterar número de camadas e neurônios
- **Learning rate**: Taxa de aprendizado
- **Funções de ativação**: Sigmoid, ReLU, Linear
- **Visualizações**: Adicionar novas métricas

## 📚 Conceitos Demonstrados

- Redes neurais artificiais
- Forward e backward propagation
- Gradiente descendente
- Funções de ativação
- Normalização de dados
- Visualização de dados
- Programação orientada a objetos

---

**Desenvolvido com ❤️ para aprendizado de Machine Learning**

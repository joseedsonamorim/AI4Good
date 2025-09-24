# 📋 Instruções de Uso - MLP from Scratch

## 🚀 Execução Rápida

### Opção 1: Script Automático (Recomendado)
```bash
./run.sh
```

### Opção 2: Manual
```bash
# Criar ambiente virtual
python3 -m venv mlp_env

# Instalar dependências
mlp_env/bin/pip install numpy matplotlib pandas seaborn

# Executar
mlp_env/bin/python mlp_from_scratch.py
```

## 📁 Estrutura do Projeto

```
Criando uma MLP/
├── mlp_from_scratch.py      # Implementação principal da MLP
├── run.sh                   # Script de execução automática
├── requirements.txt         # Dependências Python
├── README.md               # Documentação principal
├── INSTRUCOES.md           # Este arquivo
└── mlp_env/                # Ambiente virtual Python
```

## 🎯 Execução

### Treinamento Completo
- **Arquivo**: `mlp_from_scratch.py`
- **Dataset**: Heart Disease real do Kaggle (1,025 amostras)
- **Tempo**: ~2-5 minutos
- **Objetivo**: Treinamento completo com visualização em tempo real

## 🔧 Personalização

### Modificar Arquitetura
Edite diretamente no `mlp_from_scratch.py`:
```python
# Na função main(), linha ~450
layers = [13, 16, 8, 1]  # Sua arquitetura
mlp = MLP(layers, learning_rate=0.05)  # Sua taxa de aprendizado
```

### Modificar Parâmetros de Treinamento
```python
# Na função main(), linha ~460
history = mlp.train(X_train, y_train, epochs=1000, visualize=True)
```

## 📊 Interpretando os Resultados

### Métricas Importantes
- **Loss**: Erro quadrático médio (menor = melhor)
- **Accuracy**: Taxa de acerto (maior = melhor)
- **Confiança**: Probabilidade da predição (0-1)

### Visualizações (Com Explicações Detalhadas)
1. **🏗️ Arquitetura**: Grafo com neurônios e conexões coloridas + legendas explicativas
2. **📈 Progresso**: Curvas de loss/accuracy + valores atuais em tempo real
3. **⚖️ Pesos**: Histograma separando pesos positivos/negativos + estatísticas
4. **🔥 Ativações**: Barras coloridas por intensidade + descrições das camadas

**📚 Painel Explicativo**: Texto na parte inferior explica todo o processo de treinamento

## 🐛 Solução de Problemas

### Erro: "ModuleNotFoundError"
```bash
# Reinstalar dependências
mlp_env/bin/pip install --upgrade numpy matplotlib pandas seaborn
```

### Erro: "Permission denied" no run.sh
```bash
chmod +x run.sh
```

### Visualização não aparece
- Verifique se está usando ambiente gráfico
- Tente executar sem visualização: `visualize=False`

### Performance lenta
- Reduza número de épocas
- Aumente batch_size
- Desative visualização

## 🎓 Conceitos Demonstrados

### Forward Propagation
1. Valores de entrada → primeira camada
2. Soma ponderada + bias
3. Aplicação de função de ativação
4. Propagação para próxima camada

### Backward Propagation
1. Cálculo do erro na saída
2. Propagação reversa dos gradientes
3. Atualização de pesos e bias
4. Otimização por gradiente descendente

### Estrutura de Grafo
- **Nós**: Representam neurônios
- **Conexões**: Representam pesos
- **Camadas**: Agrupamento de nós

## 🔬 Experimentos Sugeridos

### 1. Teste de Arquiteturas
```python
# Teste diferentes tamanhos na função main()
layers = [13, 16, 8, 4, 1]  # Arquitetura mais profunda
```

### 2. Teste de Learning Rates
```python
# Teste diferentes taxas
mlp = MLP(layers, learning_rate=0.01)  # Taxa menor
mlp = MLP(layers, learning_rate=0.5)    # Taxa maior
```

### 3. Teste de Funções de Ativação
```python
# Modifique em Node.activate() para testar ReLU ou tanh
def activate(self, x: float) -> float:
    return max(0, x)  # ReLU
```

## 📈 Resultados Esperados

### Dataset Heart Disease
- **Fonte**: Kaggle (johnsmith88/heart-disease-dataset)
- **Amostras**: 1,025 casos reais de pacientes
- **Features**: 13 características clínicas
- **Classes**: 499 casos sem doença, 526 com doença cardíaca
- **Accuracy**: 80-90% (testado: 81.46%)
- **Convergência**: 300-500 épocas
- **Loss final**: 0.1-0.2

## 🎯 Próximos Passos

1. **Experimente** diferentes configurações
2. **Analise** as visualizações em tempo real
3. **Compare** diferentes arquiteturas
4. **Implemente** novas funções de ativação
5. **Adicione** regularização (L1/L2)
6. **Teste** com outros datasets

## 💡 Dicas de Otimização

- Use **learning rate decay** para melhor convergência
- Implemente **early stopping** para evitar overfitting
- Adicione **momentum** ao gradiente descendente
- Experimente **batch normalization**
- Teste **dropout** para regularização

---

**Boa sorte com seus experimentos! 🚀**

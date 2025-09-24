# 📊 Menu de Seleção de Datasets

## 🎯 **Dois Datasets Disponíveis**

### ❤️ **Heart Disease (Kaggle)**
- **Fonte**: Kaggle (johnsmith88/heart-disease-dataset)
- **Amostras**: 1,025 casos reais de pacientes
- **Features**: 13 características clínicas
- **Problema**: Classificação de doença cardíaca (SIM/NÃO)
- **Classes**: 499 casos sem doença, 526 com doença cardíaca
- **Performance**: ~81% accuracy
- **Tempo**: ~58 segundos (500 épocas)

### 🩺 **Diabetes (UCI)**
- **Fonte**: UCI ML Repository (ID: 296)
- **Amostras**: 101,766 casos reais de pacientes
- **Features**: 11 características numéricas selecionadas
- **Problema**: Classificação de readmissão hospitalar (SIM/NÃO)
- **Classes**: 54,864 sem readmissão, 46,902 com readmissão
- **Performance**: ~63% accuracy
- **Tempo**: ~2-3 minutos (500 épocas)

## 🚀 **Como Usar o Menu**

### 1. **Executar o Programa**
```bash
./run.sh
# ou
mlp_env/bin/python mlp_from_scratch.py
```

### 2. **Selecionar Dataset**
```
🧠 MLP from Scratch - Seleção de Dataset
=============================================
📊 Escolha o dataset para treinamento:

1. ❤️ Heart Disease (Kaggle)
   • 1,025 amostras de pacientes cardíacos
   • 13 features clínicas
   • Classificação: Doença cardíaca (SIM/NÃO)

2. 🩺 Diabetes (UCI)
   • 101,766 amostras de pacientes diabéticos
   • 11 features numéricas selecionadas
   • Classificação: Readmissão hospitalar (SIM/NÃO)

3. 🚪 Sair

Digite sua escolha (1-3): 
```

### 3. **Escolher Opção**
- **Digite 1**: Para Heart Disease
- **Digite 2**: Para Diabetes
- **Digite 3**: Para sair

## 📊 **Comparação dos Datasets**

| Característica | Heart Disease | Diabetes |
|----------------|---------------|----------|
| **Amostras** | 1,025 | 101,766 |
| **Features** | 13 | 11 |
| **Complexidade** | Baixa | Alta |
| **Accuracy** | ~81% | ~63% |
| **Tempo** | ~1 min | ~3 min |
| **Uso** | Demonstração | Pesquisa |

## 🎯 **Recomendações de Uso**

### 🎓 **Para Aprendizado/Demonstração**
- **Escolha**: Heart Disease (opção 1)
- **Motivo**: Dataset menor, mais rápido, accuracy alta
- **Tempo**: ~1 minuto para treinamento completo

### 🔬 **Para Pesquisa/Desenvolvimento**
- **Escolha**: Diabetes (opção 2)
- **Motivo**: Dataset grande, mais desafiador, dados reais
- **Tempo**: ~3 minutos para treinamento completo

## 💡 **Dicas de Uso**

### ⚡ **Teste Rápido**
- Use Heart Disease para testes rápidos
- Treine com 50-100 épocas
- Tempo: ~5-10 segundos

### 🏆 **Treinamento Completo**
- Use Diabetes para resultados robustos
- Treine com 500+ épocas
- Tempo: ~3 minutos

### 🎨 **Visualização**
- Ambos datasets suportam visualização em tempo real
- Gráficos explicativos para cada dataset
- Acompanhe o progresso do treinamento

## 🔧 **Personalização**

### Modificar Arquitetura
```python
# Na função main(), linha ~673
layers = [X.shape[1], 16, 8, 1]  # Arquitetura maior
mlp = MLP(layers, learning_rate=0.05)  # Learning rate menor
```

### Modificar Épocas
```python
# Na função main(), linha ~680
history = mlp.train(X_train, y_train, epochs=1000, visualize=True)
```

---

**🎉 Agora você pode escolher entre dois datasets médicos reais para treinar sua MLP!**

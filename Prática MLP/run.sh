#!/bin/bash

echo "🧠 MLP from Scratch - Heart Disease Classification"
echo "=================================================="

# Verificar se o ambiente virtual existe
if [ ! -d "mlp_env" ]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv mlp_env
fi

# Instalar dependências
echo "📥 Instalando dependências..."
mlp_env/bin/pip install numpy matplotlib pandas seaborn kagglehub ucimlrepo

echo ""
echo "🚀 Iniciando MLP com seleção de dataset..."
echo "📊 Opções: Heart Disease (Kaggle) ou Diabetes (UCI)"
echo "🏗️ Arquitetura: [features, 8, 4, 1]"
echo ""

# Executar treinamento completo
mlp_env/bin/python mlp_from_scratch.py

echo ""
echo "✅ Treinamento concluído!"

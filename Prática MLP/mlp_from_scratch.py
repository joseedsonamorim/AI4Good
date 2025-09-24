import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import time
import random
import kagglehub
import os
from ucimlrepo import fetch_ucirepo

class Node:
    """Representa um nó na rede neural"""
    def __init__(self, layer: int, index: int, activation: str = 'sigmoid'):
        self.layer = layer
        self.index = index
        self.activation = activation
        self.value = 0.0
        self.gradient = 0.0
        self.bias = random.uniform(-0.5, 0.5)
        
    def activate(self, x: float) -> float:
        """Aplica função de ativação"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return max(0.0, x)
        elif self.activation == 'linear':
            return x
        return x
    
    def activation_derivative(self, x: float) -> float:
        """Derivada da função de ativação"""
        if self.activation == 'sigmoid':
            s = self.activate(x)
            return s * (1 - s)
        elif self.activation == 'relu':
            return 1 if x > 0 else 0
        elif self.activation == 'linear':
            return 1
        return 1

class Connection:
    """Representa uma conexão entre dois nós"""
    def __init__(self, from_node: Node, to_node: Node):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = random.uniform(-0.5, 0.5)
        self.gradient = 0.0

class MLP:
    """Multi-Layer Perceptron implementada como grafo"""
    
    def __init__(self, layers: List[int], learning_rate: float = 0.1, dataset_name: str = 'heart'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.current_dataset = dataset_name
        self.nodes = []
        self.connections = []
        self.training_history = []
        
        # Criar nós
        self._create_nodes()
        
        # Criar conexões
        self._create_connections()
        
    def _create_nodes(self):
        """Cria todos os nós da rede"""
        for layer_idx, num_nodes in enumerate(self.layers):
            layer_nodes = []
            for node_idx in range(num_nodes):
                if layer_idx == 0:
                    activation = 'linear'  # Camada de entrada
                elif layer_idx == len(self.layers) - 1:
                    activation = 'sigmoid'  # Camada de saída para classificação binária
                else:
                    activation = 'relu'    # Camadas ocultas
                node = Node(layer_idx, node_idx, activation)
                layer_nodes.append(node)
            self.nodes.append(layer_nodes)
    
    def _create_connections(self):
        """Cria todas as conexões entre camadas adjacentes"""
        for layer_idx in range(len(self.layers) - 1):
            current_layer = self.nodes[layer_idx]
            next_layer = self.nodes[layer_idx + 1]
            
            for from_node in current_layer:
                for to_node in next_layer:
                    connection = Connection(from_node, to_node)
                    self.connections.append(connection)
    
    def forward_propagation(self, inputs: List[float]) -> List[float]:
        """Propagação para frente"""
        # Definir valores de entrada
        for i, input_val in enumerate(inputs):
            if i < len(self.nodes[0]):
                self.nodes[0][i].value = input_val
        
        # Propagação através das camadas
        for layer_idx in range(1, len(self.layers)):
            current_layer = self.nodes[layer_idx]
            
            for node in current_layer:
                # Calcular soma ponderada
                weighted_sum = node.bias
                for connection in self.connections:
                    if connection.to_node == node:
                        weighted_sum += connection.from_node.value * connection.weight
                
                # Aplicar ativação
                node.value = node.activate(weighted_sum)
        
        # Retornar saídas da última camada
        return [node.value for node in self.nodes[-1]]
    
    def backward_propagation(self, targets: List[float]):
        """Propagação para trás"""
        # Calcular gradientes da camada de saída
        output_layer = self.nodes[-1]
        for i, node in enumerate(output_layer):
            error = targets[i] - node.value
            node.gradient = error * node.activation_derivative(node.value)
        
        # Propagação reversa através das camadas ocultas
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            current_layer = self.nodes[layer_idx]
            
            for node in current_layer:
                gradient_sum = 0.0
                for connection in self.connections:
                    if connection.from_node == node:
                        gradient_sum += connection.to_node.gradient * connection.weight
                
                node.gradient = gradient_sum * node.activation_derivative(node.value)
        
        # Atualizar pesos e bias
        for connection in self.connections:
            gradient = connection.to_node.gradient * connection.from_node.value
            connection.weight += self.learning_rate * gradient
        
        for layer in self.nodes[1:]:  # Skip input layer
            for node in layer:
                node.bias += self.learning_rate * node.gradient
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              batch_size: int = 32, visualize: bool = True):
        """Treina a rede neural"""
        print("🚀 Iniciando treinamento da MLP...")
        print("📚 O que você está vendo:")
        print("   🧠 Rede Neural aprendendo padrões em dados de pacientes cardíacos")
        print("   ➡️ Forward: Dados fluem da entrada para a saída")
        print("   ⬅️ Backward: Erro é propagado para ajustar os pesos")
        print("   🎯 Objetivo: Classificar doença cardíaca (SIM/NÃO)")
        print("")
        
        if visualize:
            print("🎨 Configurando visualização...")
            self._setup_visualization()
            print("✅ Visualização configurada!")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            
            # Embaralhar dados
            indices = np.random.permutation(len(X))
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                batch_loss = 0.0
                
                for j in range(len(batch_X)):
                    # Forward pass
                    outputs = self.forward_propagation(batch_X[j].tolist())
                    
                    # Calcular erro
                    target = batch_y[j]
                    if isinstance(target, (list, np.ndarray)):
                        target = target[0]
                    
                    error = target - outputs[0]
                    batch_loss += error ** 2
                    
                    # Backward pass
                    self.backward_propagation([target])
                    
                    # Contar predições corretas
                    prediction = 1 if outputs[0] > 0.5 else 0
                    if prediction == target:
                        correct_predictions += 1
                
                epoch_loss += batch_loss / len(batch_X)
            
            # Calcular métricas
            num_batches = max(1, len(X) // batch_size)
            avg_loss = epoch_loss / num_batches
            accuracy = correct_predictions / len(X)
            
            # Salvar histórico
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
            # Visualizar progresso (otimizado para melhor performance)
            if visualize and (epoch % 25 == 0 or epoch == 0 or epoch == epochs-1):
                print(f"🔄 Atualizando visualização (época {epoch})...")
                self._update_visualization(epoch, avg_loss, accuracy)
            
            # Log de progresso
            if epoch % 100 == 0:
                print(f"Época {epoch:4d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        print("✅ Treinamento concluído!")
        
        # Mostrar resultados finais na visualização
        if visualize:
            self._show_final_results()
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        predictions = []
        for sample in X:
            outputs = self.forward_propagation(sample.tolist())
            prediction = 1 if outputs[0] > 0.5 else 0
            predictions.append(prediction)
        return np.array(predictions)
    
    def _setup_visualization(self):
        """Configura a visualização similar ao exemplo fornecido"""
        plt.style.use('default')
        self.fig = plt.figure(figsize=(16, 10))
        
        # Criar layout principal: rede neural à esquerda, controles à direita
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1], hspace=0.1, wspace=0.1)
        
        # Área principal da rede neural
        self.network_ax = self.fig.add_subplot(gs[0, 0])
        self.network_ax.set_title('Visualizador de Rede Neural', fontsize=16, fontweight='bold', pad=20)
        
        # Painel de controle à direita
        self.control_ax = self.fig.add_subplot(gs[0, 1])
        self.control_ax.set_title('Painel de Controle', fontsize=14, fontweight='bold', pad=20)
        self.control_ax.axis('off')
        
        # Configurar área da rede neural
        self.network_ax.set_facecolor('white')
        self.network_ax.grid(True, alpha=0.3)
        
        # Configurar painel de controle
        self.control_ax.set_facecolor('#f0f0f0')
        
        # Adicionar texto inicial
        self.network_ax.text(0.5, 0.5, 'Iniciando treinamento...', 
                           ha='center', va='center', fontsize=14, 
                           transform=self.network_ax.transAxes)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Pequena pausa para garantir que a janela apareça
    
    def _update_visualization(self, epoch: int, loss: float, accuracy: float):
        """Atualiza a visualização"""
        try:
            # Limpar área da rede neural
            self.network_ax.clear()
            
            # Reconfigurar área da rede neural
            self.network_ax.set_facecolor('white')
            self.network_ax.grid(True, alpha=0.3)
            self.network_ax.set_title('Visualizador de Rede Neural', fontsize=16, fontweight='bold', pad=20)
            
            # Plotar arquitetura da rede neural
            self._plot_network_architecture()
            
            # Atualizar painel de controle
            self._update_control_panel(epoch, loss, accuracy)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Pausa um pouco mais longa para garantir atualização
            
        except Exception as e:
            print(f"⚠️ Erro na visualização: {e}")
            # Continuar mesmo com erro de visualização
    
    def _plot_network_architecture(self):
        """Plota a arquitetura da rede neural similar ao exemplo fornecido"""
        ax = self.network_ax
        
        # Configurações visuais
        layer_spacing = 2.5
        node_radius = 0.15
        max_nodes = max(self.layers)
        
        # Nomes das features de entrada (simplificados)
        input_names = ['age', 'sex', 'cp', 'pressao', 'colesterol', 'acucar', 
                      'ecg', 'freq', 'angina', 'oldpeak', 'slope', 'vessels', 'thal']
        
        # Desenhar neurônios (nós) organizados por camadas
        for layer_idx, layer_nodes in enumerate(self.nodes):
            x_pos = layer_idx * layer_spacing
            
            for node_idx, node in enumerate(layer_nodes):
                # Posicionar neurônios verticalmente centrados
                y_pos = (node_idx - len(layer_nodes) / 2) * 0.4
                
                # Cor baseada na ativação do neurônio
                activation_intensity = min(1.0, max(0.0, abs(node.value)))
                if activation_intensity > 0.7:
                    node_color = '#ff6b6b'  # Vermelho para alta ativação
                elif activation_intensity > 0.3:
                    node_color = '#ffa726'  # Laranja para média ativação
                else:
                    node_color = '#90a4ae'  # Cinza para baixa ativação
                
                # Desenhar neurônio
                neuron_circle = Circle((x_pos, y_pos), node_radius, 
                                     facecolor=node_color, alpha=0.8, 
                                     edgecolor='black', linewidth=1)
                ax.add_patch(neuron_circle)
                
                # Adicionar valor de ativação
                ax.text(x_pos, y_pos, f'{node.value:.2f}', 
                       ha='center', va='center', fontsize=8, color='black', weight='bold')
                
                # Adicionar labels para neurônios de entrada
                if layer_idx == 0 and node_idx < len(input_names):
                    ax.text(x_pos, y_pos - node_radius - 0.1, input_names[node_idx], 
                           ha='center', va='top', fontsize=7, color='black')
        
        # Desenhar conexões (pesos) com valores
        for i, connection in enumerate(self.connections):
            from_x = connection.from_node.layer * layer_spacing
            from_y = (connection.from_node.index - len(self.nodes[connection.from_node.layer]) / 2) * 0.4
            to_x = connection.to_node.layer * layer_spacing
            to_y = (connection.to_node.index - len(self.nodes[connection.to_node.layer]) / 2) * 0.4
            
            # Espessura da linha baseada na magnitude do peso
            weight_magnitude = abs(connection.weight)
            line_width = max(0.5, min(2.0, weight_magnitude * 3))
            
            # Cor baseada no sinal do peso
            if connection.weight > 0:
                line_color = '#2196f3'  # Azul para pesos positivos
            else:
                line_color = '#f44336'  # Vermelho para pesos negativos
            
            # Desenhar conexão
            ax.plot([from_x, to_x], [from_y, to_y], 
                   color=line_color, alpha=0.7, linewidth=line_width)
            
            # Adicionar valor do peso em algumas conexões (para não poluir)
            if i % 5 == 0:  # Mostrar apenas alguns pesos
                mid_x = (from_x + to_x) / 2
                mid_y = (from_y + to_y) / 2
                ax.text(mid_x, mid_y, f'{connection.weight:.2f}', 
                       ha='center', va='center', fontsize=6, color='black',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Adicionar labels das camadas
        layer_labels = ['Camada de Entrada', 'Camada Oculta 1', 'Camada Oculta 2', 'Camada de Saída']
        for i, label in enumerate(layer_labels):
            x_pos = i * layer_spacing
            y_pos = max_nodes / 2 * 0.4 + 0.3
            
            ax.text(x_pos, y_pos, label, ha='center', va='bottom', 
                   fontsize=10, color='black', weight='bold')
        
        # Configurar limites e aspecto
        ax.set_xlim(-0.3, (len(self.layers) - 1) * layer_spacing + 0.3)
        ax.set_ylim(-max_nodes / 2 * 0.4 - 0.5, max_nodes / 2 * 0.4 + 0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _update_control_panel(self, epoch: int, loss: float, accuracy: float):
        """Atualiza o painel de controle com informações da rede"""
        ax = self.control_ax
        ax.clear()
        ax.axis('off')
        
        # Título do painel
        ax.text(0.5, 0.95, 'Painel de Controle', ha='center', va='top', 
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Informações específicas solicitadas
        ax.text(0.05, 0.90, 'Accuracy:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.87, f'{accuracy:.3f}', ha='left', va='top', 
               fontsize=11, color='green', transform=ax.transAxes)
        
        ax.text(0.05, 0.82, 'Learning Rate:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.79, f'{self.learning_rate:.2f}', ha='left', va='top', 
               fontsize=11, transform=ax.transAxes)
        
        ax.text(0.05, 0.74, 'Activation:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.71, 'relu (ocultas) + sigmoid (saída)', ha='left', va='top', 
               fontsize=10, transform=ax.transAxes)
        
        ax.text(0.05, 0.66, 'Architecture:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        arch_str = ' | '.join(map(str, self.layers))
        ax.text(0.05, 0.63, arch_str, ha='left', va='top', 
               fontsize=11, transform=ax.transAxes)
        
        ax.text(0.05, 0.58, 'Pre-processing:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.55, 'standardize', ha='left', va='top', 
               fontsize=11, transform=ax.transAxes)
        
        # Comentários sobre análise dos datasets
        ax.text(0.05, 0.50, 'Análise dos Datasets:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        # Comentários específicos baseados no dataset atual
        if hasattr(self, 'current_dataset'):
            if self.current_dataset == 'heart':
                ax.text(0.05, 0.47, 'Heart Disease: Padronização', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
                ax.text(0.05, 0.44, 'resolve gradientes explodindo', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
                ax.text(0.05, 0.41, 'em dados médicos com', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
                ax.text(0.05, 0.38, 'escalas diferentes', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
            else:
                ax.text(0.05, 0.47, 'Diabetes: Normalização', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
                ax.text(0.05, 0.44, 'essencial para datasets', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
                ax.text(0.05, 0.41, 'com muitas features', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
                ax.text(0.05, 0.38, 'numéricas', ha='left', va='top', 
                       fontsize=9, transform=ax.transAxes)
        else:
            ax.text(0.05, 0.47, 'Padronização resolve', ha='left', va='top', 
                   fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.44, 'problemas de escala em', ha='left', va='top', 
                   fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.41, 'dados médicos', ha='left', va='top', 
                   fontsize=9, transform=ax.transAxes)
        
        # Informações de treinamento
        ax.text(0.05, 0.28, 'Status do Treinamento:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        ax.text(0.05, 0.25, f'Época: {epoch}', ha='left', va='top', 
               fontsize=10, transform=ax.transAxes)
        
        ax.text(0.05, 0.22, f'Loss: {loss:.4f}', ha='left', va='top', 
               fontsize=10, transform=ax.transAxes)
        
        # Estatísticas dos pesos
        ax.text(0.05, 0.16, 'Estatísticas dos Pesos:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        weights = [conn.weight for conn in self.connections]
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        
        ax.text(0.05, 0.13, f'Média: {mean_weight:.3f}', ha='left', va='top', 
               fontsize=9, transform=ax.transAxes)
        
        ax.text(0.05, 0.10, f'Desvio: {std_weight:.3f}', ha='left', va='top', 
               fontsize=9, transform=ax.transAxes)
        
        ax.text(0.05, 0.07, f'Intervalo: [{min_weight:.3f}, {max_weight:.3f}]', ha='left', va='top', 
               fontsize=9, transform=ax.transAxes)
        
        # Ativações das camadas
        ax.text(0.05, 0.01, 'Ativações das Camadas:', ha='left', va='top', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        layer_names = ['Entrada', 'Oculta 1', 'Oculta 2', 'Saída']
        for i, (layer_nodes, name) in enumerate(zip(self.nodes, layer_names)):
            avg_activation = np.mean([node.value for node in layer_nodes])
            ax.text(0.05, -0.02 - i*0.03, f"{name}: {avg_activation:.3f}", 
                   ha='left', va='top', fontsize=9, transform=ax.transAxes)
        
        # Legenda de cores
        ax.text(0.05, -0.18, 'Legenda:', ha='left', va='top', 
               fontsize=10, fontweight='bold', transform=ax.transAxes)
        
        ax.text(0.05, -0.21, '• Vermelho: Alta ativação', ha='left', va='top', 
               fontsize=8, color='#ff6b6b', transform=ax.transAxes)
        
        ax.text(0.05, -0.24, '• Laranja: Média ativação', ha='left', va='top', 
               fontsize=8, color='#ffa726', transform=ax.transAxes)
        
        ax.text(0.05, -0.27, '• Cinza: Baixa ativação', ha='left', va='top', 
               fontsize=8, color='#90a4ae', transform=ax.transAxes)
    
    def _show_final_results(self):
        """Mostra os resultados finais na visualização"""
        # Atualizar visualização final
        self._update_visualization(
            len(self.training_history) - 1,
            self.training_history[-1]['loss'],
            self.training_history[-1]['accuracy']
        )
        
        # Adicionar texto de resultados finais
        final_accuracy = self.training_history[-1]['accuracy']
        final_loss = self.training_history[-1]['loss']
        
        # Adicionar banner de resultados finais
        self.network_ax.text(0.5, 0.95, '🎯 RESULTADOS FINAIS', 
                           ha='center', va='top', fontsize=16, fontweight='bold',
                           color='green', transform=self.network_ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        self.network_ax.text(0.5, 0.88, f'Accuracy Final: {final_accuracy:.1%}', 
                           ha='center', va='top', fontsize=14, fontweight='bold',
                           color='darkgreen', transform=self.network_ax.transAxes)
        
        self.network_ax.text(0.5, 0.84, f'Loss Final: {final_loss:.4f}', 
                           ha='center', va='top', fontsize=12,
                           color='darkred', transform=self.network_ax.transAxes)
        
        # Adicionar informações sobre o dataset
        if hasattr(self, 'current_dataset'):
            dataset_name = "Heart Disease" if self.current_dataset == 'heart' else "Diabetes"
            self.network_ax.text(0.5, 0.80, f'Dataset: {dataset_name}', 
                               ha='center', va='top', fontsize=12,
                               color='blue', transform=self.network_ax.transAxes)
        
        plt.draw()
        plt.pause(3)  # Pausa mais longa para mostrar os resultados
        
        # Adicionar mensagem no terminal
        print(f"\n🎯 RESULTADOS FINAIS EXIBIDOS NA TELA:")
        print(f"   📊 Accuracy Final: {final_accuracy:.1%}")
        print(f"   📉 Loss Final: {final_loss:.4f}")
        print(f"   📋 Dataset: {dataset_name}")
        print("   ⏱️ Visualização será mantida aberta por alguns segundos...")


def load_heart_disease_data():
    """Carrega e prepara o dataset Heart Disease do Kaggle"""
    print("📊 Carregando dataset Heart Disease do Kaggle...")
    
    try:
        # Download do dataset do Kaggle
        path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
        csv_path = os.path.join(path, 'heart.csv')
        
        # Carregar dados
        df = pd.read_csv(csv_path)
        print("✅ Dataset carregado do Kaggle com sucesso!")
        
    except Exception as e:
        print(f"⚠️ Erro ao carregar dataset do Kaggle: {e}")
        print("🔄 Criando dataset sintético como fallback...")
        df = create_synthetic_heart_data()
    
    # Preparar dados
    print(f"📋 Dataset original: {df.shape[0]} amostras, {df.shape[1]} colunas")
    print(f"📊 Distribuição do target: {df['target'].value_counts().to_dict()}")
    
    # Remover linhas com valores faltantes
    df = df.dropna()
    
    # Separar features e target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Normalizar features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Evitar divisão por zero
    X = (X - X_mean) / X_std
    
    # Converter target para binário (0 ou 1)
    y = (y > 0).astype(int)
    
    print(f"📈 Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"🎯 Classes: {np.bincount(y)}")
    
    return X, y

def load_diabetes_data():
    """Carrega e prepara o dataset Diabetes otimizado"""
    print("📊 Carregando dataset Diabetes otimizado...")
    
    try:
        # Tentar carregar dataset UCI com tratamento de erros
        print("🔄 Tentando carregar dataset UCI (versão otimizada)...")
        
        # Tentar com timeout implícito
        diabetes_data = fetch_ucirepo(id=296)
        
        # Data
        X = diabetes_data.data.features
        y = diabetes_data.data.targets
        
        print("✅ Dataset carregado do UCI com sucesso!")
        
        # Preparar dados
        print(f"📋 Dataset original: {X.shape[0]} amostras, {X.shape[1]} colunas")
        print(f"📊 Target: {y.columns[0]}")
        
        # Selecionar apenas features numéricas para simplificar
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        # Remover linhas com valores faltantes
        X_clean = X_numeric.dropna()
        y_clean = y.loc[X_clean.index]
        
        # Converter target para binário (NO = 0, outros = 1)
        y_binary = (y_clean.iloc[:, 0] != 'NO').astype(int)
        
        # OTIMIZAÇÃO: Amostrar apenas uma parte dos dados para execução mais rápida
        print("⚡ Otimizando dataset para execução mais rápida...")
        n_samples = min(5000, len(X_clean))  # Máximo 5000 amostras
        
        # Amostragem estratificada para manter proporção das classes
        from sklearn.model_selection import train_test_split
        X_sample, _, y_sample, _ = train_test_split(
            X_clean, y_binary, 
            train_size=n_samples, 
            stratify=y_binary, 
            random_state=42
        )
        
        # Normalizar features
        X_values = X_sample.values
        X_mean = X_values.mean(axis=0)
        X_std = X_values.std(axis=0)
        X_std[X_std == 0] = 1  # Evitar divisão por zero
        X_normalized = (X_values - X_mean) / X_std
        
        print(f"📈 Dataset otimizado: {X_normalized.shape[0]} amostras, {X_normalized.shape[1]} features")
        print(f"🎯 Classes: {np.bincount(y_sample)}")
        print(f"📊 Features selecionadas: {len(numeric_features)} (apenas numéricas)")
        print("⚡ Dataset otimizado para execução rápida!")
        
        return X_normalized, y_sample.values
        
    except Exception as e:
        print(f"⚠️ Erro ao carregar dataset do UCI: {e}")
        print("🔄 Criando dataset sintético otimizado como fallback...")
        return create_synthetic_diabetes_data()

def create_synthetic_heart_data():
    """Cria dados sintéticos para demonstração"""
    np.random.seed(42)
    n_samples = 1000
    
    # Gerar features sintéticas baseadas em distribuições realistas
    data = {
        'age': np.random.normal(55, 15, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(130, 20, n_samples),
        'chol': np.random.normal(250, 50, n_samples),
        'fbs': np.random.choice([0, 1], n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples),
        'thalach': np.random.normal(150, 25, n_samples),
        'exang': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.exponential(1, n_samples),
        'slope': np.random.choice([0, 1, 2], n_samples),
        'ca': np.random.choice([0, 1, 2, 3], n_samples),
        'thal': np.random.choice([0, 1, 2], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    return pd.DataFrame(data)

def create_synthetic_diabetes_data():
    """Cria dados sintéticos de diabetes otimizados para demonstração"""
    print("🔄 Criando dataset sintético de diabetes otimizado...")
    np.random.seed(42)
    n_samples = 3000  # Aumentado para melhor qualidade, mas ainda otimizado
    
    # Gerar features sintéticas baseadas em distribuições realistas
    data = {
        'age': np.random.normal(65, 15, n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_procedures': np.random.randint(0, 10, n_samples),
        'num_medications': np.random.randint(1, 50, n_samples),
        'number_outpatient': np.random.randint(0, 20, n_samples),
        'number_emergency': np.random.randint(0, 20, n_samples),
        'number_inpatient': np.random.randint(0, 20, n_samples),
        'number_diagnoses': np.random.randint(1, 20, n_samples),
        'glucose_level': np.random.normal(150, 50, n_samples),
        'blood_pressure': np.random.normal(80, 15, n_samples),
        'bmi': np.random.normal(28, 6, n_samples)
    }
    
    # Criar target com alguma lógica (pacientes com mais diagnósticos têm maior chance de readmissão)
    readmission_prob = np.clip(data['number_diagnoses'] / 20.0 + np.random.normal(0, 0.1, n_samples), 0, 1)
    data['target'] = np.random.binomial(1, readmission_prob, n_samples)
    
    df = pd.DataFrame(data)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Normalizar features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Evitar divisão por zero
    X_normalized = (X - X_mean) / X_std
    
    print(f"📈 Dataset sintético otimizado: {X_normalized.shape[0]} amostras, {X_normalized.shape[1]} features")
    print(f"🎯 Classes: {np.bincount(y)}")
    print("⚡ Dataset sintético otimizado pronto para uso!")
    
    return X_normalized, y

def show_dataset_menu():
    """Mostra menu de seleção de datasets"""
    print("🧠 MLP from Scratch - Seleção de Dataset")
    print("=" * 45)
    print("📊 Escolha o dataset para treinamento:")
    print("")
    print("1. ❤️ Heart Disease (Kaggle)")
    print("   • 1,025 amostras de pacientes cardíacos")
    print("   • 13 features clínicas")
    print("   • Classificação: Doença cardíaca (SIM/NÃO)")
    print("")
    print("2. 🩺 Diabetes (UCI)")
    print("   • 101,766 amostras de pacientes diabéticos")
    print("   • 11 features numéricas selecionadas")
    print("   • Classificação: Readmissão hospitalar (SIM/NÃO)")
    print("")
    print("3. 🚪 Sair")
    print("")
    
    while True:
        try:
            choice = input("Digite sua escolha (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            else:
                print("❌ Opção inválida! Digite 1, 2 ou 3.")
        except KeyboardInterrupt:
            print("\n👋 Saindo...")
            return '3'

def main():
    """Função principal"""
    # Mostrar menu de seleção
    dataset_choice = show_dataset_menu()
    
    if dataset_choice == '3':
        print("👋 Até logo!")
        return
    
    # Carregar dados baseado na escolha
    if dataset_choice == '1':
        print("\n❤️ Carregando dataset Heart Disease...")
        X, y = load_heart_disease_data()
        dataset_name = "Heart Disease"
        problem_type = "Doença Cardíaca"
    else:
        print("\n🩺 Carregando dataset Diabetes...")
        X, y = load_diabetes_data()
        dataset_name = "Diabetes"
        problem_type = "Readmissão Hospitalar"
    
    print(f"\n🎯 Problema: Classificação de {problem_type}")
    print("=" * 50)
    
    # Dividir dados
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Dados de treino: {X_train.shape[0]} amostras")
    print(f"📊 Dados de teste: {X_test.shape[0]} amostras")
    
    # Criar MLP
    layers = [X.shape[1], 8, 5, 1]  # Input, Hidden1, Hidden2, Output
    dataset_name = 'heart' if dataset_choice == '1' else 'diabetes'
    mlp = MLP(layers, learning_rate=0.01, dataset_name=dataset_name)
    
    print(f"🏗️ Arquitetura: {layers}")
    print(f"🔗 Total de conexões: {len(mlp.connections)}")
    
    # Treinar com épocas otimizadas para cada dataset
    epochs = 200 if dataset_choice == '1' else 100  # Heart Disease: 200, Diabetes: 100 (otimizado)
    print(f"⏱️ Tempo estimado: ~{epochs//20} segundos")
    
    # Treinar
    history = mlp.train(X_train, y_train, epochs=epochs, visualize=True)
    
    # Testar
    predictions = mlp.predict(X_test)
    test_accuracy = np.mean(predictions == y_test)
    
    print(f"\n🎯 Resultados - Dataset {dataset_name}:")
    print(f"   Accuracy no teste: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"   Loss final: {history[-1]['loss']:.4f}")
    
    # Mostrar algumas predições
    print(f"\n🔍 Exemplos de predições:")
    for i in range(min(5, len(X_test))):
        actual = y_test[i]
        predicted = predictions[i]
        confidence = mlp.forward_propagation(X_test[i].tolist())[0]
        actual_label = "SIM" if actual == 1 else "NÃO"
        predicted_label = "SIM" if predicted == 1 else "NÃO"
        print(f"   Amostra {i+1}: Real={actual_label}, Predito={predicted_label}, Confiança={confidence:.3f}")
    
    # Manter visualização aberta
    print("\n🎨 Visualização final mantida aberta!")
    print("   💡 Feche a janela da visualização para finalizar o programa.")
    plt.show()

if __name__ == "__main__":
    main()

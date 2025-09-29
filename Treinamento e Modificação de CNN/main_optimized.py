"""
Projeto: Treinamento e Modificação de CNN - Versão Otimizada
Objetivo: Otimizar desempenho para superar 90% de acurácia com treinamento mais rápido
Datasets: CIFAR-10 e CIFAR-100
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Usar backend sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary

# Configurações otimizadas
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

# Criar diretórios para resultados
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class OptimizedBaseCNN(nn.Module):
    """
    Modelo CNN base otimizado para treinamento mais rápido
    """
    def __init__(self, num_classes=10):
        super(OptimizedBaseCNN, self).__init__()
        
        # Camadas convolucionais otimizadas
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout otimizado
        self.dropout = nn.Dropout(0.3)
        
        # Camadas fully connected otimizadas
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        # Primeira camada convolucional
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Segunda camada convolucional
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Terceira camada convolucional
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class OptimizedModifiedCNN(nn.Module):
    """
    Modelo CNN modificado otimizado com melhorias na arquitetura
    """
    def __init__(self, num_classes=10):
        super(OptimizedModifiedCNN, self).__init__()
        
        # Camadas convolucionais expandidas e otimizadas
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout otimizado
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Camadas fully connected otimizadas
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        
    def forward(self, x):
        # Primeira camada convolucional
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Segunda camada convolucional
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Terceira camada convolucional
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Quarta camada convolucional
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten
        x = x.view(-1, 512 * 2 * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class OptimizedResNet(nn.Module):
    """
    ResNet otimizado para CIFAR
    """
    def __init__(self, num_classes=10):
        super(OptimizedResNet, self).__init__()
        
        # Primeira camada adaptada para CIFAR
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Blocos ResNet simplificados
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Pooling e classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # Primeiro bloco com stride
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Blocos restantes
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class BasicBlock(nn.Module):
    """
    Bloco básico do ResNet
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

def get_optimized_data_loaders(dataset_name='cifar10', batch_size=256, num_workers=4):
    """
    Carrega e prepara os datasets CIFAR com otimizações
    """
    # Transformações otimizadas para treinamento
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Transformações para teste
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    if dataset_name == 'cifar10':
        # CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        # CIFAR-100
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 100
        class_names = [f'class_{i}' for i in range(100)]
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes, class_names

def train_model_optimized(model, train_loader, test_loader, num_epochs=30, learning_rate=0.001, model_name="model"):
    """
    Treina o modelo com otimizações para velocidade
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    model.to(DEVICE)
    
    for epoch in range(num_epochs):
        # Treinamento
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        # Avaliação
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for data, target in test_pbar:
                data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
                
                test_pbar.set_postfix({
                    'Acc': f'{100.*correct_test/total_test:.2f}%'
                })
        
        # Calcular métricas
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        test_acc = 100. * correct_test / total_test
        
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # Salvar modelo
    torch.save(model.state_dict(), f'models/{model_name}.pth')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def evaluate_model_optimized(model, test_loader, class_names, model_name="model"):
    """
    Avalia o modelo e gera métricas detalhadas
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Avaliando modelo"):
            data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calcular acurácia
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    
    # Matriz de confusão (implementação manual)
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for target, pred in zip(all_targets, all_predictions):
        cm[target, pred] += 1
    
    # Relatório de classificação (implementação manual)
    report = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        report[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': cm[i, :].sum()
        }
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_training_history(history, model_name="model"):
    """
    Plota o histórico de treinamento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot de loss
    ax1.plot(history['train_losses'])
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot de acurácia
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['test_accuracies'], label='Test Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Histórico de treinamento salvo em 'plots/{model_name}_training_history.png'")

def plot_confusion_matrix(cm, class_names, model_name="model", dataset_name="dataset"):
    """
    Plota a matriz de confusão
    """
    plt.figure(figsize=(12, 10))
    
    if len(class_names) > 20:  # Para CIFAR-100, mostrar apenas algumas classes
        # Mostrar apenas as primeiras 20 classes para visualização
        cm_subset = cm[:20, :20]
        class_names_subset = class_names[:20]
        sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_subset, yticklabels=class_names_subset)
        plt.title(f'{model_name} - Confusion Matrix ({dataset_name}) - Primeiras 20 classes')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix ({dataset_name})')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix_{dataset_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Matriz de confusão salva em 'plots/{model_name}_confusion_matrix_{dataset_name}.png'")

def plot_model_architecture(model, input_size=(3, 32, 32), model_name="model"):
    """
    Plota a arquitetura do modelo
    """
    try:
        # Tentar usar torchsummary se disponível
        from torchsummary import summary
        summary_str = summary(model, input_size, device='cpu')
        print(f"\nArquitetura do {model_name}:")
        print(summary_str)
        
        # Salvar resumo em arquivo
        with open(f'plots/{model_name}_architecture.txt', 'w') as f:
            f.write(f"Arquitetura do {model_name}:\n")
            f.write(str(summary_str))
            
    except Exception as e:
        # Fallback: mostrar informações básicas
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nArquitetura do {model_name}:")
        print(f"Total de parâmetros: {total_params:,}")
        print(f"Parâmetros treináveis: {trainable_params:,}")
        print(f"Modelo: {model}")
        
        # Salvar informações básicas
        with open(f'plots/{model_name}_architecture.txt', 'w') as f:
            f.write(f"Arquitetura do {model_name}:\n")
            f.write(f"Total de parâmetros: {total_params:,}\n")
            f.write(f"Parâmetros treináveis: {trainable_params:,}\n")
            f.write(f"Modelo: {model}\n")

def create_architecture_diagram():
    """
    Cria diagramas das arquiteturas utilizadas
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Arquiteturas CNN Utilizadas no Projeto', fontsize=16, fontweight='bold')
    
    # CNN Base
    ax1 = axes[0, 0]
    ax1.set_title('CNN Base', fontweight='bold')
    ax1.text(0.5, 0.9, 'Input (3, 32, 32)', ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.text(0.5, 0.7, 'Conv2d(3→32) + BN + ReLU + MaxPool', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax1.text(0.5, 0.5, 'Conv2d(32→64) + BN + ReLU + MaxPool', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax1.text(0.5, 0.3, 'Conv2d(64→128) + BN + ReLU + MaxPool', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax1.text(0.5, 0.1, 'FC(512) + Dropout + FC(10)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # CNN Modificada
    ax2 = axes[0, 1]
    ax2.set_title('CNN Modificada', fontweight='bold')
    ax2.text(0.5, 0.9, 'Input (3, 32, 32)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax2.text(0.5, 0.75, 'Conv2d(3→64) + BN + ReLU + MaxPool + Dropout', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.text(0.5, 0.6, 'Conv2d(64→128) + BN + ReLU + MaxPool + Dropout', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.text(0.5, 0.45, 'Conv2d(128→256) + BN + ReLU + MaxPool + Dropout', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.text(0.5, 0.3, 'Conv2d(256→512) + BN + ReLU + MaxPool + Dropout', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.text(0.5, 0.15, 'GlobalAvgPool + FC(512) + FC(256) + FC(10)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax2.text(0.5, 0.05, 'GlobalAvgPool + FC(512) + FC(256) + FC(10)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # ResNet
    ax3 = axes[1, 0]
    ax3.set_title('ResNet Otimizado', fontweight='bold')
    ax3.text(0.5, 0.9, 'Input (3, 32, 32)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax3.text(0.5, 0.75, 'Conv2d(3→64) + BN + ReLU', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax3.text(0.5, 0.6, 'BasicBlock(64→64) x2', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax3.text(0.5, 0.45, 'BasicBlock(64→128) x2', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax3.text(0.5, 0.3, 'BasicBlock(128→256) x2', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax3.text(0.5, 0.15, 'AdaptiveAvgPool + FC(256→10)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Comparação de Performance
    ax4 = axes[1, 1]
    ax4.set_title('Otimizações Implementadas', fontweight='bold')
    optimizations = [
        '• Batch Size: 256 (vs 128)',
        '• Workers: 4 (vs 2)',
        '• Pin Memory: True',
        '• Non-blocking: True',
        '• AdamW Optimizer',
        '• CosineAnnealingLR',
        '• Epochs: 30 (vs 50)',
        '• Dropout reduzido',
        '• FC layers otimizadas'
    ]
    
    for i, opt in enumerate(optimizations):
        ax4.text(0.05, 0.9 - i*0.1, opt, ha='left', va='center', fontsize=10)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/architectures_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Diagrama de arquiteturas salvo em 'plots/architectures_comparison.png'")

def main_optimized():
    """
    Função principal otimizada que executa todas as etapas do projeto
    """
    print("=== PROJETO: TREINAMENTO E MODIFICAÇÃO DE CNN - VERSÃO OTIMIZADA ===")
    print("Objetivo: Otimizar desempenho para superar 90% de acurácia com treinamento mais rápido")
    print(f"Dispositivo: {DEVICE}")
    print()
    
    # Criar diagramas das arquiteturas
    print("Criando diagramas das arquiteturas...")
    create_architecture_diagram()
    
    # ETAPA 1: Modelo Base Otimizado
    print("\nETAPA 1: Preparação do Modelo Base Otimizado")
    print("=" * 60)
    
    # Carregar CIFAR-10 com otimizações
    train_loader_cifar10, test_loader_cifar10, num_classes_cifar10, class_names_cifar10 = get_optimized_data_loaders('cifar10')
    
    # Criar e treinar modelo base otimizado
    base_model = OptimizedBaseCNN(num_classes=num_classes_cifar10)
    plot_model_architecture(base_model, model_name="optimized_base_cnn")
    
    print(f"Modelo base otimizado criado com {sum(p.numel() for p in base_model.parameters())} parâmetros")
    
    print("\nTreinando modelo base otimizado no CIFAR-10...")
    start_time = time.time()
    base_history = train_model_optimized(base_model, train_loader_cifar10, test_loader_cifar10, 
                                        num_epochs=30, model_name="optimized_base_cnn_cifar10")
    training_time = time.time() - start_time
    
    # Avaliar modelo base
    base_results_cifar10 = evaluate_model_optimized(base_model, test_loader_cifar10, 
                                                   class_names_cifar10, "optimized_base_cnn_cifar10")
    
    print(f"\nResultados do Modelo Base Otimizado no CIFAR-10:")
    print(f"Acurácia: {base_results_cifar10['accuracy']:.2f}%")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    # Plotar resultados do modelo base
    plot_training_history(base_history, "optimized_base_cnn_cifar10")
    plot_confusion_matrix(base_results_cifar10['confusion_matrix'], 
                         class_names_cifar10, "optimized_base_cnn_cifar10", "CIFAR-10")
    
    # ETAPA 2: Modelo Modificado Otimizado
    print("\nETAPA 2: Modificação da Arquitetura Otimizada")
    print("=" * 60)
    
    # Criar e treinar modelo modificado otimizado
    modified_model = OptimizedModifiedCNN(num_classes=num_classes_cifar10)
    plot_model_architecture(modified_model, model_name="optimized_modified_cnn")
    
    print(f"Modelo modificado otimizado criado com {sum(p.numel() for p in modified_model.parameters())} parâmetros")
    
    print("\nTreinando modelo modificado otimizado no CIFAR-10...")
    start_time = time.time()
    modified_history = train_model_optimized(modified_model, train_loader_cifar10, test_loader_cifar10, 
                                            num_epochs=30, model_name="optimized_modified_cnn_cifar10")
    training_time = time.time() - start_time
    
    # Avaliar modelo modificado
    modified_results_cifar10 = evaluate_model_optimized(modified_model, test_loader_cifar10, 
                                                       class_names_cifar10, "optimized_modified_cnn_cifar10")
    
    print(f"\nResultados do Modelo Modificado Otimizado no CIFAR-10:")
    print(f"Acurácia: {modified_results_cifar10['accuracy']:.2f}%")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    # Plotar resultados do modelo modificado
    plot_training_history(modified_history, "optimized_modified_cnn_cifar10")
    plot_confusion_matrix(modified_results_cifar10['confusion_matrix'], 
                         class_names_cifar10, "optimized_modified_cnn_cifar10", "CIFAR-10")
    
    # Comparação CIFAR-10
    print(f"\nComparação CIFAR-10 (Otimizada):")
    print(f"Modelo Base: {base_results_cifar10['accuracy']:.2f}%")
    print(f"Modelo Modificado: {modified_results_cifar10['accuracy']:.2f}%")
    print(f"Melhoria: {modified_results_cifar10['accuracy'] - base_results_cifar10['accuracy']:.2f}%")
    
    # ETAPA 3: Teste com CIFAR-100 Otimizado
    print("\nETAPA 3: Teste com Segundo Dataset (CIFAR-100) Otimizado")
    print("=" * 60)
    
    # Carregar CIFAR-100 com otimizações
    train_loader_cifar100, test_loader_cifar100, num_classes_cifar100, class_names_cifar100 = get_optimized_data_loaders('cifar100')
    
    # Testar modelo base no CIFAR-100
    base_model_cifar100 = OptimizedBaseCNN(num_classes=num_classes_cifar100)
    # Carregar pesos do modelo treinado no CIFAR-10, ignorando camadas incompatíveis
    state_dict = torch.load('models/optimized_base_cnn_cifar10.pth')
    # Remover as chaves das camadas finais incompatíveis
    state_dict.pop('fc2.weight', None)
    state_dict.pop('fc2.bias', None)
    base_model_cifar100.load_state_dict(state_dict, strict=False)
    base_model_cifar100.fc2 = nn.Linear(256, num_classes_cifar100)
    
    print("\nAvaliando modelo base otimizado no CIFAR-100...")
    base_results_cifar100 = evaluate_model_optimized(base_model_cifar100, test_loader_cifar100, 
                                                     class_names_cifar100, "optimized_base_cnn_cifar100")
    
    # Testar modelo modificado no CIFAR-100
    modified_model_cifar100 = OptimizedModifiedCNN(num_classes=num_classes_cifar100)
    # Carregar pesos do modelo modificado treinado no CIFAR-10, ignorando camadas incompatíveis
    state_dict_mod = torch.load('models/optimized_modified_cnn_cifar10.pth')
    state_dict_mod.pop('fc3.weight', None)
    state_dict_mod.pop('fc3.bias', None)
    modified_model_cifar100.load_state_dict(state_dict_mod, strict=False)
    modified_model_cifar100.fc3 = nn.Linear(256, num_classes_cifar100)
    
    print("\nAvaliando modelo modificado otimizado no CIFAR-100...")
    modified_results_cifar100 = evaluate_model_optimized(modified_model_cifar100, test_loader_cifar100, 
                                                         class_names_cifar100, "optimized_modified_cnn_cifar100")
    
    print(f"\nResultados no CIFAR-100 (Otimizado):")
    print(f"Modelo Base: {base_results_cifar100['accuracy']:.2f}%")
    print(f"Modelo Modificado: {modified_results_cifar100['accuracy']:.2f}%")
    
    # Plotar matrizes de confusão para CIFAR-100
    plot_confusion_matrix(base_results_cifar100['confusion_matrix'], 
                         class_names_cifar100, "optimized_base_cnn_cifar100", "CIFAR-100")
    plot_confusion_matrix(modified_results_cifar100['confusion_matrix'], 
                         class_names_cifar100, "optimized_modified_cnn_cifar100", "CIFAR-100")
    
    # ETAPA 4: Técnica SOTA Otimizada (ResNet)
    print("\nETAPA 4: Técnica de Estado da Arte Otimizada (ResNet)")
    print("=" * 60)
    
    # ResNet otimizado para CIFAR-10
    resnet_cifar10 = OptimizedResNet(num_classes=num_classes_cifar10)
    plot_model_architecture(resnet_cifar10, model_name="optimized_resnet")
    
    print(f"ResNet otimizado criado com {sum(p.numel() for p in resnet_cifar10.parameters())} parâmetros")
    
    print("\nTreinando ResNet otimizado no CIFAR-10...")
    start_time = time.time()
    resnet_history_cifar10 = train_model_optimized(resnet_cifar10, train_loader_cifar10, test_loader_cifar10, 
                                                  num_epochs=30, model_name="optimized_resnet_cifar10")
    training_time = time.time() - start_time
    
    # Avaliar ResNet no CIFAR-10
    resnet_results_cifar10 = evaluate_model_optimized(resnet_cifar10, test_loader_cifar10, 
                                                     class_names_cifar10, "optimized_resnet_cifar10")
    
    print(f"\nResultados do ResNet Otimizado no CIFAR-10:")
    print(f"Acurácia: {resnet_results_cifar10['accuracy']:.2f}%")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    # ResNet otimizado para CIFAR-100
    resnet_cifar100 = OptimizedResNet(num_classes=num_classes_cifar100)
    
    print("\nTreinando ResNet otimizado no CIFAR-100...")
    start_time = time.time()
    resnet_history_cifar100 = train_model_optimized(resnet_cifar100, train_loader_cifar100, test_loader_cifar100, 
                                                   num_epochs=30, model_name="optimized_resnet_cifar100")
    training_time = time.time() - start_time
    
    # Avaliar ResNet no CIFAR-100
    resnet_results_cifar100 = evaluate_model_optimized(resnet_cifar100, test_loader_cifar100, 
                                                       class_names_cifar100, "optimized_resnet_cifar100")
    
    print(f"\nResultados do ResNet Otimizado no CIFAR-100:")
    print(f"Acurácia: {resnet_results_cifar100['accuracy']:.2f}%")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    # Plotar resultados do ResNet
    plot_training_history(resnet_history_cifar10, "optimized_resnet_cifar10")
    plot_training_history(resnet_history_cifar100, "optimized_resnet_cifar100")
    plot_confusion_matrix(resnet_results_cifar10['confusion_matrix'], 
                         class_names_cifar10, "optimized_resnet_cifar10", "CIFAR-10")
    plot_confusion_matrix(resnet_results_cifar100['confusion_matrix'], 
                         class_names_cifar100, "optimized_resnet_cifar100", "CIFAR-100")
    
    # COMPARAÇÃO FINAL OTIMIZADA
    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL DE RESULTADOS OTIMIZADOS")
    print("=" * 70)
    
    print("\nCIFAR-10 (Otimizado):")
    print(f"Modelo Base:        {base_results_cifar10['accuracy']:.2f}%")
    print(f"Modelo Modificado:  {modified_results_cifar10['accuracy']:.2f}%")
    print(f"ResNet (SOTA):      {resnet_results_cifar10['accuracy']:.2f}%")
    
    print("\nCIFAR-100 (Otimizado):")
    print(f"Modelo Base:        {base_results_cifar100['accuracy']:.2f}%")
    print(f"Modelo Modificado:  {modified_results_cifar100['accuracy']:.2f}%")
    print(f"ResNet (SOTA):      {resnet_results_cifar100['accuracy']:.2f}%")
    
    # Verificar se atingiu 90% de acurácia
    print("\n" + "=" * 70)
    print("VERIFICAÇÃO DO OBJETIVO (90% DE ACURÁCIA) - VERSÃO OTIMIZADA")
    print("=" * 70)
    
    target_accuracy = 90.0
    models_cifar10 = {
        'Base': base_results_cifar10['accuracy'],
        'Modificado': modified_results_cifar10['accuracy'],
        'ResNet': resnet_results_cifar10['accuracy']
    }
    
    for model_name, accuracy in models_cifar10.items():
        if accuracy >= target_accuracy:
            print(f"✅ {model_name}: {accuracy:.2f}% (OBJETIVO ATINGIDO!)")
        else:
            print(f"❌ {model_name}: {accuracy:.2f}% (Faltam {target_accuracy - accuracy:.2f}%)")
    
    print("\n" + "=" * 70)
    print("OTIMIZAÇÕES IMPLEMENTADAS:")
    print("=" * 70)
    print("• Batch Size aumentado para 256")
    print("• Número de workers aumentado para 4")
    print("• Pin Memory habilitado")
    print("• Non-blocking transfers")
    print("• AdamW optimizer com weight decay")
    print("• CosineAnnealingLR scheduler")
    print("• Epochs reduzidos para 30")
    print("• Dropout otimizado")
    print("• Camadas FC otimizadas")
    print("• ResNet customizado para CIFAR")
    
    print("\nProjeto otimizado concluído! Verifique a pasta 'plots' para visualizações e 'models' para os modelos salvos.")

if __name__ == "__main__":
    main_optimized()

"""
Projeto: Treinamento e Modificação de CNN
Objetivo: Otimizar desempenho para superar 90% de acurácia
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
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
from tqdm import tqdm

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

# Criar diretórios para resultados
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class BaseCNN(nn.Module):
    """
    Modelo CNN base para classificação de imagens CIFAR
    """
    def __init__(self, num_classes=10):
        super(BaseCNN, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        # Primeira camada convolucional
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Segunda camada convolucional
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Terceira camada convolucional
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModifiedCNN(nn.Module):
    """
    Modelo CNN modificado com melhorias na arquitetura
    """
    def __init__(self, num_classes=10):
        super(ModifiedCNN, self).__init__()
        
        # Camadas convolucionais expandidas
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout com diferentes taxas
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Camadas fully connected expandidas
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        
    def forward(self, x):
        # Primeira camada convolucional
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Segunda camada convolucional
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Terceira camada convolucional
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Quarta camada convolucional
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten
        x = x.view(-1, 512 * 2 * 2)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def get_data_loaders(dataset_name='cifar10', batch_size=128, num_workers=2):
    """
    Carrega e prepara os datasets CIFAR-10 e CIFAR-100
    """
    # Transformações para treinamento
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, num_classes, class_names

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, model_name="model"):
    """
    Treina o modelo e retorna histórico de treinamento
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
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
            data, target = data.to(DEVICE), target.to(DEVICE)
            
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
                data, target = data.to(DEVICE), target.to(DEVICE)
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

def evaluate_model(model, test_loader, class_names, model_name="model"):
    """
    Avalia o modelo e gera métricas detalhadas
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Avaliando modelo"):
            data, target = data.to(DEVICE), target.to(DEVICE)
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

def main():
    """
    Função principal que executa todas as etapas do projeto
    """
    print("=== PROJETO: TREINAMENTO E MODIFICAÇÃO DE CNN ===")
    print("Objetivo: Otimizar desempenho para superar 90% de acurácia")
    print(f"Dispositivo: {DEVICE}")
    print()
    
    # ETAPA 1: Modelo Base
    print("ETAPA 1: Preparação do Modelo Base")
    print("=" * 50)
    
    # Carregar CIFAR-10
    train_loader_cifar10, test_loader_cifar10, num_classes_cifar10, class_names_cifar10 = get_data_loaders('cifar10')
    
    # Criar e treinar modelo base
    base_model = BaseCNN(num_classes=num_classes_cifar10)
    print(f"Modelo base criado com {sum(p.numel() for p in base_model.parameters())} parâmetros")
    
    print("\nTreinando modelo base no CIFAR-10...")
    base_history = train_model(base_model, train_loader_cifar10, test_loader_cifar10, 
                              num_epochs=50, model_name="base_cnn_cifar10")
    
    # Avaliar modelo base
    base_results_cifar10 = evaluate_model(base_model, test_loader_cifar10, 
                                         class_names_cifar10, "base_cnn_cifar10")
    
    print(f"\nResultados do Modelo Base no CIFAR-10:")
    print(f"Acurácia: {base_results_cifar10['accuracy']:.2f}%")
    
    # Plotar resultados do modelo base
    plot_training_history(base_history, "base_cnn_cifar10")
    plot_confusion_matrix(base_results_cifar10['confusion_matrix'], 
                         class_names_cifar10, "base_cnn_cifar10", "CIFAR-10")
    
    # ETAPA 2: Modelo Modificado
    print("\nETAPA 2: Modificação da Arquitetura")
    print("=" * 50)
    
    # Criar e treinar modelo modificado
    modified_model = ModifiedCNN(num_classes=num_classes_cifar10)
    print(f"Modelo modificado criado com {sum(p.numel() for p in modified_model.parameters())} parâmetros")
    
    print("\nTreinando modelo modificado no CIFAR-10...")
    modified_history = train_model(modified_model, train_loader_cifar10, test_loader_cifar10, 
                                  num_epochs=50, model_name="modified_cnn_cifar10")
    
    # Avaliar modelo modificado
    modified_results_cifar10 = evaluate_model(modified_model, test_loader_cifar10, 
                                             class_names_cifar10, "modified_cnn_cifar10")
    
    print(f"\nResultados do Modelo Modificado no CIFAR-10:")
    print(f"Acurácia: {modified_results_cifar10['accuracy']:.2f}%")
    
    # Plotar resultados do modelo modificado
    plot_training_history(modified_history, "modified_cnn_cifar10")
    plot_confusion_matrix(modified_results_cifar10['confusion_matrix'], 
                         class_names_cifar10, "modified_cnn_cifar10", "CIFAR-10")
    
    # Comparação CIFAR-10
    print(f"\nComparação CIFAR-10:")
    print(f"Modelo Base: {base_results_cifar10['accuracy']:.2f}%")
    print(f"Modelo Modificado: {modified_results_cifar10['accuracy']:.2f}%")
    print(f"Melhoria: {modified_results_cifar10['accuracy'] - base_results_cifar10['accuracy']:.2f}%")
    
    # ETAPA 3: Teste com CIFAR-100
    print("\nETAPA 3: Teste com Segundo Dataset (CIFAR-100)")
    print("=" * 50)
    
    # Carregar CIFAR-100
    train_loader_cifar100, test_loader_cifar100, num_classes_cifar100, class_names_cifar100 = get_data_loaders('cifar100')
    
    # Testar modelo base no CIFAR-100
    base_model_cifar100 = BaseCNN(num_classes=num_classes_cifar100)
    base_model_cifar100.load_state_dict(torch.load('models/base_cnn_cifar10.pth'))
    base_model_cifar100.fc2 = nn.Linear(512, num_classes_cifar100)
    
    print("\nAvaliando modelo base no CIFAR-100...")
    base_results_cifar100 = evaluate_model(base_model_cifar100, test_loader_cifar100, 
                                          class_names_cifar100, "base_cnn_cifar100")
    
    # Testar modelo modificado no CIFAR-100
    modified_model_cifar100 = ModifiedCNN(num_classes=num_classes_cifar100)
    modified_model_cifar100.load_state_dict(torch.load('models/modified_cnn_cifar10.pth'))
    modified_model_cifar100.fc3 = nn.Linear(512, num_classes_cifar100)
    
    print("\nAvaliando modelo modificado no CIFAR-100...")
    modified_results_cifar100 = evaluate_model(modified_model_cifar100, test_loader_cifar100, 
                                              class_names_cifar100, "modified_cnn_cifar100")
    
    print(f"\nResultados no CIFAR-100:")
    print(f"Modelo Base: {base_results_cifar100['accuracy']:.2f}%")
    print(f"Modelo Modificado: {modified_results_cifar100['accuracy']:.2f}%")
    
    # Plotar matrizes de confusão para CIFAR-100
    plot_confusion_matrix(base_results_cifar100['confusion_matrix'], 
                         class_names_cifar100, "base_cnn_cifar100", "CIFAR-100")
    plot_confusion_matrix(modified_results_cifar100['confusion_matrix'], 
                         class_names_cifar100, "modified_cnn_cifar100", "CIFAR-100")
    
    # ETAPA 4: Técnica SOTA (ResNet)
    print("\nETAPA 4: Técnica de Estado da Arte (ResNet)")
    print("=" * 50)
    
    # Implementar ResNet18 como técnica SOTA
    from torchvision.models import resnet18
    
    # ResNet para CIFAR-10
    resnet_cifar10 = resnet18(pretrained=False)
    resnet_cifar10.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet_cifar10.maxpool = nn.Identity()
    resnet_cifar10.fc = nn.Linear(resnet_cifar10.fc.in_features, num_classes_cifar10)
    
    print(f"ResNet criado com {sum(p.numel() for p in resnet_cifar10.parameters())} parâmetros")
    
    print("\nTreinando ResNet no CIFAR-10...")
    resnet_history_cifar10 = train_model(resnet_cifar10, train_loader_cifar10, test_loader_cifar10, 
                                        num_epochs=50, model_name="resnet_cifar10")
    
    # Avaliar ResNet no CIFAR-10
    resnet_results_cifar10 = evaluate_model(resnet_cifar10, test_loader_cifar10, 
                                           class_names_cifar10, "resnet_cifar10")
    
    print(f"\nResultados do ResNet no CIFAR-10:")
    print(f"Acurácia: {resnet_results_cifar10['accuracy']:.2f}%")
    
    # ResNet para CIFAR-100
    resnet_cifar100 = resnet18(pretrained=False)
    resnet_cifar100.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet_cifar100.maxpool = nn.Identity()
    resnet_cifar100.fc = nn.Linear(resnet_cifar100.fc.in_features, num_classes_cifar100)
    
    print("\nTreinando ResNet no CIFAR-100...")
    resnet_history_cifar100 = train_model(resnet_cifar100, train_loader_cifar100, test_loader_cifar100, 
                                         num_epochs=50, model_name="resnet_cifar100")
    
    # Avaliar ResNet no CIFAR-100
    resnet_results_cifar100 = evaluate_model(resnet_cifar100, test_loader_cifar100, 
                                            class_names_cifar100, "resnet_cifar100")
    
    print(f"\nResultados do ResNet no CIFAR-100:")
    print(f"Acurácia: {resnet_results_cifar100['accuracy']:.2f}%")
    
    # Plotar resultados do ResNet
    plot_training_history(resnet_history_cifar10, "resnet_cifar10")
    plot_training_history(resnet_history_cifar100, "resnet_cifar100")
    plot_confusion_matrix(resnet_results_cifar10['confusion_matrix'], 
                         class_names_cifar10, "resnet_cifar10", "CIFAR-10")
    plot_confusion_matrix(resnet_results_cifar100['confusion_matrix'], 
                         class_names_cifar100, "resnet_cifar100", "CIFAR-100")
    
    # COMPARAÇÃO FINAL
    print("\n" + "=" * 60)
    print("COMPARAÇÃO FINAL DE RESULTADOS")
    print("=" * 60)
    
    print("\nCIFAR-10:")
    print(f"Modelo Base:        {base_results_cifar10['accuracy']:.2f}%")
    print(f"Modelo Modificado:  {modified_results_cifar10['accuracy']:.2f}%")
    print(f"ResNet (SOTA):      {resnet_results_cifar10['accuracy']:.2f}%")
    
    print("\nCIFAR-100:")
    print(f"Modelo Base:        {base_results_cifar100['accuracy']:.2f}%")
    print(f"Modelo Modificado:  {modified_results_cifar100['accuracy']:.2f}%")
    print(f"ResNet (SOTA):      {resnet_results_cifar100['accuracy']:.2f}%")
    
    # Verificar se atingiu 90% de acurácia
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DO OBJETIVO (90% DE ACURÁCIA)")
    print("=" * 60)
    
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
    
    print("\nProjeto concluído! Verifique a pasta 'plots' para visualizações e 'models' para os modelos salvos.")

if __name__ == "__main__":
    main()

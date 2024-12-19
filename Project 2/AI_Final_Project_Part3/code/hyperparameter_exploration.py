import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

class FashionMNISTCNN(nn.Module):
    def __init__(self, num_filters=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(num_filters*2 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_and_evaluate(model, train_loader, test_loader, epochs=3, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def hyperparameter_exploration():
    # Hyperparameter configurations to explore
    num_filters_list = [8, 16, 32]
    batch_sizes = [32, 64, 128]
    epochs_list = [3, 5, 7]
    
    results = {
        'num_filters': [],
        'batch_size': [],
        'epochs': [],
        'accuracies': []
    }
    
    # Explore number of filters
    for num_filters in num_filters_list:
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        
        model = FashionMNISTCNN(num_filters=num_filters).to(device)
        accuracy = train_and_evaluate(model, train_loader, test_loader)
        
        torch.save(model.state_dict(), f'model_filters_{num_filters}.pt')
        
        results['num_filters'].append(num_filters)
        results['accuracies'].append(accuracy)
        print(f'Filters: {num_filters}, Accuracy: {accuracy:.2f}%')
    
    # Explore batch sizes
    for batch_size in batch_sizes:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        model = FashionMNISTCNN().to(device)
        accuracy = train_and_evaluate(model, train_loader, test_loader)
        
        torch.save(model.state_dict(), f'model_batchsize_{batch_size}.pt')
        
        results['batch_size'].append(batch_size)
        results['accuracies'].append(accuracy)
        print(f'Batch Size: {batch_size}, Accuracy: {accuracy:.2f}%')
    
    # Explore epochs
    for epochs in epochs_list:
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        
        model = FashionMNISTCNN().to(device)
        accuracy = train_and_evaluate(model, train_loader, test_loader, epochs=epochs)
        
        torch.save(model.state_dict(), f'model_epochs_{epochs}.pt')
        
        results['epochs'].append(epochs)
        results['accuracies'].append(accuracy)
        print(f'Epochs: {epochs}, Accuracy: {accuracy:.2f}%')
    
    # Plotting results
    plt.figure(figsize=(15, 5))
    
    # Number of Filters Plot
    plt.subplot(1, 3, 1)
    plt.plot(results['num_filters'], results['accuracies'][:len(num_filters_list)], marker='o')
    plt.title('Number of Filters vs Accuracy')
    plt.xlabel('Number of Filters')
    plt.ylabel('Accuracy (%)')
    
    # Batch Size Plot
    plt.subplot(1, 3, 2)
    plt.plot(results['batch_size'], results['accuracies'][len(num_filters_list):len(num_filters_list)+len(batch_sizes)], marker='o')
    plt.title('Batch Size vs Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy (%)')
    
    # Epochs Plot
    plt.subplot(1, 3, 3)
    plt.plot(results['epochs'], results['accuracies'][-len(epochs_list):], marker='o')
    plt.title('Number of Epochs vs Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png')
    plt.close()

# Run the exploration
hyperparameter_exploration()
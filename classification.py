import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import pickle

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data directories
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ResNet18 model
class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)

# Instantiate the model
model = ResNetClassifier().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Training the model
train_model(model, criterion, optimizer)

# Representations of each tissue patch from pre-final layer
def get_representations(loader):
    model.eval()
    representations = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model.resnet[:-1](inputs)
            representations.append(outputs)
    return torch.cat(representations)

train_representations = get_representations(train_loader)
test_representations = get_representations(test_loader)
val_representations = get_representations(val_loader)

# Probability of case in each patch
def get_probabilities(loader):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = torch.softmax(model(inputs), dim=1)
            probabilities.append(outputs[:, 0])
    return torch.cat(probabilities)

train_probabilities = get_probabilities(train_loader)
test_probabilities = get_probabilities(test_loader)
val_probabilities = get_probabilities(val_loader)

# Whole slide level representation
def aggregate_representations(representations, probabilities):
    aggregated_representations = {}
    for i in range(len(representations)):
        slide_name = train_dataset.samples[i][0].split('/')[-2]  # Extract slide name from path
        if slide_name not in aggregated_representations:
            aggregated_representations[slide_name] = {
                'representation': representations[i],
                'weight': probabilities[i]
            }
        else:
            aggregated_representations[slide_name]['representation'] += representations[i] * probabilities[i]
            aggregated_representations[slide_name]['weight'] += probabilities[i]
    for slide_name in aggregated_representations:
        aggregated_representations[slide_name]['representation'] /= aggregated_representations[slide_name]['weight']
    return aggregated_representations


test_aggregated_representations = aggregate_representations(test_representations, test_probabilities)

with open("outputs/wsi_representation", 'wb') as f:
    pickle.dump(test_aggregated_representations, f)
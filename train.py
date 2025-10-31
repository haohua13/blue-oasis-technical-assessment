import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import librosa
import os
from data_processing import load_metadata, preprocess_for_cnn

class ESC50Dataset(Dataset):
    """Custom PyTorch Dataset for pre-computed ESC-50 features."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        # add a channel dimension (C, H, W) and convert to tensor
        return torch.tensor(feature, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
    
class SimpleCNN(nn.Module):
    """A simple CNN for audio classification."""
    def __init__(self, num_classes=50):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.fc(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
    return running_loss / len(loader.dataset), correct_preds.double() / len(loader.dataset)

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
    return running_loss / len(loader.dataset), correct_preds.double() / len(loader.dataset)

if __name__ == '__main__':
    CONFIG = {
        "base_data_path": "ESC-50-master",
        "sr": 22050,
        "duration": 5,
        "n_mels": 128,
        "hop_length": 512,
        "epochs": 30, # Number of training epochs
        "batch_size": 32, # Adjust based on your memory capacity
        "learning_rate": 0.001,
        "validation_fold": 1
    }

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load metadata
    metadata = load_metadata(CONFIG["base_data_path"])
    
    # pre-compute features for all files
    print("Preprocessing all audio files. This may take a while...")
    max_len = int(np.ceil(CONFIG["sr"] * CONFIG["duration"] / CONFIG["hop_length"]))
    all_features = []
    all_labels = []

    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        try:
            y, sr = librosa.load(row['filepath'], sr=CONFIG["sr"], duration=CONFIG["duration"])
            mel_spec = preprocess_for_cnn(y, sr, n_mels=CONFIG["n_mels"], hop_length=CONFIG["hop_length"])
            
            # pad or truncate to ensure fixed size
            if mel_spec.shape[1] < max_len:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])), mode='constant')
            else:
                mel_spec = mel_spec[:, :max_len]
            
            all_features.append(mel_spec)
            all_labels.append(row['category'])
        except Exception as e:
            print(f"Could not process {row['filepath']}: {e}")

    # encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    num_classes = len(label_encoder.classes_)

    # split data based on fold (thanks to ESC-50 design)
    train_indices = metadata[metadata['fold'] != CONFIG["validation_fold"]].index
    val_indices = metadata[metadata['fold'] == CONFIG["validation_fold"]].index

    X_train = [all_features[i] for i in train_indices]
    y_train = encoded_labels[train_indices]
    X_val = [all_features[i] for i in val_indices]
    y_val = encoded_labels[val_indices]

    # create dataloaders
    train_dataset = ESC50Dataset(X_train, y_train)
    val_dataset = ESC50Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # initialize model, loss, and optimizer
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # training loop
    best_val_acc = 0.0
    print("\n--- Starting Model Training ---")
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'esc50_cnn_best_model.pth')
            print(f"  -> New best model saved with accuracy: {val_acc:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
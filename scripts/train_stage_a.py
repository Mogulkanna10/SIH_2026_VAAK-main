import os
import sys
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import shared feature extractor
sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

# Hardware / CPU Threads Configuration
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on Device: {DEVICE}')

# Dataset Paths
DATASET_DIR = '/media/yuvan/2095-DDEC/dracarys_kws_data'
TARGET_CLASSES = ['bed', 'bird', 'cat', 'dog', 'down', 'go', 'house', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
CLASS_MAP = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

print(f'Stage A Pre-training on {NUM_CLASSES} classes: {TARGET_CLASSES}')

class SpeechCommandsDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav_path = self.file_list[idx]
        label = self.labels[idx]
        
        try:
            with wave.open(wav_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
            
        feat = extract_logmel_features(audio) # (49, 40)
        feat = np.expand_dims(feat, axis=0)   # (1, 49, 40) for Conv2d
        return torch.from_numpy(feat), label

# Load files & create train/val split
file_list = []
labels = []

for cls_name in TARGET_CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    if os.path.exists(cls_dir):
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.wav')]
        for f in files:
            file_list.append(f)
            labels.append(CLASS_MAP[cls_name])

print(f'Total pretraining samples collected: {len(file_list)}')

# Shuffle and split 85% Train / 15% Val
np.random.seed(42)
indices = np.random.permutation(len(file_list))
split_idx = int(len(file_list) * 0.85)

train_files = [file_list[i] for i in indices[:split_idx]]
train_labels = [labels[i] for i in indices[:split_idx]]

val_files = [file_list[i] for i in indices[split_idx:]]
val_labels = [labels[i] for i in indices[split_idx:]]

print(f'Train samples: {len(train_files)}, Val samples: {len(val_files)}')

train_loader = DataLoader(SpeechCommandsDataset(train_files, train_labels), batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(SpeechCommandsDataset(val_files, val_labels), batch_size=64, shuffle=False, num_workers=2)

# Lightweight DS-CNN (Depthwise-Separable CNN) Model (~45k params)
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.dw_conv(x)))
        x = self.relu(self.bn2(self.pw_conv(x)))
        return x

class DSCNN_Backbone(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Input shape: (B, 1, 49, 40)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block1 = DepthwiseSeparableBlock(64, 64)
        self.block2 = DepthwiseSeparableBlock(64, 64)
        self.block3 = DepthwiseSeparableBlock(64, 64)
        self.block4 = DepthwiseSeparableBlock(64, 64)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = DSCNN_Backbone().to(DEVICE)
print(f'DS-CNN Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
best_val_acc = 0.0
os.makedirs('/media/yuvan/2095-DDEC/models', exist_ok=True)
checkpoint_path = '/media/yuvan/2095-DDEC/models/pretrained_dscnn_stage_a.pth'

print('\n=== Starting Stage A Pretraining ===')
EPOCHS = 12

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for feats, target_labels in train_loader:
        feats, target_labels = feats.to(DEVICE), target_labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, target_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * feats.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == target_labels).sum().item()
        total += feats.size(0)
        
    train_acc = correct / total
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for feats, target_labels in val_loader:
            feats, target_labels = feats.to(DEVICE), target_labels.to(DEVICE)
            outputs = model(feats)
            loss = criterion(outputs, target_labels)
            val_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == target_labels).sum().item()
            val_total += feats.size(0)
            
    val_acc = val_correct / val_total
    print(f'Epoch {epoch:02d}/{EPOCHS:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)

print(f'\n=== Stage A Pretraining Completed ===')
print(f'Best Validation Accuracy achieved: {best_val_acc*100:.2f}%')

# Verify Gate 3 Target (≥ 88%)
if best_val_acc >= 0.88:
    print('GATE 3 STATUS: PASSED! (Val Acc >= 88%)')
else:
    print('GATE 3 STATUS: FAILED (Val Acc < 88%)')

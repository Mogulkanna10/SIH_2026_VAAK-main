#!/usr/bin/env python3
"""
train_stage_b.py — Stage B: Transfer Learning for Dracarys KWS
Loads the pretrained DS-CNN backbone from Stage A, freezes it,
and trains a 3-class classification head: [background, dracarys, unknown]

Uses existing speaker-disjoint splits:
  train: speakers 1-7
  val:   speakers 8-9
  test:  speaker 10
"""

import os
import sys
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features
from train_stage_a import DSCNN_Backbone, DepthwiseSeparableBlock

torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3-class labels for Dracarys KWS
DRACARYS_CLASSES = ['background', 'dracarys', 'unknown']
CLASS_MAP = {cls: i for i, cls in enumerate(DRACARYS_CLASSES)}
NUM_CLASSES = len(DRACARYS_CLASSES)

# Paths
SPLIT_DIR = '/home/yuvan/split'
STAGE_A_CHECKPOINT = '/media/yuvan/2095-DDEC/models/pretrained_dscnn_stage_a.pth'
STAGE_B_CHECKPOINT = '/media/yuvan/2095-DDEC/models/dracarys_kws_stage_b.pth'

class DracarysDataset(Dataset):
    def __init__(self, split='train'):
        self.files = []
        self.labels = []
        
        split_dir = os.path.join(SPLIT_DIR, split)
        
        # Load positive (dracarys) files
        pos_dir = os.path.join(split_dir, 'positive')
        if os.path.exists(pos_dir):
            for f in sorted(os.listdir(pos_dir)):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(pos_dir, f))
                    self.labels.append(CLASS_MAP['dracarys'])
        
        # Load negative files — classify as background or unknown
        neg_dir = os.path.join(split_dir, 'negative')
        if os.path.exists(neg_dir):
            for f in sorted(os.listdir(neg_dir)):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(neg_dir, f))
                    self.labels.append(CLASS_MAP['unknown'])
        
        # Load dedicated background files
        bg_dir = os.path.join(split_dir, 'background')
        if os.path.exists(bg_dir):
            for f in sorted(os.listdir(bg_dir)):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(bg_dir, f))
                    self.labels.append(CLASS_MAP['background'])
        
        print(f'  {split}: {len(self.files)} total samples '
              f'(dracarys={self.labels.count(1)}, '
              f'unknown={self.labels.count(2)}, '
              f'background={self.labels.count(0)})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            with wave.open(wav_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
        
        feat = extract_logmel_features(audio)
        feat = np.expand_dims(feat, axis=0)
        return torch.from_numpy(feat), label


class DSCNN_TransferHead(nn.Module):
    """DS-CNN with frozen backbone + trainable 3-class head."""
    
    def __init__(self, backbone_checkpoint, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Load pretrained backbone
        backbone = DSCNN_Backbone(num_classes=15)  # Stage A was 15-class
        backbone.load_state_dict(torch.load(backbone_checkpoint, map_location='cpu'))
        
        # Extract feature layers (everything except final FC)
        self.init_conv = backbone.init_conv
        self.block1 = backbone.block1
        self.block2 = backbone.block2
        self.block3 = backbone.block3
        self.block4 = backbone.block4
        self.avg_pool = backbone.avg_pool
        
        # Freeze backbone
        for param in self.init_conv.parameters():
            param.requires_grad = False
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False
        for param in self.block3.parameters():
            param.requires_grad = False
        for param in self.block4.parameters():
            param.requires_grad = False
        
        # New trainable head for 3-class Dracarys KWS
        self.fc = nn.Linear(64, num_classes)
        
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f'Frozen layers: {frozen}, Trainable layers: {trainable}')
    
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


def main():
    print(f'=== Stage B: Dracarys Transfer Learning ===')
    print(f'Device: {DEVICE}')
    print(f'Loading backbone from: {STAGE_A_CHECKPOINT}')
    
    # Load datasets
    train_ds = DracarysDataset('train')
    val_ds = DracarysDataset('validation')
    test_ds = DracarysDataset('test')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize transfer model
    model = DSCNN_TransferHead(STAGE_A_CHECKPOINT).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    best_val_acc = 0.0
    EPOCHS = 20
    
    print(f'\nTraining {EPOCHS} epochs...')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total = 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += feats.size(0)
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                outputs = model(feats)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += feats.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        print(f'Epoch {epoch:02d}/{EPOCHS} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), STAGE_B_CHECKPOINT)
    
    # === Gate 4 Report ===
    print(f'\n{"="*60}')
    print(f'GATE 4 REPORT — Stage B Transfer Learning Results')
    print(f'{"="*60}')
    print(f'Best Validation Accuracy: {best_val_acc*100:.2f}%')
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(STAGE_B_CHECKPOINT, map_location='cpu'))
    model.eval()
    
    # Validation confusion matrix & classification report
    val_preds, val_true = [], []
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(DEVICE)
            outputs = model(feats)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.numpy())
    
    print(f'\n--- Validation Set (speakers 8-9) ---')
    print(classification_report(val_true, val_preds, target_names=DRACARYS_CLASSES))
    print('Confusion Matrix:')
    print(confusion_matrix(val_true, val_preds))
    
    # False Activation Rate: non-dracarys samples predicted as dracarys
    val_true_np = np.array(val_true)
    val_preds_np = np.array(val_preds)
    non_dracarys_mask = val_true_np != CLASS_MAP['dracarys']
    false_activations = np.sum(val_preds_np[non_dracarys_mask] == CLASS_MAP['dracarys'])
    total_non_dracarys = np.sum(non_dracarys_mask)
    fa_rate = false_activations / total_non_dracarys if total_non_dracarys > 0 else 0
    print(f'\nFalse Activation Rate (val): {fa_rate*100:.2f}% ({false_activations}/{total_non_dracarys})')
    
    # Test set evaluation (held out, report separately)
    test_preds, test_true = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(DEVICE)
            outputs = model(feats)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.numpy())
    
    print(f'\n--- Test Set (speaker 10, HELD OUT) ---')
    print(classification_report(test_true, test_preds, target_names=DRACARYS_CLASSES))
    print('Confusion Matrix:')
    print(confusion_matrix(test_true, test_preds))
    
    test_true_np = np.array(test_true)
    test_preds_np = np.array(test_preds)
    non_dracarys_test = test_true_np != CLASS_MAP['dracarys']
    fa_test = np.sum(test_preds_np[non_dracarys_test] == CLASS_MAP['dracarys'])
    total_non_test = np.sum(non_dracarys_test)
    fa_rate_test = fa_test / total_non_test if total_non_test > 0 else 0
    print(f'False Activation Rate (test): {fa_rate_test*100:.2f}% ({fa_test}/{total_non_test})')
    
    # Gate 4 Pass/Fail
    dracarys_recall = 0.0
    for i, cls in enumerate(DRACARYS_CLASSES):
        if cls == 'dracarys':
            mask = np.array(val_true) == i
            if np.sum(mask) > 0:
                dracarys_recall = np.sum(np.array(val_preds)[mask] == i) / np.sum(mask)
    
    print(f'\n--- Gate 4 Summary ---')
    print(f'Dracarys Recall (val): {dracarys_recall*100:.2f}% (target >= 90%)')
    print(f'False Activation Rate (val): {fa_rate*100:.2f}% (target <= 2%)')
    
    if dracarys_recall >= 0.90 and fa_rate <= 0.02:
        print('GATE 4 STATUS: PASSED!')
    else:
        print('GATE 4 STATUS: NEEDS REVIEW')


if __name__ == '__main__':
    main()

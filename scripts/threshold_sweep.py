#!/usr/bin/env python3
"""
threshold_sweep.py — Step 2: Precision-Recall & ROC sweep for current Stage B model.
Sweeps thresholds 0.50 to 0.99 on validation set.
Reports whether any threshold achieves FA ≤ 2% with dracarys recall ≥ 85%.
"""

import os
import sys
import wave
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

SPLIT_DIR = '/home/yuvan/split'
STAGE_B_CHECKPOINT = '/media/yuvan/2095-DDEC/models/dracarys_kws_stage_b.keras'

# 3-class labels matching Stage B training
# CLASS_MAP: background=0, dracarys(positive)=1, unknown(negative)=2
CLASS_MAP = {'background': 0, 'positive': 1, 'negative': 2}
CLASS_NAMES = ['background', 'dracarys', 'unknown']

def load_split(split):
    files, labels = [], []
    split_dir = os.path.join(SPLIT_DIR, split)
    for cls_folder, label_idx in CLASS_MAP.items():
        folder_path = os.path.join(split_dir, cls_folder)
        if not os.path.exists(folder_path):
            continue
        # Use only RAW (non-augmented) files for evaluation
        raw_files = [f for f in os.listdir(folder_path)
                     if f.endswith('.wav') and not f.startswith('aug_')]
        for f in raw_files:
            files.append(os.path.join(folder_path, f))
            labels.append(label_idx)
    return files, labels

def extract_features(files):
    feats = []
    for fp in files:
        try:
            with wave.open(fp, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
        feat = extract_logmel_features(audio)
        feats.append(np.expand_dims(feat, axis=-1))
    return np.array(feats, dtype=np.float32)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def main():
    print("=== Step 2: Threshold Sweep on Validation Set ===\n")

    # Load validation set (raw files only)
    val_files, val_labels = load_split('validation')
    print(f"Validation raw samples: {len(val_files)}")
    print(f"  background: {val_labels.count(0)}, dracarys: {val_labels.count(1)}, unknown: {val_labels.count(2)}")

    print("\nExtracting features...")
    x_val = extract_features(val_files)
    y_val = np.array(val_labels)

    # Load model
    model = models.load_model(STAGE_B_CHECKPOINT)
    logits = model.predict(x_val, batch_size=64, verbose=0)
    probs = softmax(logits)

    dracarys_probs = probs[:, 1]  # probability of class 1 = dracarys

    # Threshold sweep
    print(f"\n{'Thresh':>6} | {'Dracarys Recall':>16} | {'FA Rate (val)':>14} | {'Precision':>10} | {'Status':>6}")
    print("-" * 65)

    best_threshold = None
    best_fa = 1.0
    best_recall = 0.0

    for thresh in np.arange(0.50, 1.00, 0.01):
        preds = (dracarys_probs >= thresh).astype(int)

        # Dracarys recall: among actual dracarys (label=1), how many predicted as 1
        dracarys_mask = (y_val == 1)
        recall = preds[dracarys_mask].mean() if dracarys_mask.sum() > 0 else 0.0

        # FA rate: among NON-dracarys samples, how many predicted as dracarys
        non_dracarys_mask = (y_val != 1)
        fa_rate = preds[non_dracarys_mask].mean() if non_dracarys_mask.sum() > 0 else 0.0

        # Precision: among predicted dracarys, how many are actually dracarys
        pred_pos = preds.sum()
        precision = (preds[dracarys_mask].sum() / pred_pos) if pred_pos > 0 else 0.0

        meets_fa = fa_rate <= 0.02
        meets_recall = recall >= 0.85
        status = "PASS" if (meets_fa and meets_recall) else ("FA-OK" if meets_fa else ("RCL-OK" if meets_recall else "FAIL"))

        print(f"  {thresh:.2f}  | {recall*100:>14.1f}% | {fa_rate*100:>12.2f}% | {precision*100:>8.1f}% | {status}")

        if meets_fa and recall > best_recall:
            best_recall = recall
            best_fa = fa_rate
            best_threshold = thresh

    print("\n=== DIAGNOSIS ===")
    if best_threshold is not None:
        print(f"THRESHOLD-TUNABLE: Threshold {best_threshold:.2f} achieves FA={best_fa*100:.2f}% with recall={best_recall*100:.1f}%")
        print("→ A threshold calibration fix CAN partially address FA without retraining.")
    else:
        print("NO THRESHOLD achieves FA ≤ 2% with dracarys recall ≥ 85%.")
        print("→ This is a data/architecture problem, not threshold calibration.")
        print("→ Background data expansion + partial unfreezing required.")

    # Also sweep on test set for comparison
    print("\n=== Test Set Sweep (held out) ===")
    test_files, test_labels = load_split('test')
    x_test = extract_features(test_files)
    y_test = np.array(test_labels)
    logits_test = model.predict(x_test, batch_size=64, verbose=0)
    probs_test = softmax(logits_test)
    dracarys_probs_test = probs_test[:, 1]

    print(f"{'Thresh':>6} | {'Dracarys Recall':>16} | {'FA Rate (test)':>15}")
    print("-" * 45)
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        preds_t = (dracarys_probs_test >= thresh).astype(int)
        dm = (y_test == 1); ndm = (y_test != 1)
        r = preds_t[dm].mean() if dm.sum() > 0 else 0.0
        fa = preds_t[ndm].mean() if ndm.sum() > 0 else 0.0
        print(f"  {thresh:.2f}  | {r*100:>14.1f}% | {fa*100:>13.2f}%")

if __name__ == '__main__':
    main()

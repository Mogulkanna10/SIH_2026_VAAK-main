#!/usr/bin/env python3
"""
train_small.py — Stage 2: Train Small Companion DS-CNN (2 blocks) for Dracarys KWS.

Architecture:
  - Init Conv: 64 filters, (10,4) kernel, stride (2,2)
  - 2x Depthwise-Separable blocks: 3x3 DW + 1x1 PW, 64 ch each
  - GlobalAveragePooling2D → Dense(3)  [background=0, unknown=1, dracarys=2]
  - Total params: ~13K (vs ~50K for the full 4-block model)

Dataset:
  /home/mogul/split/{train,validation,test}/{positive,negative,background}/*.wav
  3-class labels: background=0, negative/unknown=1, positive/dracarys=2

Output:
  pi_deploy/dracarys_small.keras  (best val checkpoint)
"""

import os
import sys
import wave
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

SPLIT_DIR       = '/home/mogul/split'
SMALL_CHECKPOINT = 'pi_deploy/dracarys_small.keras'

# 3-class: 0 = background, 1 = unknown (negative), 2 = dracarys (positive)
CLASS_NAMES     = ['background', 'unknown', 'dracarys']
FOLDER_TO_LABEL = {'background': 0, 'negative': 1, 'positive': 2}
NUM_CLASSES = 3


def load_dataset(split='train'):
    """Load split as 3-class (0=background, 1=unknown, 2=dracarys)."""
    files, labels = [], []
    split_dir = os.path.join(SPLIT_DIR, split)

    for folder, label_idx in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(split_dir, folder)
        if not os.path.exists(folder_path):
            print(f"  WARNING: folder missing — {folder_path}")
            continue
        all_files = [os.path.join(folder_path, f)
                     for f in os.listdir(folder_path) if f.endswith('.wav')]
        files.extend(all_files)
        labels.extend([label_idx] * len(all_files))
        print(f"    {folder:12s} → class {label_idx} : {len(all_files)} files")

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

    x = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Loaded {split:12s}: {len(x)} samples  "
          f"(bg={np.sum(y==0)}, unknown={np.sum(y==1)}, dracarys={np.sum(y==2)})")
    return x, y


def build_small_dscnn(input_shape=(49, 40, 1), num_classes=NUM_CLASSES):
    """
    Small DS-CNN: 2 depthwise-separable blocks (vs 4 in the full model).
      Init conv : 64 filters, (10,4) kernel, stride (2,2)
      DS block 1: DW(3,3) + PW(1,1), 64 ch
      DS block 2: DW(3,3) + PW(1,1), 64 ch
      GAP → Dense(num_classes)
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=(10, 4), strides=(2, 2),
                      padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def dscnn_block(feat, ch):
        dw = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(feat)
        dw = layers.BatchNormalization()(dw)
        dw = layers.ReLU()(dw)
        pw = layers.Conv2D(ch, kernel_size=(1, 1), use_bias=False)(dw)
        pw = layers.BatchNormalization()(pw)
        pw = layers.ReLU()(pw)
        return pw

    x = dscnn_block(x, 64)
    x = dscnn_block(x, 64)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return Model(inputs, outputs, name='dracarys_small')


def softmax_np(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    print("=" * 70)
    print("STAGE 2 — Small Companion Model Training (2-Block DS-CNN, 3-Class)")
    print("Classes: 0=background  1=unknown  2=dracarys")
    print("=" * 70)
    print(f"  Split dir : {SPLIT_DIR}")
    print(f"  Output    : {SMALL_CHECKPOINT}\n")

    print("Loading datasets...")
    x_train, y_train = load_dataset('train')
    x_val,   y_val   = load_dataset('validation')
    x_test,  y_test  = load_dataset('test')

    model = build_small_dscnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=SMALL_CHECKPOINT,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
    )
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=8, monitor='val_accuracy', mode='max',
        restore_best_weights=True, verbose=1,
    )

    EPOCHS = 40
    print(f"\nTraining for up to {EPOCHS} epochs (early-stop patience=8)...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=EPOCHS,
        callbacks=[cp_callback, es_callback],
    )

    best_model = models.load_model(SMALL_CHECKPOINT)

    print(f"\n{'='*70}")
    print("SMALL MODEL EVALUATION (3-Class: 0=background, 1=unknown, 2=dracarys)")
    print(f"{'='*70}")

    for split_name, x, y in [('Validation', x_val, y_val), ('Test (held-out)', x_test, y_test)]:
        logits = best_model.predict(x, batch_size=64, verbose=0)
        probs  = softmax_np(logits)
        preds  = np.argmax(logits, axis=-1)

        print(f"\n--- {split_name} ---")
        print(classification_report(y, preds,
                                    labels=[0, 1, 2],
                                    target_names=CLASS_NAMES,
                                    zero_division=0))
        print("Confusion matrix (rows=actual, cols=predicted [bg=0, unknown=1, dracarys=2]):")
        print(confusion_matrix(y, preds, labels=[0, 1, 2]))

        # Output shape verification
        assert logits.shape[1] == 3, f"FAIL: output shape is {logits.shape}, expected (N,3)"
        print(f"  Output shape check: {logits.shape[1]} classes — PASS")

        # Threshold-based Dracarys recall & false activation rate
        thresh = 0.85
        dracarys_mask  = (probs[:, 2] >= thresh)          # dracarys=class 2
        dracarys_true  = (y == 2)
        non_dracarys   = (y != 2)
        recall  = (np.sum(dracarys_mask & dracarys_true) / dracarys_true.sum()
                   if dracarys_true.sum() > 0 else float('nan'))
        fa_rate = (np.sum(dracarys_mask & non_dracarys) / non_dracarys.sum()
                   if non_dracarys.sum() > 0 else float('nan'))
        print(f"  @ threshold 0.85 → Dracarys Recall: {recall*100:.2f}%   "
              f"False-Activation Rate: {fa_rate*100:.2f}%")

        # Latency & RAM
        import time, tracemalloc
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = best_model.predict(x[:32], batch_size=32, verbose=0)
        t1 = time.perf_counter()
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        latency_per_sample_ms = (t1 - t0) / 32 * 1000
        print(f"  Keras inference latency (32 samples): {latency_per_sample_ms:.2f} ms/sample")
        print(f"  Peak RAM (tracemalloc): {mem_peak/1024:.1f} KB")

    print(f"\n{'='*70}")
    print("Training complete.")
    print(f"  Checkpoint : {SMALL_CHECKPOINT}")
    print("  Classes    : 0=background  1=unknown  2=dracarys")
    print("  Next step  : verify Gate 2, then run scripts/quantize_small.py")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

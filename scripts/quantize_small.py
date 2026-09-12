#!/usr/bin/env python3
"""
quantize_small.py — INT8 Post-Training Quantization for Small Companion Model.
Converts pi_deploy/dracarys_small.keras → pi_deploy/dracarys_kws_small.tflite
with full INT8 precision using 200 representative samples.
"""

import os
import sys
import wave
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

SMALL_KERAS   = 'pi_deploy/dracarys_small.keras'
SMALL_TFLITE  = 'pi_deploy/dracarys_kws_small.tflite'
SPLIT_DIR     = '/home/mogul/split'


def representative_dataset_gen():
    """Sample proportionally across positive and negative from training set."""
    samples = []
    np.random.seed(42)
    for cls_folder in ['positive', 'negative']:
        folder_path = os.path.join(SPLIT_DIR, 'train', cls_folder)
        if os.path.exists(folder_path):
            files = [os.path.join(folder_path, f)
                     for f in os.listdir(folder_path) if f.endswith('.wav')]
            chosen = list(np.random.choice(files, min(100, len(files)), replace=False))
            samples.extend(chosen)
    np.random.shuffle(samples)
    print(f"Representative dataset: {len(samples)} samples.")

    for fp in samples:
        try:
            with wave.open(fp, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)

        feat = extract_logmel_features(audio)
        feat = np.expand_dims(feat, axis=0)   # (1, 49, 40)
        feat = np.expand_dims(feat, axis=-1)  # (1, 49, 40, 1)
        yield [feat.astype(np.float32)]


def main():
    print("=" * 70)
    print("STAGE 2 — Quantize Small Model to INT8 TFLite")
    print("=" * 70)
    print(f"  Input  : {SMALL_KERAS}")
    print(f"  Output : {SMALL_TFLITE}")

    print(f"\nLoading Keras model from: {SMALL_KERAS}")
    model = tf.keras.models.load_model(SMALL_KERAS)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    print("\nConverting to full INT8 TFLite...")
    tflite_model = converter.convert()

    with open(SMALL_TFLITE, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(SMALL_TFLITE) / 1024
    print(f"\n✅ INT8 TFLite model saved to: {SMALL_TFLITE}")
    print(f"   File size: {size_kb:.1f} KB")
    print("\nNext step: update live_kws.py for dual interpreter allocation.")


if __name__ == '__main__':
    main()

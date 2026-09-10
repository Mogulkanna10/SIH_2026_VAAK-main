#!/usr/bin/env python3
"""
quantize.py — Phase 5: INT8 Post-Training Quantization of Dracarys KWS Model.
Converts dracarys_kws_stage_b.keras to dracarys_kws.tflite with full INT8 precision.
Uses proportional representative dataset generator across all 3 classes.
"""

import os
import sys
import wave
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

STAGE_B_CHECKPOINT = '/media/yuvan/2095-DDEC/models/dracarys_kws_stage_b.keras'
TFLITE_OUTPUT_PATH = '/media/yuvan/2095-DDEC/models/dracarys_kws.tflite'
SPLIT_DIR = '/home/yuvan/split/train'

CLASS_MAP = {'background': 0, 'positive': 1, 'negative': 2}

def representative_dataset_gen():
    """Sample proportionally across all 3 classes from training set."""
    samples = []
    np.random.seed(42)
    
    for cls_folder in ['background', 'positive', 'negative']:
        folder_path = os.path.join(SPLIT_DIR, cls_folder)
        if os.path.exists(folder_path):
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
            # Sample 100 clips per class
            chosen = list(np.random.choice(files, min(100, len(files)), replace=False))
            samples.extend(chosen)
            
    np.random.shuffle(samples)
    print(f"Representative dataset generator initialized with {len(samples)} samples.")
    
    for fp in samples:
        try:
            with wave.open(fp, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
            
        feat = extract_logmel_features(audio)
        feat = np.expand_dims(feat, axis=0)      # (1, 49, 40)
        feat = np.expand_dims(feat, axis=-1)     # (1, 49, 40, 1)
        yield [feat.astype(np.float32)]

def main():
    print("=== Phase 5: INT8 Post-Training Quantization ===")
    print(f"Loading Keras model from: {STAGE_B_CHECKPOINT}")
    model = tf.keras.models.load_model(STAGE_B_CHECKPOINT)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("\nConverting model to full INT8 TFLite...")
    tflite_quant_model = converter.convert()
    
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH), exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"INT8 TFLite model successfully saved to: {TFLITE_OUTPUT_PATH}")

if __name__ == '__main__':
    main()

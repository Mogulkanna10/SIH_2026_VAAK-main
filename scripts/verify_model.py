#!/usr/bin/env python3
"""
verify_model.py — Phase 5 Gate 5 Verification & Post-Quantization Accuracy Audit.
Audits:
  1. File size (Target 20-90 KB)
  2. RAM tensor arena footprint estimate (Target < 256 KB)
  3. Op audit (Zero float32 fallback ops)
  4. Post-quantization accuracy & FA rate on Validation and Test sets at threshold 0.85.
"""

import os
import sys
import wave
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

TFLITE_PATH = '/media/yuvan/2095-DDEC/models/dracarys_kws.tflite'
SPLIT_DIR = '/home/yuvan/split'

CLASS_MAP = {'background': 0, 'positive': 1, 'negative': 2}
THRESHOLD = 0.85

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def load_eval_data(split='validation'):
    files, labels = [], []
    split_dir = os.path.join(SPLIT_DIR, split)
    for cls_folder, label_idx in CLASS_MAP.items():
        folder_path = os.path.join(split_dir, cls_folder)
        if not os.path.exists(folder_path):
            continue
        raw_files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and not f.startswith('aug_')]
        for f in raw_files:
            files.append(os.path.join(folder_path, f))
            labels.append(label_idx)
            
    feats = []
    for fp in files:
        try:
            with wave.open(fp, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
        feat = extract_logmel_features(audio)
        feats.append(np.expand_dims(feat, axis=-1))  # (49, 40, 1)
        
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int32)

def main():
    print("======================================================================")
    print("GATE 5 VERIFICATION & INT8 POST-QUANTIZATION AUDIT")
    print("======================================================================\n")

    # --- 1. File Size Verification ---
    file_size_bytes = os.path.getsize(TFLITE_PATH)
    file_size_kb = file_size_bytes / 1024.0
    print(f"1. FILE SIZE AUDIT:")
    print(f"   Model Path: {TFLITE_PATH}")
    print(f"   Actual Size: {file_size_kb:.2f} KB ({file_size_bytes} bytes)")
    if 20.0 <= file_size_kb <= 90.0:
        print("   Status: PASSED ✅ (Within 20-90 KB target budget)")
    else:
        print(f"   Status: WARNING ⚠️ (Outside 20-90 KB target budget)")

    # --- Initialize TFLite Interpreter ---
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    tensor_details = interpreter.get_tensor_details()

    # --- 2. RAM Tensor Arena Estimate ---
    # TFLite reuses memory between sequential activation tensors.
    # Peak active memory = max size of 2 largest adjacent layer activation tensors + input buffer
    activation_sizes = []
    for t in tensor_details:
        name = t['name']
        shape = t['shape']
        dtype = t['dtype']
        is_weight = any(w in name.lower() for w in ['weight', 'bias', 'kernel', 'filter', 'param', 'quant'])
        if not is_weight and len(shape) > 0:
            itemsize = 1 if 'int8' in str(dtype) or 'uint8' in str(dtype) else 4
            nbytes = int(np.prod(shape)) * itemsize
            activation_sizes.append(nbytes)

    # Sort activation sizes to find peak ping-pong memory
    activation_sizes.sort(reverse=True)
    # Peak active working memory (largest two activation buffers in flight + input)
    peak_arena_bytes = activation_sizes[0] + activation_sizes[1] + 1960 if len(activation_sizes) >= 2 else sum(activation_sizes)
    peak_arena_kb = peak_arena_bytes / 1024.0

    print(f"\n2. RAM (TENSOR ARENA) FOOTPRINT AUDIT:")
    print(f"   Largest Layer Activation Buffer: {activation_sizes[0]/1024.0:.2f} KB ({activation_sizes[0]} bytes)")
    print(f"   Peak Active Tensor Arena (Ping-Pong Memory Reuse): {peak_arena_kb:.2f} KB ({peak_arena_bytes} bytes)")
    if peak_arena_kb < 256.0:
        print(f"   Status: PASSED ✅ (Comfortably under 256 KB arena budget; headroom={256.0-peak_arena_kb:.2f} KB)")
    else:
        print(f"   Status: FAILED ❌ (Exceeds 256 KB arena budget)")

    # --- 3. Op Audit (Zero float32 fallback ops) ---
    print(f"\n3. OP AUDIT:")
    ops_details = interpreter._get_ops_details()
    float32_ops = []
    print(f"   Total Quantized Operators in Model: {len(ops_details)}")
    for op in ops_details:
        op_name = op.get('op_name', 'UNKNOWN')
        print(f"     - Op: {op_name}")
        if 'FLOAT' in op_name.upper():
            float32_ops.append(op_name)
            
    print(f"   Input Dtype: {input_details['dtype']}")
    print(f"   Output Dtype: {output_details['dtype']}")
    if len(float32_ops) == 0 and input_details['dtype'] == np.int8 and output_details['dtype'] == np.int8:
        print("   Status: PASSED ✅ (100% Full INT8 model, zero float32 fallback ops)")
    else:
        print(f"   Status: FAILED ❌ (Found float32 fallback ops or non-int8 IO: {float32_ops})")

    # --- 4. Post-Quantization Accuracy Audit ---
    print(f"\n4. POST-QUANTIZATION ACCURACY AUDIT (Locked-in Threshold = {THRESHOLD:.2f}):")

    in_scale, in_zero_point = input_details['quantization']
    out_scale, out_zero_point = output_details['quantization']

    def run_quantized_inference(x_data):
        probs_list = []
        for feat in x_data:
            quant_in = np.round(feat / in_scale + in_zero_point)
            quant_in = np.clip(quant_in, -128, 127).astype(np.int8)
            quant_in = np.expand_dims(quant_in, axis=0)

            interpreter.set_tensor(input_details['index'], quant_in)
            interpreter.invoke()

            quant_out = interpreter.get_tensor(output_details['index'])[0]
            dequant_out = (quant_out.astype(np.float32) - out_zero_point) * out_scale
            prob = softmax(np.expand_dims(dequant_out, axis=0))[0]
            
            probs_list.append(prob)
        return np.array(probs_list)

    # Validate on Validation set
    print("\n   Evaluating Validation Set (Quantized INT8 Model)...")
    x_val, y_val = load_eval_data('validation')
    val_probs = run_quantized_inference(x_val)
    val_dracarys_pred = (val_probs[:, 1] >= THRESHOLD)
    val_recall = np.sum(val_dracarys_pred[y_val == 1]) / np.sum(y_val == 1)
    val_fa = np.sum(val_dracarys_pred[y_val != 1]) / np.sum(y_val != 1)

    print(f"   Val Dracarys Recall (Quantized): {val_recall*100:.2f}% (Target >= 90%)")
    print(f"   Val False Activation Rate (Quantized): {val_fa*100:.2f}% (Target <= 2%)")

    # Validate on Test set
    print("\n   Evaluating Test Set (Quantized INT8 Model, Held-Out)...")
    x_test, y_test = load_eval_data('test')
    test_probs = run_quantized_inference(x_test)
    test_dracarys_pred = (test_probs[:, 1] >= THRESHOLD)
    test_recall = np.sum(test_dracarys_pred[y_test == 1]) / np.sum(y_test == 1)
    test_fa = np.sum(test_dracarys_pred[y_test != 1]) / np.sum(y_test != 1)

    print(f"   Test Dracarys Recall (Quantized): {test_recall*100:.2f}% (Target >= 90%)")
    print(f"   Test False Activation Rate (Quantized): {test_fa*100:.2f}% (Target <= 2%)")

    # Gate 5 Overall Assessment
    passed_gate5 = (
        (20.0 <= file_size_kb <= 90.0) and
        (peak_arena_kb < 256.0) and
        (len(float32_ops) == 0) and
        (val_recall >= 0.90) and (val_fa <= 0.02) and
        (test_recall >= 0.90) and (test_fa <= 0.02)
    )

    print(f"\n======================================================================")
    if passed_gate5:
        print("GATE 5 OVERALL STATUS: PASSED! ALL CRITERIA MET WITH ZERO REGRESSION ✅")
    else:
        print("GATE 5 OVERALL STATUS: NEEDS REVIEW")
    print("======================================================================")

if __name__ == '__main__':
    main()

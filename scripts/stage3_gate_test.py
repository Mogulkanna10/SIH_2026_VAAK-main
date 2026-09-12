#!/usr/bin/env python3
"""
stage3_gate_test.py — Gate 3 offline test harness.
Classes: 0=background, 1=unknown, 2=dracarys
"""
import os, sys, wave, random
import numpy as np

PI_DEPLOY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pi_deploy'))
sys.path.insert(0, PI_DEPLOY)

from features import extract_logmel_features
from capacity_controller import CapacityController

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

SMALL_TFLITE = os.path.join(PI_DEPLOY, 'dracarys_kws_small.tflite')
SPLIT_DIR    = '/home/mogul/split'

print(f"Loading TFLite: {SMALL_TFLITE}")
interp = tflite.Interpreter(model_path=SMALL_TFLITE)
interp.allocate_tensors()
in_d  = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
in_scale, in_zp   = in_d['quantization']
out_scale, out_zp = out_d['quantization']
print(f"  Input  INT8  scale={in_scale:.5f}  zp={in_zp}")
print(f"  Output INT8  scale={out_scale:.5f}  zp={out_zp}")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def infer_tflite(feat):
    q = np.round(feat / in_scale + in_zp)
    q = np.clip(q, -128, 127).astype(np.int8)[np.newaxis, :, :, np.newaxis]
    interp.set_tensor(in_d['index'], q)
    interp.invoke()
    raw = interp.get_tensor(out_d['index'])[0]
    return softmax((raw.astype(np.float32) - out_zp) * out_scale)

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

def pick_wavs(folder, n, seed=42):
    rng = random.Random(seed)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
    return rng.sample(files, min(n, len(files)))

def run_sequence(wav_paths):
    cc = CapacityController()
    records = []
    for i, path in enumerate(wav_paths):
        feat   = extract_logmel_features(load_wav(path))
        probs  = infer_tflite(feat)
        conf   = float(probs[2])
        energy = float(np.mean(feat))
        sel    = cc.select_model(feat, conf)
        cls    = ['BG', 'UNK', 'DRAC'][int(np.argmax(probs))]
        records.append((i, cls, energy, conf, sel))
    return records

def print_seq(label, records):
    print(f"\n{'='*72}")
    print(f"SEQUENCE: {label}  ({len(records)} windows)")
    print(f"{'='*72}")
    print(f"{'Win':>4}  {'Pred':5}  {'Energy':>8}  {'DrConf':>7}  {'Selected':>8}")
    print(f"{'----':>4}  {'-----':5}  {'--------':>8}  {'-------':>7}  {'--------':>8}")
    for i, cls, energy, conf, sel in records:
        print(f"{i:>4}  {cls:5}  {energy:>8.4f}  {conf:>7.4f}  {sel:>8}")
    sels  = [r[4] for r in records]
    flips = sum(1 for a, b in zip(sels, sels[1:]) if a != b)
    print(f"\n  SMALL/LARGE transitions: {flips}")
    print(f"  Final state: {sels[-1]}")
    return flips, sels


bg_files  = pick_wavs(f'{SPLIT_DIR}/test/background', 15, seed=1)
pos_files = pick_wavs(f'{SPLIT_DIR}/test/positive',   7,  seed=2)
unk_files = pick_wavs(f'{SPLIT_DIR}/test/negative',   20, seed=3)

recs_quiet  = run_sequence(bg_files)
f1, s1 = print_seq("TEST 1 — Quiet  (15x BG)", recs_quiet)

mixed_files = bg_files[:8] + pos_files
recs_mixed  = run_sequence(mixed_files)
f2, s2 = print_seq("TEST 2 — Mixed  (8xBG + 7xSpeech)", recs_mixed)

cycle_files = bg_files[:8] + pos_files[:4] + bg_files[8:15]
recs_cycle  = run_sequence(cycle_files)
f3, s3 = print_seq("TEST 3 — Cycle  (8xBG → 4xSpeech → 7xBG)", recs_cycle)

recs_border = run_sequence(unk_files)
f4, s4 = print_seq("TEST 4 — Borderline  (20x Unknown, anti-oscillation)", recs_border)

print(f"\n{'='*72}")
print("GATE 3 VERDICT")
print(f"{'='*72}")
c1 = 'SMALL' in s1
c2 = 'LARGE' in s2[8:]
c3 = 'SMALL' in s3[12:]
c4 = f4 <= 3
print(f"[1] Quiet seq reaches SMALL:                  {'PASS' if c1 else 'FAIL'}")
print(f"[2] Mixed seq goes LARGE on speech:           {'PASS' if c2 else 'FAIL'}")
print(f"[3] Cycle seq returns to SMALL after BG:      {'PASS' if c3 else 'FAIL'}")
print(f"[4] Borderline anti-oscillation ({f4} flips≤3): {'PASS' if c4 else 'FAIL'}")
ok = c1 and c2 and c3 and c4
print(f"\n{'GATE 3: PASS' if ok else 'GATE 3: FAIL'}")

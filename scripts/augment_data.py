#!/usr/bin/env python3
"""
augment_data.py — Augment training dataset classes using audiomentations.
Applies Time shift (+/- 100ms), Pitch shift (+/- 2 semitones), and Gaussian noise.
Augments positive (dracarys) and negative (unknown) 8x.
Leaves background unaugmented as it already has 8,828 real diverse environmental clips.
"""

import os
import wave
import numpy as np
from audiomentations import Compose, AddGaussianNoise, Shift, PitchShift

SAMPLE_RATE = 16000
TRAIN_DIR = '/home/yuvan/split/train'

# Augmentation pipeline
augmenter = Compose([
    Shift(min_shift=-0.1, max_shift=0.1, p=0.8),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
])

def read_wav(path):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sampwidth, framerate

def write_wav(path, audio_data, sampwidth, framerate):
    audio_int16 = np.clip(audio_data * 32768.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(audio_int16.tobytes())

def main():
    print("=== Training Data Augmentation ===")
    
    # Augment positive (dracarys) and negative (unknown)
    for cls in ['positive', 'negative']:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
            
        files = os.listdir(cls_dir)
        aug_files = [f for f in files if f.startswith('aug_')]
        print(f"\nProcessing class '{cls}':")
        print(f"  Removing {len(aug_files)} old augmented files...")
        for f in aug_files:
            os.remove(os.path.join(cls_dir, f))
            
        raw_files = [f for f in os.listdir(cls_dir) if f.endswith('.wav') and not f.startswith('aug_')]
        print(f"  Found {len(raw_files)} raw source clips.")
        
        aug_factor = 8
        generated_count = 0
        
        for f in raw_files:
            src_path = os.path.join(cls_dir, f)
            try:
                audio, sampwidth, framerate = read_wav(src_path)
            except Exception as e:
                print(f"    Error reading {f}: {e}")
                continue
                
            for i in range(aug_factor):
                augmented_audio = augmenter(samples=audio, sample_rate=framerate)
                if len(augmented_audio) < SAMPLE_RATE:
                    augmented_audio = np.pad(augmented_audio, (0, SAMPLE_RATE - len(augmented_audio)))
                elif len(augmented_audio) > SAMPLE_RATE:
                    augmented_audio = augmented_audio[:SAMPLE_RATE]
                    
                aug_filename = f"aug_{i}_{f}"
                dest_path = os.path.join(cls_dir, aug_filename)
                write_wav(dest_path, augmented_audio, sampwidth, framerate)
                generated_count += 1
                
        print(f"  Generated {generated_count} augmented files for '{cls}'. Total: {len(os.listdir(cls_dir))}")

    # For background, clean old aug_ files if any exist
    bg_dir = os.path.join(TRAIN_DIR, 'background')
    if os.path.exists(bg_dir):
        old_bg_aug = [f for f in os.listdir(bg_dir) if f.startswith('aug_')]
        for f in old_bg_aug:
            os.remove(os.path.join(bg_dir, f))
        print(f"\nBackground class: {len(os.listdir(bg_dir))} real environmental raw clips (unaugmented).")

if __name__ == '__main__':
    main()

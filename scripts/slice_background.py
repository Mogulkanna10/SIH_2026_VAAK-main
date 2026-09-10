#!/usr/bin/env python3
"""
slice_background.py — Slice Speech Commands _background_noise_ into 1-second wavs.
Splits files disjointly across train, validation, and test splits.
"""

import os
import wave
import numpy as np

BG_SRC_DIR = '/media/yuvan/2095-DDEC/dracarys_kws_data/_background_noise_'
OUTPUT_BASE_DIR = '/home/yuvan/split'

# Disjoint partitioning of source background files
SPLIT_CONFIG = {
    'train': ['doing_the_dishes.wav', 'exercise_bike.wav', 'pink_noise.wav', 'white_noise.wav'],
    'validation': ['running_tap.wav'],
    'test': ['dude_miaowing.wav']
}

def slice_wav_file(src_path, dest_dir, prefix, overlap_ratio=0.5):
    if not os.path.exists(src_path):
        print(f"Warning: Source file {src_path} not found.")
        return 0
        
    os.makedirs(dest_dir, exist_ok=True)
    
    with wave.open(src_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Read all audio
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16)
        
        # If stereo, convert to mono
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio.mean(axis=1).astype(np.int16)
            
    # Slicing parameters (1 second = framerate frames)
    slice_len = framerate
    step = int(slice_len * (1 - overlap_ratio))
    
    slice_count = 0
    for start in range(0, len(audio) - slice_len + 1, step):
        chunk = audio[start:start + slice_len]
        filename = f"{prefix}_{slice_count:04d}.wav"
        filepath = os.path.join(dest_dir, filename)
        
        with wave.open(filepath, 'wb') as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(chunk.tobytes())
            
        slice_count += 1
        
    return slice_count

def main():
    print("=== Slicing Background Noise Files ===")
    for split, files in SPLIT_CONFIG.items():
        dest_dir = os.path.join(OUTPUT_BASE_DIR, split, 'background')
        print(f"\nProcessing {split} split (saving to {dest_dir}):")
        
        total_slices = 0
        for f in files:
            src_path = os.path.join(BG_SRC_DIR, f)
            prefix = os.path.splitext(f)[0]
            slices = slice_wav_file(src_path, dest_dir, prefix)
            print(f"  Sliced {f} into {slices} 1-second files.")
            total_slices += slices
            
        print(f"Total background slices for {split}: {total_slices}")

if __name__ == '__main__':
    main()

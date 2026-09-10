#!/usr/bin/env python3
"""
prepare_expanded_background.py — Extract diverse environmental background clips from ESC-50 dataset
and Speech Commands _background_noise_, partitioning categories disjointly across train, val, and test splits.
Outputs 1-second 16kHz mono WAVs. Uses standard csv library.
"""

import os
import wave
import csv
import numpy as np
import scipy.signal

ESC50_DIR = '/home/yuvan/esc50/ESC-50-master'
ESC50_AUDIO = os.path.join(ESC50_DIR, 'audio')
ESC50_META = os.path.join(ESC50_DIR, 'meta/esc50.csv')

SPEECH_CMDS_BG = '/media/yuvan/2095-DDEC/dracarys_kws_data/_background_noise_'

OUTPUT_BASE = '/home/yuvan/split'

# Disjoint Category Mapping for ESC-50
TRAIN_CATEGORIES = [
    'vacuum_cleaner', 'washing_machine', 'rain', 'sea_waves', 'crackling_fire',
    'wind', 'engine', 'car_horn', 'train', 'keyboard_typing', 'door_wood_knock',
    'footsteps', 'insects', 'crickets', 'dog', 'chirping_birds', 'coughing',
    'clapping', 'water_drops', 'clock_tick', 'toilet_flush', 'pouring_water', 'glass_breaking'
]

VAL_CATEGORIES = [
    'siren', 'airplane', 'thunderstorm', 'crying_baby', 'clock_alarm',
    'mouse_click', 'rooster', 'can_opening', 'sneezing', 'snoring'
]

TEST_CATEGORIES = [
    'cat', 'helicopter', 'chainsaw', 'fireworks', 'church_bells', 'crow',
    'cow', 'frog', 'hen', 'pig', 'sheep', 'drinking_sipping',
    'door_wood_creaks', 'brushing_teeth', 'breathing', 'laughing'
]

# Speech Commands _background_noise_ files mapping
SC_BG_MAPPING = {
    'train': ['doing_the_dishes.wav', 'exercise_bike.wav', 'pink_noise.wav', 'white_noise.wav'],
    'validation': ['running_tap.wav'],
    'test': ['dude_miaowing.wav']
}

def load_resample_16k(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
                
            if sr != 16000:
                num_samples = int(len(audio) * 16000 / sr)
                audio = scipy.signal.resample(audio, num_samples)
                
            return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def write_wav_16k(filepath, audio_1s):
    audio_int16 = np.clip(audio_1s * 32768.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())

def slice_into_1s(audio_full, dest_dir, prefix, stride_s=0.5):
    os.makedirs(dest_dir, exist_ok=True)
    target_len = 16000
    stride_len = int(stride_s * 16000)
    
    count = 0
    for start in range(0, len(audio_full) - target_len + 1, stride_len):
        chunk = audio_full[start:start + target_len]
        filename = f"{prefix}_{count:04d}.wav"
        write_wav_16k(os.path.join(dest_dir, filename), chunk)
        count += 1
    return count

def main():
    print("=== Extracting & Partitioning Expanded Background Data ===")
    
    # Read ESC-50 csv metadata
    esc_records = []
    with open(ESC50_META, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            esc_records.append(row)
            
    # Process each split
    split_map = {
        'train': TRAIN_CATEGORIES,
        'validation': VAL_CATEGORIES,
        'test': TEST_CATEGORIES
    }
    
    for split_name, categories in split_map.items():
        out_dir = os.path.join(OUTPUT_BASE, split_name, 'background')
        print(f"\n--- Processing {split_name.upper()} split (saving to {out_dir}) ---")
        
        # Clean existing background clips in split directory
        if os.path.exists(out_dir):
            for existing_f in os.listdir(out_dir):
                if existing_f.endswith('.wav'):
                    os.remove(os.path.join(out_dir, existing_f))
                    
        total_clips = 0
        categories_used = []
        
        # 1. Process ESC-50 categories for this split
        for cat in categories:
            cat_files = [r['filename'] for r in esc_records if r['category'] == cat]
            cat_clips = 0
            for fn in cat_files:
                fp = os.path.join(ESC50_AUDIO, fn)
                audio_16k = load_resample_16k(fp)
                if audio_16k is not None:
                    # 5-second audio sliced with 0.5s stride yields 9 1-second clips per file
                    n = slice_into_1s(audio_16k, out_dir, prefix=f"esc50_{cat}_{fn[:-4]}", stride_s=0.5)
                    cat_clips += n
            total_clips += cat_clips
            categories_used.append(f"ESC-50:{cat} ({cat_clips} raw 1s clips)")
            
        # 2. Process Speech Commands _background_noise_ files for this split
        for sc_file in SC_BG_MAPPING[split_name]:
            sc_path = os.path.join(SPEECH_CMDS_BG, sc_file)
            if os.path.exists(sc_path):
                audio_16k = load_resample_16k(sc_path)
                if audio_16k is not None:
                    prefix_sc = f"sc_bg_{sc_file[:-4]}"
                    n = slice_into_1s(audio_16k, out_dir, prefix=prefix_sc, stride_s=0.5)
                    total_clips += n
                    categories_used.append(f"SpeechCommands:{sc_file[:-4]} ({n} raw 1s clips)")
                    
        print(f"Total RAW background clips generated for {split_name}: {total_clips}")
        print(f"Categories breakdown ({len(categories_used)} distinct sources):")
        for cat_info in categories_used:
            print(f"  - {cat_info}")

if __name__ == '__main__':
    main()

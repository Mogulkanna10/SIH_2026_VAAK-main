import numpy as np
import os
import glob
import time
import wave
import sys
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pi_deploy'))
from live_kws import DracarysKWS

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

def evaluate_config(config_name, dataset_split='validation'):
    kws = DracarysKWS()
    
    # Configure the controller based on config_name
    if config_name == 'always-large':
        kws.controller.current_model = "LARGE"
        kws.controller.conf_thresh_high = 2.0  # Impossible to trigger SMALL
        kws.controller.noise_thresh_low = -100.0 # Will always stay LARGE
        kws.controller.large_hold_frames = 99999
    elif config_name == 'always-small':
        kws.controller.current_model = "SMALL"
        kws.controller.noise_thresh_high = 100.0 # Will always stay SMALL
        kws.controller.conf_thresh_high = 2.0
        kws.controller.min_large_frames = 0
        kws.controller.large_hold_frames = 0
    elif config_name == 'noise-only':
        kws.controller.conf_thresh_high = 2.0 # Disable confidence trigger
        kws.controller.conf_thresh_low = 2.0  # Disable confidence hold
    elif config_name == 'full':
        pass # Default controller settings
    
    split_dir = f'/home/mogul/split/{dataset_split}'
    
    dracarys_files = glob.glob(os.path.join(split_dir, 'positive', '*.wav'))
    negative_files = glob.glob(os.path.join(split_dir, 'negative', '*.wav'))
            
    hop_size = 3200 # 0.2s
    total_large_calls = 0
    total_windows = 0
    total_latency = 0.0
    
    true_positives = 0
    false_positives = 0
    
    # Evaluate Dracarys
    for path in dracarys_files:
        kws.audio_buffer.fill(0)
        kws.last_detection_time = 0 # RESET COOLDOWN
        
        # In noise-only, simulate initial noise condition accurately
        if config_name == 'noise-only' or config_name == 'full':
            kws.controller.current_model = "SMALL"
            kws.controller.rolling_noise_floor = -15.0
            
        audio = load_wav(path)
        triggered = False
        for i in range(0, len(audio) - hop_size + 1, hop_size):
            chunk = audio[i:i+hop_size]
            detected, score, probs, latency, t1, selected_model = kws.process_chunk(chunk)
            
            total_windows += 1
            total_latency += latency
            if selected_model == "LARGE":
                total_large_calls += 1
                
            if detected:
                triggered = True
        if triggered:
            true_positives += 1
            
    # Evaluate Negative
    for path in negative_files:
        kws.audio_buffer.fill(0)
        kws.last_detection_time = 0 # RESET COOLDOWN
        
        if config_name == 'noise-only' or config_name == 'full':
            kws.controller.current_model = "SMALL"
            kws.controller.rolling_noise_floor = -15.0
            
        audio = load_wav(path)
        triggered = False
        for i in range(0, len(audio) - hop_size + 1, hop_size):
            chunk = audio[i:i+hop_size]
            detected, score, probs, latency, t1, selected_model = kws.process_chunk(chunk)
            
            total_windows += 1
            total_latency += latency
            if selected_model == "LARGE":
                total_large_calls += 1
                
            if detected:
                triggered = True
        if triggered:
            false_positives += 1
            
    recall = true_positives / len(dracarys_files) if dracarys_files else 0
    fa_rate = false_positives / len(negative_files) if negative_files else 0
    large_rate = total_large_calls / total_windows if total_windows else 0
    mean_latency = total_latency / total_windows if total_windows else 0
    
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent()
    
    return {
        'recall': recall,
        'fa_rate': fa_rate,
        'large_rate': large_rate,
        'mean_latency': mean_latency,
        'ram_mb': rss_mb,
        'cpu_percent': cpu_percent
    }

if __name__ == '__main__':
    configs = ['always-large', 'always-small', 'noise-only', 'full']
    splits = ['validation', 'test']
    
    print("Configuration | Split      | Recall | FA Rate | Large% | Latency | RAM (MB) | CPU%")
    print("-" * 88)
    for c in configs:
        for s in splits:
            # warmup cpu profiling
            psutil.Process(os.getpid()).cpu_percent()
            res = evaluate_config(c, s)
            print(f"{c:13s} | {s:10s} | {res['recall']*100:5.1f}% | {res['fa_rate']*100:6.2f}% | {res['large_rate']*100:5.1f}% | {res['mean_latency']:5.1f}ms | {res['ram_mb']:6.1f}   | {res['cpu_percent']:4.1f}%")

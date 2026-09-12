import numpy as np
import sys
import wave
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pi_deploy'))
from live_kws import DracarysKWS

def run_test(wav_path):
    kws = DracarysKWS()
    
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
    hop_size = int(sr * 0.2)
    
    print(f"Testing {wav_path} with REAL INFERENCE...")
    
    for i in range(0, len(audio) - hop_size, hop_size):
        chunk = audio[i:i+hop_size]
        detected, score, probs, latency, t1, selected_model = kws.process_chunk(chunk)
        
        # We need the controller's internal rolling noise to print it
        rolling = kws.controller.rolling_noise_floor
        energy = np.mean(kws.audio_buffer) # approximate, not exact log-mel mean, but we can print the rolling
        
        print(f"Window {i//hop_size:03d} | Dracarys Score: {score*100:5.1f}% | Rolling: {rolling:7.3f} | Model: {selected_model}")

if __name__ == '__main__':
    run_test(os.path.join(os.path.dirname(__file__), '..', 'test.wav'))

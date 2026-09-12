import numpy as np
import sys
import wave
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pi_deploy'))
from features import extract_logmel_features
from capacity_controller import CapacityController

def run_test(wav_path):
    controller = CapacityController()
    
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
    buffer_size = sr
    hop_size = int(sr * 0.2)
    
    print(f"Testing {wav_path}...")
    
    # Simulate a fake recent confidence of 0.0 (no keyword spoken)
    # to purely test the noise floor logic.
    recent_conf = 0.0
    
    for i in range(0, len(audio) - buffer_size, hop_size):
        chunk = audio[i:i+buffer_size]
        feat = extract_logmel_features(chunk)
        
        # Test controller
        model = controller.select_model(feat, recent_conf)
        
        # We can also dynamically change recent_conf for testing
        # If we reach halfway, pretend we hear a keyword
        if i > len(audio) // 2:
            recent_conf = 0.15
        
        mean_energy = np.mean(feat)
        print(f"Window {i//hop_size:03d} | Energy: {mean_energy:7.3f} | Rolling: {controller.rolling_noise_floor:7.3f} | Model: {model}")

if __name__ == '__main__':
    run_test(os.path.join(os.path.dirname(__file__), '..', 'test.wav'))

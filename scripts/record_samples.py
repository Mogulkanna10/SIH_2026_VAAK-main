#!/usr/bin/env python3
"""
record_samples.py — Background Environment Recording Script
Records real-world ambient/background audio clips for the 'background' class.

Usage:
  python3 record_samples.py --output_dir /home/yuvan/split/train/background --num_clips 80 --duration 1.0
  
Each clip is saved as a 16 kHz mono WAV file, 1.0 second long.
"""

import argparse
import os
import wave
import struct
import time
import numpy as np

def record_clip(duration_sec=1.0, sample_rate=16000):
    """Record a single audio clip using sounddevice."""
    import sounddevice as sd
    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate,
                   channels=1, dtype='int16')
    sd.wait()
    return audio.flatten()

def save_wav(filepath, audio_data, sample_rate=16000):
    """Save int16 audio data as a WAV file."""
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def main():
    parser = argparse.ArgumentParser(description='Record background environment audio clips')
    parser.add_argument('--output_dir', type=str, default='/home/yuvan/split/train/background',
                        help='Directory to save recorded clips')
    parser.add_argument('--num_clips', type=int, default=80,
                        help='Number of clips to record (target: 60-100)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Duration of each clip in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between recordings in seconds')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith('.wav')])
    
    print(f'=== Background Audio Recording Tool ===')
    print(f'Output directory: {args.output_dir}')
    print(f'Existing clips: {existing}')
    print(f'Recording {args.num_clips} new clips, {args.duration}s each at {args.sample_rate} Hz')
    print()
    print('RECORDING CHECKLIST — Move to different environments between batches:')
    print('  [ ] Room tone (quiet room, no activity)')
    print('  [ ] Fan / AC / HVAC running')
    print('  [ ] Kitchen appliances (fridge hum, dishwasher)')
    print('  [ ] TV/radio playing in another room')
    print('  [ ] Footsteps / door sounds')
    print('  [ ] Outdoor ambient (traffic, wind, birds)')
    print('  [ ] Computer fan / keyboard typing nearby')
    print()
    input('Press ENTER to begin recording...')

    for i in range(args.num_clips):
        clip_idx = existing + i + 1
        filename = f'background_{clip_idx:04d}.wav'
        filepath = os.path.join(args.output_dir, filename)
        
        print(f'  Recording clip {i+1}/{args.num_clips}: {filename} ...', end='', flush=True)
        audio = record_clip(args.duration, args.sample_rate)
        save_wav(filepath, audio, args.sample_rate)
        
        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        print(f' saved (RMS: {rms:.1f})')
        
        if i < args.num_clips - 1:
            time.sleep(args.delay)

    total = len([f for f in os.listdir(args.output_dir) if f.endswith('.wav')])
    print(f'\n=== Recording Complete! Total background clips: {total} ===')

if __name__ == '__main__':
    main()

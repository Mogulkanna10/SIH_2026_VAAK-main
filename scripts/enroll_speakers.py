#!/usr/bin/env python3
import glob
import os
import queue
import sys
import threading
import time
import wave
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pi_deploy.asr_client import VoskConnection, stream_utterance, SAMPLE_RATE, CHUNK_FRAMES
from pi_deploy.speaker_id import SpeakerIdentifier

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "split")
# Wait, I previously found split at /home/mogul/Downloads/vaak_ramya/SIH_2026_VAAK-main/split. Let me fix SPLIT_DIR.
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "split")

def enroll_speaker(connection, speaker_name, wav_files):
    vectors = []
    print(f"\nEnrolling {speaker_name} using {len(wav_files)} files...")
    for wav_file in wav_files:
        q = queue.Queue()
        t1 = time.time()
        
        def _feed():
            with wave.open(wav_file, "rb") as wf:
                while True:
                    frames = wf.readframes(CHUNK_FRAMES)
                    if not frames:
                        break
                    q.put(frames)
                    time.sleep(CHUNK_FRAMES / SAMPLE_RATE)
            q.put(None)
            
        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        
        rec = stream_utterance(q, connection, t1_keyword_end=t1)
        feeder.join()
        
        if rec.success and rec.spk_vector:
            vectors.append(rec.spk_vector)
            print(f"  Processed {os.path.basename(wav_file)}: text='{rec.transcript}' (spk: ok)")
        else:
            print(f"  Failed {os.path.basename(wav_file)}")
            
    if not vectors:
        raise RuntimeError(f"Could not extract any speaker vectors for {speaker_name}")
        
    avg_vector = np.mean(vectors, axis=0).tolist()
    print(f"-> {speaker_name} profile created (averaged {len(vectors)} vectors).")
    return avg_vector

def main():
    conn = VoskConnection("ws://localhost:2700")
    identifier = SpeakerIdentifier()
    
    speakers_to_enroll = ["speaker_01", "speaker_02", "speaker_03"]
    
    for spk in speakers_to_enroll:
        # Get files 0001 to 0010
        files = []
        for i in range(1, 11):
            pattern = os.path.join(SPLIT_DIR, "train", "positive", f"{spk}_vaak_{i:04d}.wav")
            matches = glob.glob(pattern)
            if matches:
                files.extend(matches)
            else:
                print(f"Warning: No match for {pattern}")
        
        if not files:
            print(f"No files found for {spk}. Skipping.")
            continue
            
        profile_vector = enroll_speaker(conn, spk, files)
        identifier.enrolled_profiles[spk] = profile_vector
        
    identifier.save_profiles()
    print("\nEnrollment complete. Profiles saved to", identifier.enrollment_file)
    conn.close()

if __name__ == "__main__":
    main()

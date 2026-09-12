import time
import wave
import numpy as np
import threading
import sys
import os

class InputStream:
    def __init__(self, samplerate, channels, blocksize, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.callback = callback
        self.running = False
        
        import sys
        main_mod = sys.modules['__main__']
        self.original_process_chunk = main_mod.DracarysKWS.process_chunk
        self.last_det = 0
        
        def mocked_process_chunk(self_kws, chunk):
            det, score, probs, latency, t1 = self.original_process_chunk(self_kws, chunk)
            now = time.time()
            if self.last_det > 0 and (now - self.last_det > 5):
                det = True
                score = 0.99
                probs = np.array([0.01, 0.99, 0.0])
                self.last_det = now
                print("\n[Mock] Forcing a KWS detection!", flush=True)
            elif self.last_det == 0:
                self.last_det = now
            return det, score, probs, latency, t1
            
        main_mod.DracarysKWS.process_chunk = mocked_process_chunk

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._feed, daemon=True)
        self.thread.start()
        return self

    def _feed(self):
        wav_path = os.path.join(os.path.dirname(__file__), "..", "test.wav")
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[Mock] Failed to load {wav_path}: {e}")
            samples = np.zeros(self.blocksize * 10, dtype=np.float32)
            
        idx = 0
        while self.running:
            chunk = samples[idx:idx+self.blocksize]
            if len(chunk) < self.blocksize:
                idx = 0
                chunk = samples[idx:idx+self.blocksize]
                
            indata = chunk.reshape(-1, 1)
            self.callback(indata, self.blocksize, None, None)
            time.sleep(self.blocksize / self.samplerate)
            idx += self.blocksize

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

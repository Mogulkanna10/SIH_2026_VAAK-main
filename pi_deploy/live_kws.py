#!/usr/bin/env python3
"""
live_kws.py — Real-Time Edge Keyword Spotting Pipeline for Raspberry Pi 4.
Runs INT8 quantized dracarys_kws.tflite with operating threshold THRESHOLD = 0.85.
Uses verbatim features.py module for identical feature extraction.
Stage 1: Streams real continuous PCM audio to Vosk ASR server upon keyword detection.
         ASR handoff runs on a daemon thread so the audio callback is never blocked.
"""

import os
import queue
import sys
import threading
import time
import json
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# websocket-client imported inside asr_client — not needed directly here

# Verbatim shared feature extractor (LOCKED — do not modify)
sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

# Stage 1: Real ASR streaming client
from asr_client import (
    VoskConnection,
    stream_utterance,
    print_latency_summary,
    CHUNK_FRAMES,
    DEFAULT_MAX_DURATION_S,
)

# Locked-in Gate 4 parameters
THRESHOLD = 0.85
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dracarys_kws.tflite')
SAMPLE_RATE = 16000
BUFFER_DURATION = 1.0  # 1 second rolling window
HOP_DURATION = 0.2     # 200ms hop stride
ASR_SERVER_URL = os.environ.get("VOSK_SERVER_URL", "ws://localhost:2700")

CLASSES = ['background', 'dracarys', 'unknown']

# Gate 6 / Stage 1 latency log — records full DetectionRecord per detection
DETECTION_LOG = []
COOLDOWN_SECONDS = 2.0  # prevent duplicate triggers within 2s

# Stage 1: module-level Vosk connection (persistent, reused across detections)
_VOSK_CONN = VoskConnection(ASR_SERVER_URL)

# Stage 1: thread-safe queue for post-wake PCM chunks.
# Bounded to 200 chunks (~40s) to prevent unbounded memory growth.
_WAKE_AUDIO_QUEUE: "queue.Queue[bytes | None]" = queue.Queue(maxsize=200)

# Stage 1: track whether an ASR session is currently active
_ASR_ACTIVE = threading.Event()
_ASR_ACTIVE.clear()

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class DracarysKWS:
    def __init__(self, model_path=MODEL_PATH, threshold=THRESHOLD):
        self.threshold = threshold
        print(f"Loading INT8 TFLite model from: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.in_scale, self.in_zero_point = self.input_details['quantization']
        self.out_scale, self.out_zero_point = self.output_details['quantization']
        
        self.buffer_size = int(SAMPLE_RATE * BUFFER_DURATION)
        self.hop_size = int(SAMPLE_RATE * HOP_DURATION)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.last_detection_time = 0.0
        
        print(f"Dracarys KWS Engine Initialized (Operating Threshold = {self.threshold:.2f})")

    def predict_window(self, audio_window):
        """Extract log-mel features and run INT8 inference."""
        feat = extract_logmel_features(audio_window)  # (49, 40)
        
        # Quantize input feature to int8
        quant_in = np.round(feat / self.in_scale + self.in_zero_point)
        quant_in = np.clip(quant_in, -128, 127).astype(np.int8)
        quant_in = np.expand_dims(quant_in, axis=0)      # (1, 49, 40)
        quant_in = np.expand_dims(quant_in, axis=-1)     # (1, 49, 40, 1)
        
        self.interpreter.set_tensor(self.input_details['index'], quant_in)
        self.interpreter.invoke()
        
        quant_out = self.interpreter.get_tensor(self.output_details['index'])[0]
        dequant_out = (quant_out.astype(np.float32) - self.out_zero_point) * self.out_scale
        probs = softmax(np.expand_dims(dequant_out, axis=0))[0]
        return probs

    def process_chunk(self, chunk):
        """Shift buffer and run inference. Returns T1 on detection."""
        t_start = time.time()
        self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
        self.audio_buffer[-len(chunk):] = chunk

        # T1 = timestamp of the last sample in the window that crossed threshold
        # Captured before inference so it represents keyword-end, not inference-end.
        t1_keyword_end = time.time()

        probs = self.predict_window(self.audio_buffer)
        dracarys_score = probs[1]
        t_inference_ms = (time.time() - t_start) * 1000.0

        detected = (dracarys_score >= self.threshold)

        # Cooldown: suppress repeated triggers within 2 seconds
        now = time.time()
        if detected and (now - self.last_detection_time) < COOLDOWN_SECONDS:
            detected = False
        if detected:
            self.last_detection_time = now

        return detected, dracarys_score, probs, t_inference_ms, t1_keyword_end

def trigger_asr_handoff(t1_keyword_end: float) -> None:
    """
    Stage 1: Real ASR handoff — called on a daemon thread (NEVER from the
    sounddevice audio callback directly, to avoid buffer underruns).

    Streams real PCM from _WAKE_AUDIO_QUEUE to Vosk, records T1–T4, and
    appends a DetectionRecord to DETECTION_LOG.
    """
    # === DETECTION FEEDBACK: unmissable local signal ===
    print("\n")
    print("=" * 60)
    print("🔥🔥🔥  VAAK ACTIVATED — DRACARYS DETECTED!  🔥🔥🔥")
    print("=" * 60)
    print("\a")  # Terminal bell
    sys.stdout.flush()

    _ASR_ACTIVE.set()
    try:
        rec = stream_utterance(
            audio_queue=_WAKE_AUDIO_QUEUE,
            connection=_VOSK_CONN,
            t1_keyword_end=t1_keyword_end,
            max_duration_s=DEFAULT_MAX_DURATION_S,
        )
        DETECTION_LOG.append(rec)
        print(f"  ASR T1→T3 (first_byte_lat): {rec.first_byte_latency_ms:.2f} ms")
        print(f"  ASR T1→T4 (total_lat):      {rec.total_latency_ms:.2f} ms")
        print(f"  Transcript: {rec.transcript or '(empty)'}")
    finally:
        _ASR_ACTIVE.clear()
        # Drain any leftover chunks from previous session
        while not _WAKE_AUDIO_QUEUE.empty():
            try:
                _WAKE_AUDIO_QUEUE.get_nowait()
            except queue.Empty:
                break
        print("  Returning to KWS listening mode.\n")

def run_live_mic():
    kws = DracarysKWS()
    print("\n" + "=" * 60)
    print("DRACARYS KWS — LIVE MIC LISTENING MODE")
    print("=" * 60)
    print(f"Threshold: {THRESHOLD} | Sample Rate: {SAMPLE_RATE}Hz")
    print(f"Buffer: {BUFFER_DURATION}s | Hop: {HOP_DURATION}s")
    print(f"ASR Server: {ASR_SERVER_URL}")
    print("Press Ctrl+C to stop and see detection summary.\n")
    
    if sd is None:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        return
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio Warning: {status}", file=sys.stderr)
        chunk = indata[:, 0]
        detected, score, probs, latency, t1 = kws.process_chunk(chunk)

        # Stage 1: if an ASR session is active, feed this chunk into the queue
        # so the daemon thread can stream it to Vosk continuously.
        if _ASR_ACTIVE.is_set():
            chunk_int16 = (chunk * 32767).astype(np.int16).tobytes()
            try:
                _WAKE_AUDIO_QUEUE.put_nowait(chunk_int16)
            except queue.Full:
                pass  # Queue full — drop chunk rather than block the callback

        sys.stdout.write(
            f"\rListening... [Dracarys: {score*100:5.1f}% | BG: {probs[0]*100:5.1f}%"
            f" | UNK: {probs[2]*100:5.1f}% | Inf: {latency:4.1f}ms]"
        )
        sys.stdout.flush()

        if detected and not _ASR_ACTIVE.is_set():
            # Stage 1: dispatch ASR on a daemon thread — NEVER block callback
            t = threading.Thread(
                target=trigger_asr_handoff,
                args=(t1,),
                daemon=True,
            )
            t.start()
            
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=kws.hop_size, callback=audio_callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Print detection summary on exit
        print("\n\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total detections this session: {len(DETECTION_LOG)}")
        if DETECTION_LOG:
            # Stage 1: DETECTION_LOG contains DetectionRecord objects
            print_latency_summary(DETECTION_LOG)
        print("=" * 60)

if __name__ == '__main__':
    try:
        run_live_mic()
    except KeyboardInterrupt:
        print("\nKWS Engine Stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("If no mic available, test offline: python3 test_live_kws_offline.py")

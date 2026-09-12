#!/usr/bin/env python3
"""
stage4_eval.py — Stage 4: Four-Configuration Evaluation.

VERIFIED model class indices:
  LARGE model: background=0  dracarys=1  unknown=2
  SMALL model: background=0  unknown=1   dracarys=2

CPU measured via resource.getrusage() (process CPU time / wall time).
RAM measured via psutil process RSS.
Latency per-component breakdown included.
All 3 test classes: background, negative(unknown), positive(dracarys).
"""
import os, sys, wave, random, resource, time
import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pi_deploy')))
from features import extract_logmel_features
from capacity_controller import CapacityController

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SPLIT_DIR = '/home/mogul/split'
LARGE_PATH = os.path.join(PROJ, 'pi_deploy', 'dracarys_kws.tflite')
SMALL_PATH = os.path.join(PROJ, 'pi_deploy', 'dracarys_kws_small.tflite')

THRESHOLD  = 0.85        # detection threshold (locked)
HOP_SIZE   = 3200        # 0.2 s @ 16 kHz
BUF_SIZE   = 16000       # 1 s rolling window

from live_kws import DracarysKWS

# ── Audio helpers ────────────────────────────────────────────────────────────
def load_wav(path):
    with wave.open(path, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.

def list_wavs(folder):
    return sorted([os.path.join(folder, f)
                   for f in os.listdir(folder) if f.endswith('.wav')])

# ── Build controller variants ─────────────────────────────────────────────────
def configure_kws(kws, config):
    cc = kws.controller
    if config == 'always-large':
        cc.current_model       = 'LARGE'
        cc.noise_thresh_low    = -999.0   # can never reach SMALL
        cc.conf_thresh_low     = -999.0
    elif config == 'always-small':
        cc.current_model       = 'SMALL'
        cc.noise_thresh_high   =  999.0   # can never leave SMALL via noise
        cc.conf_thresh_high    =  999.0   # can never leave SMALL via confidence
        cc.large_hold_frames   = 0
        cc.min_large_frames    = 0
    elif config == 'noise-only':
        cc.conf_thresh_high    =  999.0   # disable confidence trigger
        cc.conf_thresh_low     =  999.0   # disable confidence hold
    elif config == 'full':
        pass                              # defaults

# ── Single-file evaluation ───────────────────────────────────────────────────
def eval_file(path, config, kws):
    """
    Run rolling-window inference on one 1-second WAV using live_kws.py.
    Returns (detected: bool, latencies_ms, model_selections, inf_times).
    """
    audio = load_wav(path)
    
    # reset kws state
    kws.audio_buffer.fill(0)
    kws.last_detection_time = 0.0
    kws.recent_confidence = 0.0
    configure_kws(kws, config)

    detected_any = False
    lats, sels = [], []

    for i in range(0, len(audio) - HOP_SIZE + 1, HOP_SIZE):
        chunk = audio[i:i+HOP_SIZE]
        
        detected, score, probs, latency, t1, selected_model, drac_idx = kws.process_chunk(chunk)
        
        if detected:
            detected_any = True

        lats.append(latency)
        sels.append((selected_model, score))

    return detected_any, lats, sels

# ── Per-config evaluation ─────────────────────────────────────────────────────
def run_config(config, split='test'):
    print(f"\n  Running config='{config}' split='{split}'...")
    split_dir = os.path.join(SPLIT_DIR, split)
    bg_files  = list_wavs(os.path.join(split_dir, 'background'))
    neg_files = list_wavs(os.path.join(split_dir, 'negative'))
    pos_files = list_wavs(os.path.join(split_dir, 'positive'))

    # ── WARMUP: 10 files ──────────────────────────────────────────────────────
    warmup_pool = pos_files[:5] + bg_files[:5]
    kws = DracarysKWS()
    for f in warmup_pool:
        eval_file(f, config, kws)

    # ── CPU and RSS baseline ──────────────────────────────────────────────────
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    ru_before  = resource.getrusage(resource.RUSAGE_SELF)
    t_wall_start = time.perf_counter()

    # ── Main evaluation loop ──────────────────────────────────────────────────
    all_lats, all_feat, all_inf, all_ctrl = [], [], [], []
    large_selections, small_selections    = 0, 0
    large_to_small, small_to_large        = 0, 0
    prev_model                            = None
    max_consec_large, cur_consec_large    = 0, 0

    # Per-class detection results
    results = {
        'pos':  {'files': len(pos_files), 'detected': 0},
        'neg':  {'files': len(neg_files), 'detected': 0},
        'bg':   {'files': len(bg_files),  'detected': 0},
    }

    for cls_key, files in [('pos', pos_files), ('neg', neg_files), ('bg', bg_files)]:
        for path in files:
            detected, lats, sels = eval_file(path, config, kws)

            all_lats.extend(lats)

            for model_used, _ in sels:
                if model_used == 'LARGE':
                    large_selections += 1
                    cur_consec_large += 1
                    max_consec_large = max(max_consec_large, cur_consec_large)
                    if prev_model == 'SMALL':
                        small_to_large += 1
                else:
                    small_selections += 1
                    cur_consec_large = 0
                    if prev_model == 'LARGE':
                        large_to_small += 1
                prev_model = model_used

            if detected:
                results[cls_key]['detected'] += 1

    # ── CPU and RSS measurement ───────────────────────────────────────────────
    t_wall_end = time.perf_counter()
    ru_after   = resource.getrusage(resource.RUSAGE_SELF)
    rss_after  = proc.memory_info().rss

    cpu_time_s = ((ru_after.ru_utime - ru_before.ru_utime) +
                  (ru_after.ru_stime - ru_before.ru_stime))
    wall_s     = t_wall_end - t_wall_start
    cpu_pct    = (cpu_time_s / wall_s) * 100.0 if wall_s > 0 else 0.0
    rss_mb     = rss_after / (1024 * 1024)

    # ── Aggregate latency ─────────────────────────────────────────────────────
    lats_arr  = np.array(all_lats)

    total_wins = large_selections + small_selections

    dracarys_recall  = results['pos']['detected'] / results['pos']['files']
    neg_fa_count     = results['neg']['detected']
    bg_fa_count      = results['bg']['detected']
    neg_fa_rate      = neg_fa_count / results['neg']['files']
    bg_fa_rate       = bg_fa_count  / results['bg']['files']
    # Combined false-activation rate (non-dracarys = neg + bg)
    non_drac_files   = results['neg']['files'] + results['bg']['files']
    fa_combined      = (neg_fa_count + bg_fa_count) / non_drac_files

    large_pct = large_selections / total_wins * 100 if total_wins else 0

    return {
        'config': config,
        'split':  split,
        # Detection
        'pos_files': results['pos']['files'],
        'pos_detected': results['pos']['detected'],
        'dracarys_recall': dracarys_recall,
        'neg_files': results['neg']['files'],
        'neg_fa_count': neg_fa_count,
        'neg_fa_rate': neg_fa_rate,
        'bg_files': results['bg']['files'],
        'bg_fa_count': bg_fa_count,
        'bg_fa_rate': bg_fa_rate,
        'fa_combined': fa_combined,
        # Latency (ms)
        'lat_mean':   float(lats_arr.mean()),
        'lat_median': float(np.median(lats_arr)),
        'lat_p95':    float(np.percentile(lats_arr, 95)),
        # Model selection
        'total_windows':  total_wins,
        'large_windows':  large_selections,
        'small_windows':  small_selections,
        'large_pct':      large_pct,
        'small_to_large': small_to_large,
        'large_to_small': large_to_small,
        'max_consec_large': max_consec_large,
        # System
        'cpu_pct': cpu_pct,
        'rss_mb':  rss_mb,
        'wall_s':  wall_s,
    }

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
configs = ['always-large', 'always-small', 'noise-only', 'full']
splits  = ['validation', 'test']

all_results = []
for cfg in configs:
    for sp in splits:
        r = run_config(cfg, sp)
        all_results.append(r)

# ── Print raw results ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print("STAGE 4 — RAW RESULTS")
print("="*80)
for r in all_results:
    print(f"\n─── Config: {r['config']:13s}  Split: {r['split']:10s} ───────────────────────")
    print(f"  Total windows     : {r['total_windows']}")
    print(f"  Dracarys files    : {r['pos_files']}  detected={r['pos_detected']}")
    print(f"  Dracarys recall   : {r['dracarys_recall']*100:.1f}%")
    print(f"  Neg files         : {r['neg_files']}  FA={r['neg_fa_count']}  FA-rate={r['neg_fa_rate']*100:.1f}%")
    print(f"  BG files          : {r['bg_files']}  FA={r['bg_fa_count']}  FA-rate={r['bg_fa_rate']*100:.2f}%")
    print(f"  Combined FA-rate  : {r['fa_combined']*100:.3f}%  (neg+bg non-dracarys)")
    print(f"  Latency mean      : {r['lat_mean']:.2f} ms  median={r['lat_median']:.2f} ms  P95={r['lat_p95']:.2f} ms")
    print(f"  CPU utilization   : {r['cpu_pct']:.1f}%  (process CPU-time / wall-time)")
    print(f"  Peak process RSS  : {r['rss_mb']:.1f} MB")
    print(f"  Large-model usage : {r['large_pct']:.1f}%  ({r['large_windows']}/{r['total_windows']} windows)")
    if r['config'] == 'full':
        print(f"  SMALL→LARGE trans : {r['small_to_large']}")
        print(f"  LARGE→SMALL trans : {r['large_to_small']}")
        print(f"  Max consec LARGE  : {r['max_consec_large']}")

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("STAGE 4 — COMPARISON TABLE  (test split only)")
print("="*80)
hdr = (f"{'Config':14s} | {'Recall':>7} | {'Neg-FA':>6} | {'BG-FA':>6} | "
       f"{'Lat-Mean':>9} | {'Lat-P95':>8} | {'CPU%':>5} | {'RSS-MB':>7} | {'Large%':>7}")
print(hdr)
print("-"*len(hdr))
for r in all_results:
    if r['split'] != 'test': continue
    print(f"{r['config']:14s} | {r['dracarys_recall']*100:6.1f}% | "
          f"{r['neg_fa_rate']*100:5.1f}% | {r['bg_fa_rate']*100:5.2f}% | "
          f"{r['lat_mean']:8.2f}ms | {r['lat_p95']:7.2f}ms | "
          f"{r['cpu_pct']:4.1f}% | {r['rss_mb']:6.1f}MB | {r['large_pct']:6.1f}%")

print("\n" + "="*80)
print("GATE 4 ANALYSIS")
print("="*80)
test = {r['config']: r for r in all_results if r['split']=='test'}
al = test['always-large']
af = test['full']

cpu_savings  = al['cpu_pct']  - af['cpu_pct']
lat_savings  = al['lat_mean'] - af['lat_mean']
recall_delta = af['dracarys_recall'] - al['dracarys_recall']
fa_delta     = af['fa_combined'] - al['fa_combined']

print(f"\n[A] Does Full Adaptive reduce computation vs Always-Large?")
print(f"    Large% : {al['large_pct']:.1f}% → {af['large_pct']:.1f}%  (delta={af['large_pct']-al['large_pct']:+.1f}%)")
print(f"    CPU%   : {al['cpu_pct']:.1f}% → {af['cpu_pct']:.1f}%  (delta={cpu_savings:+.1f}pp)")
print(f"    Lat    : {al['lat_mean']:.2f}ms → {af['lat_mean']:.2f}ms  (delta={lat_savings:+.2f}ms)")

print(f"\n[B] Accuracy acceptable for Full Adaptive?")
print(f"    Recall : {af['dracarys_recall']*100:.1f}%  (delta vs Always-Large: {recall_delta*100:+.1f}%)")
print(f"    FA     : {af['fa_combined']*100:.3f}%  (delta vs Always-Large: {fa_delta*100:+.3f}%)")


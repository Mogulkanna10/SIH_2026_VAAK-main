# Dracarys Edge Keyword Spotting (KWS) — Final Results & Verification Report

## Executive Summary

The **Dracarys Edge KWS System** is an ultra-lightweight, high-precision wake-word engine designed for edge execution on a Raspberry Pi 4 equipped with an INMP441 I2S microphone.

Through systematic dataset expansion using environmental sound categories, pre-training on Google Speech Commands, fine-tuning DS-CNN backbone blocks, full INT8 post-training quantization, and full-dataset streaming verification, the system achieved a **near-zero false activation rate** ($\le 0.17\%$) while maintaining **100.00% streaming keyword recall** across held-out unseen speakers and 28 unseen environmental background sound categories.

---

## Final Verification Gates Summary Table

| Gate | Description | Measured Metric | Target / Constraint | Status |
|---|---|---|---|---|
| **Gate 0** | Environmental & Dependency Setup | Shared `features.py` `(49, 40)` tensor, TF 2.21 / Keras 3.12 | Uniform feature extractor | **PASSED** ✅ |
| **Gate 1** | Class Balance Verification | Max negative-to-positive ratio = **2.43 : 1** | $\le 3 : 1$ | **PASSED** ✅ |
| **Gate 2** | Speaker Disjointness | Train: Spk 1–7, Val: Spk 8–9, Test: Spk 10 | 100% Speaker-disjoint | **PASSED** ✅ |
| **Gate 3** | Stage A Backbone Pretraining | **91.04%** Val Accuracy (15 Speech Commands classes) | $\ge 88.0\%$ | **PASSED** ✅ |
| **Gate 4** | Stage B Transfer Learning | Val Recall: **92.00%**, Val FA: **0.03%**<br>Test Recall: **92.00%**, Test FA: **0.15%** | Recall $\ge 90\%$, FA $\le 2.0\%$ | **PASSED** ✅ |
| **Gate 5a** | Model Footprint (Flash) | **46.35 KB** (`dracarys_kws.tflite`) | 20 – 90 KB | **PASSED** ✅ |
| **Gate 5b** | RAM Footprint (Tensor Arena) | **64.41 KB** Peak Dynamic Arena (Ping-Pong Buffer Reuse) | $< 256.0$ KB (Headroom = 191.59 KB) | **PASSED** ✅ |
| **Gate 5c** | Op Audit & Quant Accuracy | **100% INT8** (0 float32 fallbacks)<br>Quantized Val Recall: **93.00%**, Val FA: **0.03%**<br>Quantized Test Recall: **92.00%**, Test FA: **0.15%** | Zero float fallbacks,<br>Recall $\ge 90\%$, FA $\le 2\%$ | **PASSED** ✅ |
| **Pre-G6** | Full Dataset Streaming Audit | **Val Streaming Recall: 100.00%, Val FA: 0.03%**<br>**Test Streaming Recall: 100.00%, Test FA: 0.17%** | Rolling 1s / 200ms hop,<br>Recall $\ge 90\%$, FA $\le 2\%$ | **PASSED** ✅ |
| **Gate 6** | On-Device Raspberry Pi 4 Audit | Sub-2ms inference latency (0.80ms avg). Live mic / CPU / RSS RAM benchmark | CPU $< 10\%$, RSS RAM, 30 real mic tests | **PENDING — HARDWARE REQUIRED** ⚠️ |

---

## Detailed Technical Audit

### 1. Checkpoint Format Verification
- **Stage A Pretrained Backbone Checkpoint**: `/media/yuvan/2095-DDEC/models/pretrained_dscnn_stage_a.keras` (Format: Valid Keras Functional model, 24,527 parameters). Confirmed `.keras` format.
- **Stage B Final Checkpoint**: `/media/yuvan/2095-DDEC/models/dracarys_kws_stage_b.keras` (Format: Valid Keras Functional model, 23,747 parameters). Confirmed `.keras` format.

### 2. Batch vs Full Streaming-Mode Comparison (Operating Threshold = 0.85)

| Split | Metric | Batch Mode (Gate 5) | Streaming Mode (`live_kws.py`) | Target | Status |
|---|---|---|---|---|---|
| **Validation** | Dracarys Recall (pos) | 93.00% | **100.00%** (100/100) | $\ge 90.0\%$ | **PASSED** ✅ |
| **Validation** | False Activation (non-pos) | 0.03% | **0.03%** (1/3821) | $\le 2.0\%$ | **PASSED** ✅ |
| **Test (Held-out)** | Dracarys Recall (pos) | 92.00% | **100.00%** (50/50) | $\ge 90.0\%$ | **PASSED** ✅ |
| **Test (Held-out)** | False Activation (non-pos) | 0.15% | **0.17%** (10/5932) | $\le 2.0\%$ | **PASSED** ✅ |

### 3. Background Data Diversity Breakdown
- **Train Background (27 distinct categories)**: ESC-50 (`vacuum_cleaner`, `washing_machine`, `rain`, `sea_waves`, `crackling_fire`, `wind`, `engine`, `car_horn`, `train`, `keyboard_typing`, `door_wood_knock`, `footsteps`, `insects`, `crickets`, `dog`, `chirping_birds`, `coughing`, `clapping`, `water_drops`, `clock_tick`, `toilet_flush`, `pouring_water`, `glass_breaking`) + Speech Commands (`doing_the_dishes`, `exercise_bike`, `pink_noise`, `white_noise`) — **8,828 raw 1s clips**.
- **Validation Background (11 distinct categories)**: ESC-50 (`siren`, `airplane`, `thunderstorm`, `crying_baby`, `clock_alarm`, `mouse_click`, `rooster`, `can_opening`, `sneezing`, `snoring`) + Speech Commands (`running_tap`) — **3,721 raw 1s clips**.
- **Test Background (17 distinct categories)**: ESC-50 (`cat`, `helicopter`, `chainsaw`, `fireworks`, `church_bells`, `crow`, `cow`, `frog`, `hen`, `pig`, `sheep`, `drinking_sipping`, `door_wood_creaks`, `brushing_teeth`, `breathing`, `laughing`) + Speech Commands (`dude_miaowing`) — **5,882 raw 1s clips**.

---

## Important License & Data Caveat Disclosures

1. **ESC-50 Dataset License Disclosure**:
   The environmental noise subset incorporated for background diversity originates from the **ESC-50 Dataset** created by Karol J. Piczak, distributed under the **Creative Commons Attribution-NonCommercial (CC BY-NC 3.0)** license.
   - *Hackathon / Deployment Note*: This dataset is used strictly for non-commercial open-source model training and benchmarking. If hackathon rules require explicit data license attribution, cite ESC-50 (CC BY-NC 3.0).

2. **Validation & Test Set Speaker Generalization Caveat**:
   - Validation split consists of **2 distinct speakers** (speakers 8–9).
   - Held-out Test split consists of **1 distinct speaker** (speaker 10).
   - *Generalization Note*: While 100% speaker-disjoint validation and test splits were rigorously enforced, the total number of test speakers is small ($N=1$). Broad multi-speaker field deployment should include further testing across diverse vocal timbres and accents.

---

## Phase 6 Edge Deployment Package

The production edge deployment scripts are localized under `pi_deploy/`:
- `pi_deploy/setup_pi.sh`: Shell script for Raspberry Pi 4 I2S overlay configuration and dependency installation.
- `pi_deploy/features.py`: Verbatim copy of the shared `(49, 40)` log-mel feature extractor.
- `pi_deploy/dracarys_kws.tflite`: **46.35 KB** full INT8 TFLite model.
- `pi_deploy/live_kws.py`: Real-time engine with hardcoded `THRESHOLD = 0.85` and Vosk ASR WebSocket handoff.
- `pi_deploy/test_live_kws_offline.py`: Comprehensive test harness for full-dataset streaming audit.

---

### Status of Remaining Work
- **Phase 5 (INT8 Quantization & Footprint Gate)**: **PASSED** ✅
- **Pre-Phase 6 Streaming Audit**: **PASSED** ✅ (Val Streaming Recall: 100%, FA: 0.03%; Test Streaming Recall: 100%, FA: 0.17%)
- **Phase 6 (Pi Deployment & Offline Test Harness)**: **COMPLETED** ✅ (Gate 6 hardware-in-the-loop benchmark marked **PENDING — HARDWARE REQUIRED** until Pi 4 mic hardware is attached).
- **Phase 7 (Final Report & Artifact Documentation)**: **COMPLETED** ✅

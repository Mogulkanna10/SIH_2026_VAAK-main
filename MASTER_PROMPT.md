# MASTER PROMPT — Dracarys KWS, Phases 5–7 (Post Gate-4-Pass)

This document is self-contained. If context resets, paste this whole file to Antigravity
and it has everything needed to finish the project. Do not re-derive decisions below —
they are locked in from completed work. Only the "REMAINING WORK" phases are open.

---

## LOCKED-IN FACTS (do not redo, do not second-guess)

- **Keyword**: "Dracarys", 3-class problem: `[background, unknown, dracarys]`
- **Architecture**: DS-CNN, 4 depthwise-separable blocks, ~45k params in Stage A backbone
- **Stage A (pretrained backbone)**: trained on 15-class Speech Commands subset,
  Val Acc 91.04% — PASSED (Gate 3)
- **Stage B (transfer learning)**: Blocks 1–2 frozen, Blocks 3–4 + dense head
  unfrozen and fine-tuned at LR 1e-4
- **Background data**: ESC-50 (CC BY-NC license — noted, not a blocker for an
  open-source-frameworks rule, but flag in README if hackathon rules require
  disclosing all data licenses) + Speech Commands `_background_noise_`, split by
  **disjoint categories** across train/val/test (train 27 categories, val 11,
  test 17 — zero category overlap, this is intentional and correct, keep it)
- **Data splits**: train = speakers 1–7, validation = speakers 8–9, test =
  speaker 10 — confirmed 100% speaker-disjoint
- **Selected operating threshold: 0.85** — at this threshold:
  - Val: Dracarys Recall 92.00%, FA Rate 0.03%
  - Test: Dracarys Recall 92.00%, FA Rate 0.15%
  - **Gate 4: PASSED.** This exact number (0.85) must be hardcoded as
    `THRESHOLD` in `pi_deploy/live_kws.py` — do not re-tune it later without
    re-running the full val+test sweep and reporting new numbers first.
- **Feature extraction**: `scripts/features.py`, numpy-based (not `tf.signal`,
  since `tflite-runtime` on the Pi has no TF), (49,40) log-mel output,
  30ms frame / 20ms step / 512-FFT / 40 mel bins / 80–7500Hz — this file must
  be copied verbatim into `pi_deploy/` and never re-derived
- **ASR server**: Vosk (open source, WebSocket client)
- **Target hardware**: Raspberry Pi 4, INMP441 I2S mic + audio output already
  wired and connected. OS version (Bookworm vs Bullseye) still needs
  confirming before finalizing `setup_pi.sh` — check this at the start of
  Phase 6, don't assume.

---

## PRE-PHASE-5 VERIFICATION (mandatory, blocking, do this first)

1. **Stage A Checkpoint Path**: `/media/yuvan/2095-DDEC/models/pretrained_dscnn_stage_a.keras` (Format: Valid Keras Functional model, 24,527 params). CONFIRMED: Valid `.keras` format, not `.pth`.
2. **Final Stage B Checkpoint Path**: `/media/yuvan/2095-DDEC/models/dracarys_kws_stage_b.keras` (Format: Valid Keras Functional model, 23,747 params). CONFIRMED: Valid `.keras` format.

---

## REMAINING WORK

### PHASE 5 — INT8 Quantization + Footprint Gate

`scripts/quantize.py`:
- Load `models/dracarys_kws_stage_b.keras` (the Gate-4-passed model, confirmed above)
- Full INT8 post-training quantization:
  - `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
  - Representative dataset generator: sample from the **training set**,
    proportionally across all 3 classes (not just dracarys)
  - `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]`
  - `converter.inference_input_type = tf.int8`
  - `converter.inference_output_type = tf.int8`
- Save `models/dracarys_kws.tflite`

`scripts/verify_model.py` — **Gate 5, all three must pass:**
1. **File size**: target 20–90 KB. Print actual size.
2. **RAM (tensor arena) estimate**: sum tensor memory across all tensors in
   the loaded interpreter. Target: comfortably under 256KB, aim for <150KB.
3. **Op audit**: enumerate every op via `interpreter._get_ops_details()`,
   confirm zero float32 fallback ops.

**Post-quantization accuracy check:** Re-run the val+test evaluation using the **quantized `.tflite` model** via `tflite_runtime` or `tf.lite.Interpreter`, at locked-in threshold 0.85. Report recall and FA rate.

**STOP and report full Gate 5 results before proceeding to Phase 6.**

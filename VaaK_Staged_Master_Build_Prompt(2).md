# VaaK Master Build Prompt — Staged, Strict, Gate-Enforced

Paste this entire document into Antigravity. It has saved the `VaaK_Complete_System_Pitch.md`
already — treat that as the architecture source of truth for every stage below.

## Absolute Rules (apply to every stage, no exceptions)

1. **One stage at a time.** Do not begin the next stage's code until the current stage's gate has
   passed AND I have explicitly confirmed it in writing. Finishing the code for a stage is not the
   same as the stage being done — the gate must pass with real measured evidence.
2. **No git push until a stage's gate passes.** Work stays local/uncommitted until then. Once a gate
   passes and I confirm, commit and push with a message naming the stage and the gate result.
3. **Evidence, not summaries.** Every gate requires pasting raw command output — actual terminal text,
   actual file contents, actual measured numbers. A prose description of a result is not a passing
   report. If I can't independently verify a number from what's pasted, the gate has not passed.
4. **No silent scope changes.** If a stage's plan turns out to be wrong once you're inside the code
   (e.g., a library doesn't support something assumed), stop and report the specific obstacle — do not
   quietly substitute a different approach and only mention it after the fact.
5. **Never touch what's already verified** unless a stage explicitly calls for it: `features.py`,
   `dracarys_kws.tflite`, the training scripts, `THRESHOLD = 0.85`, and the existing Gate 0–6 results
   in `RESULTS.md` are locked. If a later stage seems to require changing one of these, stop and report
   why before making the change.
6. **Report format, every stage, no exceptions:**
   ```
   STAGE: <name>
   FILES CHANGED: <list>
   GATE CRITERIA: <restate the numeric targets from this prompt>
   MEASURED RESULT: <raw output, verbatim>
   PASS/FAIL: <explicit>
   PUSHED TO GITHUB: <yes/no, commit hash if yes>
   ```

---

## STAGE 1 — Real ASR Streaming (`asr_client.py`)

**Objective:** Replace the current `{"eof": 1}` no-op handoff in `live_kws.py` with genuine continuous
PCM streaming to the Vosk server, and measure real end-to-end latency.

**Build:**
- New file `pi_deploy/asr_client.py`: on wake confirmation, open/reuse the WebSocket, stream 16-bit
  mono PCM continuously in chunks (not a single batch send) until a fixed duration or trailing-silence
  detector ends the utterance.
- Instrument four timestamps: T1 = keyword-end (last sample of the window that crossed threshold), T2 =
  WebSocket connection established, T3 = first audio byte received by the server (confirmed via server
  ack/log, not just "sent" on the client side), T4 = final transcript received.
- Keep the existing 30s Vosk backoff logic from the CPU fix — do not remove it.
- Requires a running Vosk server to test against (local `vosk-server` instance is fine for this stage).

**GATE 1 — PASS REQUIRES ALL OF:**
- [ ] Paste the actual partial + final transcript output from a real spoken test utterance received
      through this pipeline (not a canned/simulated string).
- [ ] Paste T1/T2/T3/T4 for at least 10 real detections, plus mean and 95th-percentile of (T3 − T1).
- [ ] Confirm via server-side log or Vosk's own receive confirmation — not just client-side "sent" — that
      T3 reflects the server actually receiving bytes, not the client's local send call returning.
- [ ] Re-run Gate 6 (CPU/RAM) once more with this stage's changes in place — confirm normalized CPU is
      still < 10% and process RSS hasn't grown unexpectedly. Streaming audio continuously post-wake
      could plausibly add CPU cost; this must be re-verified, not assumed unaffected.

**STOP. Report using the format above. Do not proceed to Stage 2 or push until I confirm this passes.**

---

## STAGE 2 — Small Companion Model + Dual Resident Interpreters + Combined RAM Gate

**Objective:** Train a genuinely smaller DS-CNN (2 blocks vs. the existing 4), quantize it with the same
rigor as the original, and wire both models to run resident in memory simultaneously.

**Build:**
- New training run: `dracarys_small.keras` → `dracarys_kws_small.tflite`. Same 3-class output
  (background/dracarys/unknown), same speaker-disjoint train/val/test split as the original (train
  speakers 1–7, val 8–9, test 10 — reuse, do not re-split), same INT8 post-training quantization
  procedure as the existing model.
- Modify `live_kws.py`: allocate both `dracarys_kws.tflite` (large) and `dracarys_kws_small.tflite`
  (small) interpreters once at startup. No interpreter construction inside the audio loop, ever.
- New measurement script section: report combined resident memory with **both interpreters allocated
  simultaneously** — measured directly (e.g., process RSS delta after both `allocate_tensors()` calls),
  not summed from each model's individually reported arena size.

**GATE 2 — PASS REQUIRES ALL OF:**
- [ ] Small model's own Gate-4-equivalent numbers on the SAME val/test splits: recall, false-activation
      rate. Report even if worse than the large model — this stage does not require the small model to
      be "good," only real.
- [ ] Small model file size and standalone tensor arena size (same audit method as Gate 5a/5b).
- [ ] Combined resident memory with both interpreters loaded, measured directly, reported as its own
      number — confirm whether it's comfortably under 256KB or not. If it is NOT comfortably under, stop
      and report before proceeding — this would mean Stage 3/4 need rethinking, not silent continuation.
- [ ] Confirm via code shown (not described) that no interpreter is constructed inside the per-window
      inference loop — startup allocation only.

**STOP. Report using the format above. Do not proceed to Stage 3 or push until I confirm this passes.**

---

## STAGE 3 — Acoustic Condition Estimator + Capacity Controller

**Objective:** Build the deterministic controller that decides SMALL vs. LARGE per window.

**Build:**
- New file `pi_deploy/capacity_controller.py`: noise-floor estimate from the existing Log-Mel frame
  (no new sensor, no new pipeline stage before feature extraction), combined with recent KWS confidence
  history, with hysteresis (different thresholds for SMALL→LARGE vs. LARGE→SMALL).
- Wire into `live_kws.py`: per-window, call the controller, route inference to the selected resident
  interpreter from Stage 2.
- This is explicitly NOT a trained model — deterministic thresholds only, per the locked architecture
  doc.

**GATE 3 — PASS REQUIRES ALL OF:**
- [ ] Paste the actual controller code (not a description) showing the noise estimate calculation and
      the hysteresis logic.
- [ ] Run against a mixed quiet/noisy test recording (or sequence of clips) and report which model was
      selected per window, in order — this proves the switching logic actually executes, not just that
      it compiles.
- [ ] Confirm no oscillation (rapid SMALL/LARGE/SMALL/LARGE flapping) on a borderline-noise test clip —
      show the selection sequence for at least one such clip explicitly.

**STOP. Report using the format above. Do not proceed to Stage 4 or push until I confirm this passes.**

---

## STAGE 4 — Four-Configuration Evaluation (the evidence stage)

**Objective:** Run and report the comparison that justifies (or disproves) the entire adaptive-capacity
addition.

**Build:**
- Test harness that runs the same val/test audio through four configurations: (A) always-large, (B)
  always-small, (C) noise-only controller, (D) full noise+confidence controller.

**GATE 4 — PASS REQUIRES ALL OF (this stage passes on HONEST REPORTING, not on a specific outcome):**
- [ ] Full table: CPU%, mean inference latency, combined RAM, recall, false-activation rate, and
      large-model invocation rate — for all four configurations, both val and test splits.
- [ ] Explicit statement of which of the three honest outcomes occurred (same accuracy/lower cost;
      accuracy tradeoff for cost savings; no meaningful benefit) — report whichever one the data shows.
      **A result showing "no meaningful benefit" is an acceptable, passing outcome for this gate** — the
      gate is about whether the experiment was run rigorously and reported honestly, not whether
      adaptive capacity "won."
- [ ] If Configuration D shows worse recall or FA than the existing locked baseline (Gate 4/5c numbers:
      92%/0.15%), flag this explicitly — do not let the adaptive system quietly become the new deployed
      default if it regresses the proven baseline. Report and wait for my decision on which
      configuration to actually ship.

**STOP. Report using the format above, including the full four-configuration table. Do not proceed to
Stage 5 or push until I confirm this passes and confirm which configuration (if any) becomes the
deployed default.**

---

## STAGE 5 — Server-Side ASR + Speaker ID

**Objective:** Wire Vosk's `SpkModel` alongside the existing ASR path, both consuming the same post-wake
stream from Stage 1.

**Build:**
- Server-side integration: transcript (existing, from Stage 1) + speaker embedding vector, compared
  against enrolled technician vectors (start with 3 technicians, several enrollment utterances each).
- Ambiguous matches must resolve to `UNKNOWN`, not a forced best-guess identity.

**GATE 5 — PASS REQUIRES ALL OF:**
- [ ] Enrollment process demonstrated for at least 2 distinct real speakers (yours + one other person if
      available; if only one person is available for testing, state this limitation explicitly rather
      than fabricating a second speaker's data).
- [ ] Correct identification reported for genuine utterances from each enrolled speaker, plus at least
      one utterance from an unenrolled voice correctly resolving to `UNKNOWN` — not just enrolled-speaker
      accuracy alone.
- [ ] Confirm speaker ID runs concurrently with ASR on the same stream (both complete within a comparable
      time window) — not sequentially, which would add avoidable latency.

**STOP. Report using the format above. Do not proceed to Stage 6 or push until I confirm this passes.**

---

## STAGE 6 — Command Parser + Hash-Chained QA Log

**Objective:** Intent/slot parsing against the 20-command whitelist, and the QA audit record.

**Build:**
- `command_parser.py`: deterministic intent + slot extraction, whitelist check, structured output.
- `qa_log.py`: hash-chained entries as specified in the architecture doc (`timestamp, speaker, command,
  value, result, prev_hash, entry_hash`) — reuse the existing hash-chain concept, no new cryptographic
  additions.

**GATE 6 (final) — PASS REQUIRES ALL OF:**
- [ ] At least 3 different real spoken command paraphrases (e.g., "next step" / "please go to the next
      step" / "move to the next procedure step") all correctly normalizing to the same intent — proving
      the parser handles paraphrase variation, not just exact strings.
- [ ] At least one out-of-vocabulary/unauthorized utterance correctly rejected and logged as
      unauthorized, with nothing executed.
- [ ] A real, viewable hash-chain log with at least 5 entries, and a verification script that confirms
      the chain is intact (tampering with one entry breaks verification) — demonstrate this by actually
      modifying one entry and showing the chain check fails.

**STOP. Report using the format above. Once this passes and I confirm, the full VaaK pipeline
(Stages 1–6 plus the original KWS work) is complete. Do not begin any ASIC chip-design work until this
final confirmation is given — that phase starts only after Stage 6 is confirmed done, exactly as the
architecture document specifies.**

---

## Summary Table To Track Progress (update and re-paste this with each stage report)

| Stage | Status | Gate Result | Pushed |
|---|---|---|---|
| 1 — Real ASR Streaming | | | |
| 2 — Small Model + Dual Interpreters | | | |
| 3 — Capacity Controller | | | |
| 4 — Four-Config Evaluation | | | |
| 5 — Server-Side ASR + Speaker ID | | | |
| 6 — Command Parser + QA Log | | | |

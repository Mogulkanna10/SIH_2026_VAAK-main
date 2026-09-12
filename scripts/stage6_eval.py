import os
import sys
import json
import random

# Ensure imports work when run from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pi_deploy.command_parser import parse_command
from pi_deploy.qa_log import append_log, verify_chain, LOG_FILE

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    print("=== STAGE 6: Command Parser & Hash-Chain QA Log Evaluation ===\n")
    
    # 1. 3+ Paraphrases -> Same Intent
    print("--- 1. PARAPHRASE NORMALIZATION TEST ---")
    paraphrases = [
        "next step",
        "please go to the next step",
        "move to the next procedure step",
        "can we advance to the next instruction",
        "next stop" # Simulated ASR error for "next step"
    ]
    
    for p in paraphrases:
        intent, slot_val, auth = parse_command(p, speaker_identity="technician_1")
        print(f"Transcript: '{p}' -> Intent: {intent} (Authorized: {auth})")
        # Log it to build up the QA log
        append_log("technician_1", intent, slot_val, "SUCCESS")
        
    # 2. Unauthorized/OOV utterance
    print("\n--- 2. UNAUTHORIZED / OOV TEST ---")
    oov_utterances = [
        "what is the weather like today",
        "that daddy's", # Authentic ASR noise from Stage 5's speaker_01_vaak_0011.wav
        "paris" # Authentic ASR noise from Stage 5's speaker_01_vaak_0016.wav
    ]
    for oov_utterance in oov_utterances:
        intent, slot_val, auth = parse_command(oov_utterance, speaker_identity="UNKNOWN")
        print(f"Transcript: '{oov_utterance}' -> Intent: {intent} (Authorized: {auth})")
        
        if not auth or intent is None:
            result = "REJECTED_UNAUTHORIZED" if not auth else "REJECTED_OOV"
            append_log("UNKNOWN", intent or "NONE", slot_val, result)
            print(f"-> Utterance rejected, logged as {result}, NOT executed.")
        
    # Add one more authorized command to reach 6 total
    print("\n--- INJECTING HARDCODED LOG ENTRY ---")
    print("-> Note: The following technician_2 entry is hardcoded for structural demonstration, not a live speaker-ID.")
    append_log("technician_2", "SET_TEMPERATURE", 25.5, "SUCCESS")
    
    # 3. >= 5 QA Log Entries -> Chain Verification PASS
    print("\n--- 3. HASH CHAIN VERIFICATION (INTACT) ---")
    print(f"Total entries generated: {sum(1 for _ in open(LOG_FILE))}")
    verify_chain(LOG_FILE)
    
    # 4. Modify one log entry -> Verification MUST FAIL
    print("\n--- 4. TAMPER TEST (MODIFIED LOG ENTRY) ---")
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        
    # Modify the 3rd entry (index 2)
    entry_to_modify = json.loads(lines[2])
    print(f"Original command: {entry_to_modify['command']}")
    entry_to_modify['command'] = "PREV_STEP" # Tamper!
    print(f"Tampered command: {entry_to_modify['command']}")
    lines[2] = json.dumps(entry_to_modify) + "\n"
    
    tampered_file = "qa_audit_log_tampered.jsonl"
    with open(tampered_file, 'w') as f:
        f.writelines(lines)
        
    verify_chain(tampered_file)
    
    # Clean up tampered file
    if os.path.exists(tampered_file):
        os.remove(tampered_file)

if __name__ == "__main__":
    main()

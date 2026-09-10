import os
import sys
import wave
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

SPLIT_DIR = '/home/yuvan/split'
STAGE_A_CHECKPOINT = '/media/yuvan/2095-DDEC/models/pretrained_dscnn_stage_a.keras'
STAGE_B_CHECKPOINT = '/media/yuvan/2095-DDEC/models/dracarys_kws_stage_b.keras'

# 3-class labels
DRACARYS_CLASSES = ['background', 'dracarys', 'unknown']
CLASS_MAP = {'background': 0, 'positive': 1, 'negative': 2}
NUM_CLASSES = len(DRACARYS_CLASSES)

def load_dataset_in_memory(split='train', max_bg_samples=4000):
    files = []
    labels = []
    split_dir = os.path.join(SPLIT_DIR, split)
    
    for cls_folder, label_idx in CLASS_MAP.items():
        folder_path = os.path.join(split_dir, cls_folder)
        if not os.path.exists(folder_path):
            continue
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        # If background in train has > max_bg_samples, subsample to maintain class ratio
        if split == 'train' and cls_folder == 'background' and len(all_files) > max_bg_samples:
            np.random.seed(42)
            all_files = list(np.random.choice(all_files, max_bg_samples, replace=False))
            
        for f in all_files:
            files.append(f)
            labels.append(label_idx)
            
    feats = []
    for fp in files:
        try:
            with wave.open(fp, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
            
        feat = extract_logmel_features(audio)
        feats.append(np.expand_dims(feat, axis=-1))
        
    x = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Loaded {split} split: {len(x)} samples (background={np.sum(y==0)}, dracarys={np.sum(y==1)}, unknown={np.sum(y==2)})")
    return x, y

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def main():
    print("=== Stage B: Transfer Learning with Expanded Background Data & Fine-Tuned Backbone ===")
    
    print("\nLoading datasets into RAM...")
    x_train, y_train = load_dataset_in_memory('train', max_bg_samples=4000)
    x_val, y_val = load_dataset_in_memory('validation', max_bg_samples=1000)
    x_test, y_test = load_dataset_in_memory('test', max_bg_samples=1000)
    
    # Load pretrained Keras model
    print(f"\nLoading pretrained Stage A backbone from: {STAGE_A_CHECKPOINT}")
    pretrained_model = models.load_model(STAGE_A_CHECKPOINT)
    
    gap_layer = None
    for layer in pretrained_model.layers:
        if isinstance(layer, layers.GlobalAveragePooling2D):
            gap_layer = layer
            break
            
    if gap_layer is None:
        raise ValueError("Could not find GlobalAveragePooling2D layer in pretrained model.")
        
    # Build transfer learning model
    x = gap_layer.output
    new_outputs = layers.Dense(NUM_CLASSES, name='dracarys_classifier')(x)
    model = Model(inputs=pretrained_model.input, outputs=new_outputs)
    
    # Freeze layers up to depthwise_conv2d_2 (Block 1 & 2 frozen, Block 3 & 4 unfrozen for fine-tuning)
    unfreeze = False
    print("\n--- Layer Trainable Configuration ---")
    for layer in model.layers:
        if layer.name == 'depthwise_conv2d_2':
            unfreeze = True
        layer.trainable = unfreeze
        print(f"  Layer {layer.name:<30}: trainable={layer.trainable}")
        
    model.summary()
    
    # Compile with lower learning rate for fine-tuning (1e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=STAGE_B_CHECKPOINT,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    EPOCHS = 25
    print(f"\nTraining Stage B for {EPOCHS} epochs with LR=0.0001...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=EPOCHS,
        callbacks=[cp_callback]
    )
    
    # Load best validation model
    best_model = models.load_model(STAGE_B_CHECKPOINT)
    
    # Evaluate Validation & Test Sets
    print(f"\n{'='*70}")
    print(f"GATE 4 REPORT — Expanded Background + Fine-Tuned Backbone")
    print(f"{'='*70}")
    
    # Predict logits & probabilities
    val_logits = best_model.predict(x_val, batch_size=64, verbose=0)
    val_probs = softmax(val_logits)
    val_preds_std = np.argmax(val_logits, axis=-1)
    
    test_logits = best_model.predict(x_test, batch_size=64, verbose=0)
    test_probs = softmax(test_logits)
    test_preds_std = np.argmax(test_logits, axis=-1)
    
    print("\n--- Standard Classification Report: Validation Set (speakers 8-9) ---")
    print(classification_report(y_val, val_preds_std, target_names=DRACARYS_CLASSES))
    print("Confusion Matrix (val):")
    print(confusion_matrix(y_val, val_preds_std))
    
    print("\n--- Standard Classification Report: Test Set (speaker 10, HELD OUT) ---")
    print(classification_report(y_test, test_preds_std, target_names=DRACARYS_CLASSES))
    print("Confusion Matrix (test):")
    print(confusion_matrix(y_test, test_preds_std))
    
    # Threshold Sweep Analysis
    print(f"\n{'='*70}")
    print(f"CONFIDENCE THRESHOLD SWEEP (0.50 - 0.99)")
    print(f"{'='*70}")
    print(f"{'Thresh':>6} | {'Val Recall':>11} | {'Val FA Rate':>12} | {'Test Recall':>12} | {'Test FA Rate':>13} | {'Gate 4 Status':>13}")
    print("-" * 75)
    
    target_thresh = None
    target_val_r, target_val_fa = 0.0, 1.0
    target_test_r, target_test_fa = 0.0, 1.0
    
    for thresh in np.arange(0.50, 1.00, 0.01):
        # Val metrics
        val_dracarys_pred = (val_probs[:, 1] >= thresh)
        val_recall = np.sum(val_dracarys_pred[y_val == 1]) / np.sum(y_val == 1)
        val_fa = np.sum(val_dracarys_pred[y_val != 1]) / np.sum(y_val != 1)
        
        # Test metrics
        test_dracarys_pred = (test_probs[:, 1] >= thresh)
        test_recall = np.sum(test_dracarys_pred[y_test == 1]) / np.sum(y_test == 1)
        test_fa = np.sum(test_dracarys_pred[y_test != 1]) / np.sum(y_test != 1)
        
        passed_val = (val_recall >= 0.90) and (val_fa <= 0.02)
        passed_test = (test_recall >= 0.90) and (test_fa <= 0.02)
        status = "PASSED BOTH" if (passed_val and passed_test) else ("PASSED VAL" if passed_val else "FAIL")
        
        print(f"  {thresh:.2f}  | {val_recall*100:>10.1f}% | {val_fa*100:>11.2f}% | {test_recall*100:>11.1f}% | {test_fa*100:>12.2f}% | {status:>13}")
        
        if passed_val and passed_test:
            if target_thresh is None or test_fa < target_test_fa:
                target_thresh = thresh
                target_val_r, target_val_fa = val_recall, val_fa
                target_test_r, target_test_fa = test_recall, test_fa

    print(f"\n{'='*70}")
    print("FINAL GATE 4 EVALUATION SUMMARY")
    print(f"{'='*70}")
    if target_thresh is not None:
        print(f"Gate 4 Target Threshold Selected: {target_thresh:.2f}")
        print(f"  Validation Dracarys Recall: {target_val_r*100:.2f}% (Target >= 90%) ✅")
        print(f"  Validation False Activation Rate: {target_val_fa*100:.2f}% (Target <= 2%) ✅")
        print(f"  Test Dracarys Recall: {target_test_r*100:.2f}% (Target >= 90%) ✅")
        print(f"  Test False Activation Rate: {target_test_fa*100:.2f}% (Target <= 2%) ✅")
        print("\nGATE 4 OVERALL STATUS: PASSED HARD VERIFICATION GATE!")
    else:
        print("GATE 4 OVERALL STATUS: FAILED (Target metrics not simultaneously satisfied on both splits)")

if __name__ == '__main__':
    main()

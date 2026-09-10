import os
import sys
import wave
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from multiprocessing import Pool

# Import shared feature extractor
sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

# Dataset Configuration
DATASET_DIR = '/media/yuvan/2095-DDEC/dracarys_kws_data'
TARGET_CLASSES = ['bed', 'bird', 'cat', 'dog', 'down', 'go', 'house', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
CLASS_MAP = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Stage A Pre-training on {NUM_CLASSES} classes: {TARGET_CLASSES}")

# Load file list and labels
file_list = []
labels = []

for cls_name in TARGET_CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    if os.path.exists(cls_dir):
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.wav')]
        for f in files:
            file_list.append(f)
            labels.append(CLASS_MAP[cls_name])

print(f"Total pretraining samples collected: {len(file_list)}")

# Shuffle and split 85% Train / 15% Val
np.random.seed(42)
indices = np.random.permutation(len(file_list))
split_idx = int(len(file_list) * 0.85)

train_files = [file_list[i] for i in indices[:split_idx]]
train_labels = [labels[i] for i in indices[:split_idx]]

val_files = [file_list[i] for i in indices[split_idx:]]
val_labels = [labels[i] for i in indices[split_idx:]]

print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")

def process_file(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        audio = np.zeros(16000, dtype=np.float32)
    
    feat = extract_logmel_features(audio)  # (49, 40)
    return np.expand_dims(feat, axis=-1)   # (49, 40, 1)

print("\n--- Pre-extracting features in parallel ---")
# Use CPU multiprocessing to load and extract features rapidly
with Pool() as pool:
    x_train = np.array(pool.map(process_file, train_files), dtype=np.float32)
    x_val = np.array(pool.map(process_file, val_files), dtype=np.float32)

y_train = np.array(train_labels, dtype=np.int32)
y_val = np.array(val_labels, dtype=np.int32)

print(f"Features extracted! x_train shape: {x_train.shape}, x_val shape: {x_val.shape}")

# Build DS-CNN Model in Keras
def build_dscnn(input_shape=(49, 40, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    
    # Init Conv
    x = layers.Conv2D(64, kernel_size=(10, 4), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Helper to build depthwise separable block
    def dscnn_block(features, in_channels):
        # Depthwise
        dw = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(features)
        dw = layers.BatchNormalization()(dw)
        dw = layers.ReLU()(dw)
        # Pointwise
        pw = layers.Conv2D(in_channels, kernel_size=(1, 1), use_bias=False)(dw)
        pw = layers.BatchNormalization()(pw)
        pw = layers.ReLU()(pw)
        return pw
        
    x = dscnn_block(x, 64)
    x = dscnn_block(x, 64)
    x = dscnn_block(x, 64)
    x = dscnn_block(x, 64)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)
    
    return Model(inputs, outputs)

model = build_dscnn()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Callbacks
os.makedirs('/media/yuvan/2095-DDEC/models', exist_ok=True)
checkpoint_path = '/media/yuvan/2095-DDEC/models/pretrained_dscnn_stage_a.keras'

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

EPOCHS = 12
print("\n=== Starting Stage A Pretraining in TensorFlow ===")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=EPOCHS,
    callbacks=[cp_callback]
)

# Validate Gate 3
best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")
if best_val_acc >= 0.88:
    print("GATE 3 STATUS: PASSED! (Val Acc >= 88%)")
else:
    print("GATE 3 STATUS: FAILED (Val Acc < 88%)")

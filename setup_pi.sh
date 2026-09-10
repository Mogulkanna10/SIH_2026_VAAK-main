#!/usr/bin/env bash
# ==============================================================================
# setup_pi.sh - Raspberry Pi 4 Environment Setup Script for Dracarys KWS
# Target Hardware: Raspberry Pi 4 Model B (INMP441 I2S Mic + Audio Output)
# ==============================================================================

set -e

echo "=== [Phase 6 Prep] Setting up Raspberry Pi 4 KWS Environment ==="

# 1. Detect OS Version & Config File Path
CONFIG_PATH="/boot/config.txt"
if [ -f "/boot/firmware/config.txt" ]; then
    CONFIG_PATH="/boot/firmware/config.txt"
    echo "[Info] Detected Raspberry Pi OS Bookworm (Config path: $CONFIG_PATH)"
else
    echo "[Info] Detected Raspberry Pi OS Bullseye/earlier (Config path: $CONFIG_PATH)"
fi

# 2. Add INMP441 I2S Microphone Device Tree Overlays
echo "[Step 1/4] Configuring I2S Device Tree Overlay in $CONFIG_PATH..."

# Backup existing config.txt
if [ ! -f "${CONFIG_PATH}.bak" ]; then
    cp "$CONFIG_PATH" "${CONFIG_PATH}.bak"
    echo "[Info] Backed up original config.txt to ${CONFIG_PATH}.bak"
fi

# Check and add I2S overlay
if ! grep -q "dtoverlay=i2s-mmap" "$CONFIG_PATH"; then
    echo "dtoverlay=i2s-mmap" >> "$CONFIG_PATH"
fi

if ! grep -q "dtoverlay=googlevoicehat-soundcard" "$CONFIG_PATH"; then
    # Alternative standard overlay for simple I2S MEMS mic (INMP441)
    echo "# INMP441 I2S Microphone Overlay" >> "$CONFIG_PATH"
    echo "dtoverlay=i2s-mmap" >> "$CONFIG_PATH"
    echo "dtparam=i2s=on" >> "$CONFIG_PATH"
fi

# 3. System Dependencies
echo "[Step 2/4] Installing system audio libraries (ALSA, PortAudio)..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    alsa-utils

# 4. Python Environment Dependencies
echo "[Step 3/4] Installing Python lightweight runtime dependencies..."
pip3 install --upgrade pip
pip3 install \
    numpy==1.26.4 \
    tflite-runtime \
    sounddevice \
    vosk \
    websockets

# 5. ALSA Audio Configuration (.asoundrc)
echo "[Step 4/4] Generating ALSA audio configuration for INMP441 I2S mic..."
cat << 'EOF' > ~/.asoundrc
pcm.dmic_hw {
    type hw
    card 0
    device 0
}

pcm.dmic_sv {
    type softvol
    slave.pcm "dmic_hw"
    control {
        name "Boost Capture Volume"
        card 0
    }
    min_dB -5.0
    max_dB 30.0
}

pcm.cap {
    type plug
    slave.pcm "dmic_sv"
}

pcm.!default {
    type asym
    capture.pcm "cap"
}
EOF

echo "=== Raspberry Pi 4 Setup Completed Successfully! ==="
echo "Note: If I2S overlays were added to $CONFIG_PATH, please reboot the Pi (sudo reboot)."

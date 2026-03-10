#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# PostureGuard — Dependency installer for Raspberry Pi 5
# Run once:  chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────
set -euo pipefail

echo "╔══════════════════════════════════════════════╗"
echo "║  PostureGuard — Pi 5 Setup                   ║"
echo "╚══════════════════════════════════════════════╝"

# System packages needed by OpenCV and MediaPipe
echo "[1/3] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    libcap-dev \
    libopenjp2-7 \
    libtiff6 \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libhdf5-dev

# Create a virtual environment that inherits system picamera2
echo "[2/3] Creating virtual environment..."
VENV_DIR="$HOME/postureguard-venv"
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install Python packages
echo "[3/3] Installing Python packages..."
pip install --upgrade pip --quiet
pip install \
    mediapipe \
    opencv-python-headless \
    numpy \
    --quiet

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  To run PostureGuard:"
echo "    source $VENV_DIR/bin/activate"
echo "    python3 posture_tracker.py"
echo ""
echo "  Optional flags:"
echo "    --threshold 140   Slouch angle (default 140°)"
echo "    --model 0         Lite model (faster on Pi)"
echo "    --no-flip         Disable mirror mode"
echo "══════════════════════════════════════════════"

#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  AI MUSIC FEEDBACK — Linux / macOS Run Script
#  Auto-detects: Ubuntu · Kali Linux · macOS (Intel + Apple Silicon)
#
#  Usage:
#    chmod +x run.sh
#    ./run.sh              ← full pipeline + demo
#    ./run.sh --demo-only  ← launch demo only
#    ./run.sh --pipeline   ← pipeline only, no demo
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detect OS ──────────────────────────────────────────────────
OS_TYPE="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
fi

MODE="full"
if [[ "$1" == "--demo-only" ]]; then MODE="demo"; fi
if [[ "$1" == "--pipeline" ]];  then MODE="pipeline"; fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
if [[ "$OS_TYPE" == "macos" ]]; then
echo "║       AI MUSIC FEEDBACK — macOS                         ║"
else
echo "║       AI MUSIC FEEDBACK — Linux (Kali/Ubuntu)           ║"
fi
echo "║  NSynth + MAESTRO v3 + GuitarSet + Visualizations       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  OS detected : $OS_TYPE"
echo "  Mode        : $MODE"
echo ""

# ── Check Python ───────────────────────────────────────────────
echo "[check] Python version:"
if command -v python3 &>/dev/null; then
    python3 --version
    PYTHON=python3
elif command -v python &>/dev/null; then
    python --version
    PYTHON=python
else
    echo "ERROR: Python not found. Install Python 3.9+ first."
    exit 1
fi

# ── Check datasets ─────────────────────────────────────────────
echo ""
echo "[check] datasets/ contents:"
ls datasets/ 2>/dev/null || echo "  (empty — synthetic data will be used)"
echo ""

# ── Install dependencies ───────────────────────────────────────
echo "[1/4] Installing dependencies..."

if [[ "$OS_TYPE" == "macos" ]]; then
    # macOS — check for Homebrew, use pip without --break-system-packages
    if command -v brew &>/dev/null; then
        echo "  Homebrew detected"
        # Install fluidsynth via brew for MIDI→WAV synthesis
        brew list fluidsynth &>/dev/null || brew install fluidsynth --quiet 2>/dev/null || true
    fi
    $PYTHON -m pip install -q --upgrade pip 2>/dev/null || true
    $PYTHON -m pip install -q \
        librosa soundfile scikit-learn gradio \
        sentence-transformers pandas numpy pyarrow \
        matplotlib seaborn datasets jams pretty_midi \
        music21 oemer opencv-python-headless Pillow midiutil 2>&1 | tail -3
else
    # Linux (Kali / Ubuntu) — use --break-system-packages flag
    $PYTHON -m pip install -q \
        librosa soundfile scikit-learn gradio \
        sentence-transformers pandas numpy pyarrow \
        matplotlib seaborn datasets jams pretty_midi \
        music21 oemer opencv-python-headless Pillow midiutil verovio cairosvg \
        --break-system-packages 2>&1 | tail -3
fi

echo "  ✓ Dependencies installed"
echo ""

# ── Run pipeline ───────────────────────────────────────────────
if [[ "$MODE" != "demo" ]]; then
    echo "[2/4] Running full pipeline..."
    cd src && $PYTHON pipeline/pipeline.py
    cd "$SCRIPT_DIR"

    echo "[3/4] Running tests..."
    cd src
    $PYTHON -m pytest ../tests/test_all.py -v --tb=short 2>/dev/null \
        || $PYTHON ../tests/test_all.py
    cd "$SCRIPT_DIR"
else
    echo "[2/4] Skipping pipeline (--demo-only)"
    echo "[3/4] Skipping tests"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Complete                                              ║"
echo "║  Plots  : outputs/plots/  (10 PNG files)                ║"
echo "║  Report : outputs/sample_feedback_report.txt            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [[ "$MODE" != "pipeline" ]]; then
    echo "[4/4] Launching Gradio demo → http://localhost:7861"
    echo "      Press Ctrl+C to stop."
    echo ""
    cd src && $PYTHON demo/demo.py
fi

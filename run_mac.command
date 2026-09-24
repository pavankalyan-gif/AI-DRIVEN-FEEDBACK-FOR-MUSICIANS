#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  AI MUSIC FEEDBACK — macOS Launcher
#  Double-click this file in Finder to run on macOS.
#  Works on Intel and Apple Silicon (M1/M2/M3/M4).
# ═══════════════════════════════════════════════════════════════

# Move to project directory (needed when double-clicked in Finder)
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       AI MUSIC FEEDBACK — macOS                         ║"
echo "║  NSynth + MAESTRO v3 + GuitarSet + Visualizations       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Detect Apple Silicon vs Intel
ARCH=$(uname -m)
echo "  Architecture: $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    echo "  Apple Silicon (M1/M2/M3/M4) detected"
    # Homebrew on Apple Silicon is at /opt/homebrew
    export PATH="/opt/homebrew/bin:$PATH"
else
    echo "  Intel Mac detected"
    export PATH="/usr/local/bin:$PATH"
fi

# Find Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
    echo "  Python: $(python3 --version)"
else
    echo "ERROR: Python 3 not found."
    echo "Install from https://www.python.org/ or via Homebrew: brew install python"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check datasets
echo ""
echo "[check] datasets/ contents:"
ls datasets/ 2>/dev/null || echo "  (empty — synthetic data will be used)"
echo ""

# Install dependencies
echo "[1/4] Installing dependencies..."
$PYTHON -m pip install -q --upgrade pip 2>/dev/null || true

# macOS: try without sudo first, then with --user flag
$PYTHON -m pip install -q \
    librosa soundfile scikit-learn gradio \
    sentence-transformers pandas numpy pyarrow \
    matplotlib seaborn datasets jams pretty_midi \
    music21 oemer opencv-python-headless Pillow midiutil verovio cairosvg 2>&1 | tail -3

if command -v brew &>/dev/null; then
    brew list fluidsynth &>/dev/null 2>&1 || \
        brew install fluidsynth --quiet 2>/dev/null || true
fi
echo "  ✓ Dependencies ready"
echo ""

# Run pipeline
echo "[2/4] Running full pipeline..."
cd src && $PYTHON pipeline/pipeline.py
cd ..

# Run tests
echo "[3/4] Running tests..."
cd src
$PYTHON -m pytest ../tests/test_all.py -v --tb=short 2>/dev/null \
    || $PYTHON ../tests/test_all.py
cd ..

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Complete — launching Gradio demo                     ║"
echo "║  Open browser: http://localhost:7861                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Open browser automatically on macOS
sleep 3 && open "http://localhost:7861" &

cd src && $PYTHON demo/demo.py

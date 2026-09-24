#!/usr/bin/env python3
"""
AI Music Feedback — Universal Launcher
Automatically detects OS and runs the correct commands.

Usage (works on ALL operating systems):
    python start.py
    python start.py --demo-only
    python start.py --pipeline
    python start.py --info
"""

import sys
import os
import subprocess
import platform
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── OS Detection ───────────────────────────────────────────────
SYSTEM   = platform.system()          # 'Windows', 'Darwin', 'Linux'
MACHINE  = platform.machine()         # 'x86_64', 'arm64', 'AMD64'
IS_WIN   = SYSTEM == "Windows"
IS_MAC   = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"
IS_ARM   = "arm" in MACHINE.lower() or "aarch" in MACHINE.lower()

PYTHON   = sys.executable             # exact python being used right now
PIP      = [PYTHON, "-m", "pip"]

def banner():
    os_label = {
        "Windows": "Windows",
        "Darwin":  f"macOS ({'Apple Silicon' if IS_ARM else 'Intel'})",
        "Linux":   "Linux (Kali / Ubuntu)",
    }.get(SYSTEM, SYSTEM)
    print("\n" + "═"*62)
    print("  AI MUSIC FEEDBACK — Universal Launcher")
    print(f"  OS      : {os_label}")
    print(f"  Python  : {platform.python_version()} @ {PYTHON}")
    print(f"  Root    : {ROOT}")
    print("═"*62 + "\n")

def run(cmd, cwd=None, check=True):
    """Run a command, printing it first."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT, check=check)
    return result.returncode == 0

def install_deps():
    print("[1/4] Installing dependencies...")
    packages = [
        "librosa", "soundfile", "scikit-learn", "gradio",
        "sentence-transformers", "pandas", "numpy", "pyarrow",
        "matplotlib", "seaborn", "datasets", "jams", "pretty_midi",
        "music21", "oemer", "opencv-python-headless", "Pillow", "midiutil","verovio","cairosvg",
    ]

    pip_cmd = PIP + ["install", "-q"] + packages

    # Linux: add --break-system-packages to avoid PEP 668 error
    if IS_LINUX:
        pip_cmd.append("--break-system-packages")

    try:
        subprocess.run(pip_cmd, check=True,
                       capture_output=True, text=True)
        print("  ✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ pip install warning (non-fatal): {e.stderr[-200:]}")

    # macOS: try to install fluidsynth via Homebrew for better MIDI synthesis
    if IS_MAC:
        brew = subprocess.run(["which", "brew"], capture_output=True, text=True)
        if brew.returncode == 0:
            subprocess.run(["brew", "install", "fluidsynth", "--quiet"],
                           capture_output=True, check=False)
            print("  ✓ Homebrew check done (fluidsynth)")

def check_datasets():
    print("[check] datasets/ contents:")
    ds_dir = ROOT / "datasets"
    if ds_dir.exists():
        items = [d.name for d in ds_dir.iterdir() if not d.name.startswith(".")]
        if items:
            for item in items:
                print(f"  ✓ {item}")
        else:
            print("  (empty — synthetic data will be used)")
    else:
        print("  datasets/ not found — synthetic data will be used")
    print()

def run_pipeline():
    print("[2/4] Running full pipeline...")
    src = ROOT / "src"
    ok  = run([PYTHON, "pipeline/pipeline.py"], cwd=src, check=False)
    if not ok:
        print("  ⚠ Pipeline encountered an error — check output above.")
        return False
    return True

def run_tests():
    print("[3/4] Running tests...")
    src  = ROOT / "src"
    test = ROOT / "tests" / "test_all.py"
    # Try pytest first, fall back to direct execution
    r = subprocess.run([PYTHON, "-m", "pytest", str(test), "-v", "--tb=short"],
                       cwd=src, capture_output=False, check=False)
    if r.returncode != 0:
        subprocess.run([PYTHON, str(test)], cwd=src, check=False)

def launch_demo():
    print("[4/4] Launching Gradio demo...")
    print("      Open browser at: http://localhost:7861")
    print("      Press Ctrl+C to stop.\n")

    # macOS: auto-open browser after a short delay
    if IS_MAC:
        import threading, time, webbrowser
        def _open():
            time.sleep(4)
            webbrowser.open("http://localhost:7861")
        threading.Thread(target=_open, daemon=True).start()

    src = ROOT / "src"
    subprocess.run([PYTHON, "demo/demo.py"], cwd=src, check=False)

def show_info():
    banner()
    print("System Information:")
    print(f"  OS          : {SYSTEM} {platform.release()}")
    print(f"  Architecture: {MACHINE}")
    print(f"  Python      : {platform.python_version()}")
    print(f"  Executable  : {PYTHON}")
    print(f"  Project root: {ROOT}")
    print()

    print("Available Run Scripts:")
    scripts = {
        "start.py":        "Universal — works on ALL OS (this file)",
        "run.sh":          "Linux / macOS terminal",
        "run_mac.command": "macOS — double-click in Finder",
        "run.bat":         "Windows — Command Prompt",
        "run.ps1":         "Windows — PowerShell",
    }
    for name, desc in scripts.items():
        exists = "✓" if (ROOT/name).exists() else "✗"
        print(f"  {exists} {name:22s} — {desc}")

    print()
    print("Datasets expected in datasets/:")
    for ds in ["nsynth_parquet", "maestro_v3", "GuitarSet"]:
        exists = "✓" if (ROOT/"datasets"/ds).exists() else "✗"
        items  = len(list((ROOT/"datasets"/ds).rglob("*"))) \
                 if (ROOT/"datasets"/ds).exists() else 0
        print(f"  {exists} {ds:25s} ({items} files)")

def main():
    parser = argparse.ArgumentParser(
        description="AI Music Feedback — Universal Launcher")
    parser.add_argument("--demo-only", action="store_true",
                        help="Launch demo only (skip pipeline)")
    parser.add_argument("--pipeline",  action="store_true",
                        help="Run pipeline only (no demo)")
    parser.add_argument("--info",      action="store_true",
                        help="Show system/project info and exit")
    args = parser.parse_args()

    if args.info:
        show_info(); return

    banner()
    check_datasets()
    install_deps()

    if not args.demo_only:
        run_pipeline()
        run_tests()
    else:
        print("[2/4] Skipping pipeline (--demo-only)")
        print("[3/4] Skipping tests\n")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✓ Complete                                              ║")
    print("║  Plots  : outputs/plots/  (10 PNG files)                ║")
    print("║  Report : outputs/sample_feedback_report.txt            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if not args.pipeline:
        launch_demo()


if __name__ == "__main__":
    main()

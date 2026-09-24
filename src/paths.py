"""
AI Music Feedback — Cross-Platform Path Utilities
Auto-detects OS: Windows · Linux (Kali/Ubuntu) · macOS
All paths use pathlib.Path — never hardcoded separators.
"""
import sys
import os
import platform
from pathlib import Path

# ── OS Detection ───────────────────────────────────────────────
def get_os() -> str:
    """Returns 'windows', 'linux', or 'macos'."""
    s = sys.platform
    if s.startswith("win"):    return "windows"
    if s.startswith("darwin"): return "macos"
    return "linux"

OS = get_os()

def is_windows() -> bool: return OS == "windows"
def is_linux()   -> bool: return OS == "linux"
def is_macos()   -> bool: return OS == "macos"

def get_python_cmd() -> str:
    return "python" if is_windows() else "python3"

def get_pip_cmd() -> str:
    return "pip" if is_windows() else "pip3"

def get_os_info() -> dict:
    return {
        "os":        OS,
        "platform":  sys.platform,
        "system":    platform.system(),
        "release":   platform.release(),
        "machine":   platform.machine(),
        "python":    platform.python_version(),
    }

# ── Project Root Detection ─────────────────────────────────────
def get_project_root() -> Path:
    """
    Walks up from this file until it finds the project root
    (folder containing both 'datasets/' and 'src/').
    Works from any working directory on any OS.
    """
    current = Path(__file__).resolve().parent
    for _ in range(8):
        if (current / "datasets").exists() and (current / "src").exists():
            return current
        current = current.parent
    # Fallback: this file is in src/
    return Path(__file__).resolve().parent.parent

ROOT          = get_project_root()
SRC_DIR       = ROOT / "src"
DATASETS_DIR  = ROOT / "datasets"
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = DATA_DIR / "models"
OUTPUTS_DIR   = ROOT / "outputs"
PLOTS_DIR     = OUTPUTS_DIR / "plots"
REPORTS_DIR   = OUTPUTS_DIR / "reports"
SHEET_DIR     = OUTPUTS_DIR / "sheet_music"

NSYNTH_DIR    = DATASETS_DIR / "nsynth_parquet"
MAESTRO_DIR   = DATASETS_DIR / "maestro_v3"
GUITARSET_DIR = DATASETS_DIR / "GuitarSet"

# ── Ensure all output dirs exist ───────────────────────────────
for _d in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR,
           DATA_DIR / "raw", DATASETS_DIR,
           SHEET_DIR / "musicxml", SHEET_DIR / "midi", SHEET_DIR / "scores"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── sys.path bootstrap ─────────────────────────────────────────
for _p in [str(SRC_DIR), str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── macOS-specific helpers ─────────────────────────────────────
def get_soundfont_path() -> str:
    """Find a soundfont file for MIDI→WAV synthesis."""
    candidates = []
    if is_macos():
        candidates = [
            "/usr/local/share/sounds/sf2/default.sf2",
            "/opt/homebrew/share/soundfonts/default.sf2",
            "/Library/Audio/Sounds/Banks/gs_instruments.dls",
        ]
    elif is_linux():
        candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default.sf2",
        ]
    # Windows: pretty_midi synthesises without soundfont
    return next((c for c in candidates if Path(c).exists()), "")

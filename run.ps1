# ═══════════════════════════════════════════════════════════════
#  AI MUSIC FEEDBACK — Windows PowerShell Run Script
#
#  Usage:
#    .\run.ps1              ← full pipeline + demo
#    .\run.ps1 --demo-only  ← demo only
#    .\run.ps1 --pipeline   ← pipeline only
#
#  If blocked by execution policy, run once:
#    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# ═══════════════════════════════════════════════════════════════

param([string]$Mode = "full")
if ($args -contains "--demo-only") { $Mode = "demo" }
if ($args -contains "--pipeline")  { $Mode = "pipeline" }

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "+=========================================================+" -ForegroundColor Magenta
Write-Host "|       AI MUSIC FEEDBACK -- Windows PowerShell          |" -ForegroundColor Magenta
Write-Host "|  NSynth + MAESTRO v3 + GuitarSet + Visualizations      |" -ForegroundColor Magenta
Write-Host "+=========================================================+" -ForegroundColor Magenta
Write-Host "  OS detected : Windows" -ForegroundColor Cyan
Write-Host "  Mode        : $Mode" -ForegroundColor Cyan
Write-Host ""

# ── Check Python ──────────────────────────────────────────────
Write-Host "[check] Python version:" -ForegroundColor Cyan
try {
    $v = python --version 2>&1; Write-Host "  $v" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Install from https://www.python.org/" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

# ── Activate venv if present ──────────────────────────────────
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "  Activating venv..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# ── Dataset check ─────────────────────────────────────────────
Write-Host ""; Write-Host "[check] datasets\ contents:" -ForegroundColor Cyan
if (Test-Path "datasets") {
    $items = Get-ChildItem "datasets" -Name
    if ($items) { $items | ForEach-Object { Write-Host "  $_" } }
    else { Write-Host "  (empty -- synthetic data will be used)" -ForegroundColor Yellow }
} else { Write-Host "  datasets\ not found -- synthetic data" -ForegroundColor Yellow }
Write-Host ""

# ── Install dependencies ──────────────────────────────────────
Write-Host "[1/4] Installing dependencies..." -ForegroundColor Cyan
$pkgs = @("librosa","soundfile","scikit-learn","gradio",
          "sentence-transformers","pandas","numpy","pyarrow",
          "matplotlib","seaborn","datasets","jams","pretty_midi",
          "music21","oemer","opencv-python-headless","Pillow","midiutil","verovio","cairosvg")
pip install -q @pkgs
Write-Host "  ✓ Done" -ForegroundColor Green; Write-Host ""

if ($Mode -ne "demo") {
    Write-Host "[2/4] Running full pipeline..." -ForegroundColor Cyan
    Set-Location src; python pipeline\pipeline.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Pipeline failed." -ForegroundColor Red
        Set-Location $ProjectRoot; Read-Host "Press Enter"; exit 1
    }
    Set-Location $ProjectRoot

    Write-Host "[3/4] Running tests..." -ForegroundColor Cyan
    Set-Location src
    python -m pytest ..\tests\test_all.py -v --tb=short 2>$null
    if ($LASTEXITCODE -ne 0) { python ..\tests\test_all.py }
    Set-Location $ProjectRoot
} else {
    Write-Host "[2/4] Skipping pipeline (--demo-only)" -ForegroundColor Yellow
    Write-Host "[3/4] Skipping tests" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "+=========================================================+" -ForegroundColor Green
Write-Host "|  Pipeline complete                                      |" -ForegroundColor Green
Write-Host "|  Plots  : outputs\plots\  (10 PNG files)               |" -ForegroundColor Green
Write-Host "+=========================================================+" -ForegroundColor Green
Write-Host ""

if ($Mode -ne "pipeline") {
    Write-Host "[4/4] Launching Gradio demo..." -ForegroundColor Cyan
    Write-Host "      Open: http://localhost:7861" -ForegroundColor Green
    Write-Host "      Press Ctrl+C to stop."; Write-Host ""
    Set-Location src; python demo\demo.py; Set-Location $ProjectRoot
}

@echo off
:: ═══════════════════════════════════════════════════════════════
::  AI MUSIC FEEDBACK — Windows Run Script (CMD)
::  Auto-detects Python location and venv
::
::  Usage:
::    run.bat              ← full pipeline + demo
::    run.bat --demo-only  ← demo only
::    run.bat --pipeline   ← pipeline only
:: ═══════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion
cd /d "%~dp0"

set MODE=full
if "%1"=="--demo-only" set MODE=demo
if "%1"=="--pipeline"  set MODE=pipeline

echo.
echo +=========================================================+
echo ^|       AI MUSIC FEEDBACK -- Windows                     ^|
echo ^|  NSynth + MAESTRO v3 + GuitarSet + Visualizations      ^|
echo +=========================================================+
echo.
echo   OS detected : Windows
echo   Mode        : %MODE%
echo.

:: ── Check Python ──────────────────────────────────────────────
echo [check] Python version:
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Install from https://www.python.org/ and check "Add to PATH"
    pause & exit /b 1
)

:: ── Activate venv if present ──────────────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo   Activating venv...
    call venv\Scripts\activate.bat
)

:: ── Dataset check ─────────────────────────────────────────────
echo.
echo [check] datasets\ contents:
if exist "datasets\" (dir /b datasets\ 2>nul || echo   (empty))
else (echo   datasets\ not found -- synthetic data will be used)
echo.

:: ── Install dependencies ──────────────────────────────────────
echo [1/4] Installing dependencies...
pip install -q librosa soundfile scikit-learn gradio ^
    sentence-transformers pandas numpy pyarrow ^
    matplotlib seaborn datasets jams pretty_midi ^
    music21 oemer opencv-python-headless Pillow midiutil verovio cairosvg 2>nul
if %errorlevel% neq 0 (
    echo   Retrying without -q flag...
    pip install librosa soundfile scikit-learn gradio ^
        sentence-transformers pandas numpy pyarrow ^
        matplotlib seaborn datasets jams pretty_midi ^
        music21 oemer opencv-python-headless Pillow midiutil
)
echo   Done
echo.

if "%MODE%"=="demo" goto SKIP_PIPELINE

:: ── Run pipeline ──────────────────────────────────────────────
echo [2/4] Running full pipeline...
cd src
python pipeline\pipeline.py
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed. Check output above.
    cd .. & pause & exit /b 1
)
cd ..

:: ── Run tests ─────────────────────────────────────────────────
echo [3/4] Running tests...
cd src
python -m pytest ..\tests\test_all.py -v --tb=short 2>nul
if %errorlevel% neq 0 python ..\tests\test_all.py
cd ..
goto AFTER_PIPELINE

:SKIP_PIPELINE
echo [2/4] Skipping pipeline (--demo-only)
echo [3/4] Skipping tests

:AFTER_PIPELINE
echo.
echo +=========================================================+
echo ^|  Pipeline complete                                      ^|
echo ^|  Plots  : outputs\plots\  (10 PNG files)               ^|
echo ^|  Report : outputs\sample_feedback_report.txt           ^|
echo +=========================================================+
echo.

if "%MODE%"=="pipeline" goto END

echo [4/4] Launching Gradio demo...
echo       Open: http://localhost:7861
echo       Press Ctrl+C to stop.
echo.
cd src
python demo\demo.py
cd ..

:END
endlocal
pause

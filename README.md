# 🎵 AI Music Feedback — Intelligent Practice Intelligence

**AI-Driven Feedback for Musicians — Cross-Platform**
Virtuoso Learning Ltd × Univerworx | Archie Maclennan | archie@virtuosoapp.io

---

## 🚀 Quick Start — Pick Your OS

### ▶ Universal (works on ALL operating systems)
```bash
python start.py
```

### 🐧 Linux (Kali / Ubuntu) — Terminal
```bash
chmod +x run.sh && ./run.sh
```

### 🍎 macOS — Double-click in Finder
```
run_mac.command   ← double-click this file
```
Or in Terminal:
```bash
chmod +x run.sh && ./run.sh
```

### 🪟 Windows — Command Prompt
```bat
run.bat
```

### 🪟 Windows — PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # once only
.\run.ps1
```

---

## ⚡ Run Options (all scripts support these flags)

| Flag | Effect |
|---|---|
| *(no flag)* | Full pipeline + demo |
| `--demo-only` | Launch demo only (skip training) |
| `--pipeline` | Run pipeline only (no demo) |
| `--info` | Show system info (start.py only) |

Examples:
```bash
python start.py --demo-only      # fastest — just launches the demo
python start.py --info           # check OS/Python/dataset status
./run.sh --pipeline              # Linux: train only
run.bat --demo-only              # Windows: demo only
```

---

## 📁 Dataset Setup (Optional)

Place your datasets in the `datasets/` folder **before running**.
If absent, synthetic data is auto-generated and everything still works.

```
AI_MUSIC_FEEDBACK/
└── datasets/
    ├── nsynth_parquet/          ← NSynth (Arrow cache from HuggingFace)
    ├── maestro_v3/              ← MAESTRO v3.0.0 (CSV + MIDI folders)
    │   ├── maestro-v3.0.0.csv
    │   └── maestro-v3.0.0/
    │       ├── 2004/ ... 2018/
    └── GuitarSet/               ← GuitarSet
        ├── annotation/          ← .jams files
        └── audio_mono-mic/      ← *_mic.wav files
```

---

## 🎭 Gradio Demo — 8 Tabs (http://localhost:7861)

| Tab | Function |
|---|---|
| 🎧 Audio Analysis | Upload audio → pitch, quality score, 60+ features |
| 🔄 Feedback Comparison | Student vs teacher NLP alignment |
| 📝 Practice Logger | Log session → structured feedback + JSON |
| 🎼 Sheet Music Upload | PDF/PNG/JPG → MusicXML (oemer OMR — local) |
| 📋 MusicXML & Events | Note event table, bar selection, CSV download |
| 🎹 Score & MIDI | Piano roll visualisation + MIDI playback |
| 📊 Visualizations | 10 training/testing charts gallery |
| 🖥️ Dashboard | Live dataset + model stats |

---

## 🖥️ OS Compatibility

| Feature | Windows | Linux | macOS Intel | macOS M1/M2/M3 |
|---|---|---|---|---|
| run.bat | ✅ | ❌ | ❌ | ❌ |
| run.ps1 | ✅ | ❌ | ❌ | ❌ |
| run.sh | ❌ | ✅ | ✅ | ✅ |
| run_mac.command | ❌ | ❌ | ✅ | ✅ |
| start.py | ✅ | ✅ | ✅ | ✅ |
| Python pipeline | ✅ | ✅ | ✅ | ✅ |
| Gradio demo | ✅ | ✅ | ✅ | ✅ |
| All visualizations | ✅ | ✅ | ✅ | ✅ |
| Sheet Music OMR | ✅* | ✅ | ✅ | ✅ |
| MIDI synthesis | ✅ | ✅ | ✅† | ✅† |

*NSynth audio decoding uses metadata fallback on Windows (no FFmpeg DLLs needed)
†Homebrew fluidsynth recommended for best WAV synthesis on macOS

---

## 🏗️ File Structure

```
AI_MUSIC_FEEDBACK/
├── start.py                     ← Universal launcher (ALL OS)
├── run.sh                       ← Linux / macOS terminal
├── run_mac.command              ← macOS double-click launcher
├── run.bat                      ← Windows CMD
├── run.ps1                      ← Windows PowerShell
├── requirements.txt
├── README.md
├── src/
│   ├── paths.py                 ← Cross-platform path + OS detection
│   ├── data_loader.py           ← NSynth + MAESTRO + GuitarSet + Synthetic
│   ├── audio_features/
│   │   └── extractor.py         ← librosa 60+ features
│   ├── ml_model/
│   │   └── model.py             ← GradientBoosting + RandomForest + CV
│   ├── visualizations/
│   │   └── charts.py            ← 10 charts (matplotlib version-safe)
│   ├── feedback/
│   │   └── feedback.py          ← Sentence transformer NLP
│   ├── sheet_music/
│   │   └── sheet_processor.py   ← oemer OMR + music21 + MIDI
│   ├── pipeline/
│   │   └── pipeline.py          ← 5-step orchestrator
│   └── demo/
│       └── demo.py              ← Gradio 8-tab app (port 7861)
├── datasets/                    ← PUT YOUR DATASETS HERE
├── data/                        ← Auto-generated (models, processed data)
├── outputs/                     ← Auto-generated (plots, reports, MIDI)
└── tests/
    └── test_all.py              ← 35 unit tests
```

---

## 🤖 Model Performance

| Model | Algorithm | Performance |
|---|---|---|
| Quality Scorer | GradientBoosting (200 trees) | MAE ≈ 3 pts · R² ≈ 0.95 |
| Skill Classifier | RandomForest (300 trees) | Accuracy ≈ 99.6% |

Full metrics → `data/models/model_summary.json`

---

*AI Music Feedback · Virtuoso Learning Ltd · 2026*

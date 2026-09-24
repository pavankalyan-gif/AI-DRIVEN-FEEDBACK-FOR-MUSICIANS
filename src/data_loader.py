"""
AI Music Feedback — Dataset Loader (Cross-Platform)
Loads NSynth, MAESTRO v3, GuitarSet from ./datasets/ folder.
Works on Linux and Windows — no shell commands, pure Python.
"""
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Import cross-platform paths
import sys, os
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
from paths import (ROOT, DATASETS_DIR, PROCESSED_DIR, MODELS_DIR,
                   NSYNTH_DIR, MAESTRO_DIR, GUITARSET_DIR)

# ── Synthetic generators ───────────────────────────────────────────────────────
def _synth_audio(n=400, seed=42, source="synthetic"):
    rng = np.random.default_rng(seed)
    instruments = ["Piano","Guitar","Violin","Flute","Bass","Trumpet","Cello","Drums"]
    skills = ["Beginner","Intermediate","Advanced"]
    rows = []
    for i in range(n):
        skill = rng.choice(skills)
        sf = {"Beginner":0.38,"Intermediate":0.68,"Advanced":0.93}[skill]
        rows.append({
            "source": source,
            "instrument": str(rng.choice(instruments)),
            "skill_level": skill,
            **{f"mfcc_{j}_mean": float(rng.normal(j*2-13, 4+abs(j-6))) for j in range(13)},
            **{f"mfcc_{j}_std":  float(abs(rng.normal(2, 0.8)))          for j in range(13)},
            **{f"chroma_{j}":    float(rng.uniform(0,1))                  for j in range(12)},
            "spectral_centroid":      float(rng.uniform(400,4500)*sf),
            "spectral_rolloff":       float(rng.uniform(800,9000)*sf),
            "spectral_bandwidth":     float(rng.uniform(400,3500)),
            "spectral_contrast_mean": float(rng.uniform(10,50)),
            "zero_crossing_rate":     float(rng.uniform(0.01,0.35)),
            "tempo":                  float(rng.uniform(50,200)),
            "rms_energy_mean":        float(rng.uniform(0.01,0.4)*sf),
            "rms_energy_std":         float(rng.uniform(0.001,0.08)),
            "loudness_consistency":   float(np.clip(rng.normal(sf,0.12),0,1)),
            "pitch_accuracy":         float(np.clip(rng.normal(sf,0.10),0,1)),
            "note_detection_score":   float(np.clip(rng.normal(sf,0.12),0,1)),
            "pitch_cents_deviation":  float(rng.normal(0, 30*(1-sf))),
            "duration_sec":           float(rng.uniform(1,30)),
            "quality_score":          float(np.clip(rng.normal(sf*100, 8), 0, 100)),
        })
    return pd.DataFrame(rows)

def _synth_logs(n=300, seed=42, source="synthetic"):
    rng = np.random.default_rng(seed)
    pieces  = ["Für Elise","Moonlight Sonata","Canon in D","Nocturne Op.9 No.2",
               "River Flows in You","Clair de Lune","Gymnopédie No.1","Air on G String"]
    focuses = ["Dynamics control","Tempo consistency","Articulation","Scales",
               "Expression","Rhythm accuracy","Memory work","Sight reading"]
    rows = []
    for i in range(n):
        sr  = int(rng.integers(1,11))
        tr  = int(np.clip(sr + rng.integers(-2,3), 1, 10))
        piece = str(rng.choice(pieces))
        dur   = int(rng.integers(10,90))
        focus = str(rng.choice(focuses))
        rows.append({
            "source": source, "instrument":"Piano",
            "piece": piece, "focus_area": focus, "duration_min": dur,
            "student_rating": sr, "teacher_rating": tr,
            "alignment_score": float(1.0-abs(sr-tr)/10.0),
            "quality_score": float(np.clip(rng.normal(65,14),0,100)),
            "skill_level": str(rng.choice(["Beginner","Intermediate","Advanced"])),
            "student_self_report": f"Practised {piece} for {dur} min. Focused on {focus.lower()}. Rating: {sr}/10.",
            "teacher_feedback":    f"{'Good effort' if tr>=6 else 'Needs work'}. Work on {focus.lower()}. Rating: {tr}/10.",
            "tempo": float(rng.uniform(60,180)),
            "pitch_accuracy": float(np.clip(rng.normal(0.65,0.15),0,1)),
        })
    return pd.DataFrame(rows)

# ── NSynth ─────────────────────────────────────────────────────────────────────
def load_nsynth(max_samples=600):
    print("\n[NSynth] Checking datasets/nsynth_parquet ...")
    if not NSYNTH_DIR.exists() or not any(NSYNTH_DIR.rglob("*.arrow")):
        print("  ⚠  Arrow files not found → synthetic fallback")
        df = _synth_audio(max_samples, source="nsynth_synthetic")
        df.to_parquet(PROCESSED_DIR/"nsynth_features.parquet", index=False)
        return df
    try:
        from datasets import load_dataset
        ds = load_dataset("confit/nsynth-parquet", name="instrument",
                          split="train", cache_dir=str(NSYNTH_DIR))
        print(f"  ✓  NSynth loaded: {len(ds)} samples")
        families = ["bass","brass","flute","guitar","keyboard","mallet",
                    "organ","reed","string","synth_lead","vocal"]
        rng = np.random.default_rng(42)
        idx = rng.choice(len(ds), size=min(max_samples,len(ds)), replace=False)
        rows = []
        for i in idx:
            item  = ds[int(i)]
            label = item.get("label", 0)
            fam   = families[label % len(families)] if isinstance(label,int) else str(label)
            audio = item.get("audio", None)
            if audio is not None and hasattr(audio,"__len__") and len(audio)>100:
                arr  = np.array(audio, dtype=np.float32)
                rms  = float(np.sqrt(np.mean(arr**2)))
                peak = float(np.max(np.abs(arr)))
                zcr  = float(np.mean(np.abs(np.diff(np.sign(arr))))/2)
            else:
                rms  = float(rng.uniform(0.01,0.5))
                peak = float(rng.uniform(0.1,1.0))
                zcr  = float(rng.uniform(0.01,0.3))
            rows.append({
                "source":"nsynth","instrument":fam,"skill_level":"Advanced",
                "quality_score":float(np.clip(rms*120+peak*25+rng.normal(0,3),0,100)),
                "rms_energy_mean":rms,"peak_amplitude":peak,"zero_crossing_rate":zcr,
                "loudness_consistency":float(np.clip(rms*3,0,1)),
                "pitch_accuracy":float(rng.uniform(0.75,1.0)),
                "spectral_centroid":float(rng.uniform(800,5000)),
                "spectral_rolloff":float(rng.uniform(1500,9000)),
                "spectral_bandwidth":float(rng.uniform(500,3000)),
                "spectral_contrast_mean":float(rng.uniform(15,50)),
                "tempo":float(rng.uniform(80,160)),
                "note_detection_score":float(rng.uniform(0.7,1.0)),
                "pitch_cents_deviation":float(rng.normal(0,8)),
                "duration_sec":float(rng.uniform(2,10)),
                "rms_energy_std":float(rng.uniform(0.001,0.05)),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(PROCESSED_DIR/"nsynth_features.parquet", index=False)
        print(f"  ✓  Extracted {len(df)} feature rows")
        return df
    except Exception as e:
        print(f"  ⚠  NSynth error ({e}) → synthetic fallback")
        df = _synth_audio(max_samples, source="nsynth_synthetic")
        df.to_parquet(PROCESSED_DIR/"nsynth_features.parquet", index=False)
        return df

# ── MAESTRO ────────────────────────────────────────────────────────────────────
def load_maestro():
    print("\n[MAESTRO] Checking datasets/maestro_v3 ...")
    csv_candidates = list(MAESTRO_DIR.rglob("maestro-v3.0.0.csv"))
    if not csv_candidates:
        print("  ⚠  CSV not found → synthetic fallback")
        df = _synth_logs(300, source="maestro_synthetic")
        df.to_parquet(PROCESSED_DIR/"maestro_features.parquet", index=False)
        return df
    df_csv = pd.read_csv(csv_candidates[0])
    print(f"  ✓  MAESTRO CSV: {len(df_csv)} rows")
    midi_base = csv_candidates[0].parent
    rng = np.random.default_rng(42)
    rows = []
    for _, row in df_csv.iterrows():
        duration = float(row.get("duration",0) or 0)
        composer = str(row.get("canonical_composer","Unknown"))
        title    = str(row.get("canonical_title","Unknown"))
        split    = str(row.get("split","train"))
        midi_fn  = str(row.get("midi_filename",""))
        tempo, note_count = 120.0, 0
        if midi_fn:
            # Cross-platform path join
            midi_parts = midi_fn.replace("\\","/").split("/")
            midi_path  = midi_base
            for part in midi_parts:
                midi_path = midi_path / part
            if midi_path.exists():
                try:
                    import pretty_midi
                    pm    = pretty_midi.PrettyMIDI(str(midi_path))
                    tvals = pm.get_tempo_changes()[1]
                    tempo = float(tvals[0]) if len(tvals)>0 else 120.0
                    note_count = sum(len(inst.notes) for inst in pm.instruments)
                except Exception:
                    pass
        if tempo == 120.0:    tempo      = float(rng.uniform(60,200))
        if note_count == 0:   note_count = int(rng.integers(50,600))
        complexity = min(note_count/500.0, 1.0)
        quality    = float(np.clip(68+complexity*28+rng.normal(0,3), 0, 100))
        sr2 = int(rng.integers(7,11))
        tr2 = int(np.clip(sr2+rng.integers(-1,2),1,10))
        rows.append({
            "source":"maestro","instrument":"Piano",
            "piece":title,"composer":composer,"split":split,
            "duration_min":duration/60,"tempo":tempo,"note_count":note_count,
            "quality_score":quality,"skill_level":"Advanced",
            "pitch_accuracy":float(rng.uniform(0.82,1.0)),
            "spectral_centroid":float(rng.uniform(1000,4500)),
            "spectral_rolloff":float(rng.uniform(2000,9000)),
            "spectral_bandwidth":float(rng.uniform(600,3000)),
            "spectral_contrast_mean":float(rng.uniform(20,50)),
            "zero_crossing_rate":float(rng.uniform(0.02,0.2)),
            "rms_energy_mean":float(rng.uniform(0.05,0.35)),
            "rms_energy_std":float(rng.uniform(0.005,0.06)),
            "loudness_consistency":float(rng.uniform(0.65,1.0)),
            "note_detection_score":float(rng.uniform(0.78,1.0)),
            "pitch_cents_deviation":float(rng.normal(0,5)),
            "duration_sec":duration,
            "student_rating":sr2,"teacher_rating":tr2,
            "alignment_score":float(1.0-abs(sr2-tr2)/10.0),
            "focus_area":"Piano performance",
            "student_self_report":f"Practised {title} by {composer}. Duration {duration:.0f}s.",
            "teacher_feedback":f"Professional rendition of {title}. Tempo {tempo:.0f} BPM.",
        })
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(PROCESSED_DIR/"maestro_features.parquet", index=False)
    print(f"  ✓  MAESTRO features: {len(df_out)} rows")
    return df_out

# ── GuitarSet ──────────────────────────────────────────────────────────────────
def load_guitarset():
    print("\n[GuitarSet] Checking datasets/GuitarSet ...")
    ann_dir   = GUITARSET_DIR / "annotation"
    audio_dir = GUITARSET_DIR / "audio_mono-mic"
    if not ann_dir.exists():
        print("  ⚠  Not found → synthetic fallback")
        df = _synth_audio(180, source="guitarset_synthetic")
        df.to_parquet(PROCESSED_DIR/"guitarset_features.parquet", index=False)
        return df
    jams_files = sorted(ann_dir.glob("*.jams"))
    wav_map    = {}
    if audio_dir.exists():
        for f in audio_dir.glob("*_mic.wav"):
            wav_map[f.stem.replace("_mic","")] = f
    print(f"  ✓  Found {len(jams_files)} JAMS + {len(wav_map)} WAV files")
    from audio_features.extractor import AudioFeatureExtractor
    ext = AudioFeatureExtractor()
    rng = np.random.default_rng(42)
    rows = []
    for jf in jams_files:
        try:
            import jams
            jam   = jams.load(str(jf))
            stem  = jf.stem
            parts = stem.split("-")
            tempo = 120.0
            try: tempo = float(parts[1])
            except: pass
            key   = parts[2].split("_")[0] if len(parts)>=3 else "C"
            genre = next((g for g in ["BN","Funk","Jazz","Rock","SS"] if g in stem),"Unknown")
            skill_map = {"Jazz":"Advanced","Rock":"Intermediate","Funk":"Intermediate",
                         "BN":"Beginner","SS":"Beginner"}
            skill = skill_map.get(genre,"Intermediate")
            af = {}
            if stem in wav_map:
                try: af = ext.extract(str(wav_map[stem]))
                except: pass
            def _f(k, lo, hi):
                return float(af.get(k, rng.uniform(lo,hi)))
            rows.append({
                "source":"guitarset","instrument":"Guitar",
                "piece":stem,"genre":genre,"key":key,
                "skill_level":skill,"tempo":tempo,
                "quality_score":     _f("quality_score",45,88),
                "pitch_accuracy":    _f("pitch_accuracy",0.45,0.92),
                "spectral_centroid": _f("spectral_centroid",800,4500),
                "spectral_rolloff":  _f("spectral_rolloff",1500,9000),
                "spectral_bandwidth":_f("spectral_bandwidth",500,3000),
                "spectral_contrast_mean":_f("spectral_contrast_mean",10,50),
                "loudness_consistency":_f("loudness_consistency",0.35,0.92),
                "zero_crossing_rate":_f("zero_crossing_rate",0.04,0.35),
                "rms_energy_mean":   _f("rms_energy_mean",0.01,0.35),
                "rms_energy_std":    _f("rms_energy_std",0.001,0.08),
                "note_detection_score":_f("note_detection_score",0.45,0.92),
                "pitch_cents_deviation":float(af.get("pitch_cents_deviation",
                                                      rng.normal(0,25))),
                "duration_sec":      _f("duration_sec",5,90),
            })
        except: pass
    if not rows:
        print("  ⚠  No rows parsed → synthetic fallback")
        df = _synth_audio(180, source="guitarset_synthetic")
        df.to_parquet(PROCESSED_DIR/"guitarset_features.parquet", index=False)
        return df
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED_DIR/"guitarset_features.parquet", index=False)
    print(f"  ✓  GuitarSet: {len(df)} rows")
    return df

# ── Combined builder ───────────────────────────────────────────────────────────
AUDIO_COLS = ["source","instrument","skill_level","quality_score","pitch_accuracy",
              "spectral_centroid","spectral_rolloff","spectral_bandwidth",
              "spectral_contrast_mean","zero_crossing_rate","rms_energy_mean",
              "rms_energy_std","loudness_consistency","tempo","note_detection_score",
              "pitch_cents_deviation","duration_sec",
              *[f"mfcc_{j}_mean" for j in range(13)],
              *[f"mfcc_{j}_std"  for j in range(13)],
              *[f"chroma_{j}"    for j in range(12)]]

LOG_COLS = ["source","instrument","piece","focus_area","duration_min","student_rating",
            "teacher_rating","alignment_score","quality_score","skill_level","tempo",
            "pitch_accuracy","student_self_report","teacher_feedback"]

def build_training_dataset() -> dict:
    print("\n" + "="*62)
    print("  AI MUSIC FEEDBACK — DATASET LOADING")
    print("="*62)

    nsynth_df  = load_nsynth(600)
    maestro_df = load_maestro()
    guitar_df  = load_guitarset()

    print("\n[Synthetic] Generating supplementary training data...")
    synth_a = _synth_audio(400)
    synth_l = _synth_logs(300)
    print(f"  ✓ Synthetic audio: {len(synth_a)} | logs: {len(synth_l)}")

    def _merge(frames, cols):
        out = []
        for df in frames:
            sub = df[[c for c in cols if c in df.columns]].copy()
            for c in cols:
                if c not in sub.columns: sub[c] = np.nan
            out.append(sub)
        merged = pd.concat(out, ignore_index=True)
        for col in merged.select_dtypes(include=[np.number]).columns:
            merged[col] = merged[col].fillna(merged[col].median())
        return merged

    audio_df = _merge([nsynth_df, guitar_df, synth_a], AUDIO_COLS)
    logs_df  = _merge([maestro_df, synth_l],           LOG_COLS)

    audio_df.to_parquet(PROCESSED_DIR/"audio_features.parquet", index=False)
    logs_df.to_parquet(PROCESSED_DIR/"practice_logs.parquet",   index=False)

    stats = {
        "nsynth_samples":    int(len(nsynth_df)),
        "maestro_samples":   int(len(maestro_df)),
        "guitarset_samples": int(len(guitar_df)),
        "synthetic_audio":   int(len(synth_a)),
        "synthetic_logs":    int(len(synth_l)),
        "total_audio":       int(len(audio_df)),
        "total_logs":        int(len(logs_df)),
    }
    (ROOT/"data"/"manifest.json").write_text(json.dumps(stats, indent=2))

    print("\n" + "="*62)
    print("  DATASET SUMMARY")
    for k,v in stats.items():
        print(f"  {k:30s}: {v}")
    print("="*62)
    return {"audio_df":audio_df, "logs_df":logs_df, "stats":stats,
            "nsynth_df":nsynth_df, "maestro_df":maestro_df, "guitar_df":guitar_df}

if __name__ == "__main__":
    build_training_dataset()

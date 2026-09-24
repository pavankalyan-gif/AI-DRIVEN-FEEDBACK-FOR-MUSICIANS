"""
AI Music Feedback — Gradio Demo (Cross-Platform: Windows · Linux · macOS)
8 Tabs based on Archie's feedback:
  1. Audio Analysis
  2. Feedback Comparison
  3. Practice Logger
  4. Sheet Music Upload   ← shows uploaded image + converts to MusicXML
  5. MusicXML & Events   ← clean note table (Archie's request)
  6. Score & MIDI        ← STANDARD NOTATION (verovio) + piano roll + MIDI
  7. Visualizations
  8. Dashboard
"""
import sys, json, warnings, time, shutil
import numpy as np, pandas as pd
import gradio as gr
from pathlib import Path
warnings.filterwarnings("ignore")

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from paths import ROOT, PLOTS_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR, is_windows
from audio_features.extractor import AudioFeatureExtractor
from feedback.feedback import FeedbackComparator, FeedbackGenerator
from sheet_music.sheet_processor import (
    save_uploaded_file, image_to_musicxml, parse_musicxml,
    render_notation_score, render_score_image, render_bar_range,
    musicxml_to_midi, midi_to_wav
)

MANIFEST_PATH  = ROOT / "data" / "manifest.json"
MODEL_SUM_PATH = MODELS_DIR / "model_summary.json"
SHEET_DIR      = OUTPUTS_DIR / "sheet_music"

print("Loading AI Music Feedback components...")
extractor  = AudioFeatureExtractor()
comparator = FeedbackComparator()
generator  = FeedbackGenerator()

logs_df   = pd.read_parquet(PROCESSED_DIR/"practice_logs.parquet") \
            if (PROCESSED_DIR/"practice_logs.parquet").exists() else pd.DataFrame()
manifest  = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) \
            if MANIFEST_PATH.exists() else {}
model_sum = json.loads(MODEL_SUM_PATH.read_text(encoding="utf-8")) \
            if MODEL_SUM_PATH.exists() else {}
session_store = []

sheet_state = {
    "uploaded_path":  None,
    "saved_path":     None,
    "musicxml_path":  None,
    "events":         [],
    "stats":          {},
    "midi_path":      None,
    "wav_path":       None,
    "filename":       "",
}

NOTES = [""] + [f"{n}{o}" for o in range(2,7)
                for n in ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]]
FOCUS = ["Scales and arpeggios","Sight reading","Dynamics control",
         "Tempo consistency","Articulation","Pedal technique",
         "Hand coordination","Memory work","Expression","Rhythm accuracy"]

print("  ✓ Ready")

# ── TAB 1: AUDIO ANALYSIS ─────────────────────────────────────
def analyze_audio(audio_file, target_note):
    if audio_file is None:
        return "Please upload an audio file.", "{}", "No audio provided."
    try:
        feat = extractor.extract(audio_file)
        fb   = extractor.generate_feedback_string(feat, target_note=target_note or None)
        display = {
            "Quality Score":        f"{feat.get('quality_score',0):.1f} / 100",
            "Detected Note":        feat.get("detected_note","N/A"),
            "Pitch Accuracy":       f"{feat.get('pitch_accuracy',0)*100:.1f}%",
            "In Tune":              "✓ Yes" if feat.get("pitch_in_tune") else "✗ No",
            "Pitch Deviation":      f"{feat.get('pitch_cents_deviation',0):+.1f} cents",
            "Tempo":                f"{feat.get('tempo',0):.0f} BPM",
            "Loudness Consistency": f"{feat.get('loudness_consistency',0)*100:.1f}%",
            "Spectral Centroid":    f"{feat.get('spectral_centroid',0):.0f} Hz",
            "RMS Energy":           f"{feat.get('rms_energy_mean',0):.4f}",
            "Duration":             f"{feat.get('duration_sec',0):.2f} s",
        }
        tbl = "| Feature | Value |\n|---|---|\n" + \
              "".join(f"| {k} | {v} |\n" for k,v in display.items())
        raw = json.dumps({k:v for k,v in feat.items()
                          if isinstance(v,(int,float,str,bool))}, indent=2)
        return tbl, raw, fb
    except Exception as e:
        return f"Error: {e}", "{}", str(e)

# ── TAB 2: FEEDBACK COMPARISON ────────────────────────────────
def compare_feedback(student, teacher):
    if not student.strip() or not teacher.strip():
        return "Please enter both texts.", ""
    result = comparator.compare_session(student, teacher)
    icon   = {"HIGH":"🟢","MODERATE":"🟡","LOW":"🔴"}.get(result["alignment"],"⚪")
    out    = f"""## {icon} Alignment: {result['alignment']}
**Similarity Score:** {result['similarity_score']:.0%}
**Interpretation:** {result['interpretation']}
Student words: {result['student_word_count']} | Teacher words: {result['teacher_word_count']}"""
    tips = generator.generate(
        session_data={"student_rating":6,"teacher_rating":6,"piece":"Current"},
        comparison_result=result)
    tips_md = "\n".join(f"- {t}" for t in tips.get("actionable_tips",[]))
    return out, f"### 💡 Actionable Tips\n{tips_md}"

def random_session():
    valid = logs_df.dropna(subset=["student_self_report","teacher_feedback"]) \
            if "student_self_report" in logs_df.columns else pd.DataFrame()
    if len(valid):
        r = valid.sample(1).iloc[0]
        return str(r.get("student_self_report","")), str(r.get("teacher_feedback",""))
    return ("Practised Nocturne for 30 min. Left hand improving.",
            "Good effort. Dynamics in section B need work.")

# ── TAB 3: PRACTICE LOGGER ───────────────────────────────────
def log_session(piece, focus, duration, report_txt, rating, audio_file):
    if not piece.strip() or not report_txt.strip():
        return "Please fill in Piece and Self-Report.", ""
    session = {"session_id":f"SESS_{len(session_store):04d}","piece":piece,
               "focus_area":focus,"duration_min":int(duration),
               "student_self_report":report_txt,"student_rating":int(rating),
               "teacher_rating":None,"teacher_feedback":""}
    af = None
    if audio_file:
        try: af = extractor.extract(audio_file)
        except: pass
    fb = generator.generate(audio_features=af, session_data=session)
    session["feedback"] = fb
    session_store.append(session)
    report = generator.format_for_display(fb)
    js_out = json.dumps({"session":{k:v for k,v in session.items() if k!="feedback"},
                         "feedback":fb}, indent=2, default=str)
    return report, js_out

# ── TAB 4: SHEET MUSIC UPLOAD ─────────────────────────────────
def upload_sheet_music(file):
    """Upload file, save it, show it back to user immediately."""
    if file is None:
        return (None, "Please upload a sheet music file.",
                "", "", gr.update(interactive=False))
    try:
        info   = save_uploaded_file(file)
        sheet_state["uploaded_path"] = file
        sheet_state["saved_path"]    = info["saved_path"]
        sheet_state["filename"]      = info["filename"]
        sheet_state["musicxml_path"] = None
        sheet_state["events"]        = []
        sheet_state["stats"]         = {}
        status = f"✅ Sheet music file uploaded successfully."
        info_txt = f"**File:** {info['filename']}  |  **Size:** {info['size_mb']} MB"
        return (info["saved_path"], status, info_txt, "",
                gr.update(interactive=True))
    except Exception as e:
        return (None, f"❌ Upload error: {e}", "", "", gr.update(interactive=False))

def convert_to_musicxml():
    """Convert uploaded sheet music to MusicXML."""
    if not sheet_state["uploaded_path"]:
        return "❌ Upload a sheet music file first.", ""
    out_name = f"{Path(sheet_state['filename']).stem}_{int(time.time())}"
    result   = image_to_musicxml(sheet_state["uploaded_path"], out_name)
    if result["success"]:
        sheet_state["musicxml_path"] = result["musicxml_path"]
        parsed = parse_musicxml(result["musicxml_path"])
        if parsed["success"]:
            sheet_state["events"] = parsed["events"]
            sheet_state["stats"]  = parsed["stats"]
        s = sheet_state["stats"]
        method = result.get("method","oemer")
        info   = (f"**Parts:** {s.get('parts',0)}  |  "
                  f"**Measures:** {s.get('measures',0)}  |  "
                  f"**Notes:** {s.get('notes',0)}  |  "
                  f"**BPM:** {s.get('bpm',120):.0f}  |  "
                  f"**Method:** {method}")
        return f"✅ {result['message']}", info
    return f"❌ {result['message']}", ""

# ── TAB 5: MUSICXML & EVENTS ──────────────────────────────────
def get_events_table():
    if not sheet_state["events"]:
        return "No events — upload and convert a sheet music file first.", None
    s = sheet_state["stats"]
    header = (f"### 🎼 MusicXML Processing\n"
              f"**Parts:** {s.get('parts',0)} | "
              f"**Measures:** {s.get('measures',0)} | "
              f"**Notes:** {s.get('notes',0)} | "
              f"**BPM:** {s.get('bpm',120):.0f}")
    rows = []
    for e in sheet_state["events"]:
        rows.append([str(e["measure"]),str(e["beat"]),str(e["time_sec"]),
                     e["note"],str(e["octave"]),str(e["duration"]),e["dur_name"]])
    return header, rows

def download_events_csv():
    if not sheet_state["events"]: return None
    import tempfile, csv
    tmp = tempfile.NamedTemporaryFile(suffix=".csv",delete=False,mode="w",newline="")
    w   = csv.DictWriter(tmp,fieldnames=["part","measure","beat","time_sec",
                                          "note","octave","duration","dur_name"])
    w.writeheader(); w.writerows(sheet_state["events"]); tmp.close()
    return tmp.name

# ── TAB 6: SCORE & MIDI ───────────────────────────────────────
def show_notation_score():
    """Render proper sheet music notation — what Archie asked for."""
    if not sheet_state["musicxml_path"]:
        return None, None, "❌ Upload and convert a sheet music file first."
    out_name = f"notation_{int(time.time())}"
    # Standard notation (verovio)
    r_notation = render_notation_score(sheet_state["musicxml_path"], out_name)
    # Piano roll (secondary)
    r_roll = render_score_image(sheet_state["musicxml_path"], out_name)
    s   = sheet_state["stats"]
    msg = (f"✅ Score visualised — "
           f"{s.get('parts',0)} parts · "
           f"{s.get('measures',0)} bars · "
           f"{s.get('notes',0)} notes")
    notation_img = r_notation.get("image_path") if r_notation["success"] else None
    roll_img     = r_roll.get("image_path")     if r_roll["success"]     else None
    return notation_img, roll_img, msg

def show_bar_range(start_bar, end_bar):
    if not sheet_state["musicxml_path"]:
        return None, None, "❌ No score loaded."
    out_name = f"bars_{int(time.time())}"
    # Notation for bar range
    r_n  = render_notation_score(sheet_state["musicxml_path"], out_name)
    r_r  = render_bar_range(sheet_state["musicxml_path"],
                            int(start_bar), int(end_bar), out_name)
    msg  = (f"✅ Bars {int(start_bar)}–{int(end_bar)}"
            if r_r["success"] else f"❌ {r_r.get('message','')}")
    return (r_n.get("image_path") if r_n["success"] else None,
            r_r.get("image_path") if r_r["success"] else None,
            msg)

def generate_midi():
    if not sheet_state["musicxml_path"]:
        return None, None, "❌ No MusicXML loaded."
    out_name  = f"midi_{int(time.time())}"
    midi_res  = musicxml_to_midi(sheet_state["musicxml_path"], out_name)
    if not midi_res["success"]:
        return None, None, f"❌ {midi_res['message']}"
    sheet_state["midi_path"] = midi_res["midi_path"]
    wav_res = midi_to_wav(midi_res["midi_path"], out_name)
    if wav_res["success"]:
        sheet_state["wav_path"] = wav_res["wav_path"]
        return (wav_res["wav_path"], midi_res["midi_path"],
                f"✅ MIDI file created successfully.")
    return None, midi_res["midi_path"], f"✅ MIDI created (WAV synthesis unavailable)"

# ── TAB 7: VISUALIZATIONS ────────────────────────────────────
def get_plot_images():
    if not PLOTS_DIR.exists(): return [], "No plots found."
    plots = sorted(PLOTS_DIR.glob("*.png"))
    if not plots: return [], "No plots. Run: python src/pipeline/pipeline.py"
    return [str(p) for p in plots], f"✓ {len(plots)} plots available"

# ── TAB 8: DASHBOARD ─────────────────────────────────────────
def get_dashboard():
    qs = model_sum.get("quality_scorer",{})
    sc = model_sum.get("skill_classifier",{})
    platform_str = {"windows":"Windows","linux":"Linux","macos":"macOS"}.get(
        "windows" if is_windows() else "linux","Linux")
    return f"""## 📊 AI Music Feedback Dashboard ({platform_str})

### 🎼 Training Datasets
| Dataset | Samples |
|---|---|
| NSynth | **{manifest.get('nsynth_samples',0)}** |
| MAESTRO v3.0.0 | **{manifest.get('maestro_samples',0)}** |
| GuitarSet | **{manifest.get('guitarset_samples',0)}** |
| Synthetic | **{manifest.get('synthetic_audio',0)}** |
| **Total** | **{manifest.get('total_audio',0)}** audio + **{manifest.get('total_logs',0)}** logs |

### 🤖 Model Performance
| Model | Performance |
|---|---|
| Quality Scorer (GBR) | MAE = {qs.get('test_mae',0):.2f} pts · R² = {qs.get('test_r2',0):.3f} |
| Skill Classifier (RF) | Accuracy = {sc.get('accuracy',0)*100:.1f}% · CV = {sc.get('cv_acc_mean',0)*100:.1f}% |

### 🎵 Current Sheet Music Session
| | |
|---|---|
| File loaded | {sheet_state['filename'] or '—'} |
| Parts | {sheet_state['stats'].get('parts','—')} |
| Measures | {sheet_state['stats'].get('measures','—')} |
| Notes | {sheet_state['stats'].get('notes','—')} |

### 🎯 Practice Sessions Logged: **{len(session_store)}**"""

def get_alignment():
    comp_path = PROCESSED_DIR/"feedback_comparisons.parquet"
    if not comp_path.exists():
        return "Run the pipeline first to generate alignment data."
    df  = pd.read_parquet(comp_path)
    col = "computed_similarity" if "computed_similarity" in df.columns else "alignment_score"
    sc  = df[col].dropna()
    hi  = (sc>0.75).sum(); mod=((sc>=0.45)&(sc<=0.75)).sum()
    lo  = (sc<0.45).sum(); n=len(sc)
    return f"""## 🔄 Feedback Alignment Distribution
| Category | Count | % |
|---|---|---|
| 🟢 HIGH (>0.75) | {hi} | {hi/n*100:.1f}% |
| 🟡 MODERATE | {mod} | {mod/n*100:.1f}% |
| 🔴 LOW (<0.45) | {lo} | {lo/n*100:.1f}% |
**Mean:** {sc.mean():.3f} · **Std:** {sc.std():.3f}"""

# ══════════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ══════════════════════════════════════════════════════════════
def build_demo():
    with gr.Blocks(title="AI Music Feedback",
                   theme=gr.themes.Soft(primary_hue="violet")) as demo:

        gr.Markdown("""
# 🎵 AI Music Feedback — Intelligent Practice Intelligence
**Audio Analysis · NLP Feedback · Sheet Music OMR · Standard Notation · MIDI**
NSynth · MAESTRO v3 · GuitarSet · oemer · verovio · music21 · sentence-transformers
---""")

        with gr.Tabs():

            # ── Tab 1 ─────────────────────────────────────
            with gr.TabItem("🎧 Audio Analysis"):
                gr.Markdown("Upload a recording → pitch accuracy, quality score, 60+ features.")
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_in = gr.Audio(label="Upload Audio (WAV/MP3)", type="filepath")
                        note_in  = gr.Dropdown(choices=NOTES, label="Target Note (optional)", value="")
                        an_btn   = gr.Button("🔍 Analyse", variant="primary")
                    with gr.Column(scale=2):
                        feat_tbl = gr.Markdown()
                        fb_txt   = gr.Textbox(label="Feedback", lines=9, interactive=False)
                        raw_json = gr.Code(label="Raw JSON", language="json", lines=10)
                an_btn.click(analyze_audio, [audio_in,note_in], [feat_tbl,raw_json,fb_txt])

            # ── Tab 2 ─────────────────────────────────────
            with gr.TabItem("🔄 Feedback Comparison"):
                gr.Markdown("Compare student self-report vs teacher feedback using NLP.")
                with gr.Row():
                    with gr.Column():
                        stu_in  = gr.Textbox(label="Student Self-Report", lines=5)
                        tch_in  = gr.Textbox(label="Teacher Feedback", lines=5)
                        with gr.Row():
                            cmp_btn = gr.Button("🔍 Compare", variant="primary")
                            rnd_btn = gr.Button("🎲 Load Sample")
                    with gr.Column():
                        cmp_out  = gr.Markdown()
                        tips_out = gr.Markdown()
                cmp_btn.click(compare_feedback,[stu_in,tch_in],[cmp_out,tips_out])
                rnd_btn.click(random_session,[],[stu_in,tch_in])

            # ── Tab 3 ─────────────────────────────────────
            with gr.TabItem("📝 Practice Logger"):
                gr.Markdown("Log a practice session and receive AI-generated feedback.")
                with gr.Row():
                    with gr.Column(scale=1):
                        piece_in = gr.Textbox(label="Piece / Exercise",
                                              placeholder="e.g. Nocturne Op.9 No.2")
                        focus_in = gr.Dropdown(choices=FOCUS, value="Dynamics control",
                                               label="Focus Area")
                        dur_in   = gr.Slider(5,120,value=30,step=5,label="Duration (min)")
                        rep_in   = gr.Textbox(label="Self-Report", lines=4,
                            placeholder="Describe what you worked on...")
                        rat_in   = gr.Slider(1,10,value=6,step=1,label="Self-Rating (1–10)")
                        rec_in   = gr.Audio(label="Optional Recording", type="filepath")
                        log_btn  = gr.Button("📤 Log Session", variant="primary")
                    with gr.Column(scale=2):
                        rep_out = gr.Textbox(label="Feedback Report",lines=22,interactive=False)
                        js_out  = gr.Code(label="Session JSON",language="json",lines=16)
                log_btn.click(log_session,
                    [piece_in,focus_in,dur_in,rep_in,rat_in,rec_in],[rep_out,js_out])

            # ── Tab 4: SHEET MUSIC UPLOAD ─────────────────
            with gr.TabItem("🎼 Sheet Music Upload"):
                gr.Markdown("""### Upload Sheet Music — PDF · PNG · JPG · JPEG
Upload your sheet music. It will be displayed back so you can confirm it's the correct source,
then converted to MusicXML for note extraction.""")
                with gr.Row():
                    with gr.Column(scale=1):
                        sheet_file = gr.File(
                            label="Upload Sheet Music PDF/PNG/JPG",
                            file_types=[".pdf",".png",".jpg",".jpeg"])
                        upload_status = gr.Markdown()
                        upload_info   = gr.Markdown()
                        gr.Markdown("---")
                        gr.Markdown("### MusicXML Conversion")
                        convert_btn    = gr.Button("🎵 Convert to MusicXML",
                                                    variant="primary", interactive=False)
                        convert_status = gr.Markdown()
                        convert_info   = gr.Markdown()
                        gr.Markdown("---")
                        gr.Markdown("""**Workflow:**
1. **Upload** — Select PDF/PNG/JPG file
2. **Convert** — Extract notes using OMR
3. **View** — See events table in next tab
4. **Score** — View notation + piano roll
5. **Play** — Generate MIDI playback""")
                    with gr.Column(scale=2):
                        gr.Markdown("### 📄 Uploaded Sheet Music (Source)")
                        sheet_preview = gr.Image(
                            label="Your uploaded sheet music — this is the source for all data",
                            type="filepath", interactive=False)

                sheet_file.change(upload_sheet_music, [sheet_file],
                                  [sheet_preview, upload_status, upload_info,
                                   convert_status, convert_btn])
                convert_btn.click(convert_to_musicxml, [],
                                  [convert_status, convert_info])

            # ── Tab 5: MUSICXML & EVENTS ──────────────────
            with gr.TabItem("📋 MusicXML & Events"):
                gr.Markdown("""### Recognised Musical Events
Clean note-by-note breakdown: Measure · Beat · Note · Octave · Duration""")
                with gr.Row():
                    extract_btn  = gr.Button("📊 Load Note Events", variant="primary")
                    download_btn = gr.Button("⬇️ Download CSV")
                events_header = gr.Markdown()
                events_table  = gr.Dataframe(
                    headers=["Measure","Beat","Time (s)","Note","Octave","Duration","Duration Name"],
                    interactive=False, wrap=True)
                download_file = gr.File(label="Download CSV", visible=False)

                gr.Markdown("---")
                gr.Markdown("### Bar Identification")
                with gr.Row():
                    start_bar = gr.Number(label="Start Bar", value=1, minimum=1, precision=0)
                    end_bar   = gr.Number(label="End Bar",   value=9, minimum=1, precision=0)
                bar_info_md = gr.Markdown()

                extract_btn.click(get_events_table, [], [events_header, events_table])
                download_btn.click(lambda: gr.update(value=download_events_csv(), visible=True),
                                   [], [download_file])
                def bar_info(s,e):
                    n=max(0,int(e)-int(s)+1)
                    return f"**Selected:** {n} bars ({int(s)} to {int(e)})"
                start_bar.change(bar_info,[start_bar,end_bar],[bar_info_md])
                end_bar.change(bar_info,[start_bar,end_bar],[bar_info_md])

            # ── Tab 6: SCORE & MIDI ───────────────────────
            with gr.TabItem("🎹 Score & MIDI"):
                gr.Markdown("""### Music Score Visualisation + MIDI Playback
**Standard notation** (what musicians read) and piano roll view.""")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Full Score")
                        viz_btn    = gr.Button("🎼 Show Score", variant="primary")
                        viz_status = gr.Markdown()

                        gr.Markdown("#### Bar Range View")
                        with gr.Row():
                            v_start = gr.Number(label="Start Bar",value=1,minimum=1,precision=0)
                            v_end   = gr.Number(label="End Bar",  value=9,minimum=1,precision=0)
                        range_btn    = gr.Button("🔍 Show Bar Range", variant="secondary")
                        range_status = gr.Markdown()

                        gr.Markdown("#### MIDI Playback")
                        midi_btn    = gr.Button("▶️ Generate MIDI", variant="primary")
                        midi_status = gr.Markdown()
                        audio_player= gr.Audio(label="MIDI Playback",type="filepath",
                                               interactive=False)
                        midi_file   = gr.File(label="Download MIDI", visible=False)

                    with gr.Column(scale=2):
                        gr.Markdown("#### 🎵 Standard Music Notation (Staff View)")
                        notation_img = gr.Image(
                            label="Standard Notation — treble/bass clef as printed",
                            type="filepath", interactive=False)
                        gr.Markdown("#### 🎹 Piano Roll View")
                        roll_img = gr.Image(
                            label="Piano Roll — notes on timeline",
                            type="filepath", interactive=False)

                def do_score():
                    n,r,m = show_notation_score()
                    return n,r,m
                def do_range(s,e):
                    n,r,m = show_bar_range(int(s),int(e))
                    return n,r,m
                def do_midi():
                    wav,mid,msg = generate_midi()
                    return wav, gr.update(value=mid,visible=mid is not None), msg

                viz_btn.click(do_score,[],[notation_img,roll_img,viz_status])
                range_btn.click(do_range,[v_start,v_end],[notation_img,roll_img,range_status])
                midi_btn.click(do_midi,[],[audio_player,midi_file,midi_status])

            # ── Tab 7 ─────────────────────────────────────
            with gr.TabItem("📊 Visualizations"):
                gr.Markdown("### All 10 training and testing charts from the pipeline.")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Load Plots", variant="primary")
                    plot_status = gr.Markdown()
                gallery = gr.Gallery(label="Charts",columns=2,height=750,
                                     object_fit="contain")
                def load_plots():
                    imgs,status=get_plot_images(); return imgs,status
                refresh_btn.click(load_plots,[],[gallery,plot_status])
                demo.load(load_plots,[],[gallery,plot_status])

            # ── Tab 8 ─────────────────────────────────────
            with gr.TabItem("🖥️ Dashboard"):
                gr.Markdown("### Live dataset, model, and session stats.")
                ref_btn = gr.Button("🔄 Refresh")
                with gr.Row():
                    with gr.Column(): dash_md = gr.Markdown()
                    with gr.Column(): aln_md  = gr.Markdown()
                def refresh():
                    return get_dashboard(), get_alignment()
                ref_btn.click(refresh,[],[dash_md,aln_md])
                demo.load(refresh,[],[dash_md,aln_md])

        gr.Markdown("---\n*AI Music Feedback · Virtuoso Learning Ltd · "
                    "Archie Maclennan · archie@virtuosoapp.io · 2026*")
    return demo

if __name__ == "__main__":
    demo = build_demo()
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        allowed_paths=[
            str(PLOTS_DIR),
            str(OUTPUTS_DIR),
            str(SHEET_DIR),
            str(SHEET_DIR/"scores"),
            str(SHEET_DIR/"midi"),
            str(SHEET_DIR/"uploads"),
        ],
    )

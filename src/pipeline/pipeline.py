"""
AI Music Feedback — Pipeline Orchestrator (Cross-Platform)
Runs on Linux (Kali) and Windows without modification.
"""
import sys, json, time, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

# Bootstrap cross-platform paths
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
_root = _src.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from paths import ROOT, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR


class AIMusFeedbackPipeline:
    def __init__(self):
        self.audio_df = None; self.logs_df = None
        self.scorer   = None; self.clf     = None
        self.stats    = {}

    def step1_load_datasets(self):
        print("\n" + "─"*62)
        print("STEP 1 — Loading Datasets")
        print("─"*62)
        from data_loader import build_training_dataset
        data = build_training_dataset()
        self.audio_df = data["audio_df"]
        self.logs_df  = data["logs_df"]
        self.stats    = data["stats"]
        print(f"\n  Audio features : {self.audio_df.shape}")
        print(f"  Practice logs  : {self.logs_df.shape}")

    def step2_train_models(self):
        print("\n" + "─"*62)
        print("STEP 2 — Training ML Models")
        print("─"*62)
        from ml_model.model import train_all_models
        results = train_all_models(self.audio_df)
        self.scorer = results["scorer"]
        self.clf    = results["classifier"]

    def step3_visualizations(self):
        print("\n" + "─"*62)
        print("STEP 3 — Generating All Visualizations")
        print("─"*62)
        from visualizations.charts import generate_all_plots
        generate_all_plots(self.audio_df, self.logs_df,
                           self.stats, self.scorer, self.clf)

    def step4_feedback_comparator(self):
        print("\n" + "─"*62)
        print("STEP 4 — Feedback Comparator")
        print("─"*62)
        from feedback.feedback import FeedbackComparator
        comp   = FeedbackComparator()
        sample = self.logs_df.dropna(
            subset=["student_self_report","teacher_feedback"]
        ).sample(min(80, len(self.logs_df)), random_state=42)
        scores = comp.similarity_batch(sample)
        sample = sample.copy()
        sample["computed_similarity"] = scores
        sample.to_parquet(PROCESSED_DIR/"feedback_comparisons.parquet", index=False)
        print(f"  Mean alignment : {scores.mean():.3f}")
        print(f"  HIGH (>0.75)   : {(scores>0.75).sum()}")
        print(f"  LOW  (<0.45)   : {(scores<0.45).sum()}")
        self.stats.update({
            "mean_alignment": float(scores.mean()),
            "high_alignment": int((scores>0.75).sum()),
            "low_alignment":  int((scores<0.45).sum()),
        })

    def step5_sample_report(self):
        print("\n" + "─"*62)
        print("STEP 5 — Generating Sample Output")
        print("─"*62)
        from audio_features.extractor import AudioFeatureExtractor
        from feedback.feedback import FeedbackGenerator, FeedbackComparator
        import numpy as np

        ext  = AudioFeatureExtractor()
        gen  = FeedbackGenerator()
        comp = FeedbackComparator()

        sr = 22050
        t  = np.linspace(0, 2.0, sr*2)
        y  = (0.5*np.sin(2*np.pi*440*t) +
              0.08*np.random.default_rng(1).normal(size=sr*2))
        feat = ext.extract(y, sr=sr)

        logs_with_text = self.logs_df.dropna(
            subset=["student_self_report","teacher_feedback"])
        row = logs_with_text.iloc[0] if len(logs_with_text) else self.logs_df.iloc[0]

        session = {
            "piece":        str(row.get("piece","Unknown")),
            "focus_area":   str(row.get("focus_area","Dynamics")),
            "duration_min": int(row.get("duration_min",30) or 30),
            "student_rating": int(row.get("student_rating",7) or 7),
            "teacher_rating": int(row.get("teacher_rating",7) or 7),
            "student_self_report": str(row.get("student_self_report","")),
            "teacher_feedback":    str(row.get("teacher_feedback","")),
        }
        comparison = comp.compare_session(
            session["student_self_report"], session["teacher_feedback"])
        feedback = gen.generate(
            audio_features=feat, session_data=session,
            comparison_result=comparison)
        report = gen.format_for_display(feedback)
        print(report)

        (OUTPUTS_DIR/"sample_feedback_report.txt").write_text(
            report, encoding="utf-8")
        with open(OUTPUTS_DIR/"sample_feedback.json","w",encoding="utf-8") as f:
            json.dump({**feedback,"audio_features":feat,"comparison":comparison},
                      f, indent=2, default=str)
        print("  ✓ Sample report saved")

    def step6_save_summary(self, elapsed:float):
        summary = {
            "pipeline_version": "AI_MUSIC_FEEDBACK_v2.0",
            "platform": sys.platform,
            "elapsed_seconds": round(elapsed,1),
            "datasets": self.stats,
            "models": {
                "quality_scorer":  {k:v for k,v in self.scorer.metrics.items()
                                    if not isinstance(v,list)},
                "skill_classifier":{k:v for k,v in self.clf.metrics.items()
                                    if not isinstance(v,list)},
            },
        }
        (OUTPUTS_DIR/"pipeline_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("\n  ✓ pipeline_summary.json saved")

    def run(self):
        print("\n" + "="*62)
        print("  AI MUSIC FEEDBACK — FULL PIPELINE")
        print(f"  Platform: {sys.platform}")
        print("="*62)
        t0 = time.time()
        self.step1_load_datasets()
        self.step2_train_models()
        self.step3_visualizations()
        self.step4_feedback_comparator()
        self.step5_sample_report()
        elapsed = time.time() - t0
        self.step6_save_summary(elapsed)
        print(f"\n{'='*62}")
        print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
        print(f"  Plots   → outputs/plots/  (10 PNG files)")
        print(f"  Report  → outputs/sample_feedback_report.txt")
        print(f"  Summary → outputs/pipeline_summary.json")
        print(f"{'='*62}")
        return self


if __name__ == "__main__":
    AIMusFeedbackPipeline().run()

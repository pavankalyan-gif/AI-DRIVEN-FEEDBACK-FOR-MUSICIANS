"""AI Music Feedback — Test Suite (Cross-Platform, 32 tests)"""
import sys, unittest
import numpy as np, pandas as pd
from pathlib import Path

# Bootstrap cross-platform paths
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
_root = _src.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class TestExtractor(unittest.TestCase):
    def setUp(self):
        from audio_features.extractor import AudioFeatureExtractor
        self.ext = AudioFeatureExtractor()
        sr = 22050; t = np.linspace(0,1,sr)
        self.y  = (0.5*np.sin(2*np.pi*440*t)).astype(np.float32)
        self.sr = sr

    def test_extract_returns_dict(self):
        self.assertIsInstance(self.ext.extract(self.y, sr=self.sr), dict)
    def test_mfcc_present(self):
        f = self.ext.extract(self.y, sr=self.sr)
        for j in range(13): self.assertIn(f"mfcc_{j}_mean", f)
    def test_chroma_present(self):
        f = self.ext.extract(self.y, sr=self.sr)
        for j in range(12): self.assertIn(f"chroma_{j}", f)
    def test_spectral_present(self):
        f = self.ext.extract(self.y, sr=self.sr)
        for k in ["spectral_centroid","spectral_rolloff","spectral_bandwidth"]:
            self.assertIn(k, f)
    def test_quality_range(self):
        f = self.ext.extract(self.y, sr=self.sr)
        self.assertGreaterEqual(f["quality_score"], 0)
        self.assertLessEqual(f["quality_score"], 100)
    def test_pitch_present(self):
        f = self.ext.extract(self.y, sr=self.sr)
        self.assertIn("pitch_accuracy", f); self.assertIn("detected_note", f)
    def test_feedback_string(self):
        f = self.ext.extract(self.y, sr=self.sr)
        s = self.ext.generate_feedback_string(f, target_note="A4")
        self.assertIsInstance(s, str); self.assertGreater(len(s), 10)
    def test_batch_extraction(self):
        import soundfile as sf, tempfile, os
        paths = []
        for freq in [261.63, 329.63, 392.0]:
            sig = (0.5*np.sin(2*np.pi*freq*np.linspace(0,1,22050))).astype(np.float32)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, sig, 22050); paths.append(tmp.name); tmp.close()
        try:
            df = self.ext.extract_batch(paths, verbose=False)
            self.assertEqual(len(df), 3)
        finally:
            for p in paths: os.unlink(p)
    def test_rhythm_features(self):
        f = self.ext.extract(self.y, sr=self.sr)
        self.assertIn("zero_crossing_rate", f); self.assertIn("tempo", f)
    def test_loudness_consistency(self):
        f = self.ext.extract(self.y, sr=self.sr)
        self.assertIn("loudness_consistency", f)
        self.assertGreaterEqual(f["loudness_consistency"], 0)
        self.assertLessEqual(f["loudness_consistency"], 1)


class TestMLModels(unittest.TestCase):
    def setUp(self):
        from data_loader import _synth_audio
        self.df = _synth_audio(200, seed=7)

    def test_quality_scorer_trains(self):
        from ml_model.model import QualityScorer
        s = QualityScorer(); m = s.train(self.df)
        self.assertIn("test_mae", m); self.assertLess(m["test_mae"], 30)
    def test_quality_scorer_predicts(self):
        from ml_model.model import QualityScorer
        s = QualityScorer(); s.train(self.df)
        p = s.predict(self.df.iloc[0].to_dict())
        self.assertGreaterEqual(p, 0); self.assertLessEqual(p, 100)
    def test_quality_scorer_r2(self):
        from ml_model.model import QualityScorer
        s = QualityScorer(); m = s.train(self.df)
        self.assertGreater(m["test_r2"], 0.5)
    def test_skill_classifier_trains(self):
        from ml_model.model import SkillLevelClassifier
        c = SkillLevelClassifier(); m = c.train(self.df)
        self.assertGreater(m["accuracy"], 0.3)
    def test_skill_classifier_predicts(self):
        from ml_model.model import SkillLevelClassifier
        c = SkillLevelClassifier(); c.train(self.df)
        label, proba = c.predict(self.df.iloc[0].to_dict())
        self.assertIn(label, ["Beginner","Intermediate","Advanced"])
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=3)
    def test_feature_importance(self):
        from ml_model.model import QualityScorer
        s = QualityScorer(); s.train(self.df)
        fi = s.feature_importance()
        self.assertIsInstance(fi, pd.DataFrame)
        self.assertIn("feature", fi.columns); self.assertIn("importance", fi.columns)
    def test_cv_metrics_present(self):
        from ml_model.model import QualityScorer
        s = QualityScorer(); m = s.train(self.df)
        self.assertIn("cv_mae_mean", m); self.assertIn("cv_mae_std", m)
    def test_confusion_matrix_shape(self):
        from ml_model.model import SkillLevelClassifier
        c = SkillLevelClassifier(); m = c.train(self.df)
        cm = np.array(m["confusion_matrix"])
        self.assertEqual(cm.shape, (3,3))


class TestFeedbackComparator(unittest.TestCase):
    def setUp(self):
        from feedback.feedback import FeedbackComparator
        self.comp = FeedbackComparator()

    def test_returns_float(self):
        s = self.comp.similarity("I practised today","Good session")
        self.assertIsInstance(s, float)
        self.assertGreaterEqual(s, 0); self.assertLessEqual(s, 1)
    def test_identical_high(self):
        t = "I worked on dynamics and tempo"
        self.assertGreater(self.comp.similarity(t, t), 0.8)
    def test_compare_session_dict(self):
        r = self.comp.compare_session("Practised 30 min","Good effort, watch tempo")
        self.assertIn("similarity_score", r); self.assertIn("alignment", r)
        self.assertIn(r["alignment"], ["HIGH","MODERATE","LOW"])
    def test_batch_similarity(self):
        from data_loader import _synth_logs
        df = _synth_logs(10, seed=3)
        sc = self.comp.similarity_batch(df)
        self.assertEqual(len(sc), 10)
        self.assertTrue(all(0 <= s <= 1 for s in sc))
    def test_word_count_present(self):
        r = self.comp.compare_session("Short report here","Teacher says this")
        self.assertIn("student_word_count", r)
        self.assertIn("teacher_word_count", r)


class TestFeedbackGenerator(unittest.TestCase):
    def setUp(self):
        from feedback.feedback import FeedbackGenerator
        self.gen  = FeedbackGenerator()
        self.af   = {"pitch_accuracy":0.75,"quality_score":70.0,
                     "detected_note":"A4","pitch_cents_deviation":-10.0,
                     "loudness_consistency":0.65,"tempo":88.0}
        self.sess = {"piece":"Für Elise","focus_area":"Tempo",
                     "duration_min":25,"student_rating":7,"teacher_rating":6}
        self.cmp  = {"similarity_score":0.6,"alignment":"MODERATE",
                     "interpretation":"Some overlap.","student_word_count":12,
                     "teacher_word_count":10}

    def test_returns_dict(self):
        self.assertIsInstance(self.gen.generate(self.af,self.sess,self.cmp), dict)
    def test_required_keys(self):
        fb = self.gen.generate(self.af,self.sess,self.cmp)
        for k in ["summary","audio_feedback","alignment_feedback","actionable_tips","flags"]:
            self.assertIn(k, fb)
    def test_tips_not_empty(self):
        fb = self.gen.generate(self.af,self.sess,self.cmp)
        self.assertGreater(len(fb["actionable_tips"]), 0)
    def test_format_display(self):
        fb = self.gen.generate(self.af,self.sess,self.cmp)
        s  = self.gen.format_for_display(fb)
        self.assertIsInstance(s, str); self.assertIn("FEEDBACK REPORT", s)
    def test_no_teacher_rating(self):
        sess2 = {**self.sess,"teacher_rating":None}
        fb = self.gen.generate(self.af, sess2)
        self.assertIsInstance(fb, dict)
    def test_audio_only(self):
        self.assertIsInstance(self.gen.generate(audio_features=self.af), dict)


class TestDataLoader(unittest.TestCase):
    def test_synth_audio(self):
        from data_loader import _synth_audio
        df = _synth_audio(100, seed=1)
        self.assertEqual(len(df), 100)
        self.assertIn("quality_score", df.columns)
    def test_synth_logs(self):
        from data_loader import _synth_logs
        df = _synth_logs(50, seed=1)
        self.assertEqual(len(df), 50)
        self.assertIn("student_self_report", df.columns)
    def test_synth_skill_balance(self):
        from data_loader import _synth_audio
        df = _synth_audio(300, seed=5)
        vc = df["skill_level"].value_counts()
        for s in ["Beginner","Intermediate","Advanced"]:
            self.assertIn(s, vc.index)

class TestPaths(unittest.TestCase):
    def test_root_exists(self):
        from paths import ROOT
        self.assertTrue(ROOT.exists())
    def test_src_in_root(self):
        from paths import ROOT, SRC_DIR
        self.assertTrue(SRC_DIR.exists())
    def test_datasets_in_root(self):
        from paths import DATASETS_DIR
        self.assertTrue(DATASETS_DIR.exists())


if __name__ == "__main__":
    print("\n" + "="*62)
    print("  AI MUSIC FEEDBACK — TEST SUITE (Cross-Platform)")
    print("="*62)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestExtractor, TestMLModels, TestFeedbackComparator,
                TestFeedbackGenerator, TestDataLoader, TestPaths]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    unittest.TextTestRunner(verbosity=2).run(suite)

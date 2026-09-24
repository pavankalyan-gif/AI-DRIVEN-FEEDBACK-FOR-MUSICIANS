"""
Virtuoso AI — Feedback Module
Compares student self-report vs teacher feedback using sentence transformers.
Generates structured, actionable feedback for students.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# SENTENCE SIMILARITY ENGINE
# ─────────────────────────────────────────────────────────────────

class FeedbackComparator:
    """
    Compares student self-reports with teacher feedback using
    sentence-transformer embeddings and cosine similarity.
    Falls back to keyword matching if model unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading sentence transformer: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"  ✓ Sentence transformer loaded")
        except Exception as e:
            print(f"  ⚠ Sentence transformer unavailable ({e}), using keyword fallback")
            self.model = None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword overlap fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.5
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        # Jaccard + length similarity bonus
        jaccard = intersection / union
        len_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
        return (jaccard * 0.7 + len_ratio * 0.3)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity score (0–1) between two texts."""
        if self.model is not None:
            emb1 = self.model.encode([text1])[0]
            emb2 = self.model.encode([text2])[0]
            return self._cosine_similarity(emb1, emb2)
        else:
            return self._keyword_similarity(text1, text2)

    def similarity_batch(self, df: pd.DataFrame,
                          col1: str = "student_self_report",
                          col2: str = "teacher_feedback") -> np.ndarray:
        """Compute similarity for all rows in a DataFrame."""
        if self.model is not None:
            texts1 = df[col1].fillna("").tolist()
            texts2 = df[col2].fillna("").tolist()
            emb1 = self.model.encode(texts1, batch_size=32, show_progress_bar=True)
            emb2 = self.model.encode(texts2, batch_size=32, show_progress_bar=True)
            scores = []
            for a, b in zip(emb1, emb2):
                scores.append(self._cosine_similarity(a, b))
            return np.array(scores)
        else:
            return np.array([
                self._keyword_similarity(r[col1], r[col2])
                for _, r in df.iterrows()
            ])

    def compare_session(self, student_report: str, teacher_feedback: str) -> dict:
        """Full comparison of one session."""
        score = self.similarity(student_report, teacher_feedback)

        if score >= 0.75:
            alignment = "HIGH"
            interpretation = "Student and teacher are well-aligned. Student has good self-awareness."
        elif score >= 0.45:
            alignment = "MODERATE"
            interpretation = "Some overlap detected. Teacher identified areas student may have missed."
        else:
            alignment = "LOW"
            interpretation = "Significant gap between self-report and teacher feedback. Student may be over- or under-estimating progress."

        return {
            "similarity_score": round(score, 4),
            "alignment": alignment,
            "interpretation": interpretation,
            "student_word_count": len(student_report.split()),
            "teacher_word_count": len(teacher_feedback.split()),
        }


# ─────────────────────────────────────────────────────────────────
# ACTIONABLE FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────────

class FeedbackGenerator:
    """
    Generates structured, actionable feedback for students
    combining audio analysis results and session data.
    """

    PRACTICE_TIPS = {
        "pitch": [
            "Record yourself and listen back critically — your ear improves with active listening.",
            "Use a tuner app while practising to get real-time pitch feedback.",
            "Slow down difficult passages to ensure pitch accuracy before building speed.",
            "Sing the note before playing it to engage your inner hearing.",
        ],
        "dynamics": [
            "Practise the same passage at three levels: pp, mf, and ff — then blend.",
            "Mark dynamic instructions in your score and actively think about them while playing.",
            "Record yourself — dynamics often feel bigger than they sound to others.",
            "Isolate one hand at a time and focus purely on tonal control.",
        ],
        "tempo": [
            "Use a metronome consistently — start at 60% of target tempo.",
            "Identify the 'problem bar' and loop it with a gradual tempo increase.",
            "Practise with a drone or backing track to internalise the pulse.",
            "Count aloud while playing — it forces awareness of rhythmic placement.",
        ],
        "general": [
            "Set one specific achievable goal per practice session.",
            "End each session by noting what improved and what to tackle next time.",
            "Short, focused 20-minute sessions often outperform long, unfocused ones.",
            "Reflect before and after each session: what did I want to achieve?",
        ],
    }

    def generate(self,
                 audio_features: Optional[dict] = None,
                 session_data: Optional[dict] = None,
                 comparison_result: Optional[dict] = None) -> dict:
        """
        Generate structured feedback from available data.
        Returns a JSON-serialisable feedback object.
        """
        feedback = {
            "summary": "",
            "audio_feedback": "",
            "alignment_feedback": "",
            "actionable_tips": [],
            "flags": [],
            "overall_rating": 5,
        }

        # ── Audio analysis feedback ──────────────────────
        if audio_features:
            audio_lines = []
            score = audio_features.get("quality_score", 50)
            feedback["overall_rating"] = max(1, min(10, int(score / 10)))

            pitch_acc = audio_features.get("pitch_accuracy", 0.5)
            if pitch_acc >= 0.8:
                audio_lines.append(f"✓ Excellent pitch accuracy ({pitch_acc*100:.0f}%)")
            elif pitch_acc >= 0.6:
                audio_lines.append(f"~ Pitch accuracy is moderate ({pitch_acc*100:.0f}%) — keep working on intonation")
                feedback["actionable_tips"].extend(np.random.default_rng(42).choice(
                    self.PRACTICE_TIPS["pitch"], size=2, replace=False).tolist())
            else:
                audio_lines.append(f"✗ Pitch accuracy needs improvement ({pitch_acc*100:.0f}%)")
                feedback["flags"].append("PITCH_CONCERN")
                feedback["actionable_tips"].extend(self.PRACTICE_TIPS["pitch"][:2])

            loudness = audio_features.get("loudness_consistency", 0.5)
            if loudness > 0.75:
                audio_lines.append("✓ Dynamics are well-controlled")
            elif loudness > 0.5:
                audio_lines.append("~ Dynamic control could be more consistent")
                feedback["actionable_tips"].append(self.PRACTICE_TIPS["dynamics"][0])
            else:
                audio_lines.append("✗ Dynamics need significant work")
                feedback["flags"].append("DYNAMICS_CONCERN")
                feedback["actionable_tips"].extend(self.PRACTICE_TIPS["dynamics"][:2])

            detected = audio_features.get("detected_note", "N/A")
            cents = audio_features.get("pitch_cents_deviation", 0.0)
            audio_lines.append(f"Detected note: {detected} | Deviation: {cents:+.1f} cents")

            tempo = audio_features.get("tempo", 0)
            if tempo > 0:
                audio_lines.append(f"Detected tempo: {tempo:.0f} BPM")

            audio_lines.append(f"Overall quality score: {score:.1f}/100")
            feedback["audio_feedback"] = "\n".join(audio_lines)

        # ── Alignment/comparison feedback ─────────────────
        if comparison_result:
            alignment = comparison_result.get("alignment", "MODERATE")
            score_val = comparison_result.get("similarity_score", 0.5)
            interpretation = comparison_result.get("interpretation", "")

            feedback["alignment_feedback"] = (
                f"Self-report vs Teacher alignment: {alignment} ({score_val:.0%})\n"
                f"{interpretation}"
            )

            if alignment == "LOW":
                s_rat = (session_data or {}).get("student_rating", 5) or 5
                t_rat = (session_data or {}).get("teacher_rating", None)
                if t_rat is not None:
                    flag = "OVERCONFIDENCE" if int(s_rat) > int(t_rat) else "UNDERESTIMATING"
                else:
                    flag = "UNDERESTIMATING"
                feedback["flags"].append(flag)
                feedback["actionable_tips"].extend(self.PRACTICE_TIPS["general"][:2])

        # ── Session data feedback ─────────────────────────
        if session_data:
            duration = session_data.get("duration_min", 0)
            student_rating = int(session_data.get("student_rating", 5) or 5)
            teacher_rating_raw = session_data.get("teacher_rating", None)
            teacher_rating = int(teacher_rating_raw) if teacher_rating_raw is not None else None
            piece = session_data.get("piece", "")
            focus = session_data.get("focus_area", "")

            session_lines = [f"Session: {piece}" if piece else ""]
            if focus:
                session_lines.append(f"Focus area: {focus}")
            if duration:
                session_lines.append(f"Duration: {duration} min")

            if teacher_rating is not None:
                session_lines.append(
                    f"Self-rating: {student_rating}/10 | Teacher rating: {teacher_rating}/10"
                )
                rating_gap = abs(student_rating - (teacher_rating or student_rating))
                if rating_gap > 3:
                    session_lines.append(
                        f"⚠ Large rating gap ({rating_gap} points) — discuss expectations with teacher"
                    )
                    feedback["flags"].append(f"RATING_GAP_{rating_gap}")
            else:
                session_lines.append(f"Self-rating: {student_rating}/10")

            feedback["session_summary"] = "\n".join([l for l in session_lines if l])

        # ── Summary ──────────────────────────────────────
        if not feedback["actionable_tips"]:
            feedback["actionable_tips"] = self.PRACTICE_TIPS["general"][:2]

        # Deduplicate tips
        seen = set()
        unique_tips = []
        for tip in feedback["actionable_tips"]:
            if tip not in seen:
                seen.add(tip)
                unique_tips.append(tip)
        feedback["actionable_tips"] = unique_tips[:4]

        score_display = audio_features.get("quality_score", 50) if audio_features else None
        rating = feedback["overall_rating"]

        if rating >= 8:
            mood = "Excellent session"
        elif rating >= 6:
            mood = "Good progress"
        elif rating >= 4:
            mood = "Developing"
        else:
            mood = "Needs focused work"

        feedback["summary"] = f"{mood} — {len(feedback['flags'])} flag(s) detected"
        return feedback

    def format_for_display(self, feedback: dict) -> str:
        """Format feedback dict as readable string."""
        lines = ["=" * 50, "  VIRTUOSO FEEDBACK REPORT", "=" * 50]

        lines.append(f"\n📊 Summary: {feedback.get('summary', 'N/A')}")
        lines.append(f"⭐ Overall Rating: {feedback.get('overall_rating', 5)}/10")

        if feedback.get("audio_feedback"):
            lines.append("\n🎵 Audio Analysis:")
            lines.append(feedback["audio_feedback"])

        if feedback.get("alignment_feedback"):
            lines.append("\n🔄 Alignment:")
            lines.append(feedback["alignment_feedback"])

        if feedback.get("session_summary"):
            lines.append("\n📅 Session:")
            lines.append(feedback["session_summary"])

        if feedback.get("actionable_tips"):
            lines.append("\n💡 Actionable Tips:")
            for i, tip in enumerate(feedback["actionable_tips"], 1):
                lines.append(f"  {i}. {tip}")

        if feedback.get("flags"):
            lines.append(f"\n🚩 Flags: {', '.join(feedback['flags'])}")

        lines.append("=" * 50)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# VIRTUOSO SESSION SCHEMA
# ─────────────────────────────────────────────────────────────────

VIRTUOSO_SESSION_SCHEMA = {
    "session_id": "string — unique identifier (UUID)",
    "student_id": "string — student identifier",
    "teacher_id": "string — teacher identifier (optional)",
    "piece": "string — name of piece being practised",
    "focus_area": "string — what the student focused on",
    "duration_min": "int — practice duration in minutes",
    "practice_date": "string — ISO date YYYY-MM-DD",
    "student_self_report": "string — free text from student",
    "student_rating": "int — self-rating 1-10",
    "audio_file_path": "string — path to recorded audio (optional)",
    "audio_features": {
        "pitch_accuracy": "float 0-1",
        "quality_score": "float 0-100",
        "tempo": "float BPM",
        "detected_note": "string",
        "loudness_consistency": "float 0-1",
    },
    "teacher_feedback": "string — teacher free text (optional)",
    "teacher_rating": "int — teacher rating 1-10 (optional)",
    "alignment_score": "float 0-1 — auto-computed",
    "ml_skill_level": "string — Beginner/Intermediate/Advanced",
    "ml_quality_score": "float 0-100",
    "feedback_report": "dict — generated feedback object",
    "flags": "list[string] — auto-generated flags",
    "created_at": "string — ISO timestamp",
}


if __name__ == "__main__":
    print("Testing FeedbackComparator...")
    comp = FeedbackComparator()
    result = comp.compare_session(
        "I practised for 30 minutes, felt it went well, tempo was a bit fast",
        "Good effort today. Tempo was rushed in section B. Watch the dynamics."
    )
    print(json.dumps(result, indent=2))

    print("\nTesting FeedbackGenerator...")
    gen = FeedbackGenerator()
    fake_audio = {
        "pitch_accuracy": 0.72, "quality_score": 65.0,
        "detected_note": "A4", "pitch_cents_deviation": -15.0,
        "loudness_consistency": 0.6, "tempo": 88.0
    }
    fake_session = {
        "piece": "Nocturne Op.9 No.2", "focus_area": "Dynamics",
        "duration_min": 30, "student_rating": 8, "teacher_rating": 6
    }
    feedback = gen.generate(
        audio_features=fake_audio,
        session_data=fake_session,
        comparison_result=result
    )
    print(gen.format_for_display(feedback))

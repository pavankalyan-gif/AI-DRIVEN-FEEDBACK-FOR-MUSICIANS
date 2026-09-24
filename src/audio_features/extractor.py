"""
Virtuoso AI — Audio Feature Extractor
Extracts MFCC, Chroma, Spectral, Pitch and Rhythm features from audio files.
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# TARGET NOTE FREQUENCIES (C2 – B7)
# ─────────────────────────────────────────────────────────────────
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = range(2, 8)
NOTE_FREQ_MAP = {}
for octave in OCTAVES:
    for semitone, name in enumerate(NOTE_NAMES):
        midi = 12 * (octave + 1) + semitone
        freq = 440.0 * (2 ** ((midi - 69) / 12))
        NOTE_FREQ_MAP[f"{name}{octave}"] = freq


class AudioFeatureExtractor:
    """
    Extracts a rich feature vector from an audio file or numpy array.
    Features: MFCCs (13), Chroma (12), Spectral (4), ZCR, Tempo,
              Pitch Assessment, Energy/Dynamics, Rhythm/Beat.
    """

    def __init__(self, sr: int = 22050, n_mfcc: int = 13, hop_length: int = 512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    # ──────────────────────────────────────────
    # LOAD AUDIO
    # ──────────────────────────────────────────

    def load(self, path: Union[str, Path]) -> tuple:
        """Load audio from file path, returning (y, sr)."""
        path = str(path)
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        return y, sr

    def load_array(self, array: np.ndarray, sr: int = None) -> tuple:
        """Use a pre-loaded numpy array as audio signal."""
        sr = sr or self.sr
        if sr != self.sr:
            array = librosa.resample(array, orig_sr=sr, target_sr=self.sr)
        return array, self.sr

    # ──────────────────────────────────────────
    # FEATURE BLOCKS
    # ──────────────────────────────────────────

    def extract_mfcc(self, y: np.ndarray) -> dict:
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                                      hop_length=self.hop_length)
        features = {}
        for i in range(self.n_mfcc):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
        return features

    def extract_chroma(self, y: np.ndarray) -> dict:
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=self.hop_length)
        return {f"chroma_{i}": float(np.mean(chroma[i])) for i in range(12)}

    def extract_spectral(self, y: np.ndarray) -> dict:
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=self.hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, hop_length=self.hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, hop_length=self.hop_length)
        return {
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_rolloff": float(np.mean(rolloff)),
            "spectral_bandwidth": float(np.mean(bandwidth)),
            "spectral_contrast_mean": float(np.mean(contrast)),
        }

    def extract_rhythm(self, y: np.ndarray) -> dict:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        # Handle both scalar and array returns across librosa versions
        tempo_val = float(np.squeeze(np.atleast_1d(tempo_raw))[()]) if hasattr(tempo_raw, '__len__') else float(tempo_raw)
        if np.isnan(tempo_val) or np.isinf(tempo_val):
            tempo_val = 0.0
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        # Loudness consistency (inverse of std/mean)
        rms_vals = rms[0]
        loudness_consistency = float(1.0 - (np.std(rms_vals) / (np.mean(rms_vals) + 1e-6)))
        loudness_consistency = float(np.clip(loudness_consistency, 0, 1))
        return {
            "zero_crossing_rate": float(np.mean(zcr)),
            "tempo": tempo_val,
            "rms_energy_mean": float(np.mean(rms)),
            "rms_energy_std": float(np.std(rms)),
            "loudness_consistency": loudness_consistency,
        }

    def extract_pitch(self, y: np.ndarray) -> dict:
        """
        Estimate pitch accuracy via pyin.
        Returns: detected_note, target_note, in_tune_flag, accuracy_score
        """
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr, hop_length=self.hop_length
            )
            voiced_f0 = f0[voiced_flag]

            if len(voiced_f0) == 0:
                return {
                    "pitch_accuracy": 0.5,
                    "detected_note": "N/A",
                    "pitch_in_tune": False,
                    "note_detection_score": 0.5,
                    "pitch_cents_deviation": 0.0,
                }

            median_f0 = float(np.nanmedian(voiced_f0))

            # Closest note
            best_note = min(NOTE_FREQ_MAP.items(), key=lambda x: abs(x[1] - median_f0))
            best_name, best_freq = best_note

            # Cents deviation
            if median_f0 > 0 and best_freq > 0:
                cents_dev = 1200 * np.log2(median_f0 / best_freq)
            else:
                cents_dev = 0.0

            in_tune = abs(cents_dev) <= 50  # within half semitone
            accuracy = float(np.clip(1.0 - abs(cents_dev) / 100.0, 0, 1))

            return {
                "pitch_accuracy": accuracy,
                "detected_note": best_name,
                "pitch_in_tune": bool(in_tune),
                "note_detection_score": float(np.mean(voiced_prob[voiced_flag])),
                "pitch_cents_deviation": float(cents_dev),
            }
        except Exception as e:
            return {
                "pitch_accuracy": 0.5,
                "detected_note": "N/A",
                "pitch_in_tune": False,
                "note_detection_score": 0.5,
                "pitch_cents_deviation": 0.0,
            }

    def compute_quality_score(self, features: dict) -> float:
        """
        Aggregate quality score (0–100) from extracted features.
        Weighted: pitch (40%), dynamics (20%), rhythm (20%), spectral (20%)
        """
        pitch = features.get("pitch_accuracy", 0.5) * 40
        dynamics = features.get("loudness_consistency", 0.5) * 20
        rhythm_score = min(features.get("tempo", 0) / 200, 1.0) * 20
        spectral = min(features.get("spectral_centroid", 1000) / 4000, 1.0) * 20
        return round(pitch + dynamics + rhythm_score + spectral, 2)

    # ──────────────────────────────────────────
    # MAIN EXTRACT METHOD
    # ──────────────────────────────────────────

    def extract(self, source: Union[str, Path, np.ndarray], sr: int = None) -> dict:
        """Extract all features from a file path or numpy array."""
        if isinstance(source, (str, Path)):
            y, _ = self.load(source)
            source_label = str(Path(source).name)
        else:
            y, _ = self.load_array(source, sr)
            source_label = "array_input"

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        features = {"source": source_label, "duration_sec": float(len(y) / self.sr)}
        features.update(self.extract_mfcc(y))
        features.update(self.extract_chroma(y))
        features.update(self.extract_spectral(y))
        features.update(self.extract_rhythm(y))
        features.update(self.extract_pitch(y))
        features["quality_score"] = self.compute_quality_score(features)

        return features

    def extract_batch(self, paths: list, verbose: bool = True) -> pd.DataFrame:
        """Extract features from a list of audio file paths."""
        results = []
        for i, p in enumerate(paths):
            try:
                feat = self.extract(p)
                results.append(feat)
                if verbose:
                    print(f"  [{i+1}/{len(paths)}] {Path(p).name} — quality: {feat['quality_score']:.1f}")
            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(paths)}] ERROR: {p} — {e}")
        return pd.DataFrame(results)

    def generate_feedback_string(self, features: dict, target_note: str = None) -> str:
        """Generate a human-readable feedback string from features."""
        lines = []
        score = features.get("quality_score", 50)
        lines.append(f"Overall Quality Score: {score:.1f}/100")

        pitch_acc = features.get("pitch_accuracy", 0.5)
        detected = features.get("detected_note", "N/A")
        in_tune = features.get("pitch_in_tune", False)
        cents_dev = features.get("pitch_cents_deviation", 0.0)

        if target_note:
            lines.append(f"Target Note: {target_note} | Detected: {detected}")
        else:
            lines.append(f"Detected Note: {detected}")

        if in_tune:
            lines.append(f"✓ Pitch is in tune (deviation: {cents_dev:+.1f} cents)")
        else:
            lines.append(f"✗ Pitch is off — deviation: {cents_dev:+.1f} cents. Aim for within ±50 cents.")

        loudness = features.get("loudness_consistency", 0.5)
        if loudness > 0.75:
            lines.append("✓ Dynamics are consistent — good control.")
        elif loudness > 0.5:
            lines.append("~ Dynamics vary slightly — focus on even tone production.")
        else:
            lines.append("✗ Dynamics are inconsistent — work on steady tone control.")

        tempo = features.get("tempo", 0)
        if tempo > 0:
            lines.append(f"Detected Tempo: {tempo:.0f} BPM")

        return "\n".join(lines)


# ──────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    import sys

    print("Testing AudioFeatureExtractor...")
    extractor = AudioFeatureExtractor()

    # Test with synthetic audio
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    y_test = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note

    features = extractor.extract(y_test, sr=sr)
    print("\nExtracted Features:")
    for k, v in sorted(features.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nFeedback:")
    print(extractor.generate_feedback_string(features, target_note="A4"))

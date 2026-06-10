Early detection of Developmental Language Disorder in children is a clinical challenge, traditionally relying on subjective assessments by trained professionals. This paper presents an automated machine learning pipeline for DLD detection using the ENNI dataset comprising 352 children. Child-only speech segments are isolated from raw narrative recordings, followed by noise reduction and volume normalization. Three complementary feature groups are extracted: 206 acoustic features, 11 prosodic features via Praat, and 14 temporal features, yielding a 231- dimensional feature vector. Five classifiers — Support Vector Machine (SVM), Random Forest, XGBoost, Logistic Regression, and Linear Discriminant Analysis (LDA) — are evaluated on a stratified child-level train/test split to prevent data leakage. Logistic Regression achieved the highest performance with = AUC = 0.813 and accuracy = 85.9%. The results demonstrate that multi-modal feature fusion provides richer DLD markers than any single feature group alone, offering a scalable and interpretable alternative to manual clinical assessment.

# Automated Detection of Developmental Language Disorder (DLD) in Children

A machine learning pipeline that detects Developmental Language Disorder (DLD) in children using narrative speech recordings from the [ENNI dataset](https://talkbank.org/childes/access/Clinical-Eng/ENNI.html). The system extracts acoustic, prosodic, and temporal features from child-only audio segments and classifies each child as either **Typically Developing (TD)** or **DLD**.

---

## Results

| Model | Accuracy | DLD F1 | TD F1 | AUC | CV F1 |
|---|---|---|---|---|---|
| SVM | 77.5% | 0.333 | 0.864 | 0.769 | 0.781 ± 0.099 |
| Random Forest | 84.5% | 0.353 | 0.912 | 0.781 | 0.603 ± 0.072 |
| XGBoost | 83.1% | 0.250 | 0.905 | 0.796 | 0.734 ± 0.085 |
| **Logistic Regression** | **85.9%** | **0.643** | **0.912** | **0.813** | **0.781 ± 0.038** |
| LDA | 70.4% | 0.323 | 0.811 | 0.641 | 0.469 ± 0.061 |

> **Best model: Logistic Regression** — AUC = 0.813, Accuracy = 85.9%

---

## Dataset

The [Edmonton Narrative Norms Instrument (ENNI)](https://talkbank.org/childes/access/Clinical-Eng/ENNI.html) corpus from the TalkBank CHILDES repository.

- **352 children** — 66 DLD, 286 Typically Developing
- **Two subsets** — A and B, each with three story levels
- Each child has an `.mp3` audio file and a `.cha` CLAN transcript with millisecond-level speaker timestamps

> The dataset must be downloaded separately from [TalkBank].

---

## Pipeline Overview

```
ENNI dataset (MP3 + .cha)
        │
        ├──── Audio cleaning (noise reduction, normalize)
        │           │
        │     ┌─────┴──────────────┐
        │     ▼                    ▼
        │  Acoustic (206)     Prosodic (11)
        │  MFCC + Log-Mel     F0, jitter, HNR
        │
        └──── .cha parsing (CHI timestamps)
                    │
                    ▼
              Temporal (14)
              Rate, pauses, TTR
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  StandardScaler         Combined vector
  (train split only)     (231 features)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
             SVM          Random Forest      XGBoost
        Log. Regression      LDA
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          Typically Developing           DLD
```

---

## Features Extracted

### Acoustic (206 features)
Extracted from **individual utterance clips** using `librosa`.

| Group | Description | Count |
|---|---|---|
| MFCC mean + std | Coefficients C1–C13 | 26 |
| Delta MFCC mean + std | First-order delta | 26 |
| Delta-delta MFCC mean + std | Second-order delta | 26 |
| Log-Mel mean + std | 64 mel frequency bands | 128 |

### Prosodic (11 features)
Extracted from **individual utterance clips** using `praat-parselmouth`.

| Feature | Description |
|---|---|
| `f0_mean/std/min/max/range` | Pitch (fundamental frequency) |
| `jitter_local`, `jitter_rap` | Cycle-to-cycle pitch variation |
| `shimmer_local`, `shimmer_apq3` | Amplitude variation |
| `hnr` | Harmonics-to-noise ratio |
| `cpp` | Cepstral peak prominence proxy |

### Temporal (14 features)
Derived from **`.cha` transcript timestamps** using the per-child merged audio.

| Feature | Description |
|---|---|
| `speech_rate_wps` | Words per second |
| `articulation_rate` | Words/sec excluding pauses |
| `pause_count` | Number of pauses > 100ms |
| `mean/max_pause_duration` | Pause length statistics |
| `pause_ratio` | Proportion of time in silence |
| `mean_utt_duration` | Average utterance length |
| `total_utterances` | Total CHI turns |
| `mean_words_per_utt` | Average words per utterance |
| `total_words` | Total words in session |
| `type_token_ratio` | Lexical diversity |
| `narrative_length_s` | Total session duration |

---

## Installation

```bash
pip install librosa praat-parselmouth noisereduce scikit-learn xgboost imageio-ffmpeg numpy matplotlib
```

> **Windows users:** `noisereduce` must be run with `n_jobs=1` to avoid memory issues.  
> `ffmpeg` is handled automatically via `imageio_ffmpeg` — no manual install needed.

---

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| **Child-level train/test split** | Prevents data leakage — same child cannot appear in both train and test |
| **Examiner segments zeroed, not trimmed (merged files)** | Preserves natural pause durations between utterances |
| **Utterance clips trimmed (individual files)** | Removes leading/trailing silence for cleaner acoustic/prosodic features |
| **StandardScaler fit on train only** | Prevents test set information leaking into normalization |
| **`class_weight='balanced'`** | Compensates for 1:4.3 DLD:TD class imbalance |
| **`n_jobs=1` for noisereduce** | Avoids Windows paging file exhaustion |

---

## Requirements

- Python 3.8+
- Windows / Linux / macOS
- ~8 GB RAM recommended for noise reduction on full dataset

---

## Citation

If you use this code or pipeline, please cite:

```
@misc{dld_enni_pipeline,
  author = {Manasa Balaji},
  title  = {Automated Detection of Developmental Language Disorder Using
            Multi-Modal Speech Features},
  year   = {2025},
  note   = {CUNY Graduate Center Spring 2026 — Speech and Audio Learning}
}
```

**Dataset citation:**
> Schneider, P., Hayward, D., & Dubé, R. V. (2006). Storytelling from pictures using the Edmonton Narrative Norms Instrument. *Journal of Speech-Language Pathology and Audiology*, 30, 224–238.

---

## References

- [TalkBank CHILDES](https://talkbank.org/) for the ENNI corpus
- [librosa](https://librosa.org/) for audio feature extraction
- [praat-parselmouth](https://parselmouth.readthedocs.io/) for prosodic analysis
- [noisereduce](https://github.com/timsainburg/noisereduce) for noise reduction

# Features and Detection Experiments

This folder contains all feature extraction modules and documentation for running detection experiments in the lyrics detection project. Below is an overview of the available feature extractors, their usage, and a guide to the main detection experiment scripts.

---

## 1. Feature Extraction Modules

All feature extractors in this folder inherit from the `FeatureExtractor` base class (see `feature_extractor.py`). Each extractor provides a unified interface for embedding extraction and downstream classification. Key methods:

- `predict`: Loads or computes features as needed.
- `classify_and_save`: Trains or loads a classifier (MLP), evaluates on the test set, and saves predictions.
- `get_metrics`: Logs basic classification metrics (for quick reference, not for publication).

### Available Feature Extractors

- **Text-based:**
  - `bert.py`: BERT-based embeddings (SentenceTransformers)
  - `sbert.py`: Sentence-BERT embeddings
  - `binoculars.py`: Binoculars model features
  - `ensemble.py`: Ensemble of multiple extractors
  - `entropy.py`: Entropy-based features
  - `gritlm.py`: GritLM model features
  - `llm2vec.py`: LLM2Vec model features
  - `loglikelihood.py`: Log-likelihood and perplexity features
  - `luar.py`: LUAR model features

- **Audio-based:**
  - `mms.py`: MMS (audio) model features
  - `w2v2.py`: Wav2Vec2 audio features
  - `xeus.py`: XEUS audio features

### Usage Example

```python
from features.bert import Bert
extractor = Bert(dataset, bert_variant="all-mpnet-base-v2", ...)
model = extractor.load_model()
embeddings = extractor.compute_embeddings(model, lyrics)
```

### Adding a New Feature Extractor

1. Create a new file (e.g., `myfeature.py`).
2. Implement a class inheriting from `FeatureExtractor`.
3. Implement the required methods: `load_model` and `compute_embeddings`.
4. Add your file to this folder.

---

## 2. Detection Experiments

Detection experiments are primarily run using scripts in `features/`. The main script, `run_features.py`, trains and evaluates features on various datasets and settings. You can select transcribers, data setups, and features by (un-)commenting options in the script.

### Core Scripts Overview

- `run_features.py`: Main script for training and evaluating features.
- `predict_features.py`: For OOD experiments; loads test data and evaluates pre-trained classifiers.
- `predict.py`: Handles dataset loading, model setup, and running the detector (relies on the `Detector` class).
- `feature_extractor.py`: Base class for all features (lyrics, speech, ensemble); handles feature computation, classifier training/evaluation, and metrics logging.

### Main Detection Experiments

- **Purpose:** Trains and evaluates all features (transcript-based, speech-based, multimodal) on a training dataset and tests on a dataset of the same distribution.
- **Customization:** Select transcribers, data setups, and features by (un-)commenting in the script.
- **Default:** Runs all features with Whisper-large-v2 in the main real vs. fully fake scenario.

### Out-of-Distribution (OOD) Experiments

- **Audio OOD (Table 4):**
  - Use `predict_features.py` to evaluate models on OOD audio attacks or the Udio subset.
  - Set `predict_only=True` to evaluate only on test data.

- **Text OOD (ISMIR Table 3):**
  - Use `robust_detection/utils/stratify_by_model.py` to redistribute train/test splits by generator models.
  - Then run as above.

---

## 4. Evaluation

- **Script:** `evaluation/get_eval_results.py`
  - Computes macro-average recall scores for all runs in a directory and saves them to CSV.
  - Generates plots for per-language performance and macro recall across setups.

- **Notebook:** `evaluation/get_main_table_results.ipynb`
  - Filters and formats results for publication tables, based on the output of `get_eval_results.py`.

---

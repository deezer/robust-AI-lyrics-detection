# Transcription Module

This folder contains scripts and notebooks for transcribing, analyzing, and evaluating lyrics and audio data as part of the robust AI lyrics detecton project. It includes tools for both transcript generation (using state-of-the-art models) and transcript analysis.

## Overview

Transcription is performed using scripts for OpenAI Whisper (large-v2, large-v3, large-v3-turbo) and Meta models (Seamless-large, mms-1b), located in `robust_detection/transcription`.

## Contents

- `adversarial_augmenter.py`: Generate adversarial augmentations of transcripts.
- `analyze_transcripts_real.ipynb`: Analyze real transcript data.
- `compare_transcripts.ipynb`: Compare different transcript sources or models.
- `eval_transcripts.py`, `eval_transcripts_all.py`: Evaluate transcript quality and accuracy (single or batch).
- `run_meta.py`, `run_meta_fake.py`: Transcribe with Meta models (real and fake/synthetic audio).
- `run_whisper.py`, `run_whisper_fake.py`: Transcribe with Whisper (real and fake/synthetic audio).
- `transcript_utils.py`, `utils.py`: Utility functions for transcript processing and analysis.

## Transcription Scripts

### Whisper

- `run_whisper.py`: For real songs.
- `run_whisper_fake.py`: For AI-generated (fake & half-fake) audio. Also supports downloading from Suno/Udio if not present locally.

Both scripts use `faster-whisper` models to transcribe songs to a specified output directory, retaining the .json structure of the input dataset. They support advanced options such as demucs source separation, VAD, language specification, and audio perturbation (for robustness experiments). To run on Udio or a non-Suno dataset, set `--generation_model`.

### Meta

- `run_meta.py` and `run_meta_fake.py` provide similar functionality for Meta models. Note: output directories must be set manually.

## Usage Example

Use the provided scripts to transcribe audio, generate adversarial examples, or evaluate transcript quality. Notebooks are available for in-depth analysis and comparison.

```bash
python run_whisper.py --input data/audio_file.wav --output transcripts/output.txt
```

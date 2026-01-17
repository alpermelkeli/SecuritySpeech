# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SecuritySpeech is a speaker identification/verification system using SpeechBrain's ECAPA-TDNN model pretrained on VoxCeleb. It enrolls speakers from audio samples and identifies unknown speakers via cosine similarity of embeddings.

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Speaker Identification
```bash
python pretrained/speech_brain.py
```

### Download Test Audio Data
```bash
python scripts/download_test_auido.py
```

## Environment Variables

Requires a `.env` file with:
```
HF_TOKEN=your_huggingface_token
```

## Architecture

### Core Component: `pretrained/speech_brain.py`
- `SpeechBrain` class: Main speaker identification system
  - Uses `speechbrain/spkrec-ecapa-voxceleb` model from HuggingFace
  - `enroll_speakers()`: Scans `data/` directory, creates mean embedding per speaker folder
  - `identify(audio_path, threshold)`: Returns recognition result with confidence scores
  - Embeddings are L2-normalized; comparison uses cosine similarity

### Data Organization
- `data/`: Speaker enrollment data - each subfolder is a speaker name containing `.wav` files
- `pretrained_models/`: Cached SpeechBrain model weights
- `scripts/download_test_auido.py`: Downloads LibriSpeech samples for testing

### Key Dependencies
- `speechbrain`: Speaker encoder model
- `torchaudio`: Audio loading
- `numpy`: Embedding arithmetic
- `huggingface_hub`: Model download authentication

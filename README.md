# SecuritySpeech

SecuritySpeech is a speaker verification and identification system built with [SpeechBrain](https://speechbrain.github.io/) and [Flask](https://flask.palletsprojects.com/). It uses the ECAPA-TDNN model (`speechbrain/spkrec-ecapa-voxceleb`) to generate speaker embeddings and identify users based on audio samples.

## Features

- **Speaker Enrollment**: Register new speakers by uploading multiple audio samples.
- **Speaker Identification**: Upload a voice clip to identify which enrolled speaker it belongs to.
- **Verification Threshold**: Adjustable confidence threshold for strictness of matching.
- **Web Interface**: Simple user-friendly interface for managing speakers and testing verification.
- **Dockerized**: Fully containerized setup for easy deployment.

## Prerequisites

- **Docker** and **Docker Compose** installed on your machine.
- A **Hugging Face Account** and [Access Token](https://huggingface.co/settings/tokens) (Read permission) to download the SpeechBrain model.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SecuritySpeech
```

### 2. Configure Environment

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

Open the `.env` file and replace `put_your_hugging_face_token_here` with your actual Hugging Face Access Token.

```env
HF_TOKEN=hf_your_actual_token_here
```

### 3. Run with Docker (Recommended)

Build and start the container:

```bash
docker-compose up --build
```

The application will start and listen on port **8000**.

> **Note:** The first run may take a few minutes as it downloads the pre-trained models from Hugging Face and installs dependencies (PyTorch, Torchaudio).

### 4. Access the Application

Open your web browser and go to:
[http://localhost:8000](http://localhost:8000)

## Local Development (Without Docker)

If you prefer to run locally, ensure you have Python 3.10+ and `ffmpeg` installed.

1.  **Install System Dependencies**:
    *   **MacOS**: `brew install ffmpeg libsndfile`
    *   **Ubuntu/Debian**: `sudo apt-get install ffmpeg libsndfile1`

2.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Python Packages**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**:
    ```bash
    python web/app.py
    ```

## Usage

### Enrolling a Speaker
1.  Enter the speaker's name in the "Add Speaker" section.
2.  Select one or more `.wav` audio files (ideally containing clear speech).
3.  Click **"Add Speaker"**. The system will generate an embedding for the speaker.

### Verifying Audio
1.  In the "Verify Speaker" section, upload a test audio file.
2.  (Optional) Adjust the **Threshold** slider. Higher values require a closer match (default: 0.65).
3.  Click **"Verify"**. The system will return the most likely speaker match and the confidence score.

## Project Structure

```text
/
├── data/                  # Directory storing enrolled speaker audio samples
├── model/
│   └── speech_brain.py    # Core logic for model loading, embedding, and matching
├── web/
│   ├── app.py             # Flask backend routes
│   ├── templates/         # HTML templates
│   └── static/            # CSS and JS files
├── pretrained_models/     # Cached directory for SpeechBrain models
├── scripts/               # Utility scripts (e.g., downloading test data)
├── docker-compose.yml     # Docker services configuration
├── Dockerfile             # Docker image definition
└── requirements.txt       # Python dependencies
```

## Troubleshooting

-   **Connection Refused**: Ensure the Docker container is running and port 8000 is not blocked.
-   **Model Load Error**: Verify your `HF_TOKEN` in the `.env` file is valid and has read permissions.
-   **Audio Format Issues**: The system works best with standard `.wav` files (16kHz sample rate recommended, though SpeechBrain handles resampling).

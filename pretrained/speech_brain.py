import os
import shutil
from dotenv import load_dotenv
from huggingface_hub import login
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
import torch
import numpy as np

# Get the project root directory (parent of pretrained folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "data")
TEST_FILE_PATH = os.path.join(PROJECT_ROOT, "test.wav")

class SpeechBrain():

    def __init__(self, data_folder_path=None):
        self._setup_credentials()
        self.classifier = self._load_model()
        self.enrolled_speakers = {}  # name: mean embedding
        self.data_folder_path = data_folder_path or DATA_FOLDER_PATH
        self.threshold = 0.65  # Default threshold

    def _setup_credentials(self):
        load_dotenv()
        login(token=os.getenv("HF_TOKEN"))

    def _load_model(self) -> EncoderClassifier | None:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
    
    def _enroll_speaker(self, name: str, audio_folder_path: str):
        embeddings = []

        if os.path.exists(audio_folder_path):
            auido_files = [audio_folder_path + "/" + f for f in os.listdir(audio_folder_path) if os.path.isfile(os.path.join(audio_folder_path, f))]
            for file in auido_files:
                embedding = self._get_embedding(file)
                embeddings.append(embedding)
            print(f"  Processed: {name} audio files")
        else:
            print(f"  Warning: File not found - {audio_folder_path}")
            
        
        if embeddings:
            # Average all embeddings for this speaker
            mean_embedding = np.mean(embeddings, axis=0)
            # Normalize
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            self.enrolled_speakers[name] = mean_embedding
            print(f"Enrolled '{name}' with {len(embeddings)} sample(s)")
        else:
            print(f"Failed to enroll '{name}' - no valid audio files")

    def enroll_speakers(self):
        if os.path.exists(self.data_folder_path):
            folderNames = [f for f in os.listdir(self.data_folder_path) if os.path.isdir(os.path.join(self.data_folder_path, f))]
            for folderName in folderNames:
                self._enroll_speaker(folderName, self.data_folder_path + "/" + folderName)
            print(f"Speakers enrolled {self.enrolled_speakers}")
        else:
            print("Warning: there is no data folder path")

    def get_speakers(self):
        """Get list of all enrolled speakers with their sample counts."""
        speakers = []
        if os.path.exists(self.data_folder_path):
            for folder_name in os.listdir(self.data_folder_path):
                folder_path = os.path.join(self.data_folder_path, folder_name)
                if os.path.isdir(folder_path):
                    samples = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                    speakers.append({
                        "name": folder_name,
                        "sample_count": len(samples),
                        "enrolled": folder_name in self.enrolled_speakers
                    })
        return speakers

    def add_speaker(self, speaker_name, audio_files):
        """Add a new speaker with audio files."""
        speaker_folder = os.path.join(self.data_folder_path, speaker_name)

        # Create speaker folder if it doesn't exist
        os.makedirs(speaker_folder, exist_ok=True)

        saved_files = []
        for i, audio_file in enumerate(audio_files):
            # Get next sample number
            existing_samples = [f for f in os.listdir(speaker_folder) if f.startswith('sample_')]
            next_num = len(existing_samples) + 1
            filename = f"sample_{next_num}.wav"
            filepath = os.path.join(speaker_folder, filename)
            audio_file.save(filepath)
            saved_files.append(filename)

        # Re-enroll this speaker
        self._enroll_speaker(speaker_name, speaker_folder)

        return {"speaker": speaker_name, "files_added": saved_files}

    def delete_speaker(self, speaker_name):
        """Delete a speaker and all their audio files."""
        speaker_folder = os.path.join(self.data_folder_path, speaker_name)

        if not os.path.exists(speaker_folder):
            return {"success": False, "error": "Speaker not found"}

        # Remove from enrolled speakers
        if speaker_name in self.enrolled_speakers:
            del self.enrolled_speakers[speaker_name]

        # Delete the folder
        shutil.rmtree(speaker_folder)

        return {"success": True, "speaker": speaker_name}

    def set_threshold(self, threshold):
        """Set the recognition threshold (0.0 to 1.0)."""
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            return {"success": True, "threshold": self.threshold}
        return {"success": False, "error": "Threshold must be between 0.0 and 1.0"}

    def get_threshold(self):
        """Get the current recognition threshold."""
        return self.threshold

    def _get_embedding(self, audio_path):
        signal, fs = torchaudio.load(audio_path)
        embedding = self.classifier.encode_batch(signal) # type: ignore
        return embedding.squeeze().numpy()
    
    def identify(self, audio_path, threshold=None):
        if not self.enrolled_speakers:
            return {"status": "error", "message": "No speakers enrolled"}

        # Use instance threshold if not specified
        if threshold is None:
            threshold = self.threshold

        test_embedding = self._get_embedding(audio_path)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)

        # Compare with all enrolled speakers
        scores = {}
        for name, enrolled_embedding in self.enrolled_speakers.items():
            # Cosine similarity
            similarity = np.dot(test_embedding, enrolled_embedding)
            scores[name] = float(similarity)

        # Find best match
        best_match = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_match]

        if best_score >= threshold:
            return {
                "status": "recognized",
                "name": best_match,
                "confidence": round(best_score * 100, 2),
                "all_scores": {k: round(v * 100, 2) for k, v in scores.items()}
            }
        else:
            return {
                "status": "not recognized",
                "name": None,
                "confidence": round(best_score * 100, 2),
                "all_scores": {k: round(v * 100, 2) for k, v in scores.items()}
            }


sb = SpeechBrain()

if __name__ == "__main__":
    sb.enroll_speakers()
    result = sb.identify(TEST_FILE_PATH)
    print(result)
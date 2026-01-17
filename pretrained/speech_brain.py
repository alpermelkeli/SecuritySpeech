import os
from dotenv import load_dotenv
from huggingface_hub import login
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
import torch
import numpy as np

DATA_FOLDER_PATH = "/Users/alpermelkeli/Documents/SecuritySpeech/data"
TEST_FILE_PATH = "/Users/alpermelkeli/Documents/SecuritySpeech/sample_4.wav"

class SpeechBrain():

    def __init__(self):
        self._setup_credentials()
        self.classifier = self._load_model()
        self.enrolled_speakers = {} #name: mean embeeding.

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
        if os.path.exists(DATA_FOLDER_PATH):
            folderNames = [f for f in os.listdir(DATA_FOLDER_PATH) if os.path.isdir(os.path.join(DATA_FOLDER_PATH, f))]
            for folderName in folderNames:
                self._enroll_speaker(folderName, DATA_FOLDER_PATH + "/" + folderName)
            print(f"Speakers enrolled {self.enrolled_speakers}")
        else:
            print("Warning: there is no data folder path")

    def _get_embedding(self, audio_path):
        signal, fs = torchaudio.load(audio_path)
        embedding = self.classifier.encode_batch(signal) # type: ignore
        return embedding.squeeze().numpy()
    
    def identify(self, audio_path, threshold=0.65):
        if not self.enrolled_speakers:
            return {"status": "error", "message": "No speakers enrolled"}
        
        test_embedding = self._get_embedding(audio_path)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Compare with all enrolled speakers
        scores = {}
        for name, enrolled_embedding in self.enrolled_speakers.items():
            # Cosine similarity
            similarity = np.dot(test_embedding, enrolled_embedding)
            scores[name] = float(similarity)
        
        # Find best match
        best_match = max(scores, key=scores.get) # type: ignore
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
import os
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

from datasets import load_dataset

def download_librispeech_samples(num_speakers=5, samples_per_speaker=5, output_dir="data"):
    """
    Download audio samples from LibriSpeech dataset using streaming (no full download)
    """
    print("Loading dataset in streaming mode...")
    
    # Use streaming=True to avoid downloading the entire dataset
    dataset = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        trust_remote_code=True
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track speakers and their samples
    speaker_counts = {}  # {speaker_id: count}
    downloaded = {}      # {speaker_id: [filepaths]}
    
    print(f"Looking for {num_speakers} speakers with {samples_per_speaker} samples each...\n")
    
    for sample in dataset:
        speaker_id = sample["speaker_id"]
        
        # Skip if we already have enough speakers
        if len(speaker_counts) >= num_speakers and speaker_id not in speaker_counts:
            continue
        
        # Initialize speaker if new
        if speaker_id not in speaker_counts:
            speaker_counts[speaker_id] = 0
            downloaded[speaker_id] = []
            
            # Create speaker directory
            speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
            os.makedirs(speaker_dir, exist_ok=True)
            print(f"Found new speaker: {speaker_id}")
        
        # Skip if we have enough samples for this speaker
        if speaker_counts[speaker_id] >= samples_per_speaker:
            # Check if we're done
            if all(count >= samples_per_speaker for count in speaker_counts.values()):
                if len(speaker_counts) >= num_speakers:
                    break
            continue
        
        # Save this sample
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        
        speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
        filepath = os.path.join(speaker_dir, f"sample_{speaker_counts[speaker_id] + 1}.wav")
        
        sf.write(filepath, audio_array, sample_rate)
        
        speaker_counts[speaker_id] += 1
        downloaded[speaker_id].append(filepath)
        
        print(f"  Saved: {filepath}")
        
        # Check if done
        if len(speaker_counts) >= num_speakers:
            if all(count >= samples_per_speaker for count in speaker_counts.values()):
                break
    
    print("\n" + "="*50)
    print("DOWNLOAD COMPLETE")
    print("="*50)
    
    for speaker_id, files in downloaded.items():
        print(f"\nSpeaker {speaker_id}: {len(files)} files")
        for f in files:
            print(f"  {f}")
    
    return downloaded


if __name__ == "__main__":
    downloaded_files = download_librispeech_samples(
        num_speakers=5,
        samples_per_speaker=5,
        output_dir="audio"
    )
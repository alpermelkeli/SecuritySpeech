import sys
import os
import shutil
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.speech_brain import SpeechBrain

app = Flask(__name__)

# Initialize SpeechBrain
print("Initializing SpeechBrain...")
try:
    sb = SpeechBrain()
    sb.enroll_speakers()
    print("SpeechBrain initialized successfully.")
except Exception as e:
    print(f"Error initializing SpeechBrain: {e}")
    sb = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    if not sb:
        return jsonify({"error": "Model not initialized"}), 500
    return jsonify(list(sb.enrolled_speakers.keys()))

@app.route('/api/speakers', methods=['POST'])
def add_speaker():
    if not sb:
        return jsonify({"error": "Model not initialized"}), 500
    
    name = request.form.get('name')
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files uploaded"}), 400

    # Create directory
    speaker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data', name))
    if os.path.exists(speaker_dir):
        return jsonify({"error": "Speaker already exists"}), 400
    
    os.makedirs(speaker_dir)

    saved_files = 0
    for file in files:
        if file.filename:
            file.save(os.path.join(speaker_dir, file.filename))
            saved_files += 1
            
    if saved_files == 0:
        os.rmdir(speaker_dir)
        return jsonify({"error": "No valid files saved"}), 400

    # Enroll the new speaker
    sb._enroll_speaker(name, speaker_dir)
    
    return jsonify({"message": f"Speaker {name} added with {saved_files} samples", "speakers": list(sb.enrolled_speakers.keys())})

@app.route('/api/speakers/<name>', methods=['DELETE'])
def delete_speaker(name):
    if not sb:
        return jsonify({"error": "Model not initialized"}), 500
    
    speaker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data', name))
    
    if name in sb.enrolled_speakers:
        del sb.enrolled_speakers[name]
        
    if os.path.exists(speaker_dir):
        shutil.rmtree(speaker_dir)
        return jsonify({"message": f"Speaker {name} deleted", "speakers": list(sb.enrolled_speakers.keys())})
    else:
        # If folder doesn't exist but was in memory, it's already removed from memory above.
        return jsonify({"message": f"Speaker {name} deleted (folder was missing)", "speakers": list(sb.enrolled_speakers.keys())})

@app.route('/api/verify', methods=['POST'])
def verify():
    if not sb:
        return jsonify({"error": "Model not initialized"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        threshold = float(request.form.get('threshold', 0.65))
    except ValueError:
        return jsonify({"error": "Invalid threshold value"}), 400
    
    # Save temp file
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../temp_verify.wav'))
    file.save(temp_path)
    
    try:
        result = sb.identify(temp_path, threshold=threshold)
    except Exception as e:
        result = {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return jsonify(result)

@app.route('/api/reload', methods=['POST'])
def reload_model():
    if not sb:
        return jsonify({"error": "Model not initialized"}), 500
    sb.enrolled_speakers = {}
    sb.enroll_speakers()
    return jsonify({"message": "Model reloaded", "speakers": list(sb.enrolled_speakers.keys())})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))

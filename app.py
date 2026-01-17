import os
import tempfile
from flask import Flask, request, jsonify, render_template
from pretrained.speech_brain import SpeechBrain

app = Flask(__name__)

# Initialize the SpeechBrain model
sb = SpeechBrain()
sb.enroll_speakers()


# ============== Web Panel Routes ==============

@app.route('/')
def index():
    """Render the main management panel."""
    return render_template('index.html')


# ============== REST API Routes ==============

@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    """Get list of all speakers."""
    speakers = sb.get_speakers()
    return jsonify({
        "success": True,
        "speakers": speakers,
        "total": len(speakers)
    })


@app.route('/api/speakers', methods=['POST'])
def add_speaker():
    """Add a new speaker with audio samples."""
    if 'speaker_name' not in request.form:
        return jsonify({"success": False, "error": "Speaker name is required"}), 400

    speaker_name = request.form['speaker_name']

    if not speaker_name:
        return jsonify({"success": False, "error": "Speaker name cannot be empty"}), 400

    # Get uploaded audio files
    audio_files = request.files.getlist('audio_files')

    if not audio_files or all(f.filename == '' for f in audio_files):
        return jsonify({"success": False, "error": "At least one audio file is required"}), 400

    # Filter out empty files
    audio_files = [f for f in audio_files if f.filename != '']

    result = sb.add_speaker(speaker_name, audio_files)

    return jsonify({
        "success": True,
        "message": f"Speaker '{speaker_name}' added successfully",
        "details": result
    })


@app.route('/api/speakers/<speaker_name>', methods=['DELETE'])
def delete_speaker(speaker_name):
    """Delete a speaker."""
    result = sb.delete_speaker(speaker_name)

    if result["success"]:
        return jsonify({
            "success": True,
            "message": f"Speaker '{speaker_name}' deleted successfully"
        })
    else:
        return jsonify(result), 404


@app.route('/api/threshold', methods=['GET'])
def get_threshold():
    """Get the current recognition threshold."""
    return jsonify({
        "success": True,
        "threshold": sb.get_threshold(),
        "threshold_percent": round(sb.get_threshold() * 100, 2)
    })


@app.route('/api/threshold', methods=['PUT'])
def set_threshold():
    """Set the recognition threshold."""
    data = request.get_json()

    if not data or 'threshold' not in data:
        return jsonify({"success": False, "error": "Threshold value is required"}), 400

    try:
        threshold = float(data['threshold'])
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid threshold value"}), 400

    result = sb.set_threshold(threshold)

    if result["success"]:
        return jsonify({
            "success": True,
            "message": f"Threshold set to {threshold}",
            "threshold": threshold,
            "threshold_percent": round(threshold * 100, 2)
        })
    else:
        return jsonify(result), 400


@app.route('/api/identify', methods=['POST'])
def identify_speaker():
    """Identify a speaker from an audio file."""
    if 'audio_file' not in request.files:
        return jsonify({"success": False, "error": "Audio file is required"}), 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        audio_file.save(tmp_file.name)
        tmp_path = tmp_file.name

    try:
        # Custom threshold from request (optional)
        threshold = request.form.get('threshold')
        if threshold:
            threshold = float(threshold)
        else:
            threshold = None

        result = sb.identify(tmp_path, threshold=threshold)
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/enroll', methods=['POST'])
def re_enroll_speakers():
    """Re-enroll all speakers from the data folder."""
    sb.enrolled_speakers = {}
    sb.enroll_speakers()

    return jsonify({
        "success": True,
        "message": "All speakers re-enrolled successfully",
        "enrolled_count": len(sb.enrolled_speakers)
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        "success": True,
        "enrolled_speakers": len(sb.enrolled_speakers),
        "threshold": sb.get_threshold(),
        "data_folder": sb.data_folder_path,
        "model_loaded": sb.classifier is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

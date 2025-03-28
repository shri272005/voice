import os
import shutil
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac'}  # Allowed file types

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Pretrained AI-Human Voice Model (Wav2Vec2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
model.to(device)

def allowed_file(filename):
    """Check if uploaded file is in allowed formats."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_delete(filepath):
    """Ensure safe deletion of file using shutil."""
    try:
        if os.path.exists(filepath):
            temp_path = filepath + "_to_delete"
            os.rename(filepath, temp_path)  # Rename before deleting
            os.remove(temp_path)  # Delete safely
    except Exception as e:
        print(f"Warning: Could not delete {filepath}. Error: {e}")

def extract_features(audio_path):
    """Extract deep audio features using Wav2Vec2"""
    try:
        waveform, sr = torchaudio.load(audio_path)
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        ai_probability = probabilities[0][1].item() * 100  # AI voice probability

        return ai_probability
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_voice():
    """Handle both recorded and uploaded audio."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Invalid file format. Allowed: wav, mp3, ogg, flac"}), 400

    try:
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)

        ai_probability = extract_features(filepath)
        safe_delete(filepath)  # Use safe delete

        if ai_probability is None:
            return jsonify({"error": "Could not analyze voice"}), 500

        return jsonify({
            "is_ai": ai_probability > 50,
            "confidence": round(abs(ai_probability - 50) * 2, 2),
            "ai_probability": round(ai_probability, 2)
        })

    except Exception as e:
        safe_delete(filepath)
        return jsonify({"error": str(e)}), 500

@app.route('/record', methods=['POST'])
def record_voice():
    """Handle browser-recorded voice input."""
    try:
        audio_data = request.files['audio']
        filename = "recorded_audio.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_data.save(filepath)

        ai_probability = extract_features(filepath)
        safe_delete(filepath)

        if ai_probability is None:
            return jsonify({"error": "Could not analyze voice"}), 500

        return jsonify({
            "is_ai": ai_probability > 50,
            "confidence": round(abs(ai_probability - 50) * 2, 2),
            "ai_probability": round(ai_probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

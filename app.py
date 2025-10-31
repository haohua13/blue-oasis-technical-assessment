# app.py
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from data_processing import load_metadata, preprocess_audio, generate_audio_plots
import librosa
import os
import pandas as pd
import random 
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

BASE_DATA_PATH = "ESC-50-master" 
# Load metadata once when the app starts
try:
    METADATA = load_metadata(BASE_DATA_PATH)
    app.logger.info(f"Flask App: Loaded {len(METADATA)} audio files metadata.")
    
    # Create a mapping from category names to integer labels if needed for model training
    # And get unique categories for dropdown list
    unique_categories = sorted(METADATA['category'].unique().tolist())
    category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
    id_to_category = {i: cat for cat, i in category_to_id.items()}
    METADATA['category_id'] = METADATA['category'].map(category_to_id)

    app.logger.info(f"Loaded {len(unique_categories)} unique categories.")
    app.logger.info(f"First 5 categories: {unique_categories[:5]}")

except FileNotFoundError:
    app.logger.error(f"ERROR: ESC-50 metadata not found at {os.path.join(BASE_DATA_PATH, 'meta', 'esc50.csv')}")
    app.logger.error("Please ensure the ESC-50 dataset is correctly placed in the 'ESC-50-master' directory.")
    METADATA = pd.DataFrame() # Create an empty DataFrame to prevent errors
    unique_categories = []
    category_to_id = {}
    id_to_category = {}
except Exception as e:
    app.logger.error(f"An unexpected error occurred during metadata loading: {e}")
    METADATA = pd.DataFrame()
    unique_categories = []
    category_to_id = {}
    id_to_category = {}

@app.route('/')
def index():
    """Renders the main page with a list of ESC-50 categories."""
    return render_template('index.html', categories=unique_categories)

@app.route('/explore_audio', methods=['POST'])
def explore_audio():
    """
    Endpoint to load, preprocess, and generate plots for a randomly selected audio file
    from a specified class (category_name) or a specific file_id.
    Returns base64 encoded images of the plots and audio URL.
    """
    category_name = request.form.get('category_name') # User inputs category name
    file_id = request.form.get('file_id') # User can also input a specific file ID

    if METADATA.empty:
        return jsonify(error="Dataset metadata not loaded. Check server logs."), 500

    audio_info = None
    if file_id:
        audio_info_rows = METADATA[METADATA['filename'] == file_id]
        if not audio_info_rows.empty:
            audio_info = audio_info_rows.iloc[0]
    elif category_name:
        if category_name not in unique_categories:
            return jsonify(error=f"Category '{category_name}' not found. Please choose from the list."), 400
        
        # Select a random audio file from the chosen category
        category_files = METADATA[METADATA['category'] == category_name]
        if not category_files.empty:
            audio_info = category_files.sample(1).iloc[0]
    
    if audio_info is None:
        return jsonify(error="No audio file found for the given input."), 404

    audio_filepath = audio_info['filepath']
    audio_class = audio_info['category']
    audio_filename = audio_info['filename']

    if not os.path.exists(audio_filepath):
        app.logger.error(f"Audio file not found at {audio_filepath}.")
        return jsonify(error=f"Audio file not found for {audio_filename}. Check dataset path."), 404

    try:
        # Load audio (downsample to 22050 Hz for consistency and efficiency)
        y, sr = librosa.load(audio_filepath, sr=22050) 

        # Preprocess to get features
        mel_spec_db, mfccs = preprocess_audio(y, sr)

        # Generate plots as base64 images
        plots_b64 = generate_audio_plots(y, sr, mel_spec_db, mfccs, title=f"{audio_class} ({audio_filename})")

        # Create a URL for audio playback. Flask's send_from_directory will handle serving it.
        audio_url = url_for('serve_audio', filename=audio_filename)

        return jsonify(
            waveform_image=plots_b64['waveform_b64'],
            mel_spectrogram_image=plots_b64['mel_spectrogram_b64'],
            mfcc_image=plots_b64['mfcc_b64'],
            audio_class=audio_class,
            audio_file=audio_filename,
            duration=f"{len(y)/sr:.2f} s",
            sample_rate=f"{sr} Hz",
            audio_url=audio_url
        )
    except Exception as e:
        app.logger.error(f"Error processing audio file {audio_filename}: {e}", exc_info=True)
        return jsonify(error=f"An error occurred while processing the audio: {str(e)}"), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files directly from the dataset's audio folder."""
    # This serves files from ESC-50-master/audio
    return send_from_directory(os.path.join(BASE_DATA_PATH, 'audio'), filename)

if __name__ == '__main__':
    app.run(debug=True)
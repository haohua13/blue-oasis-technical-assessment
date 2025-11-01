# app.py
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import matplotlib
import torch
import torch.nn as nn
from train import SimpleCNN
from data_processing import load_metadata, preprocess_audio, generate_audio_plots, preprocess_for_cnn
import librosa
import os
import pandas as pd
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')  # use a non-interactive backend for matplotlib
    
app = Flask(__name__)

# configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
model = None
device = torch.device("cpu")
BASE_DATA_PATH = "ESC-50-master" 
# load metadata once when the app starts
try:
    METADATA = load_metadata(BASE_DATA_PATH)
    app.logger.info(f"Flask App: Loaded {len(METADATA)} audio files metadata.")
    
    # create a mapping from category names to integer labels if needed for model training
    # and get unique categories for dropdown list
    unique_categories = sorted(METADATA['category'].unique().tolist())
    category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
    id_to_category = {i: cat for cat, i in category_to_id.items()}
    METADATA['category_id'] = METADATA['category'].map(category_to_id)

    app.logger.info(f"Loaded {len(unique_categories)} unique categories.")
    app.logger.info(f"First 5 categories: {unique_categories[:5]}")

    # load trained model here if needed for inference
    model_path = 'esc50_cnn_best_model.pth'
    if os.path.exists(model_path):
        device = torch.device("cpu")
        model = SimpleCNN(num_classes=len(unique_categories))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        app.logger.info(f"Loaded trained model from {model_path}")
    else:
        app.logger.warning(f"Trained model not found. Predictions will be unavailable.")

except FileNotFoundError:
    app.logger.error(f"ERROR: ESC-50 metadata not found at {os.path.join(BASE_DATA_PATH, 'meta', 'esc50.csv')}")
except Exception as e:
    app.logger.error(f"An unexpected error occurred: {e}")

except FileNotFoundError:
    app.logger.error(f"ERROR: ESC-50 metadata not found at {os.path.join(BASE_DATA_PATH, 'meta', 'esc50.csv')}")
    app.logger.error("Please ensure the ESC-50 dataset is correctly placed in the 'ESC-50-master' directory.")
    METADATA = pd.DataFrame() # create an empty DataFrame to prevent errors
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
    folds = sorted(METADATA['fold'].unique().tolist()) if not METADATA.empty else []
    return render_template('index.html', 
                         categories=unique_categories,
                         folds=folds,
                         model_loaded=model is not None)


@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    """Get dataset statistics."""
    if METADATA.empty:
        return jsonify(error="Dataset metadata not loaded."), 500

    class_counts = METADATA['category'].value_counts().to_dict()
    fold_counts = METADATA['fold'].value_counts().sort_index().to_dict()

    return jsonify(
        total_samples=len(METADATA),
        num_classes=len(unique_categories),
        class_distribution=class_counts,
        fold_distribution=fold_counts
    )

@app.route('/explore_audio', methods=['POST'])
def explore_audio():
    """
    Endpoint to load, preprocess, and generate plots for a randomly selected audio file
    from a specified class (category_name) or a specific file_id.
    Returns base64 encoded images of the plots and audio URL.
    """
    category_name = request.form.get('category_name') # user inputs category name
    file_id = request.form.get('file_id') # user can also input a specific file ID
    fold = request.form.get('fold')  # optional fold selection

    if METADATA.empty:
        return jsonify(error="Dataset metadata not loaded. Check server logs."), 500

    # apply fold filter if specified
    filtered_metadata = METADATA
    if fold and fold != 'all':
        try:
            fold_num = int(fold)
            filtered_metadata = METADATA[METADATA['fold'] == fold_num]
        except ValueError:
            return jsonify(error="Invalid fold number."), 400

    audio_info = None
    if file_id:
        audio_info_rows = filtered_metadata[filtered_metadata['filename'] == file_id]
        if not audio_info_rows.empty:
            audio_info = audio_info_rows.iloc[0]
    elif category_name:
        if category_name not in unique_categories:
            return jsonify(error=f"Category '{category_name}' not found."), 400
        
        category_files = filtered_metadata[filtered_metadata['category'] == category_name]
        if not category_files.empty:
            audio_info = category_files.sample(1).iloc[0]
    
    if audio_info is None:
        return jsonify(error="No audio file found for the given input."), 404

    audio_filepath = audio_info['filepath']
    audio_class = audio_info['category']
    audio_filename = audio_info['filename']
    audio_fold = int(audio_info['fold'])

    if not os.path.exists(audio_filepath):
        app.logger.error(f"Audio file not found at {audio_filepath}.")
        return jsonify(error=f"Audio file not found for {audio_filename}."), 404

    try:
        # load audio (downsample to 22050 Hz for consistency and efficiency)
        y, sr = librosa.load(audio_filepath, sr=22050) 

        # preprocess to get features
        mel_spec_db, mfccs, zrc = preprocess_audio(y, sr)

        # generate plots as base64 images
        plots_b64 = generate_audio_plots(y, sr, mel_spec_db, mfccs, zrc, title=f"{audio_class} ({audio_filename})")

        # get model prediction if available
        prediction_data = None
        if model is not None:
            try:
                mel_spec_cnn = preprocess_for_cnn(y, sr, n_mels=128, hop_length=512)
                max_len = int(np.ceil(sr * 5 / 512))
                if mel_spec_cnn.shape[1] < max_len:
                    mel_spec_cnn = np.pad(mel_spec_cnn, ((0, 0), (0, max_len - mel_spec_cnn.shape[1])), mode='constant')
                else:
                    mel_spec_cnn = mel_spec_cnn[:, :max_len]
                
                input_tensor = torch.tensor(mel_spec_cnn, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    predicted_class_id = torch.argmax(probabilities).item()
                    predicted_class = unique_categories[predicted_class_id]
                    confidence = probabilities[predicted_class_id].item()
                    
                    top5_probs, top5_indices = torch.topk(probabilities, min(5, len(probabilities)))
                    top5_predictions = [
                        {'class': unique_categories[idx.item()], 'probability': prob.item() * 100}
                        for prob, idx in zip(top5_probs, top5_indices)
                    ]
                # output the prediction results as a dictionary
                prediction_data = {
                    'predicted_class': predicted_class,
                    'confidence': confidence * 100,
                    'correct': predicted_class == audio_class,
                    'top5_predictions': top5_predictions
                }
            except Exception as e:
                app.logger.error(f"Error during prediction: {e}")

        # create a URL for audio playback. Flask's send_from_directory will handle serving it
        audio_url = url_for('serve_audio', filename=audio_filename)

        return jsonify(
            waveform_image=plots_b64['waveform_b64'],
            mel_spectrogram_image=plots_b64['mel_spectrogram_b64'],
            mfcc_image=plots_b64['mfcc_b64'],
            zcr_image=plots_b64['zcr_b64'],
            audio_class=audio_class,
            audio_file=audio_filename,
            audio_fold=audio_fold,
            duration=f"{len(y)/sr:.2f} s",
            sample_rate=f"{sr} Hz",
            audio_url=audio_url,
            prediction=prediction_data
        )
    except Exception as e:
        app.logger.error(f"Error processing audio file {audio_filename}: {e}", exc_info=True)
        return jsonify(error=f"An error occurred while processing the audio: {str(e)}"), 500

@app.route('/compare_samples', methods=['POST'])
def compare_samples():
    """Compare multiple samples from the same class and show CNN predictions."""
    category_name = request.form.get('category_name')
    num_samples = int(request.form.get('num_samples', 3))
    fold = request.form.get('fold')

    if METADATA.empty:
        return jsonify(error="Dataset metadata not loaded."), 500

    if category_name not in unique_categories:
        return jsonify(error=f"Category '{category_name}' not found."), 400

    # filter by fold
    filtered_metadata = METADATA
    if fold and fold != 'all':
        try:
            fold_num = int(fold)
            filtered_metadata = METADATA[METADATA['fold'] == fold_num]
        except ValueError:
            return jsonify(error="Invalid fold number."), 400

    category_files = filtered_metadata[filtered_metadata['category'] == category_name]

    if category_files.empty:
        return jsonify(error=f"No samples found for category '{category_name}'."), 404

    sampled_files = category_files.sample(min(num_samples, len(category_files)))

    samples = []
    for _, audio_info in sampled_files.iterrows():
        audio_filename = audio_info['filename']
        audio_filepath = audio_info['filepath']
        audio_fold = int(audio_info['fold'])

        if not os.path.exists(audio_filepath):
            continue

        try:
            y, sr = librosa.load(audio_filepath, sr=22050)
            duration = len(y) / sr

            prediction_data = None
            if model is not None:
                try:
                    mel_spec_cnn = preprocess_for_cnn(y, sr, n_mels=128, hop_length=512)
                    max_len = int(np.ceil(sr * 5 / 512))
                    if mel_spec_cnn.shape[1] < max_len:
                        mel_spec_cnn = np.pad(mel_spec_cnn, ((0, 0), (0, max_len - mel_spec_cnn.shape[1])), mode='constant')
                    else:
                        mel_spec_cnn = mel_spec_cnn[:, :max_len]

                    input_tensor = torch.tensor(mel_spec_cnn, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)[0]
                        top5_probs, top5_indices = torch.topk(probs, min(5, len(probs)))
                        top5_predictions = [
                            {'class': unique_categories[idx.item()], 'probability': prob.item() * 100}
                            for prob, idx in zip(top5_probs, top5_indices)
                        ]
                        predicted_class = unique_categories[top5_indices[0].item()]
                        confidence = top5_probs[0].item() * 100

                    prediction_data = {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'correct': predicted_class == category_name,
                        'top5_predictions': top5_predictions
                    }
                except Exception as e:
                    app.logger.error(f"Error during CNN prediction for {audio_filename}: {e}")

            samples.append({
                'filename': audio_filename,
                'fold': audio_fold,
                'duration': f"{duration:.2f} s",
                'audio_url': url_for('serve_audio', filename=audio_filename),
                'prediction': prediction_data
            })
        except Exception as e:
            app.logger.error(f"Error loading audio {audio_filename}: {e}")

    return jsonify(
        category=category_name,
        num_samples=len(samples),
        samples=samples
    )

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files directly from the dataset's audio folder."""
    # this serves files from ESC-50-master/audio
    return send_from_directory(os.path.join(BASE_DATA_PATH, 'audio'), filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
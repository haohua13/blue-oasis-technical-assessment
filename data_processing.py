import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend issues
def load_metadata(base_data_path="data/esc50"):
    metadata_path = os.path.join(base_data_path, 'meta', 'esc50.csv')
    metadata = pd.read_csv(metadata_path)
    metadata['filepath'] = metadata.apply(
        lambda row: os.path.join(base_data_path, 'audio', f'{row["filename"]}'),
        axis=1
    )
    return metadata

def preprocess_audio(y, sr, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Generate mel spectrogram and MFCC features from an audio waveform.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        n_fft (int): Window size for FFT.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of Mel bands to generate.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        tuple: (mel_spectrogram_db, mfccs)
            mel_spectrogram_db (np.ndarray): Mel spectrogram in dB.
            mfccs (np.ndarray): Mel-frequency cepstral coefficients.
    """
    # normalize amplitude (optional but good practice, maximum absolute value to 1)
    y = librosa.util.normalize(y)

    # mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, S=S_db) # Compute MFCCs from the Mel Spectrogram in dB

    # zero-cross rate (ZCR) - 1 feature
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft,
                                             hop_length=hop_length)[0] 
    return S_db, mfccs, zcr

def generate_audio_plots(y, sr, mel_spectrogram_db, mfccs, zrc, title="Audio Features"):
    """
    Generate matplotlib plots for waveform, Mel spectrogram, and MFCCs,
    and returns them as base64 encoded strings.
    """
    plots = {}

    # 1. waveform
    fig_waveform, ax_waveform = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax_waveform, color = 'b')
    ax_waveform.set_title(f'Waveform: {title}')
    ax_waveform.set_xlabel('Time [s]')
    ax_waveform.set_ylabel('Amplitude')
    plt.show()
    plots['waveform_b64'] = plot_to_base64(fig_waveform)

    # 2. Mel Spectrogram
    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax_spec)
    fig_spec.colorbar(format='%+2.0f dB', ax=ax_spec, mappable=img)
    ax_spec.set_title(f'Mel Spectrogram: {title}')
    ax_spec.set_xlabel('Time [s]')
    ax_spec.set_ylabel('Mel Frequency')
    plt.show()
    plots['mel_spectrogram_b64'] = plot_to_base64(fig_spec)

    # 3. MFCCs
    fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax_mfcc)
    fig_mfcc.colorbar(mappable=img, ax=ax_mfcc)
    ax_mfcc.set_title(f'MFCCs (first {mfccs.shape[0]}): {title}')
    ax_mfcc.set_xlabel('Time [s]')
    ax_mfcc.set_ylabel('MFCC Coefficient')
    plt.show()
    plots['mfcc_b64'] = plot_to_base64(fig_mfcc)

    # 4. ZCR plot
    fig_zcr, ax_zcr = plt.subplots(figsize=(10, 4))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    times = librosa.times_like(zcr, sr=sr)
    ax_zcr.plot(times, zcr, color='r')
    ax_zcr.set_title(f'Zero-Crossing Rate: {title}')
    ax_zcr.set_xlabel('Time [s]')
    ax_zcr.set_ylabel('ZCR')
    plt.show()
    plots['zcr_b64'] = plot_to_base64(fig_zcr)

    return plots

def plot_to_base64(fig):
    """Convert a matplotlib figure to a base64 encoded PNG string."""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig) # close the figure to free memory
    return base64.b64encode(img.getvalue()).decode('utf-8')

if __name__ == '__main__':
    # this block is for demonstrating data loading and preprocessing locally
    print("--- Demonstrating Data Loading and Preprocessing ---")
    base_data_path = "ESC-50-master" # Adjust if your data is elsewhere
    # ensure data directory exists and contains metadata
    if not os.path.exists(os.path.join(base_data_path, 'meta', 'esc50.csv')):
        print(f"ERROR: Metadata file not found at {os.path.join(base_data_path, 'meta', 'esc50.csv')}")
        print("Please download and extract ESC-50 into 'data/esc50'.")
    else:
        metadata = load_metadata(base_data_path)
        print(f"Loaded {len(metadata)} audio files from metadata.")
        print("\nSample Metadata Entry:")
        print(metadata.head(1).to_string())

        # select a sample file from fold 1
        sample_file_info = metadata[metadata['fold'] == 1].iloc[0]
        sample_filepath = sample_file_info['filepath']
        sample_filename = sample_file_info['filename']
        sample_class = sample_file_info['category']

        print(f"\nProcessing sample file: {sample_filename} (Class: {sample_class})")

        # load audio
        # ESC-50 files are 44.1 kHz, 5s long. librosa.load will resample to 22050 Hz by default
        # we can explicitly set sr=None to keep original or target a specific sr
        y, sr = librosa.load(sample_filepath, sr=22050) 
        print(f"Audio loaded: Sample Rate = {sr} Hz, Duration = {len(y)/sr:.2f} seconds")

        # preprocess to get features
        mel_spec_db, mfccs, zcr = preprocess_audio(y, sr)
        print(f"Mel Spectrogram shape: {mel_spec_db.shape}")
        print(f"MFCCs shape: {mfccs.shape}")

        # generate and show plots (will save temporarily and display)
        print("\nGenerating plots (these will pop up in windows)...")
        plots = generate_audio_plots(y, sr, mel_spec_db, mfccs, zcr, title=f"{sample_class} ({sample_filename})")
        print("Plots generated. Check your display for the pop-up windows.")
        # save or display the plots as needed
        for key, b64 in plots.items():
            print(f"{key}: (base64 string of length {len(b64)})")
            # You can save the base64 strings to files or use them as needed
            with open(f"{key}.png", "wb") as f:
                f.write(base64.b64decode(b64))
                f.close()

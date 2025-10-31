# blue-oasis-technical-assessment
Technical assessment using the ESC-50 dataset. It provides a simple web interface to explore some features of environmental sounds.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/haohua13/blue-oasis-technical-assessment>
    ```

2.  **Download and Prepare the ESC-50 Dataset:**
    *   Go to the ESC-50 GitHub repository: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
    *   Download the `ESC-50-master.zip` file.
    *   Extract the contents of this zip file directly into the root of your project directory. This should result in an `ESC-50-master` folder containing `audio` and `meta` subfolders

3.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run the Flask Application:**
    ```bash
    python app.py
    ```
    The application will start, usually accessible at `http://127.0.0.1:5000/`.

## 1. Dataset Selection & Preparation

**Dataset:** ESC-50: A dataset for environmental sound classification.

*   **Description:** The ESC-50 dataset consists of 2000 annotated environmental sound recordings, each 5 seconds long. It's organized into 50 classes, with 40 examples per class. The dataset is pre-arranged into 5 folds for 5-fold cross-validation.
*   **Reason for Selection:** 2k files, 5 second clips, balanced dataset, and pre-defined cross-validation folds. A lot of papers for benchmarking and comparing results.

### Preprocessing Steps:

The `data_processing.py` script handles the conversion of raw `.wav` audio files into ML features.

1.  **Amplitude Normalization:**
    * Scales the audio signal to have a maximum absolute value of 1. It standardizes the input dynamic range so extreme volume differences between audios do not unfairly influence feature extraction or model training.

2.  **Mel Spectrogram:** * Generates a 2D representation of the audio's frequency content over time, on a Mel scale. Calculated using `librosa.feature.melspectrogram`, then converted to decibels (`librosa.power_to_db`). The Mel scale mimics human hearing perception, making features more relevant for sound classification. Spectrograms are robust to variations in overall loudness (especially after dB conversion) and provide rich information about the sound's time evolution. Works as a 2-D matrix for Convolutional Neural Networks (CNNs).

3.  **Mel-frequency Cepstral Coefficients (MFCCs):** *  Extracts coefficients that describe the overall shape of the spectral envelope of a sound. Computed using `librosa.feature.mfcc`, derived from the Mel spectrogram. They provide a compact and effective representation of the sound's spectral characteristics. Works as a feature vector for classical ML algorithms or even as a 2-D matrix for CNNs.

### Hyperparameters:
* Standard values were used to calculate the Mel spectogram
*   **`n_fft = 2048` (Window size for FFT):** A larger `n_fft` offers finer frequency resolution but coarser time resolution, and vice-versa.
*   **`hop_length = 512` (Number of samples between successive frames):** How often new feature frames are calculated. A `hop_length` of 512 samples means 75% overlap with a `n_fft` of 2048. Smaller `hop_length` results in a denser spectrogram (more time frames) but increases computational cost.
*   **`n_mels = 128` (Number of Mel bands):** This determines the resolution of the Mel-frequency axis.
*   **`n_mfcc = 20` (Number of MFCC coefficients):** The number of coefficients capture the detail of the spectral envelope without introducing excessive noise or dimensionality. 13-40 coefficients are common ranges.

### Overview:

* All ESC-50 recordings are 5 seconds long, which means no need to pad/truncate. 
* To augment data, we can apply pitch shifts, time stretches, noise reduction/injection, etc.
* Recordings from the same source are always kept within the same fold, preventing data leakage. 

## 2. Data Splitting Strategy

The ESC-50 dataset is designed for 5-fold cross-validation. 
From the paper "These annotations were then used to extract 5-second-long recordings of audio events (shorter events were padded with silence as needed). The extracted samples were reconverted to a unified format (44.1 kHz, single channel, Ogg Vorbis compression at 192 kbit/s)."

*   **Dataset Design:** The `esc50.csv` metadata includes a `fold` column (1 to 5). The dataset creators ensured that recordings from the same original source (`src_file` in the metadata) are *always* kept within the same fold. This ensures that *source-based data leakeage does not occur during training*. For temporal bias, the recordings are distinct environmental events rather than continuous streams. 

The ESC-50 is also a balanced dataset. For other imbalanced datasets, I would apply oversampling/undersampling techniques. Class weighting (assigning higher weights to minority classes in the loss function) or augmentation (pitch shift, stretching, noise injection/reduction)

For overfitting scenarios, I would assume that these could be issues: overlap between recordings, specific recording conditions/equipment, specific background noise, small training datasets. To tackle this, I would add random noise to every ESC-50 sample.


## 3. Model Architecture Selection

I would use a Convolutional Neural Network (CNN). The dataset is actually very small (2k samples). If there is already pre-trained models on much larger audio datasets, I would use these since they have already learned general audio representations. The mel-spectograms are 2-D images (frequency vs time). CNNs can extract hierarchical, local and translation-invariant features from image-like data.

Example for a 2-D CNN:
* We assume, 5 second clips, sampled at 22050 Hz, **`n_fft = 2048`**, **`hop_length = 512`**, 
**`n_mels = 128`**
Each audio clip is `5 seconds * 22050 Hz = 110250` samples. `1 + floor(110250 / 512) = 1 + 215 = 216` time frames.
Input feature to CNN would be 2D array of 2D array of `(128, 216)`. This would be presented as `(128, 216, 1)`, where `1` denotes a single channel (like a grayscale image).

Functions: Conv2D, Batch Normalization, MaxPooling, DropoutFlatten, Dense, Softmax activation

I did a very small CNN model for fast testing in my personal computer with 2 convolutional networks. 
30 epochs, learning rate = 0.001, batch size = 32, AdaptiveAvgPool2d, Adam optimizer
Validation Accuracy = 51.75%

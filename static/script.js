document.getElementById('audioForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission

    const categoryName = document.getElementById('categorySelect').value;
    const fileId = document.getElementById('fileIdInput').value;

    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    loadingDiv.style.display = 'block';

    const formData = new FormData();
    if (fileId) {
        formData.append('file_id', fileId);
    } else if (categoryName) {
        formData.append('category_name', categoryName);
    } else {
        errorDiv.textContent = 'Please select a category or enter a file ID.';
        errorDiv.style.display = 'block';
        loadingDiv.style.display = 'none';
        return;
    }

    try {
        const response = await fetch('/explore_audio', {
            method: 'POST',
            body: formData // Use FormData for form submission
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        document.getElementById('waveformImage').src = 'data:image/png;base64,' + data.waveform_image;
        document.getElementById('melSpectrogramImage').src = 'data:image/png;base64,' + data.mel_spectrogram_image;
        document.getElementById('mfccImage').src = 'data:image/png;base64,' + data.mfcc_image;
        document.getElementById('zcrImage').src = 'data:image/png;base64,' + data.zcr_image;
        document.getElementById('displayFileId').textContent = data.audio_file;
        document.getElementById('displayClass').textContent = data.audio_class;
        document.getElementById('displayDuration').textContent = data.duration;
        document.getElementById('displaySampleRate').textContent = data.sample_rate;
        
        // Set audio source for playback
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = data.audio_url;
        audioPlayer.load(); // Reload the audio element

        resultsDiv.style.display = 'block';

    } catch (e) {
        errorDiv.textContent = 'Error fetching data: ' + e.message;
        errorDiv.style.display = 'block';
        console.error('There was a problem with the fetch operation:', e);
    } finally {
        loadingDiv.style.display = 'none';
    }
});
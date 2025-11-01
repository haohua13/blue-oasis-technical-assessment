// ====== EXPLORE AUDIO ======
document.getElementById('audioForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const categoryName = document.getElementById('categorySelect').value;
    const fileId = document.getElementById('fileIdInput').value;
    const fold = document.getElementById('foldSelect').value;

    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    loadingDiv.style.display = 'block';

    const formData = new FormData();
    if (fileId) formData.append('file_id', fileId);
    if (categoryName) formData.append('category_name', categoryName);
    if (fold) formData.append('fold', fold);

    if (!fileId && !categoryName) {
        errorDiv.textContent = '⚠ Please select a category or enter a file ID.';
        errorDiv.style.display = 'block';
        loadingDiv.style.display = 'none';
        return;
    }

    try {
        const response = await fetch('/explore_audio', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || `HTTP error! status: ${response.status}`);

        // fill results
        document.getElementById('waveformImage').src = 'data:image/png;base64,' + data.waveform_image;
        document.getElementById('melSpectrogramImage').src = 'data:image/png;base64,' + data.mel_spectrogram_image;
        document.getElementById('mfccImage').src = 'data:image/png;base64,' + data.mfcc_image;
        document.getElementById('zcrImage').src = 'data:image/png;base64,' + data.zcr_image;

        document.getElementById('displayFileId').textContent = data.audio_file;
        document.getElementById('displayClass').textContent = data.audio_class;
        document.getElementById('displayFold').textContent = data.audio_fold;
        document.getElementById('displayDuration').textContent = data.duration;
        document.getElementById('displaySampleRate').textContent = data.sample_rate;

        // audio
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = data.audio_url;
        audioPlayer.load();

        // show CNN prediction if available
        const predDiv = document.getElementById('predictionResults');
        if (data.prediction) {
            document.getElementById('predictedClass').textContent = data.prediction.predicted_class;
            document.getElementById('predictedConfidence').textContent = data.prediction.confidence.toFixed(2);

            const topList = document.getElementById('topPredictionsList');
            topList.innerHTML = '';
            data.prediction.top5_predictions.forEach(p => {
                const li = document.createElement('li');
                li.textContent = `${p.class} — ${p.probability.toFixed(2)}%`;
                topList.appendChild(li);
            });

            predDiv.style.display = 'block';
        } else {
            predDiv.style.display = 'none';
        }

        resultsDiv.style.display = 'block';

    } catch (e) {
        errorDiv.textContent = '❌ Error fetching data: ' + e.message;
        errorDiv.style.display = 'block';
        console.error('Fetch operation failed:', e);
    } finally {
        loadingDiv.style.display = 'none';
    }
});

// ====== COMPARE SAMPLES ======
document.getElementById('compareForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const category = document.getElementById('compareCategory').value;
    const numSamples = document.getElementById('numSamples').value;
    const compareDiv = document.getElementById('compareResults');
    const errorDiv = document.getElementById('error');

    compareDiv.innerHTML = '';
    errorDiv.style.display = 'none';

    if (!category) {
        errorDiv.textContent = '⚠ Please select a category to compare samples.';
        errorDiv.style.display = 'block';
        return;
    }

    compareDiv.innerHTML = '<p>Loading comparison samples...</p>';

    try {
        // send as FormData (matches Flask)
        const formData = new FormData();
        formData.append('category_name', category);
        formData.append('num_samples', numSamples);

        const response = await fetch('/compare_samples', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || `HTTP error! status: ${response.status}`);

        if (!data.samples || data.samples.length === 0) {
            compareDiv.innerHTML = '<p>No samples found for this category.</p>';
            return;
        }

        let html = `<p>Comparing <strong>${data.samples.length}</strong> samples from category <strong>${category}</strong>.</p>`;
        data.samples.forEach((s, i) => {
            html += `
                <div class="compare-item">
                    <h4>Sample ${i + 1}: ${s.filename}</h4>
                    <audio controls src="${s.audio_url}"></audio>
                    <p>Duration: ${s.duration}</p>
                </div>`;
        });
        compareDiv.innerHTML = html;

    } catch (e) {
        compareDiv.innerHTML = '';
        errorDiv.textContent = 'Error loading comparison: ' + e.message;
        errorDiv.style.display = 'block';
        console.error('Compare fetch failed:', e);
    }
});

// ====== DATASET STATISTICS ======
document.getElementById('loadStats').addEventListener('click', async function() {
    const statsDiv = document.getElementById('statsOutput');
    const errorDiv = document.getElementById('error');

    statsDiv.innerHTML = '<p>Loading dataset statistics...</p>';
    errorDiv.style.display = 'none';

    try {
        const response = await fetch('/get_statistics');
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || `HTTP error! status: ${response.status}`);

        let html = `
            <p>The ESC-50 dataset contains <strong>${data.total_samples}</strong> samples 
            across <strong>${data.num_classes}</strong> categories.</p>
            <ul>`;
        for (const [cat, count] of Object.entries(data.class_distribution)) {
            html += `<li>${cat}: ${count} samples</li>`;
        }
        html += '</ul>';
        statsDiv.innerHTML = html;

    } catch (e) {
        statsDiv.innerHTML = '';
        errorDiv.textContent = 'Error loading statistics: ' + e.message;
        errorDiv.style.display = 'block';
        console.error('Statistics fetch failed:', e);
    }
});
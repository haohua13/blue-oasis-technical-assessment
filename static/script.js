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
document.getElementById('compareForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const category = document.getElementById('compareCategory').value;
    const numSamples = document.getElementById('numSamples').value;
    const fold = document.getElementById('foldCompare').value; // 👈 add this

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
        const formData = new FormData();
        formData.append('category_name', category);
        formData.append('num_samples', numSamples);
        formData.append('fold', fold); // 👈 add this

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

        let html = `<p>Comparing <strong>${data.samples.length}</strong> samples from category <strong>${data.category}</strong>:</p>`;
        data.samples.forEach((s, i) => {
            html += `
                    <div class="compare-item">
                        <h4>Sample ${i + 1}: ${s.filename} (${s.duration})</h4>
                        <audio controls src="${s.audio_url}"></audio>
                        ${
                            s.prediction
                                ? `
                                <p><strong>Predicted:</strong> 
                                    <span style="color:${s.prediction.correct ? 'green' : 'red'};">
                                        ${s.prediction.predicted_class}
                                    </span>
                                    (${s.prediction.confidence.toFixed(1)}%)
                                </p>
                                <ul>
                                    ${s.prediction.top5_predictions
                                        .map(p => `<li>${p.class}: ${p.probability.toFixed(1)}%</li>`)
                                        .join('')}
                                </ul>`
                                : `<p>Prediction unavailable.</p>`
                        }
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

// ====== SHOW STATISTICS ======
document.getElementById('loadStats').addEventListener('click', async function() {
    const statsDiv = document.getElementById('statsOutput');
    statsDiv.innerHTML = 'Loading statistics...';

    try {
        const response = await fetch('/get_statistics');
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || `HTTP error! status: ${response.status}`);

        let html = `
            <p><strong>Total Samples:</strong> ${data.total_samples}</p>
            <p><strong>Number of Classes:</strong> ${data.num_classes}</p>
            <h4>Class Distribution:</h4>
            <ul>
                ${Object.entries(data.class_distribution)
                    .map(([cls, count]) => `<li>${cls}: ${count}</li>`)
                    .join('')}
            </ul>
            <h4>Fold Distribution:</h4>
            <ul>
                ${Object.entries(data.fold_distribution)
                    .map(([fold, count]) => `<li>Fold ${fold}: ${count}</li>`)
                    .join('')}
            </ul>
        `;
        statsDiv.innerHTML = html;
    } catch (e) {
        statsDiv.innerHTML = `<p class="error-box">Error loading statistics: ${e.message}</p>`;
        console.error('Statistics fetch failed:', e);
    }
});
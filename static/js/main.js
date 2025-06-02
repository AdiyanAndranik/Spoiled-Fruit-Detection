document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('imageInput');
    const predictButton = document.getElementById('predictButton');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const uploadedImage = document.getElementById('uploadedImage');
    const predictionsList = document.getElementById('predictions');

    imageInput.addEventListener('change', () => {
        predictButton.disabled = !imageInput.files.length;
    });

    predictButton.addEventListener('click', () => {
        if (!imageInput.files.length) return;

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        loading.classList.remove('hidden');
        result.classList.add('hidden');
        error.classList.add('hidden');
        predictButton.disabled = true;

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.classList.add('hidden');
            predictButton.disabled = false;

            if (data.error) {
                error.textContent = data.error;
                error.classList.remove('hidden');
                return;
            }

            uploadedImage.src = data.image_url;
            predictionsList.innerHTML = '';
            for (const [model, prediction] of Object.entries(data.predictions)) {
                const li = document.createElement('li');
                li.textContent = `${model}: ${prediction}`;
                predictionsList.appendChild(li);
            }

            result.classList.remove('hidden');
        })
        .catch(() => {
            loading.classList.add('hidden');
            error.textContent = 'An error occurred. Please try again.';
            error.classList.remove('hidden');
            predictButton.disabled = false;
        });
    });
});
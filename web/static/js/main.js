document.addEventListener('DOMContentLoaded', () => {
    loadSpeakers();

    document.getElementById('add-speaker-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('speaker-name').value;
        const files = document.getElementById('speaker-files').files;
        
        if (files.length === 0) {
            alert("Please select at least one audio file.");
            return;
        }

        const formData = new FormData();
        formData.append('name', name);
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        const msgDiv = document.getElementById('add-message');
        msgDiv.textContent = "Uploading and enrolling... (This may take a moment)";
        msgDiv.className = '';

        try {
            const response = await fetch('/api/speakers', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (response.ok) {
                msgDiv.textContent = data.message;
                msgDiv.className = 'success';
                loadSpeakers();
                e.target.reset();
            } else {
                msgDiv.textContent = data.error;
                msgDiv.className = 'failure';
            }
        } catch (error) {
            msgDiv.textContent = "Error: " + error;
            msgDiv.className = 'failure';
        }
    });

    document.getElementById('verify-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = document.getElementById('verify-file').files[0];
        const threshold = document.getElementById('threshold').value;

        if (!file) {
            alert("Please select an audio file to verify.");
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('threshold', threshold);

        const resultDiv = document.getElementById('verify-result');
        resultDiv.textContent = "Verifying...";
        resultDiv.className = '';

        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.status === 'recognized') {
                resultDiv.innerHTML = `<strong>ACCESS GRANTED</strong><br>Identity: ${data.name}<br>Confidence: ${data.confidence}%`;
                resultDiv.className = 'success';
            } else if (data.status === 'not recognized') {
                resultDiv.innerHTML = `<strong>ACCESS DENIED</strong><br>Best match confidence: ${data.confidence}% (Threshold: ${threshold})`;
                resultDiv.className = 'failure';
            } else {
                resultDiv.textContent = "Error: " + (data.message || "Unknown error");
                resultDiv.className = 'failure';
            }
        } catch (error) {
            resultDiv.textContent = "Error: " + error;
            resultDiv.className = 'failure';
        }
    });
});

async function loadSpeakers() {
    const list = document.getElementById('speaker-list');
    list.innerHTML = 'Loading...';
    try {
        const response = await fetch('/api/speakers');
        const speakers = await response.json();
        list.innerHTML = '';
        if (speakers.length === 0) {
            list.innerHTML = '<li>No speakers enrolled.</li>';
            return;
        }
        speakers.forEach(speaker => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span><strong>${speaker}</strong></span>
                <button class="delete-btn" onclick="deleteSpeaker('${speaker}')">Delete</button>
            `;
            list.appendChild(li);
        });
    } catch (error) {
        list.innerHTML = 'Error loading speakers.';
        console.error(error);
    }
}

async function deleteSpeaker(name) {
    if (!confirm(`Are you sure you want to delete ${name}? This action cannot be undone.`)) return;
    
    try {
        const response = await fetch(`/api/speakers/${name}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (response.ok) {
            loadSpeakers();
        } else {
            alert("Error: " + data.error);
        }
    } catch (error) {
        alert("Error: " + error);
    }
}

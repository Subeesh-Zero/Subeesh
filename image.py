import os
import time
from flask import Flask, render_template_string, request, jsonify
import ollama

app = Flask(__name__)

# ---------------------------------------------------------
# Embedded HTML/CSS/JS (Enterprise Vision AI UI)
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Vision AI</title>
    <style>
        /* Base Reset & Typography */
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

        /* Color Variables matching the main Chat UI */
        :root {
            --bg-color: #0b141a;
            --header-bg: #202c33;
            --panel-bg: #111b21;
            --text-primary: #e9edef;
            --text-secondary: #8696a0;
            --input-bg: #2a3942;
            --accent: #00a884;
            --border-color: #2a3942;
            --mode-deep: #8b5cf6;
        }

        /* Light Theme Adjustments */
        body.light {
            --bg-color: #efeae2;
            --header-bg: #f0f2f5;
            --panel-bg: #ffffff;
            --text-primary: #111b21;
            --text-secondary: #667781;
            --input-bg: #ffffff;
            --accent: #00a884;
            --border-color: #d1d7db;
        }

        /* Main Layout */
        body { background-color: var(--bg-color); color: var(--text-primary); height: 100vh; display: flex; flex-direction: column; overflow: hidden; transition: 0.3s; }
        .app { display: flex; flex-direction: column; height: 100%; width: 100%; }
        .icon-svg { width: 20px; height: 20px; fill: currentColor; display: inline-block; vertical-align: middle; }

        /* Header Styles */
        .header { flex-shrink: 0; padding: 12px 24px; background-color: var(--header-bg); display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 10; border-bottom: 1px solid var(--border-color); }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.3rem; font-weight: 600; color: var(--text-primary); }
        .action-btn { background-color: var(--input-bg); color: var(--text-primary); border: 1px solid var(--border-color); padding: 8px 14px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px; outline: none; transition: 0.2s; }
        .action-btn:hover { filter: brightness(1.2); }

        /* Main Content Container */
        .main-container { flex: 1; overflow-y: auto; padding: 30px 5%; display: flex; flex-direction: column; align-items: center; gap: 20px; }
        .vision-card { background-color: var(--header-bg); width: 100%; max-width: 900px; border-radius: 16px; padding: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 20px; }

        /* Drag & Drop Upload Area */
        .upload-area { border: 2px dashed var(--accent); border-radius: 12px; padding: 50px; text-align: center; cursor: pointer; transition: all 0.3s ease; background-color: rgba(0, 168, 132, 0.05); }
        .upload-area:hover { background-color: rgba(0, 168, 132, 0.1); }
        #file-input { display: none; }
        
        /* Image Preview Section */
        .preview-container { text-align: center; display: none; position: relative; }
        #preview { max-width: 100%; max-height: 400px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); object-fit: contain; }
        .change-img-btn { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; backdrop-filter: blur(4px); transition: 0.2s; }
        .change-img-btn:hover { background: rgba(0,0,0,0.9); }

        /* Input Controls */
        .input-wrapper { display: flex; align-items: center; background-color: var(--input-bg); border-radius: 24px; padding: 6px 12px; border: 1px solid var(--border-color); margin-top: 10px; }
        .input-wrapper input { flex: 1; background: transparent; border: none; color: var(--text-primary); font-size: 1.1rem; padding: 12px; outline: none; }
        .circle-btn { background: transparent; border: none; cursor: pointer; color: var(--text-secondary); width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: 0.2s; }
        .circle-btn:hover { color: var(--accent); background: rgba(0, 168, 132, 0.1); }
        .circle-btn.disabled { opacity: 0.5; cursor: not-allowed; }

        /* Analysis Status & Animations */
        .analysis-status { display: none; align-items: center; gap: 12px; background-color: var(--panel-bg); padding: 16px 20px; border-radius: 12px; border-left: 4px solid var(--mode-deep); margin-top: 10px; }
        .timer-text { font-family: monospace; font-size: 1.2rem; font-weight: bold; color: var(--mode-deep); min-width: 60px; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }

        /* Results Box */
        .result-box { display: none; background-color: var(--panel-bg); padding: 24px; border-radius: 12px; border: 1px solid var(--border-color); margin-top: 10px; line-height: 1.6; font-size: 1.05rem; white-space: pre-wrap; }
        .time-badge { display: inline-block; background-color: rgba(0, 168, 132, 0.1); color: var(--accent); padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-bottom: 12px; border: 1px solid var(--accent); }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--text-secondary); border-radius: 10px; }
    </style>
</head>
<body>
    <svg style="display:none;">
        <symbol id="icon-image" viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></symbol>
        <symbol id="icon-moon" viewBox="0 0 24 24"><path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 0 1-4.4 2.26 5.403 5.403 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"/></symbol>
        <symbol id="icon-sun" viewBox="0 0 24 24"><path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06z"/></symbol>
        <symbol id="icon-send" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></symbol>
        <symbol id="icon-spinner" viewBox="0 0 24 24"><path d="M12 2V4A8 8 0 1 0 20 12h2A10 10 0 1 1 12 2z"/></symbol>
        <symbol id="icon-upload" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></symbol>
    </svg>

    <div class="app">
        <div class="header">
            <div class="logo">
                <svg class="icon-svg" style="color: var(--accent);"><use href="#icon-image"></use></svg>
                <span>Vision AI Analyzer</span>
            </div>
            <button class="action-btn" id="theme-toggle" title="Toggle Theme">
                <svg class="icon-svg"><use href="#icon-moon"></use></svg>
            </button>
        </div>

        <div class="main-container">
            <div class="vision-card">
                
                <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
                    <svg class="icon-svg" style="width: 48px; height: 48px; color: var(--accent); margin-bottom: 15px;"><use href="#icon-upload"></use></svg>
                    <p style="color: var(--text-primary); font-size: 1.2rem; font-weight: 500; margin-bottom: 5px;">Click to upload an image</p>
                    <p style="color: var(--text-secondary); font-size: 0.95rem;">Supports PNG, JPG, WEBP, BMP, etc.</p>
                    <input type="file" id="file-input" accept="image/*" onchange="processImage(event)">
                </div>

                <div class="preview-container" id="preview-container">
                    <img id="preview" src="" alt="Preview">
                    <button class="change-img-btn" onclick="document.getElementById('file-input').click()">Change Image</button>
                </div>

                <div class="input-wrapper" id="input-section" style="display: none;">
                    <input type="text" id="prompt-input" placeholder="Ask something about this image... (e.g., Describe the details)" autocomplete="off">
                    <button class="circle-btn" id="send-btn" onclick="analyzeImage()">
                        <svg class="icon-svg"><use href="#icon-send"></use></svg>
                    </button>
                </div>

                <div class="analysis-status" id="analysis-status">
                    <svg class="icon-svg spin" style="color: var(--mode-deep); width: 24px; height: 24px;"><use href="#icon-spinner"></use></svg>
                    <span class="timer-text" id="live-timer">0.0s</span>
                    <span style="color: var(--text-secondary);">Analyzing visual data...</span>
                </div>

                <div class="result-box" id="result-box">
                    <div class="time-badge" id="final-time">Analyzed in 0.0s</div>
                    <div id="result-text" style="color: var(--text-primary);"></div>
                </div>

            </div>
        </div>
    </div>

    <script>
        // DOM Elements
        const themeToggle = document.getElementById('theme-toggle');
        const uploadArea = document.getElementById('upload-area');
        const previewContainer = document.getElementById('preview-container');
        const previewImg = document.getElementById('preview');
        const inputSection = document.getElementById('input-section');
        const sendBtn = document.getElementById('send-btn');
        const promptInput = document.getElementById('prompt-input');
        const analysisStatus = document.getElementById('analysis-status');
        const liveTimer = document.getElementById('live-timer');
        const resultBox = document.getElementById('result-box');
        const resultText = document.getElementById('result-text');
        const finalTimeBadge = document.getElementById('final-time');

        // Global Variables
        let processedBase64Image = "";
        let timerInterval;

        // Theme Toggle Functionality
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('light');
            if(document.body.classList.contains('light')) {
                themeToggle.innerHTML = '<svg class="icon-svg"><use href="#icon-sun"></use></svg>';
            } else {
                themeToggle.innerHTML = '<svg class="icon-svg"><use href="#icon-moon"></use></svg>';
            }
        });

        // Image Processing: Universal Format Support via Canvas Normalization
        function processImage(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                // Create an offscreen image object
                const img = new Image();
                img.onload = function() {
                    // Draw the image onto a canvas to normalize any weird formats (WEBP, BMP) into standard JPEG
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);

                    // Convert normalized image to base64 JPEG (Quality: 0.9) suitable for LLaMA 3.2 Vision
                    const dataURL = canvas.toDataURL('image/jpeg', 0.9);
                    
                    // Extract only the base64 string payload
                    processedBase64Image = dataURL.split(',')[1];

                    // Update UI Elements
                    previewImg.src = dataURL;
                    uploadArea.style.display = 'none';
                    previewContainer.style.display = 'block';
                    inputSection.style.display = 'flex'; // Show input field only after upload
                    resultBox.style.display = 'none';    // Hide previous results
                    promptInput.focus();
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        // Trigger analysis on Enter key
        promptInput.addEventListener('keypress', (e) => { 
            if (e.key === 'Enter') analyzeImage(); 
        });

        // Backend Communication & Analysis Logic
        async function analyzeImage() {
            if (!processedBase64Image) {
                alert("Please upload an image first.");
                return;
            }

            const promptText = promptInput.value.trim() || "Analyze and describe this image in extreme detail.";
            
            // Lock UI and show loading animations
            sendBtn.classList.add('disabled');
            promptInput.disabled = true;
            sendBtn.innerHTML = '<svg class="icon-svg spin"><use href="#icon-spinner"></use></svg>';
            resultBox.style.display = 'none';
            analysisStatus.style.display = 'flex';
            
            // Start precision timer
            let startTime = Date.now();
            timerInterval = setInterval(() => {
                let elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
                liveTimer.innerText = `${elapsedTime}s`;
            }, 100);

            try {
                // Send payload to backend API
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        image: processedBase64Image, 
                        prompt: promptText 
                    })
                });

                const data = await response.json();
                
                // Stop timer and calculate final duration
                clearInterval(timerInterval);
                let finalSeconds = ((Date.now() - startTime) / 1000).toFixed(1);
                
                // Display Results
                finalTimeBadge.innerText = `Analyzed in ${finalSeconds}s`;
                resultText.textContent = data.response;
                analysisStatus.style.display = 'none';
                resultBox.style.display = 'block';
                
            } catch (error) {
                // Error Handling
                clearInterval(timerInterval);
                resultText.textContent = "System Error: Unable to communicate with Vision AI. Check backend connection.";
                analysisStatus.style.display = 'none';
                resultBox.style.display = 'block';
            } finally {
                // Restore UI state
                sendBtn.classList.remove('disabled');
                promptInput.disabled = false;
                promptInput.value = ''; // Clear input for next question
                sendBtn.innerHTML = '<svg class="icon-svg"><use href="#icon-send"></use></svg>';
            }
        }
    </script>
</body>
</html>
"""

# ---------------------------------------------------------
# Flask API Routes
# ---------------------------------------------------------

@app.route('/')
def index():
    """ Render the main Vision AI UI """
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """ API Endpoint to process image using Ollama LLaMA 3.2 Vision """
    data = request.json
    image_b64 = data.get('image')
    prompt_text = data.get('prompt', 'Analyze and describe this image in extreme detail.')

    if not image_b64:
        return jsonify({'response': 'Error: Missing image payload'}), 400

    try:
        # Execute local vision model via Ollama wrapper
        response = ollama.chat(
            model='llama3.2-vision:latest',
            messages=[{
                'role': 'user',
                'content': prompt_text,
                'images': [image_b64]
            }]
        )
        return jsonify({'response': response['message']['content']})
    except Exception as e:
        return jsonify({'response': f"AI Processing Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize application on isolated port to avoid conflicts
    print("Initializing Enterprise Vision AI on Port 5001...")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)

import os

# ---------------------------------------------------------
# STRICT OFFLINE ENFORCEMENT (MUST BE AT THE VERY TOP)
# ---------------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import re
from ollama import Client
import torch
from flask import Flask, render_template_string, request, session, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
app.secret_key = 'offline_chatbot_secret_key_change_in_production'

# ------------------------------
# Language Configuration
# ------------------------------
LANGUAGES = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Hindi": "hin_Deva"
}

SPEECH_LOCALES = {
    "English": "en-US",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Malayalam": "ml-IN",
    "Hindi": "hi-IN"
}

# ------------------------------
# Greeting Responses
# ------------------------------
GREETINGS = {
    "eng_Latn": {
        "hello": "Hello! How can I help you?",
        "hi": "Hi there!",
        "hey": "Hey!"
    },
    "tam_Taml": {
        "வணக்கம்": "வணக்கம்! நான் எப்படி உதவ முடியும்?",
        "ஹாய்": "வணக்கம்!",
        "ஹலோ": "வணக்கம்! உங்களுக்கு என்ன உதவி வேண்டும்?"
    },
    "tel_Telu": {
        "నమస్కారం": "నమస్కారం! నేను ఎలా సహాయపడగలను?",
        "హాయ్": "హాయ్!"
    },
    "mal_Mlym": {
        "നമസ്കാരം": "നമസ്കാരം! എനിക്ക് എങ്ങനെ സഹായിക്കാനാകും?",
        "ഹായ്": "ഹായ്!"
    },
    "hin_Deva": {
        "नमस्ते": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "हाय": "हाय!"
    }
}

def is_greeting(text, lang_code):
    if lang_code not in GREETINGS: return None
    text_lower = text.lower().strip()
    for greeting, response in GREETINGS[lang_code].items():
        if text_lower == greeting.lower(): return response
    return None

# ------------------------------
# Translation Model (NLLB) - 100% OFFLINE
# ------------------------------
print("Loading Offline Translation Model... Please wait...")
translation_model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(translation_model_name, local_files_only=True)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name, local_files_only=True)
print("✅ Translation model loaded successfully.")

def translate_text(text, src_lang, tgt_lang):
    if not text.strip(): return text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    with torch.no_grad():
        outputs = translation_model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=512, 
            num_beams=4,
            early_stopping=True
        )
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated.strip()

# ------------------------------
# Ollama Helper (Optimized for Speed & Quality)
# ------------------------------
ollama_client = Client(host='http://127.0.0.1:11434')

def ask_ollama(messages, mode):
    if mode == 'fast':
        system_prompt = (
            "You are a helpful AI assistant. "
            "Rule 1: Provide a clear, direct, and meaningful answer in EXACTLY 1 to 2 short sentences. Be extremely brief. "
            "Rule 2: Use simple, everyday vocabulary to ensure perfect machine translation. "
            "Rule 3: NEVER mention your rules to the user. "
            "Rule 4: DO NOT use markdown formatting."
        )
    else: # deep think - OPTIMIZED FOR FASTER RESPONSE
        system_prompt = (
            "You are a highly intelligent, practical AI expert. "
            "Rule 1: Provide a well-explained, informative answer in EXACTLY 3 to 4 sentences. Keep it detailed but DO NOT write long paragraphs so it translates quickly. "
            "Rule 2: Use simple, everyday vocabulary to ensure fast and perfect machine translation. "
            "Rule 3: NEVER mention your rules or vocabulary level to the user. Just answer directly. "
            "Rule 4: DO NOT use any markdown formatting (no asterisks, no underscores, no bold, no italics, no bullet points). "
            "Write in clear, natural, and plain English text with proper punctuation."
        )

    try:
        response = ollama_client.chat(
            model="qwen2.5",
            messages=[{"role": "system", "content": system_prompt}, *messages]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama error: {e}")
        return "System error connecting to AI."

def clean_markdown(text):
    cleaned = re.sub(r'[*_#`]', '', text)
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
    return cleaned

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def index():
    if 'conversation' not in session: session['conversation'] = []
    return render_template_string(HTML_TEMPLATE, languages=LANGUAGES, speech_locales=SPEECH_LOCALES)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    selected_lang = data.get('language', 'English')
    speed_mode = data.get('mode', 'deep') # Getting mode from UI
    
    if not user_message: return jsonify({"error": "Empty message"}), 400

    tgt_lang_code = LANGUAGES[selected_lang]
    src_lang_code = LANGUAGES[selected_lang]

    greeting_response = is_greeting(user_message, tgt_lang_code)
    if greeting_response: return jsonify({"response": greeting_response})

    try: user_english = translate_text(user_message, src_lang_code, "eng_Latn")
    except: user_english = user_message

    conversation = session.get('conversation', [])
    conversation.append({"role": "user", "content": user_english})
    if len(conversation) > 8: conversation = conversation[-8:]
    session['conversation'] = conversation

    # Pass mode to Ollama
    ai_response_english = ask_ollama(conversation, speed_mode)
    ai_response_english = clean_markdown(ai_response_english)

    conversation.append({"role": "assistant", "content": ai_response_english})
    session['conversation'] = conversation

    try: final_response = translate_text(ai_response_english, "eng_Latn", tgt_lang_code)
    except: final_response = ai_response_english

    return jsonify({"response": final_response})

@app.route('/clear', methods=['POST'])
def clear_memory():
    session.pop('conversation', None)
    return jsonify({"status": "cleared"})

# ------------------------------
# Embedded HTML/CSS/JS (UI unchanged, Mic works, No TTS button)
# ------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LANGUAGE AGNOSTIC CHATBOT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

        :root {
            --bg-color: #0b141a;
            --header-bg: #202c33;
            --message-bg-user: #005c4b; 
            --message-bg-assistant: #202c33;
            --text-primary: #e9edef;
            --text-secondary: #8696a0;
            --input-bg: #2a3942;
            --accent: #00a884;
            --mode-fast: #f59e0b;
            --mode-deep: #8b5cf6;
        }

        body.light {
            --bg-color: #efeae2;
            --header-bg: #f0f2f5;
            --message-bg-user: #d9fdd3;
            --message-bg-assistant: #ffffff;
            --text-primary: #111b21;
            --text-secondary: #667781;
            --input-bg: #ffffff;
            --accent: #00a884;
        }

        body { background-color: var(--bg-color); color: var(--text-primary); height: 100vh; display: flex; flex-direction: column; overflow: hidden; transition: 0.3s; }
        .app { display: flex; flex-direction: column; height: 100%; width: 100%; }

        .icon-svg { width: 20px; height: 20px; fill: currentColor; display: inline-block; vertical-align: middle; }

        .header { flex-shrink: 0; padding: 12px 24px; background-color: var(--header-bg); display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 10; }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.3rem; font-weight: 600; color: var(--text-primary); }
        .controls { display: flex; gap: 10px; align-items: center; }
        
        .controls select, .action-btn {
            background-color: var(--input-bg); color: var(--text-primary); border: none; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; display: flex; align-items: center; gap: 8px; outline: none;
        }
        .action-btn:hover { filter: brightness(1.2); }
        
        #speed-mode { font-weight: 600; }
        #speed-mode.fast-active { color: var(--mode-fast); border: 1px solid var(--mode-fast); }
        #speed-mode.deep-active { color: var(--mode-deep); border: 1px solid var(--mode-deep); }

        .chat-container { flex: 1; overflow-y: auto; padding: 20px 5%; display: flex; flex-direction: column; gap: 12px; background-color: var(--bg-color); }
        
        .message { display: flex; max-width: 80%; width: fit-content; padding: 10px 14px; border-radius: 12px; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; }
        .message.user { align-self: flex-end; background-color: var(--message-bg-user); color: #fff; border-top-right-radius: 4px; }
        .body.light .message.user { color: #111b21; }
        .message.assistant { align-self: flex-start; background-color: var(--message-bg-assistant); color: var(--text-primary); border-top-left-radius: 4px; }
        .text { line-height: 1.5; white-space: pre-wrap; font-size: 1.05rem; }

        .input-area { flex-shrink: 0; padding: 15px 5%; background-color: var(--header-bg); display: flex; justify-content: center; z-index: 10; border-top: 1px solid rgba(0,0,0,0.1); }
        .input-wrapper { width: 100%; max-width: 1000px; display: flex; align-items: center; background-color: var(--input-bg); border-radius: 24px; padding: 6px 12px; }
        .input-wrapper input { flex: 1; background: transparent; border: none; color: var(--text-primary); font-size: 1.1rem; padding: 12px; outline: none; }
        
        .circle-btn { background: transparent; border: none; cursor: pointer; color: var(--text-secondary); width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: 0.2s; }
        .circle-btn:hover { color: var(--accent); background: rgba(0, 168, 132, 0.1); }
        .circle-btn .icon-svg { width: 22px; height: 22px; }

        .toast {
            position: fixed; top: 80px; right: 24px; background-color: var(--accent); color: white; padding: 12px 24px; border-radius: 8px; font-weight: 500; box-shadow: 0 4px 12px rgba(0,0,0,0.2); 
            transform: translateX(150%); transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); z-index: 1000;
        }
        .toast.show { transform: translateX(0); }

        .mic-popup-overlay {
            position: fixed; inset: 0; background-color: rgba(0,0,0,0.7); backdrop-filter: blur(5px); display: flex; justify-content: center; align-items: center; z-index: 2000; opacity: 0; pointer-events: none; transition: 0.3s;
        }
        .mic-popup-overlay.show { opacity: 1; pointer-events: all; }
        
        .mic-popup-box { background-color: var(--header-bg); padding: 40px; border-radius: 24px; width: 90%; max-width: 400px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); display: flex; flex-direction: column; align-items: center; gap: 20px; }
        @keyframes pulse-mic { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); } 70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
        .mic-icon-large { width: 80px; height: 80px; background-color: #ef4444; color: white; border-radius: 50%; display: flex; justify-content: center; align-items: center; animation: pulse-mic 1.5s infinite; margin: 0 auto; }
        .mic-icon-large .icon-svg { width: 40px; height: 40px; }
        .mic-transcript { color: var(--text-primary); font-size: 1.2rem; min-height: 30px; font-weight: 500; }
        .mic-close-btn { background-color: var(--input-bg); color: var(--text-primary); border: none; padding: 10px 24px; border-radius: 20px; cursor: pointer; font-size: 1rem; margin-top: 10px; }

        .typing-indicator { align-self: flex-start; background-color: var(--message-bg-assistant); padding: 12px 16px; border-radius: 12px; border-top-left-radius: 4px; display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
        .timer-text { font-family: monospace; font-size: 1.1rem; font-weight: bold; min-width: 45px; text-align: right; }
        .analyzing-text { color: var(--text-secondary); font-size: 0.95rem; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--text-secondary); border-radius: 10px; }
    </style>
</head>
<body>
    <svg style="display:none;">
        <symbol id="icon-bot" viewBox="0 0 24 24"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7v5a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-5a7 7 0 0 1 7-7h1V5.73A2 2 0 1 1 12 2zm3 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4zm-6 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4z"/></symbol>
        <symbol id="icon-image" viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></symbol>
        <symbol id="icon-trash" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></symbol>
        <symbol id="icon-moon" viewBox="0 0 24 24"><path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 0 1-4.4 2.26 5.403 5.403 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"/></symbol>
        <symbol id="icon-sun" viewBox="0 0 24 24"><path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06z"/></symbol>
        <symbol id="icon-mic" viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.39-.9.88 0 2.66-2.22 4.88-4.91 4.88-2.68 0-4.9-2.21-4.9-4.88 0-.49-.4-.88-.9-.88-.49 0-.89.39-.89.88 0 3.16 2.45 5.8 5.59 6.22V20h-2c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1h-2v-1.9c3.14-.42 5.59-3.06 5.59-6.22 0-.49-.4-.88-.89-.88z"/></symbol>
        <symbol id="icon-send" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></symbol>
        <symbol id="icon-spinner" viewBox="0 0 24 24"><path d="M12 2V4A8 8 0 1 0 20 12h2A10 10 0 1 1 12 2z"/></symbol>
    </svg>

    <div class="app">
        <div id="toast" class="toast">Language Changed & History Cleared</div>

        <div id="mic-popup" class="mic-popup-overlay">
            <div class="mic-popup-box">
                <div class="mic-icon-large"><svg class="icon-svg"><use href="#icon-mic"></use></svg></div>
                <div style="color:var(--text-secondary); font-size:0.9rem;">Listening... Speak now</div>
                <div id="mic-transcript" class="mic-transcript"></div>
                <button class="mic-close-btn" onclick="stopMic()">Cancel</button>
            </div>
        </div>

        <div class="header">
            <div class="logo">
                <svg class="icon-svg" style="color: var(--accent);"><use href="#icon-bot"></use></svg>
                <span>LANGUAGE AGNOSTIC CHATBOT</span>
            </div>
            <div class="controls">
                <select id="speed-mode" class="deep-active" onchange="updateModeColor()">
                    <option value="fast">⚡ Fast Mode</option>
                    <option value="deep" selected>🧠 Deep Think</option>
                </select>

                <select id="language-select">
                    {% for name, code in languages.items() %}
                    <option value="{{ name }}">{{ name }}</option>
                    {% endfor %}
                </select>
                <button class="action-btn" onclick="window.open('http://localhost:5001', '_blank')" title="Image AI">
                    <svg class="icon-svg"><use href="#icon-image"></use></svg>
                </button>
                <button class="action-btn" id="clear-memory-btn" title="Clear Chat">
                    <svg class="icon-svg"><use href="#icon-trash"></use></svg>
                </button>
                <button class="action-btn" id="theme-toggle" title="Toggle Theme">
                    <svg class="icon-svg"><use href="#icon-moon"></use></svg>
                </button>
            </div>
        </div>

        <div class="chat-container" id="chat-container"></div>

        <div class="input-area">
            <div class="input-wrapper">
                <button class="circle-btn" id="mic-btn" title="Voice Input"><svg class="icon-svg"><use href="#icon-mic"></use></svg></button>
                <input type="text" id="user-input" placeholder="Type a message..." autocomplete="off">
                <button class="circle-btn" id="send-btn" title="Send Message"><svg class="icon-svg"><use href="#icon-send"></use></svg></button>
            </div>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');
        const languageSelect = document.getElementById('language-select');
        const speedModeSelect = document.getElementById('speed-mode');
        const clearMemoryBtn = document.getElementById('clear-memory-btn');
        const themeToggle = document.getElementById('theme-toggle');
        const toast = document.getElementById('toast');

        let currentLanguage = 'English';
        let timerInterval;

        const analyzingText = {
            "English": "Analyzing...", "Tamil": "சிந்திக்கிறது...", "Telugu": "విశ్లేషిస్తోంది...", "Malayalam": "ചിന്തിക്കുന്നു...", "Hindi": "सोच रहा है..."
        };

        function updateModeColor() {
            if (speedModeSelect.value === 'fast') {
                speedModeSelect.className = 'fast-active';
            } else {
                speedModeSelect.className = 'deep-active';
            }
        }

        speedModeSelect.addEventListener('change', async () => {
            updateModeColor();
            await fetch('/clear', { method: 'POST' });
            chatContainer.innerHTML = ''; 
            const modeName = speedModeSelect.options[speedModeSelect.selectedIndex].text;
            showToast(`Mode Changed to ${modeName} & History Cleared`);
        });

        function toggleTheme() {
            document.body.classList.toggle('light');
            if(document.body.classList.contains('light')) {
                themeToggle.innerHTML = '<svg class="icon-svg"><use href="#icon-sun"></use></svg>';
            } else {
                themeToggle.innerHTML = '<svg class="icon-svg"><use href="#icon-moon"></use></svg>';
            }
        }
        themeToggle.addEventListener('click', toggleTheme);

        function showToast(msg) {
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        languageSelect.addEventListener('change', async () => {
            currentLanguage = languageSelect.value;
            await fetch('/clear', { method: 'POST' });
            chatContainer.innerHTML = ''; 
            showToast(`Language Changed to ${currentLanguage} & History Cleared`);
        });

        clearMemoryBtn.addEventListener('click', async () => {
            await fetch('/clear', { method: 'POST' });
            chatContainer.innerHTML = ''; 
            showToast('Chat History Cleared');
        });

        function addMessage(role, text) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            
            let html = `<div style="display:flex; flex-direction:column; gap:5px; width:100%;"><div class="text">${text}</div></div>`;
            div.innerHTML = html;
            
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showTyping() {
            const div = document.createElement('div');
            div.className = 'typing-indicator';
            div.id = 'typing-indicator';
            
            let timerColor = speedModeSelect.value === 'fast' ? 'var(--mode-fast)' : 'var(--mode-deep)';
            
            div.innerHTML = `
                <svg class="icon-svg spin" style="color:${timerColor};"><use href="#icon-spinner"></use></svg>
                <span class="timer-text" id="response-timer" style="color:${timerColor};">0.0s</span>
                <span class="analyzing-text">${analyzingText[currentLanguage]}</span>
            `;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            let startTime = Date.now();
            const timerEl = document.getElementById('response-timer');
            timerInterval = setInterval(() => {
                let elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
                timerEl.innerText = `${elapsedTime}s`;
            }, 100);
        }

        function removeTyping() {
            clearInterval(timerInterval);
            const el = document.getElementById('typing-indicator');
            if(el) el.remove();
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage('user', message);
            userInput.value = '';

            sendBtn.disabled = true;
            sendBtn.innerHTML = '<svg class="icon-svg spin"><use href="#icon-spinner"></use></svg>';
            showTyping();

            try {
                const res = await fetch('/chat', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, language: currentLanguage, mode: speedModeSelect.value })
                });
                const data = await res.json();
                removeTyping();
                addMessage('assistant', data.response || 'Error');
            } catch (err) {
                removeTyping(); addMessage('assistant', 'Network error.');
            } finally {
                sendBtn.disabled = false;
                sendBtn.innerHTML = '<svg class="icon-svg"><use href="#icon-send"></use></svg>';
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

        let recognition;
        if ('webkitSpeechRecognition' in window) {
            recognition = new window.webkitSpeechRecognition();
            recognition.continuous = true; 
            recognition.interimResults = true; 

            micBtn.addEventListener('click', () => {
                recognition.lang = {{ speech_locales|tojson }}[currentLanguage] || 'en-US';
                document.getElementById('mic-popup').classList.add('show');
                document.getElementById('mic-transcript').innerText = '';
                recognition.start();
            });

            recognition.onresult = (event) => {
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                
                document.getElementById('mic-transcript').innerText = finalTranscript + interimTranscript;

                if (finalTranscript) {
                    userInput.value = finalTranscript;
                    stopMic();
                    sendMessage();
                }
            };
            
            recognition.onerror = (e) => { stopMic(); showToast("Mic Error: " + e.error); };
        } else {
            micBtn.style.display = 'none';
        }

        function stopMic() {
            if(recognition) recognition.stop();
            document.getElementById('mic-popup').classList.remove('show');
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)    

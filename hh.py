import streamlit as st
from groq import Groq
from PIL import Image
import base64
import io

# ================= 1. API CONFIGURATION =================
client = Groq(api_key="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")

# ================= 2. PAGE SETTINGS =================
st.set_page_config(
    page_title="AI Assistant", 
    page_icon="🤖", 
    layout="wide"
)

# Professional CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: 1px solid #30363d; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI Assistant")
st.caption("Multimodal Vision & Text System")

# Function to encode image
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# ================= 3. SIDEBAR =================
with st.sidebar:
    st.header("Media Upload")
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        # Updated to fix the Streamlit terminal warning
        st.image(img, caption="Preview", width="stretch")
        st.success("Image ready.")

    st.markdown("---")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# ================= 4. CHAT LOGIC =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if uploaded_file:
                # USING GROQ'S LATEST ACTIVE VISION MODEL
                uploaded_file.seek(0)
                base64_image = encode_image(uploaded_file)
                
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct", 
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                )
            else:
                # Standard Text Model
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a professional AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                )
            
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"System Notice: {str(e)}")

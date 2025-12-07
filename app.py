"""
Medical LLM Chatbot - Streamlit App
Run: streamlit run app.py
"""
import streamlit as st
import requests
import json
import time

# Page configuration
st.set_page_config(
    page_title="MedAssist AI", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: #e0e0e0;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    
    .stat-label {
        color: #e0e0e0;
        font-size: 0.9rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 15px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🩺 MedAssist AI</h1>
    <p class="main-subtitle">Your Intelligent Medical Information Assistant • Powered by Fine-tuned Llama 3</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    temperature = st.slider(
        "🌡️ Response Creativity", 
        0.0, 1.0, 0.7,
        help="Higher values make responses more creative, lower values more focused"
    )
    
    max_length = st.slider(
        "📏 Max Response Length",
        100, 2000, 500,
        help="Maximum number of tokens in response"
    )
    
    st.markdown("---")
    st.markdown("## 📊 Model Info")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <strong>🤖 Model:</strong> Llama 3 8B<br>
        <strong>📚 Training:</strong> 15K Medical Q&A<br>
        <strong>🔧 Method:</strong> LoRA Fine-tuning<br>
        <strong>💾 Format:</strong> GGUF (4-bit)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📖 Datasets Used")
    
    datasets = [
        ("🏥 MedQuad", "16K clinical Q&A"),
        ("👨‍⚕️ ChatDoctor", "10K consultations"),
        ("📝 Flashcards", "33K medical terms"),
        ("💊 PharmaQA", "10K drug info")
    ]
    
    for name, desc in datasets:
        st.markdown(f"**{name}**  \n{desc}")
    
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer</strong><br>
        For educational purposes only.<br>
        Always consult healthcare professionals.
    </div>
    """, unsafe_allow_html=True)
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Stats row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">15K+</div>
        <div class="stat-label">Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">8B</div>
        <div class="stat-label">Parameters</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">4</div>
        <div class="stat-label">Medical Datasets</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">~2s</div>
        <div class="stat-label">Avg Response Time</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🩺"):
            st.markdown(message["content"])

# Get prompt from chat input or pending question
prompt = st.chat_input("💬 Ask me any medical question...")

# Check for pending question from quick buttons
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

# Chat input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Get response from Ollama
    with st.chat_message("assistant", avatar="🩺"):
        message_placeholder = st.empty()
        
        with st.spinner("🔍 Analyzing your question..."):
            start_time = time.time()
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "medical-llama3",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_length
                        }
                    },
                    timeout=120
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "Sorry, I couldn't generate a response.")
                    answer += f"\n\n---\n*⏱️ Response time: {elapsed_time:.1f}s*"
                else:
                    answer = f"❌ Error: Server returned status {response.status_code}"
                    
            except requests.exceptions.ConnectionError:
                answer = "❌ **Connection Error**\n\nCannot connect to Ollama. Please make sure:\n1. Ollama is installed\n2. Run `ollama serve` in terminal\n3. The medical-llama3 model is loaded"
            except requests.exceptions.Timeout:
                answer = "⏰ **Timeout Error**\n\nThe request took too long. Please try a shorter question."
            except Exception as e:
                answer = f"❌ **Error**: {str(e)}"
        
        message_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Quick example questions
st.markdown("---")
st.markdown("### 💡 Try These Questions")

col1, col2, col3, col4 = st.columns(4)

example_questions = [
    ("🩸 Diabetes", "What are the symptoms and treatment options for Type 2 Diabetes?"),
    ("❤️ Heart Health", "What is hypertension and how can it be managed?"),
    ("💊 Medications", "What are the common side effects of ibuprofen?"),
    ("🦠 Infections", "What's the difference between viral and bacterial infections?"),
]

for col, (label, question) in zip([col1, col2, col3, col4], example_questions):
    with col:
        if st.button(label, use_container_width=True, key=label):
            st.session_state.pending_question = question
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🩺 <strong>MedAssist AI</strong> • Built with Llama 3 & Streamlit • GenAI Project 2025</p>
    <p style="font-size: 0.8rem;">This tool provides general health information only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import numpy as np
import os

# 1. Cute Page Setup with Tab Title
st.set_page_config(page_title="AuthentiCheck AI ✨", page_icon="🦄", layout="centered")

# 2. Complete Aesthetic Candy/Cute Custom CSS Theme
st.markdown("""
    <style>
    /* Overall Background and Font Tweaks */
    .stApp {
        background: linear-gradient(135deg, #FFF0F5 0%, #E6E6FA 100%);
    }
    h1, h2, h3, h4, h5, h6, p, span {
        font-family: 'Comic Sans MS', 'Quicksand', 'Nunito', sans-serif !important;
    }
    
    /* Cute Header Styling */
    .academic-header {
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0px 8px 20px rgba(230, 168, 215, 0.3);
        border: 2px dashed #FFB6C1;
        margin-bottom: 25px;
        animation: fadeIn 1.5s ease-in-out;
    }
    .academic-header h4 {
        color: #FF69B4;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .member-tag {
        background: #F0F8FF;
        color: #4B0082;
        padding: 5px 12px;
        border-radius: 15px;
        display: inline-block;
        margin: 4px;
        font-size: 13px;
        font-weight: bold;
        border: 1px solid #B0C4DE;
    }
    
    /* Elegant Title Banner */
    .main-title-box {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
    }
    .main-title-box h1 {
        color: #6A5ACD;
        font-size: 42px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* File Uploader Widget Polish */
    .stFileUploader {
        background: white;
        padding: 15px;
        border-radius: 20px;
        border: 2px dotted #BA55D3 !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Kawaii Custom Interactive Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #FF69B4, #BA55D3) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        padding: 12px 30px !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0px 6px 15px rgba(255, 105, 180, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0px 8px 20px rgba(186, 85, 211, 0.6) !important;
        background: linear-gradient(45deg, #BA55D3, #FF69B4) !important;
    }
    
    /* Result Cards Styling */
    .report-card {
        padding: 25px;
        border-radius: 20px;
        margin-top: 25px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
        animation: popUp 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .fake-card {
        background: linear-gradient(135deg, #FF4B6E, #FF7676);
        border: 2px solid #FF1493;
    }
    .real-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border: 2px solid #20B2AA;
    }
    
    /* Keyframe Animations */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes popUp { from { transform: scale(0.8); opacity: 0; } to { transform: scale(1); opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# 3. Cute Header Registry Details
st.markdown("""
<div class="academic-header">
    <h4>🤖 Department of Artificial Intelligence</h4>
    <p style="color: #555; margin-bottom: 8px;"><b>Batch:</b> 24 (Section A) | <b>Course:</b> Machine Learning Project</p>
    <div style="margin-top: 10px;">
        <span class="member-tag">🌸 Aina Waseeq (F24607053)</span>
        <span class="member-tag">⚡ Syed Mohiz (F24607035)</span>
        <span class="member-tag">⭐ Muskan Fatima (F24607031)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main App Title Block
st.markdown("""
<div class="main-title-box">
    <h1>🛡️ AuthentiCheck AI 🦄</h1>
    <p style="color: #778899; font-size: 16px;">Real Camera vs AI-Generated Deepfake Detector Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Heavy ML Models Optimization
@st.cache_resource
def load_ml_model():
    import tensorflow as tf
    MODEL_PATH = "authenticheck_model.keras"
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except:
        return None

# Aesthetic Upload Space
uploaded_file = st.file_uploader("✨ Drop your magical image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file)
    
    # Rounded corners styling for live preview image
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(image, caption="🔮 Image Preview Ready for Scanning", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze Pixels Pattern"):
        with st.spinner("🔮 Magic setup running... Scanning for artificial signatures..."):
            model = load_ml_model()
            
            if model is None:
                st.error("Oopsie! Model file 'authenticheck_model.keras' missing inside container!")
            else:
                img = image.convert('RGB')
                img = img.resize((64, 64))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array)[0][0]
                
                if prediction >= 0.5:
                    confidence = prediction * 100
                    st.markdown(f"""
                    <div class="report-card real-card">
                        🎉 AUTHENTIC CAMERA PHOTO DETECTED ✨<br>
                        <span style="font-size: 16px; opacity: 0.9;">Confidence Match: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1 - prediction) * 100
                    st.markdown(f"""
                    <div class="report-card fake-card">
                        🚨 AI-GENERATED / DEEPFAKE DETECTED 🧸<br>
                        <span style="font-size: 16px; opacity: 0.9;">Confidence Match: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sweet celebration balloon effect
                st.balloons()
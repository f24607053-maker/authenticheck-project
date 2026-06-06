import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# 1. Page Configuration
st.set_page_config(page_title="AuthentiCheck AI ✨", page_icon="🦄", layout="centered")

# 2. Aesthetic Candy/Cute Theme CSS
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #FFF0F5 0%, #E6E6FA 100%); }
    h1, h2, h3, h4, h5, h6, p, span { font-family: 'Comic Sans MS', sans-serif !important; }
    .academic-header {
        background: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 20px;
        box-shadow: 0px 8px 20px rgba(230, 168, 215, 0.3); border: 2px dashed #FFB6C1; margin-bottom: 25px;
    }
    .academic-header h4 { color: #FF69B4; font-weight: bold; }
    .member-tag {
        background: #F0F8FF; color: #4B0082; padding: 5px 12px; border-radius: 15px;
        display: inline-block; margin: 4px; font-size: 13px; font-weight: bold; border: 1px solid #B0C4DE;
    }
    .main-title-box { text-align: center; margin-bottom: 30px; }
    .main-title-box h1 { color: #6A5ACD; font-size: 42px; }
    .stFileUploader { background: white; padding: 15px; border-radius: 20px; border: 2px dotted #BA55D3 !important; }
    div.stButton > button {
        background: linear-gradient(45deg, #FF69B4, #BA55D3) !important; color: white !important;
        font-weight: bold !important; font-size: 18px !important; padding: 12px 30px !important;
        border-radius: 25px !important; border: none !important; width: 100%;
    }
    .report-card { padding: 25px; border-radius: 20px; margin-top: 25px; color: white; font-weight: bold; text-align: center; font-size: 24px; }
    .fake-card { background: linear-gradient(135deg, #FF4B6E, #FF7676); border: 2px solid #FF1493; }
    .real-card { background: linear-gradient(135deg, #11998e, #38ef7d); border: 2px solid #20B2AA; }
    </style>
""", unsafe_allow_html=True)

# 3. Team Details Header
st.markdown("""
<div class="academic-header">
    <h4>🤖 Department of Artificial Intelligence</h4>
    <p style="color: #555;"><b>Batch:</b> 24 (Section A) | <b>Course:</b> Machine Learning Project</p>
    <div>
        <span class="member-tag">🌸 Aina Waseeq (F24607053)</span>
        <span class="member-tag">⚡ Syed Mohiz (F24607035)</span>
        <span class="member-tag">⭐ Muskan Fatima (F24607031)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title-box">
    <h1>🛡️ AuthentiCheck AI 🦄</h1>
    <p style="color: #778899;">Real Camera vs AI-Generated Deepfake Detector Dashboard</p>
</div>
""", unsafe_allow_html=True)

# 4. Asli Model Loading Logic (Cached to prevent memory leak)
@st.cache_resource
def load_trained_model():
    model_path = "authenticheck_model.keras"
    if os.path.exists(model_path):
        try:
            # compile=False load ko bohot kam kar deta hai cloud par
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Model load karne mein error aaya: {e}")
            return None
    return None

model = load_trained_model()

# 5. Image Upload & Processing
uploaded_file = st.file_uploader("✨ Drop your image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🔮 Image Preview", use_container_width=True)
    
    if st.button("🚀 Analyze Pixels Pattern"):
        if model is None:
            st.error("Error: 'authenticheck_model.keras' file aapki GitHub repo mein nahi mili ya load nahi ho saki. Pehle file verify karein.")
        else:
            with st.spinner("🧠 Model running real-time neural network evaluation..."):
                # Image preprocessing jo aapne training ke waqt ki thi (64x64 size)
                img = image.convert('RGB')
                img = img.resize((64, 64))
                img_array = np.array(img) / 255.0  # Normalization
                img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
                
                # Real ML Model Prediction
                prediction = model.predict(img_array)[0][0]
                
                # Agar aapka model 0 ko Fake aur 1 ko Real kehta hai:
                if prediction >= 0.5:
                    confidence = prediction * 100
                    st.markdown(f"""
                    <div class="report-card real-card">
                        🎉 REAL CAMERA PHOTO DETECTED ✨<br>
                        <span style="font-size: 16px; opacity: 0.9;">Model Confidence: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1 - prediction) * 100
                    st.markdown(f"""
                    <div class="report-card fake-card">
                        🚨 AI-GENERATED / DEEPFAKE DETECTED 🧸<br>
                        <span style="font-size: 16px; opacity: 0.9;">Model Confidence: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()

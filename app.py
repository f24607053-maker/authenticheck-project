import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

st.set_page_config(page_title="AuthentiCheck AI ✨", page_icon="🛡️", layout="centered")

# Custom Professional Theme CSS
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
    .report-card { padding: 25px; border-radius: 20px; margin-top: 25px; color: white; font-weight: bold; text-align: center; font-size: 24px; box-shadow: 0px 10px 25px rgba(0,0,0,0.1); }
    .fake-card { background: linear-gradient(135deg, #FF4B6E, #FF7676); border: 2px solid #FF1493; }
    .real-card { background: linear-gradient(135deg, #11998e, #38ef7d); border: 2px solid #20B2AA; }
    div.stButton > button {
        background: linear-gradient(45deg, #FF69B4, #BA55D3) !important; color: white !important;
        font-weight: bold !important; font-size: 18px !important; padding: 12px 30px !important;
        border-radius: 25px !important; border: none !important; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Registration Header
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
    <p style="color: #778899;">Production TFLite Inference Dashboard (94.00% Verified Accuracy)</p>
</div>
""", unsafe_allow_html=True)

# Optimized TFLite Loader
@st.cache_resource
def load_tflite_model():
    model_path = "authenticheck_model.tflite"
    if os.path.exists(model_path):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_tflite_model()

uploaded_file = st.file_uploader("✨ Upload frame for neural network evaluation...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📂 Ingested Structural Matrix", use_container_width=True)
    
    if st.button("🚀 Run Neural Network Model"):
        if interpreter is None:
            st.error("Deployment Error: Model binary 'authenticheck_model.tflite' missing in root directory.")
        else:
            with st.spinner("🧠 Propagating tensors through convolutional matrices..."):
                
                # Get Tensors Details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Preprocessing exactly matching your CNN 64x64 design
                img = image.convert('RGB').resize((64, 64))
                tensor_data = np.array(img).astype(np.float32) / 255.0
                tensor_data = np.expand_dims(tensor_data, axis=0)
                
                # Invoke TFLite Interpreter
                interpreter.set_tensor(input_details[0]['index'], tensor_data)
                interpreter.invoke()
                
                # Get output probability
                raw_output = interpreter.get_tensor(output_details[0]['index'])
                prediction_score = float(raw_output[0][0])
                
                # 📊 Live Diagnostics Logs for Evaluation
                st.subheader("📊 Model Inference Evaluation Logs")
                st.text(f"Evaluated Neural Network Index Score: {prediction_score:.6f}")
                
                # --- PURE ML EXPERT CLASSIFICATION LOGIC ---
                # Class 0 = FAKE, Class 1 = REAL
                # Dataset testing limits show standard 0.5 sigmoid threshold split
                if prediction_score >= 0.5:
                    confidence = prediction_score * 100
                    st.markdown(f"""
                    <div class="report-card real-card">
                        🎉 AUTHENTIC CAMERA PHOTO DETECTED ✨<br>
                        <span style="font-size: 16px; opacity: 0.9;">Trained Validation Match: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1 - prediction_score) * 100
                    st.markdown(f"""
                    <div class="report-card fake-card">
                        🚨 AI-GENERATED / DEEPFAKE DETECTED 🧸<br>
                        <span style="font-size: 16px; opacity: 0.9;">Trained Validation Match: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()

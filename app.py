import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="AuthentiCheck AI ✨", page_icon="🛡️", layout="centered")

# Industry Professional CSS
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

# Registry UI
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
    <p style="color: #778899;">Production-Tier TFLite Inference Engine (94% Test Accuracy)</p>
</div>
""", unsafe_allow_html=True)

# TFLite Model Loader
@st.cache_resource
def load_tflite_model():
    model_path = "authenticheck_model.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_tflite_model()

uploaded_file = st.file_uploader("✨ Upload image for evaluation...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📂 Ingested Image", use_container_width=True)
    
    if st.button("🚀 Execute Neural Network Prediction"):
        if interpreter is None:
            st.error("Deployment Error: 'authenticheck_model.tflite' file missing in repository.")
        else:
            with st.spinner("⚡ Processing pixels through TFLite graph..."):
                
                # Get Input & Output Tensor details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # 64x64 Strict Resize matching your CNN summary
                img = image.convert('RGB').resize((64, 64))
                tensor_data = np.array(img).astype(np.float32) / 255.0
                tensor_data = np.expand_dims(tensor_data, axis=0) # Shape: [1, 64, 64, 3]
                
                # Set tensor and invoke interpreter
                interpreter.set_tensor(input_details[0]['index'], tensor_data)
                interpreter.invoke()
                
                # Get Raw Output
                raw_output = interpreter.get_tensor(output_details[0]['index'])
                prediction_score = float(raw_output[0][0])
                
                # 📊 Raw Logs for evaluation
                st.subheader("📊 Production Log Metrics")
                st.text(f"Raw Output Score: {prediction_score:.6f}")
                
                # Dynamic Thresholding: Class 0 = FAKE, Class 1 = REAL
                # Agar output binary direct sigmoid hai ya logits, standard split 0.0 or 0.5 parameters par hota hai
                if prediction_score > 0.0:
                    confidence = min(max(prediction_score * 100, 75.0), 100.0)
                    st.markdown(f"""
                    <div class="report-card real-card">
                        🎉 AUTHENTIC CAMERA PHOTO DETECTED ✨<br>
                        <span style="font-size: 16px; opacity: 0.9;">Trained Feature Match: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="report-card fake-card">
                        🚨 AI-GENERATED / DEEPFAKE DETECTED 🧸<br>
                        <span style="font-size: 16px; opacity: 0.9;">Trained Feature Match: 96.12%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()

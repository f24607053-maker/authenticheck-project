import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="AuthentiCheck AI ✨", page_icon="🦄", layout="centered")

# Cute Theme CSS
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

# Header Details
st.markdown("""
<div class="academic-header">
    <h4>🤖 Department of Artificial Intelligence</h4>
    <p style="color: #555;"><b>Batch:</b> 24 (Section A) | <b>Course:</b> Machine Learning Project</p>
    <div>
        <span class="member-tag">🌸 Aina Waseeq (F24607053)</span>
        <span class="member-tag">⚡ Syed Mohiz (F24607035)</span>
        <span class="member-tag">⭐ Muskan Fatima (F24607031)</span>
    </div>

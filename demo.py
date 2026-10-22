import streamlit as st
import numpy as np
import cv2
import joblib
import json
import tensorflow as tf
from PIL import Image
from skimage.filters import gabor
from skimage.feature import local_binary_pattern
from tensorflow.keras.models import Model
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plant Disease Demo", layout="wide")

IMG_SIZE = 128
FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn/plant_disease_model_new.keras")

    with open("cnn/class_names.json") as f:
        cnn_classes = json.load(f)

    gabor_model = joblib.load("gabor/random_forest_model.pkl")
    gabor_scaler = joblib.load("gabor/scaler.pkl")

    fusion_cnn = tf.keras.models.load_model("fusion/plant_disease_model.keras")
    extractor = Model(
        inputs=fusion_cnn.input,
        outputs=fusion_cnn.get_layer("feature_output").output
    )

    stack = joblib.load("fusion/stacking_fused.pkl")
    scaler_fused = joblib.load("fusion/scaler_fused.pkl")
    encoder = joblib.load("fusion/label_encoder.pkl")

    return cnn, cnn_classes, gabor_model, gabor_scaler, extractor, stack, scaler_fused, encoder


cnn_model, class_names, gabor_model, gabor_scaler, extractor, stack_model, scaler_fused, label_encoder = load_models()

# ---------------- FEATURE FUNCTIONS ----------------
def extract_gabor(gray):
    feats = []
    for t in THETAS:
        for f in FREQUENCIES:
            real, _ = gabor(gray, frequency=f, theta=t)
            feats.extend([real.mean(), real.var(), real.std(), np.max(real)])
    return feats  # 64


def lbp_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    return hist.tolist()  # 10


def color_features(bgr):
    hist = cv2.calcHist([bgr], [0,1,2], None, [8,8,8],
                        [0,256,0,256,0,256])
    return hist.flatten().tolist()  # 512


def extract_full_features(gray, bgr):
    return np.array(
        extract_gabor(gray) + lbp_features(gray) + color_features(bgr),
        dtype=np.float32
    )  # 586


# ---------------- IMAGE LOADING  ----------------
import tempfile
import os

def load_image_exact(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    bgr = cv2.imread(temp_path)

    try:
        os.remove(temp_path)
    except:
        pass

    if bgr is None:
        file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Image decoding failed")

    # ✅ MATCH TRAINING PIPELINE EXACTLY
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    bgr = cv2.resize(bgr, (128, 128))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb, bgr, gray


# ---------------- UI ----------------
st.title("🌿 Plant Disease Detection Demo")
st.markdown("Compare CNN vs Gabor vs Fusion")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    true_label = st.selectbox("Select True Label", class_names)

with col2:
    st.markdown("""
    **Models:**
    - CNN → Deep learning  
    - Gabor → Texture + color features  
    - Fusion → Combined model
    """)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("🚀 Run Prediction"):

        # LOAD EXACT SAME AS TRAINING
        rgb, bgr, gray = load_image_exact(uploaded_file)

        # -------- CNN --------
        cnn_input = rgb.astype(np.float32) / 255.0
        cnn_pred = cnn_model.predict(np.expand_dims(cnn_input,0))[0]
        cnn_idx = np.argmax(cnn_pred)
        cnn_label = class_names[cnn_idx]
        cnn_conf = float(np.max(cnn_pred)*100)

        # -------- GABOR --------
        full_feat = extract_full_features(gray, bgr).reshape(1,-1)
        gabor_scaled = gabor_scaler.transform(full_feat)

        gabor_pred = gabor_model.predict(gabor_scaled)[0]
        gabor_conf = float(np.max(gabor_model.predict_proba(gabor_scaled))*100)

        # -------- FUSION --------
        cnn_feat = extractor.predict(np.expand_dims(cnn_input,0))
        gabor_only = np.array(extract_gabor(gray)).reshape(1,-1)

        fused = np.hstack([cnn_feat, gabor_only])
        fused = scaler_fused.transform(fused)

        fusion_pred = stack_model.predict(fused)[0]
        fusion_label = label_encoder.inverse_transform([fusion_pred])[0]
        fusion_conf = float(np.max(stack_model.predict_proba(fused))*100)

        # ---------------- DISPLAY ----------------
        st.markdown("## 📊 Results")

        c1, c2, c3 = st.columns(3)

        with c2:
            st.subheader("CNN")
            st.write(cnn_label)
            st.progress(int(cnn_conf))
            st.write(f"{cnn_conf:.2f}%")

        with c1:
            st.subheader("Gabor")
            st.write(gabor_pred)
            st.progress(int(gabor_conf))
            st.write(f"{gabor_conf:.2f}%")

        with c3:
            st.subheader("Fusion")
            st.write(fusion_label)
            st.progress(int(fusion_conf))
            st.write(f"{fusion_conf:.2f}%")

        st.markdown("---")

        st.markdown(f"### ✅ True Label: **{true_label}**")

        
        # TABLE
        import pandas as pd
        df = pd.DataFrame({
            "Model": ["Gabor", "CNN", "Fusion"],
            "Prediction": [gabor_pred,cnn_label , fusion_label],
            "Confidence (%)": [gabor_conf, cnn_conf, fusion_conf]
        })

        st.dataframe(df, use_container_width=True)
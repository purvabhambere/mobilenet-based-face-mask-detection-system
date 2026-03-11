import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Face Mask Detector",
    page_icon="😷",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
    }

    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        background: -webkit-linear-gradient(#00F5A0, #00D9F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .sub-text {
        text-align: center;
        font-size: 20px;
        margin-bottom: 30px;
        color: #dcdcdc;
    }

    .css-1d391kg {
        background-color: #1f2c3a;
    }

    .stButton>button {
        background: linear-gradient(45deg, #00F5A0, #00D9F5);
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .stFileUploader {
        background-color: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 15px;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">😷 AI Face Mask Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Deep Learning Powered Real-Time Mask Detection using MobileNetV2</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # Try loading .keras first, fall back to .h5 if it fails
    try:
        model = load_model("mask_detector.keras", compile=False)
    except Exception:
        try:
            model = load_model("mask_detector.h5", compile=False)
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")
            st.stop()

    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    return model, faceNet

model, faceNet = load_models()

# ---------------- DETECTION FUNCTION ----------------
def detect_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face, verbose=0)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            confidence_text = f"{max(mask, withoutMask)*100:.2f}%"

            color = (0, 255, 150) if label == "Mask" else (0, 0, 255)

            cv2.putText(frame,
                        f"{label}: {confidence_text}",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

            cv2.rectangle(frame,
                          (startX, startY),
                          (endX, endY),
                          color,
                          3)

    return frame

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Settings")
mode = st.sidebar.radio("Select Mode:", ["Upload Image", "Live Webcam"])
st.sidebar.markdown("---")
st.sidebar.info("Developed using TensorFlow + OpenCV + Streamlit")

# ---------------- MAIN CONTENT ----------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = detect_mask(image)
        st.image(result, channels="BGR", use_column_width=True)

elif mode == "Live Webcam":
    st.warning("⚠️ Webcam is not supported on Streamlit Cloud. Please run locally for live detection.")
    if st.button("🎥 Start Webcam (Local Only)"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_button = st.button("⏹ Stop Webcam")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            result = detect_mask(frame)
            stframe.image(result, channels="BGR")

        cap.release()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🚀 Built with Streamlit | Deep Learning Project | Attractive UI Version</center>",
    unsafe_allow_html=True
)

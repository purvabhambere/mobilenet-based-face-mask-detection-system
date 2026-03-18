import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(page_title="AI Face Mask Detector", page_icon="😷", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #141E30, #243B55); color: white; }
    .main-title {
        font-size: 48px; font-weight: bold; text-align: center;
        background: -webkit-linear-gradient(#00F5A0, #00D9F5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-text { text-align: center; font-size: 20px; margin-bottom: 30px; color: #dcdcdc; }
    .stButton>button {
        background: linear-gradient(45deg, #00F5A0, #00D9F5);
        color: black; font-weight: bold; border-radius: 10px; height: 3em; width: 100%;
    }
    .stFileUploader {
        background-color: rgba(255,255,255,0.05);
        padding: 20px; border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">😷 AI Face Mask Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Deep Learning Powered Real-Time Mask Detection using MobileNetV2</div>', unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    model = None
    for fname in ["mask_detector.h5", "mask_detector.keras"]:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            try:
                model = tf.keras.models.load_model(fpath, compile=False)
                break
            except Exception:
                continue

    if model is None:
        st.error("❌ No model file found!")
        st.code(f"Looking in: {BASE_DIR}\nFiles: {os.listdir(BASE_DIR)}")
        st.stop()

    prototxt = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
    weights  = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt) or not os.path.exists(weights):
        st.error("❌ Face detector files missing!")
        st.stop()

    faceNet = cv2.dnn.readNet(prototxt, weights)
    return model, faceNet

model, faceNet = load_models()

def detect_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = preprocess_input(face_rgb)
            face_rgb = np.expand_dims(face_rgb, axis=0)

            (mask, withoutMask) = model.predict(face_rgb, verbose=0)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 150) if label == "Mask" else (0, 0, 255)

            cv2.putText(frame, f"{label}: {max(mask,withoutMask)*100:.2f}%",
                        (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
    return frame

# Detect environment
IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("IS_STREAMLIT_CLOUD")

st.sidebar.title("⚙ Settings")

# Show different modes based on environment
if IS_CLOUD:
    mode = st.sidebar.radio("Select Mode:", [
        "📤 Upload Image",
        "📷 Camera (Photo)"
    ])
else:
    mode = st.sidebar.radio("Select Mode:", [
        "📤 Upload Image",
        "📷 Camera (Photo)",
        "🎥 Live Webcam"
    ])

st.sidebar.markdown("---")

# ---------------- UPLOAD IMAGE ----------------
if mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = detect_mask(image)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, use_container_width=True)

# ---------------- CAMERA PHOTO - WORKS ON MOBILE & DESKTOP ----------------
elif mode == "📷 Camera (Photo)":
    st.info("📱 Point your camera at a face and take a photo to detect mask. Works on mobile & desktop!")
    camera_image = st.camera_input("📷 Take a photo")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = detect_mask(image)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, use_container_width=True)

# ---------------- LIVE WEBCAM - LOCAL ONLY ----------------
elif mode == "🎥 Live Webcam":
    run = st.checkbox("▶ START Webcam")
    FRAME_WINDOW = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access webcam.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    break
                result = detect_mask(frame)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(result_rgb, use_container_width=True)
            cap.release()
    else:
        FRAME_WINDOW.empty()

st.markdown("---")

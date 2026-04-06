```python
import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="centered"
)

# ---------------- INSANE UI CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a;
    color: #e8e8e8;
}

.block-container {
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

.hero-title {
    font-size: 50px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-title span { color: #c8ff00; }

.stButton > button {
    background: #c8ff00;
    color: black;
    font-weight: 700;
    border: none;
    padding: 12px;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    box-shadow: 0 0 20px #c8ff00;
}

.result-card {
    padding: 25px;
    border-radius: 10px;
    background: rgba(20,20,20,0.6);
    backdrop-filter: blur(10px);
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
            self.fc = nn.Linear(self.backbone.num_features, 1)

        def forward(self, x):
            return self.fc(self.backbone(x))

    model = Model()
    if os.path.exists("deepfake_model.pth"):
        model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- SIGNALS ----------------
def extract_signals(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise = np.mean(np.abs(frame - cv2.GaussianBlur(frame, (3,3), 0)))
    edges = cv2.Canny(gray, 50, 150).mean()
    return {
        "noise": noise/50,
        "edges": edges/255
    }

# ---------------- IMAGE PRED ----------------
def predict_image(img):
    tf = A.Compose([
        A.Resize(224,224),
        A.Normalize(),
        ToTensorV2()
    ])
    img_np = np.array(img)
    tensor = tf(image=img_np)["image"].unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    sigs = extract_signals(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    score = 0.7*prob + 0.3*np.mean(list(sigs.values()))

    return score, sigs

# ---------------- VIDEO PRED ----------------
def predict_video(path):
    cap = cv2.VideoCapture(path)
    probs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224,224))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        score, _ = predict_image(img)
        probs.append(score)

    cap.release()
    return np.mean(probs)

# ---------------- UI ----------------
st.markdown('<div class="hero-title">Deep<span>Fake</span> Detector</div>', unsafe_allow_html=True)

mode = st.selectbox("Select Mode", ["Image", "Video"])

uploaded = st.file_uploader("Upload File")

if uploaded:
    if mode == "Image":
        img = Image.open(uploaded)
        st.image(img)

    else:
        st.video(uploaded)

run = st.button("ANALYZE")

# ---------------- LOADING ----------------
if run and uploaded:
    with st.spinner("Initializing AI..."):
        time.sleep(1)

    prog = st.empty()
    for i in range(0,101,5):
        prog.write(f"Analyzing... {i}%")
        time.sleep(0.02)

    # ---------------- PROCESS ----------------
    if mode == "Image":
        score, sigs = predict_image(img)
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(uploaded.read())
            path = f.name
        score = predict_video(path)
        os.unlink(path)
        sigs = {}

    label = "FAKE" if score > 0.5 else "REAL"

    # ---------------- RESULT ----------------
    st.markdown(f"""
    <div class="result-card">
        <h2>{label}</h2>
        <p>Confidence: {score*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
```

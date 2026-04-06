import streamlit.components.v1 as components
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

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

/* Smooth fade-in animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

.block-container {
    animation: fadeIn 0.8s ease-in-out;
}

/* Glow effect */
.glow {
    text-shadow: 0 0 10px rgba(200,255,0,0.6),
                 0 0 20px rgba(200,255,0,0.4);
}

/* Glass effect cards */
.result-card {
    backdrop-filter: blur(12px);
    background: rgba(20,20,20,0.6);
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 0 30px rgba(0,0,0,0.6);
}

/* Hover animation */
.result-card:hover {
    transform: translateY(-4px);
    transition: all 0.3s ease;
}

/* Button animation */
.stButton > button {
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: "";
    position: absolute;
    width: 0;
    height: 100%;
    left: 0;
    top: 0;
    background: rgba(255,255,255,0.15);
    transition: width 0.3s;
}

.stButton > button:hover::after {
    width: 100%;
}

/* Upload drag glow */
[data-testid="stFileUploader"] > div:hover {
    box-shadow: 0 0 20px rgba(200,255,0,0.3);
}

/* Pulse animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(200,255,0,0.4);}
    70% { box-shadow: 0 0 0 10px rgba(200,255,0,0);}
    100% { box-shadow: 0 0 0 0 rgba(200,255,0,0);}
}

.stButton > button {
    animation: pulse 2s infinite;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path="deepfake_model.pth"):
    class DeepfakeDetector(nn.Module):
        def __init__(self, backbone="efficientnet_b4", pretrained=False):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                              num_classes=0, global_pool="avg")
            dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.BatchNorm1d(dim), nn.Dropout(0.4),
                nn.Linear(dim, 512), nn.SiLU(),
                nn.BatchNorm1d(512), nn.Dropout(0.35),
                nn.Linear(512, 128), nn.SiLU(),
                nn.BatchNorm1d(128), nn.Dropout(0.25),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.head(self.backbone(x)).squeeze(1)

    device = torch.device("cpu")
    model = DeepfakeDetector(pretrained=False)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model, device


def extract_signals(frame_bgr):
    h, w = frame_bgr.shape[:2]
    median = cv2.medianBlur(frame_bgr, 3)
    noise = np.mean(np.abs(frame_bgr.astype(float) - median.astype(float)))
    noise_n = np.clip(noise / 15.0, 0, 1)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    fmag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.zeros_like(fmag, dtype=np.float32)
    mask = np.ascontiguousarray(mask)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    low_e = np.sum(fmag * mask) + 1e-8
    high_e = np.sum(fmag * (1 - mask))
    fft_r = np.clip((high_e / low_e - 12) / 6, 0, 1)

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr_var = float(np.std(ycrcb[:, :, 1]))
    cr_var_n = np.clip(cr_var / 50.0, 0, 1)

    edges = cv2.Canny(gray, 40, 120).astype(float)
    left = np.mean(edges[:, :w//4])
    right = np.mean(edges[:, 3*w//4:])
    sym = np.clip(abs(left - right) / (left + right + 1e-8), 0, 1)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    q1, q3 = np.percentile(np.abs(lap), [25, 75])
    sharpness_spread = np.clip((q3 - q1) / 30.0, 0, 1)

    def lbp_uniformity(img):
        out = np.zeros_like(img)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            out += (shifted >= img).astype(np.uint8)
        return float(np.mean(out)) / 8.0

    lbp = lbp_uniformity(gray)
    return {
        "noise": noise_n, "fft_ratio": fft_r, "cr_variance": cr_var_n,
        "edge_asymmetry": sym, "sharpness_spread": sharpness_spread, "lbp_uniformity": lbp
    }


def predict_image(img_pil, model, device, threshold=0.35):
    predict_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    SIG_WEIGHTS = {
        "noise": 0.20, "fft_ratio": 0.15, "cr_variance": 0.30,
        "edge_asymmetry": 0.15, "sharpness_spread": 0.10, "lbp_uniformity": 0.10
    }

    img_np = np.array(img_pil.convert("RGB"))
    tensor = predict_tf(image=img_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr_resized = cv2.resize(img_bgr, (224, 224))
    sigs = extract_signals(img_bgr_resized)
    sig_score = sum(sigs[k] * SIG_WEIGHTS[k] for k in SIG_WEIGHTS)

    gap = abs(prob - sig_score)
    if gap > 0.35:
        ensemble = 0.80 * prob + 0.20 * sig_score
    else:
        ensemble = 0.60 * prob + 0.40 * sig_score

    label = "FAKE" if ensemble > threshold else "REAL"
    return label, ensemble, prob, sig_score, sigs


def predict_video(video_path, model, device, threshold=0.35, max_frames=50, frame_step=3):
    predict_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    SIG_WEIGHTS = {
        "noise": 0.20, "fft_ratio": 0.15, "cr_variance": 0.30,
        "edge_asymmetry": 0.15, "sharpness_spread": 0.10, "lbp_uniformity": 0.10
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    dnn_probs, all_sigs = [], []
    frame_idx = processed = 0
    prog = st.progress(0, text="")

    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_np = cv2.resize(rgb, (224, 224))
                tensor = predict_tf(image=img_np)["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).item()
                dnn_probs.append(prob)
                sigs = extract_signals(cv2.resize(frame, (224, 224)))
                all_sigs.append(sigs)
                processed += 1
                prog.progress(min(processed / max_frames, 1.0), text="")
            except Exception:
                pass
        frame_idx += 1
    cap.release()
    prog.empty()

    if not dnn_probs:
        return "UNKNOWN", 0.5, 0.5, 0.5, {}, total_frames, fps

    dnn_score = float(np.mean(dnn_probs))
    avg_sigs = {k: float(np.mean([s[k] for s in all_sigs])) for k in all_sigs[0]}
    sig_score = sum(avg_sigs[k] * SIG_WEIGHTS[k] for k in SIG_WEIGHTS)

    gap = abs(dnn_score - sig_score)
    ensemble = (0.80 * dnn_score + 0.20 * sig_score) if gap > 0.35 else (0.60 * dnn_score + 0.40 * sig_score)

    label = "FAKE" if ensemble > threshold else "REAL"
    return label, ensemble, dnn_score, sig_score, avg_sigs, total_frames, fps


def render_result(label, ensemble, dnn_score, sig_score, sigs):
    is_fake = label == "FAKE"
    verdict_class = "verdict-fake" if is_fake else "verdict-real"
    conf = ensemble if is_fake else 1.0 - ensemble
    pct = int(ensemble * 100)

    sig_labels = {
        "noise": "Noise Residual", "fft_ratio": "FFT Artifacts",
        "cr_variance": "Color Shift", "edge_asymmetry": "Edge Asymmetry",
        "sharpness_spread": "Sharpness", "lbp_uniformity": "Texture"
    }

    rows = ""
    for k, v in sigs.items():
        bar_pct = int(v * 100)
        bar_class = "bar-high" if v > 0.6 else ("bar-mid" if v > 0.3 else "bar-low")
        rows += f"""
        <div class="signal-row">
            <span class="signal-name">{sig_labels.get(k, k)}</span>
            <div class="bar-wrap"><div class="bar-fill {bar_class}" style="width:{bar_pct}%"></div></div>
            <span class="signal-val">{v:.3f}</span>
        </div>"""

    st.markdown(f"""
    <div class="result-card">
        <div class="verdict-fake" style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.2em;color:#333;text-transform:uppercase;margin-bottom:16px;">
            {'⚠ deepfake detected' if is_fake else '✓ authentic content'}
        </div>
        <div class="result-verdict {verdict_class}">{label}</div>
        <div class="result-conf">
            ensemble probability: {pct}% &nbsp;·&nbsp;
            dnn: {int(dnn_score*100)}% &nbsp;·&nbsp;
            signals: {int(sig_score*100)}%
        </div>
        <div class="section-label">forensic signals</div>
        {rows}
    </div>
    """, unsafe_allow_html=True)


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-label">AI Forensics · v1.0</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Deep<span>Fake</span><br>Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">EfficientNet-B4 · 6-signal ensemble · CPU inference</div>', unsafe_allow_html=True)

model_path = "deepfake_model.pth"
model_loaded = os.path.exists(model_path)

if not model_loaded:
    st.markdown("""
    <div style="background:#111;border:1px solid #1e1e1e;border-radius:4px;padding:20px;margin-bottom:24px;">
        <p style="font-family:'DM Mono',monospace;font-size:11px;color:#555;letter-spacing:0.08em;">
            MODEL FILE NOT FOUND — place <code style="color:#888;">deepfake_model.pth</code> in the same directory as <code style="color:#888;">app.py</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

mode = st.selectbox("", ["Image", "Video"], label_visibility="collapsed")

uploaded = st.file_uploader(
    "DROP FILE HERE",
    type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"] if mode == "Video" else ["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)

threshold = st.slider("", 0.1, 0.9, 0.35, 0.01, label_visibility="collapsed",
                      help="Detection threshold — lower = more sensitive")

st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#2e2e2e;letter-spacing:0.15em;text-align:right;margin-top:-8px;margin-bottom:16px;">THRESHOLD · {threshold:.2f}</div>', unsafe_allow_html=True)

run = st.button("ANALYSE", disabled=(uploaded is None or not model_loaded))

if uploaded and run:
    model, device = load_model(model_path)

    if mode == "Image":
        img = Image.open(uploaded)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(img, use_container_width=True)

        with st.spinner(""):
            label, ensemble, dnn_score, sig_score, sigs = predict_image(img, model, device, threshold)

        render_result(label, ensemble, dnn_score, sig_score, sigs)

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as f:
            f.write(uploaded.read())
            tmp_path = f.name

        st.markdown('<div class="section-label">processing frames</div>', unsafe_allow_html=True)

        label, ensemble, dnn_score, sig_score, sigs, total_frames, fps = predict_video(
            tmp_path, model, device, threshold
        )
        os.unlink(tmp_path)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("TOTAL FRAMES", f"{total_frames:,}")
        with c2:
            st.metric("FPS", f"{fps:.1f}")

        render_result(label, ensemble, dnn_score, sig_score, sigs)

elif uploaded and not run:
    if mode == "Image":
        img = Image.open(uploaded)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(img, use_container_width=True)

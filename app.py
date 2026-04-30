import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# XAI Imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. Model Configuration and Loading
# ==========================================

CLASS_NAMES = ['Bona Fide (Real)', 'Morphed (Fake)']

@st.cache_resource
def load_model():
    """
    Loads the ResNet50 model once to optimize performance.
    """
    model = models.resnet50(pretrained=False)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load('models/resnet50_morph.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Error: Model file not found. Ensure 'resnet50_morph.pth' is in the 'models' directory.")
        return None
    
    model.eval() 
    return model

model = load_model()

# ==========================================
# 2. Image Preprocessing
# ==========================================
def process_image(image):
    """
    Prepares the image for the model using standard ImageNet transformations.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# ==========================================
# 3. User Interface (UI)
# ==========================================
st.set_page_config(
    page_title="S-MAD: Morphing Attack Detection",
    layout="wide",
    page_icon="🔍"
)

# ── Global Styles ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ANIMATIONS */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes verdictReveal {
    from { opacity: 0; transform: scale(0.93); }
    to   { opacity: 1; transform: scale(1);    }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 0   rgba(56,189,248,0.00); }
    50%       { box-shadow: 0 0 0 8px rgba(56,189,248,0.14); }
}
@keyframes scanBar {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

/* BASE */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main { background-color: #070C18 !important; font-family: 'Inter', sans-serif !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A1020; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111F 0%, #0A1828 100%) !important;
    border-right: 1px solid #132033 !important;
}

/* COLUMNS → cards */
[data-testid="stColumn"] {
    background: linear-gradient(160deg, #0C1A2E 0%, #0F2040 100%) !important;
    border: 1px solid #1A3050 !important;
    border-radius: 18px !important;
    padding: 1.4rem !important;
    animation: fadeInUp 0.55s ease both;
}

/* BUTTON */
div.stButton > button {
    background: linear-gradient(135deg, #1A44C8 0%, #0C9FE8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.97rem !important;
    letter-spacing: 0.4px;
    padding: 0.65rem 1.4rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.38) !important;
    animation: glowPulse 2.8s ease infinite !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(14,165,233,0.55) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* FILE UPLOADER */
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #091422 0%, #0D1C34 100%) !important;
    border: 2px dashed #1E3D64 !important;
    border-radius: 14px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #38BDF8 !important; }

/* IMAGES */
[data-testid="stImage"] img {
    max-height: 400px;
    object-fit: contain;
    width: 100%;
    border-radius: 10px;
}


/* METRIC */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0B1C35 0%, #0E2040 100%) !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricLabel"] p {
    color: #4E6A8A !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
}

/* ALERTS */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.2px;
}

/* DIVIDER */
hr { border-color: #132033 !important; margin: 1.4rem 0 !important; }

/* EXPANDER */
[data-testid="stExpander"] {
    background: #091422 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 12px !important;
}
[data-testid="stExpanderToggleIcon"] { color: #38BDF8 !important; }

/* ── CUSTOM HTML COMPONENTS ────────────────────────────────────────────── */

/* Hero header */
.smad-hero {
    background: linear-gradient(135deg, #091422 0%, #0D1E38 50%, #091828 100%);
    border: 1px solid #1A3050;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    animation: fadeInUp 0.6s ease both;
    position: relative;
    overflow: hidden;
}
.smad-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.smad-hero-inner { display: flex; align-items: center; gap: 1.4rem; }
.smad-icon {
    width: 58px; height: 58px;
    background: linear-gradient(135deg, #1A44C8, #0C9FE8);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.7rem;
    box-shadow: 0 6px 20px rgba(14,165,233,0.42);
    flex-shrink: 0;
}
.smad-title {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    background: linear-gradient(90deg, #38BDF8 0%, #818CF8 55%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}
.smad-subtitle {
    font-size: 0.82rem;
    color: #3D5A7A;
    margin-top: 0.25rem;
}
.smad-pill {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.22);
    color: #38BDF8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    padding: 0.38rem 0.85rem;
    border-radius: 50px;
    white-space: nowrap;
}

/* Sidebar brand */
.sb-brand {
    text-align: center;
    padding: 0.4rem 0 1.4rem;
    animation: fadeIn 0.5s ease both;
}
.sb-logo {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, #1A44C8, #0C9FE8);
    border-radius: 15px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    margin: 0 auto 0.8rem;
    box-shadow: 0 4px 18px rgba(14,165,233,0.38);
}
.sb-name {
    font-size: 1.05rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sb-tagline { font-size: 0.7rem; color: #2D4A6A; margin-top: 0.2rem; }

/* Sidebar chip */
.sb-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #2D4A6A;
    margin: 1.1rem 0 0.5rem;
}
.sb-chip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0C1A2E;
    border: 1px solid #1A3050;
    border-radius: 9px;
    padding: 0.48rem 0.75rem;
    margin-bottom: 0.35rem;
    font-size: 0.8rem;
    color: #4E6A8A;
}
.sb-chip span { color: #38BDF8; font-weight: 600; font-size: 0.8rem; }

/* Sidebar status */
.sb-status {
    display: flex; align-items: center; gap: 0.55rem;
    background: #0C1A2E;
    border: 1px solid #1A3050;
    border-radius: 9px;
    padding: 0.5rem 0.75rem;
    margin-top: 0.35rem;
    font-size: 0.8rem;
}
.dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.dot-green { background: #10B981; box-shadow: 0 0 6px #10B981; animation: glowPulse 2s ease infinite; }
.dot-red   { background: #EF4444; box-shadow: 0 0 6px #EF4444; }

/* Section label above images/panels */
.panel-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2D4A6A;
    margin-bottom: 0.7rem;
}

/* Verdict card */
.verdict {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    margin-bottom: 1rem;
    animation: verdictReveal 0.45s cubic-bezier(0.34,1.56,0.64,1) both;
}
.verdict-real {
    background: linear-gradient(135deg, #042010 0%, #053828 100%);
    border: 1px solid rgba(16,185,129,0.45);
    box-shadow: 0 4px 22px rgba(16,185,129,0.12);
}
.verdict-fake {
    background: linear-gradient(135deg, #300808 0%, #5C1212 100%);
    border: 1px solid rgba(239,68,68,0.45);
    box-shadow: 0 4px 22px rgba(239,68,68,0.12);
}
.v-icon {
    width: 46px; height: 46px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem; flex-shrink: 0; font-style: normal;
}
.verdict-real  .v-icon { background: rgba(16,185,129,0.18); }
.verdict-fake  .v-icon { background: rgba(239,68,68,0.18); }
.v-label { font-size: 1.1rem; font-weight: 800; letter-spacing: 0.3px; }
.verdict-real  .v-label { color: #34D399; }
.verdict-fake  .v-label { color: #F87171; }
.v-sub { font-size: 0.76rem; margin-top: 0.08rem; opacity: 0.65; }
.verdict-real  .v-sub { color: #6EE7B7; }
.verdict-fake  .v-sub { color: #FCA5A5; }

/* Grad-CAM header */
.cam-header { margin: 1rem 0 0.4rem; }
.cam-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2D4A6A;
}
.cam-desc { font-size: 0.78rem; color: #2D4A6A; margin-top: 0.2rem; }

/* Probability bars */
.prob-wrap { padding: 0.3rem 0; }
.prob-row { margin-bottom: 0.9rem; }
.prob-meta {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; font-weight: 600;
    margin-bottom: 0.35rem;
}
.prob-track {
    background: #091422;
    border: 1px solid #1A3050;
    border-radius: 50px;
    height: 9px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 50px; }
.fill-real { background: linear-gradient(90deg, #047857, #34D399); }
.fill-fake { background: linear-gradient(90deg, #B91C1C, #F87171); }
.label-real { color: #34D399; }
.label-fake { color: #F87171; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
model_loaded = model is not None
dot_cls   = "dot-green" if model_loaded else "dot-red"
status_lbl = "Model Ready" if model_loaded else "Model Not Found"
status_col = "#34D399" if model_loaded else "#F87171"

with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-logo">🔍</div>
        <div class="sb-name">S-MAD</div>
        <div class="sb-tagline">Morphing Attack Detection</div>
    </div>
    <div class="sb-section-label">System Info</div>
    <div class="sb-chip">Architecture <span>ResNet50</span></div>
    <div class="sb-chip">Input Size   <span>224 × 224 RGB</span></div>
    <div class="sb-chip">XAI Method   <span>Grad-CAM</span></div>
    <div class="sb-chip">Task         <span>Binary Classification</span></div>
    <div class="sb-section-label">Status</div>
    <div class="sb-status">
        <div class="dot {dot_cls}"></div>
        <span style="color:{status_col};font-weight:600;">{status_lbl}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="smad-hero">
    <div class="smad-hero-inner">
        <div class="smad-icon">🔍</div>
        <div>
            <div class="smad-title">Face Morphing Attack Detection</div>
            <div class="smad-subtitle">
                Upload a facial image to analyze its authenticity and detect potential morphing artifacts
            </div>
        </div>
    </div>
    <div class="smad-pill">S-MAD · v1.0</div>
</div>
""", unsafe_allow_html=True)

# ── File Uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a face image here, or click to browse",
    type=["jpg", "png", "jpeg"]
)

# ── Main Content ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="panel-label">Input Image</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="panel-label">Analysis Panel</div>', unsafe_allow_html=True)
        if st.button("Analyze Image", use_container_width=True):
            if model:
                with st.spinner("Analyzing facial artifacts and generating activation maps..."):
                    # --- Core Computation ---
                    input_tensor = process_image(image)

                    # Forward pass without torch.no_grad() because Grad-CAM needs the gradients
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)

                    confidence, predicted_class = torch.max(probabilities, 1)
                    label = CLASS_NAMES[predicted_class.item()]
                    score = confidence.item() * 100

                    # --- XAI: Grad-CAM Generation ---
                    # Target the last convolutional layer of ResNet50
                    target_layers = [model.layer4[-1]]

                    cam     = GradCAM(model=model, target_layers=target_layers)
                    targets = [ClassifierOutputTarget(predicted_class.item())]

                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

                    # Resize original image to match model input size and normalize to [0, 1]
                    img_resized       = np.array(image.resize((224, 224))) / 255.0
                    cam_visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

                st.divider()

                # --- Verdict Card ---
                if predicted_class.item() == 0:
                    st.markdown("""
                    <div class="verdict verdict-real">
                        <em class="v-icon">✓</em>
                        <div>
                            <div class="v-label">BONA FIDE</div>
                            <div class="v-sub">Authentic face detected — no morphing artifacts found</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="verdict verdict-fake">
                        <em class="v-icon">⚠</em>
                        <div>
                            <div class="v-label">MORPHING ATTACK</div>
                            <div class="v-sub">Synthetic artifacts detected — identity blending suspected</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- Confidence Metric ---
                st.metric(label="Confidence Score", value=f"{score:.2f}%")

                # --- Grad-CAM ---
                st.markdown("""
                <div class="cam-header">
                    <div class="cam-title">Explainable AI · Grad-CAM Heatmap</div>
                    <div class="cam-desc">Highlighted regions show the facial areas most influential in the model's decision.</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(cam_visualization, use_container_width=True)

                # --- Probability Distribution ---
                prob_real = probabilities[0][0].item() * 100
                prob_fake = probabilities[0][1].item() * 100

                with st.expander("Probability Distribution"):
                    st.markdown(f"""
                    <div class="prob-wrap">
                        <div class="prob-row">
                            <div class="prob-meta">
                                <span class="label-real">Bona Fide (Real)</span>
                                <span class="label-real">{prob_real:.1f}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill fill-real" style="width:{prob_real:.1f}%"></div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <div class="prob-meta">
                                <span class="label-fake">Morphed (Fake)</span>
                                <span class="label-fake">{prob_fake:.1f}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill fill-fake" style="width:{prob_fake:.1f}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
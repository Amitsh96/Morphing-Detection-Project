import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Pipeline & XAI Imports
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. Pipeline Configuration and Loading
# ==========================================

CLASS_NAMES = ['Bona Fide (Real)', 'Morphed (Fake)']

@st.cache_resource
def load_systems():
    """
    Loads MTCNN and BOTH ResNet50 models (Ensemble Architecture).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load MTCNN (Face Cropper)
    mtcnn = MTCNN(select_largest=True, post_process=False, device=device)
    
    # Function to build and load a model
    def get_model(path):
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()
            return model
        except FileNotFoundError:
            return None

    # 2. Load "The Hunter" (Original Model - sees full image)
    model_full = get_model('models/best_mad_resnet50.pth')
    
    # 3. Load "The Microscope" (Robust Model - sees cropped face only)
    model_crop = get_model('models/robust_mad_resnet50.pth')
    
    return mtcnn, model_full, model_crop, device

mtcnn, model_full, model_crop, device = load_systems()

if not model_full or not model_crop:
    st.error("⚠️ Error: Missing model files. Ensure both 'best_mad_resnet50.pth' and 'robust_mad_resnet50.pth' are in the 'models' directory.")

# ==========================================
# 2. Image Preprocessing
# ==========================================
def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def generate_cam(model, input_tensor, pil_img, predicted_idx):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    img_resized = np.array(pil_img.resize((224, 224))) / 255.0
    return show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

# ==========================================
# 3. User Interface (UI)
# ==========================================
st.set_page_config(page_title="S-MAD: Ensemble Architecture", layout="wide", page_icon="🛡️")

# ── Global Styles ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@keyframes fadeInUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes verdictReveal { from { opacity: 0; transform: scale(0.93); } to { opacity: 1; transform: scale(1); } }
@keyframes glowPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(56,189,248,0.00); } 50% { box-shadow: 0 0 0 8px rgba(56,189,248,0.14); } }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background-color: #070C18 !important; font-family: 'Inter', sans-serif !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A1020; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #07111F 0%, #0A1828 100%) !important; border-right: 1px solid #132033 !important; }
[data-testid="stColumn"] { background: linear-gradient(160deg, #0C1A2E 0%, #0F2040 100%) !important; border: 1px solid #1A3050 !important; border-radius: 18px !important; padding: 1.4rem !important; animation: fadeInUp 0.55s ease both; }
div.stButton > button { background: linear-gradient(135deg, #1A44C8 0%, #0C9FE8 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-size: 0.97rem !important; letter-spacing: 0.4px; padding: 0.65rem 1.4rem !important; transition: transform 0.2s ease, box-shadow 0.2s ease !important; box-shadow: 0 4px 20px rgba(14,165,233,0.38) !important; animation: glowPulse 2.8s infinite !important; width: 100%; }
div.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(14,165,233,0.55) !important; }
[data-testid="stFileUploaderDropzone"] { background: linear-gradient(135deg, #091422 0%, #0D1C34 100%) !important; border: 2px dashed #1E3D64 !important; border-radius: 14px !important; transition: border-color 0.3s !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #38BDF8 !important; }
[data-testid="stImage"] img { max-height: 400px; object-fit: contain; width: 100%; border-radius: 10px; }
[data-testid="stMetric"] { background: linear-gradient(135deg, #0B1C35 0%, #0E2040 100%) !important; border: 1px solid #1E3A5F !important; border-radius: 14px !important; padding: 1.2rem 1.5rem !important; }
[data-testid="stMetricLabel"] p { color: #4E6A8A !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { color: #38BDF8 !important; font-size: 2.3rem !important; font-weight: 800 !important; }
hr { border-color: #132033 !important; margin: 1.4rem 0 !important; }
.smad-hero { background: linear-gradient(135deg, #091422 0%, #0D1E38 50%, #091828 100%); border: 1px solid #1A3050; border-radius: 20px; padding: 2rem 2.4rem; margin-bottom: 1.8rem; display: flex; align-items: center; justify-content: space-between; animation: fadeInUp 0.6s ease both; position: relative; overflow: hidden; }
.smad-hero::before { content: ''; position: absolute; top: -60px; right: -60px; width: 220px; height: 220px; background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 70%); pointer-events: none; }
.smad-hero-inner { display: flex; align-items: center; gap: 1.4rem; }
.smad-icon { width: 58px; height: 58px; background: linear-gradient(135deg, #1A44C8, #0C9FE8); border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; box-shadow: 0 6px 20px rgba(14,165,233,0.42); flex-shrink: 0; }
.smad-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.6px; background: linear-gradient(90deg, #38BDF8 0%, #818CF8 55%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.15; }
.smad-subtitle { font-size: 0.82rem; color: #3D5A7A; margin-top: 0.25rem; }
.smad-pill { background: rgba(56,189,248,0.07); border: 1px solid rgba(56,189,248,0.22); color: #38BDF8; font-size: 0.7rem; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; padding: 0.38rem 0.85rem; border-radius: 50px; white-space: nowrap; }
.sb-brand { text-align: center; padding: 0.4rem 0 1.4rem; animation: fadeIn 0.5s ease both; }
.sb-logo { width: 54px; height: 54px; background: linear-gradient(135deg, #1A44C8, #0C9FE8); border-radius: 15px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin: 0 auto 0.8rem; box-shadow: 0 4px 18px rgba(14,165,233,0.38); }
.sb-name { font-size: 1.05rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sb-tagline { font-size: 0.7rem; color: #2D4A6A; margin-top: 0.2rem; }
.sb-section-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; color: #2D4A6A; margin: 1.1rem 0 0.5rem; }
.sb-chip { display: flex; justify-content: space-between; align-items: center; background: #0C1A2E; border: 1px solid #1A3050; border-radius: 9px; padding: 0.48rem 0.75rem; margin-bottom: 0.35rem; font-size: 0.8rem; color: #4E6A8A; }
.sb-chip span { color: #38BDF8; font-weight: 600; font-size: 0.8rem; }
.sb-status { display: flex; align-items: center; gap: 0.55rem; background: #0C1A2E; border: 1px solid #1A3050; border-radius: 9px; padding: 0.5rem 0.75rem; margin-top: 0.35rem; font-size: 0.8rem; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-green { background: #10B981; box-shadow: 0 0 6px #10B981; animation: glowPulse 2s ease infinite; }
.dot-red { background: #EF4444; box-shadow: 0 0 6px #EF4444; }
.panel-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #2D4A6A; margin-bottom: 0.7rem; }
.verdict { border-radius: 14px; padding: 1.2rem 1.5rem; display: flex; align-items: center; gap: 1.1rem; margin-bottom: 1rem; animation: verdictReveal 0.45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-real { background: linear-gradient(135deg, #042010 0%, #053828 100%); border: 1px solid rgba(16,185,129,0.45); box-shadow: 0 4px 22px rgba(16,185,129,0.12); }
.verdict-fake { background: linear-gradient(135deg, #300808 0%, #5C1212 100%); border: 1px solid rgba(239,68,68,0.45); box-shadow: 0 4px 22px rgba(239,68,68,0.12); }
.v-icon { width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; flex-shrink: 0; font-style: normal; }
.verdict-real .v-icon { background: rgba(16,185,129,0.18); }
.verdict-fake .v-icon { background: rgba(239,68,68,0.18); }
.v-label { font-size: 1.1rem; font-weight: 800; letter-spacing: 0.3px; }
.verdict-real .v-label { color: #34D399; }
.verdict-fake .v-label { color: #F87171; }
.v-sub { font-size: 0.76rem; margin-top: 0.08rem; opacity: 0.65; }
.verdict-real .v-sub { color: #6EE7B7; }
.verdict-fake .v-sub { color: #FCA5A5; }
.cam-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #2D4A6A; text-align: center; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
sys_ready = mtcnn and model_full and model_crop
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-logo">🛡️</div>
        <div class="sb-name">S-MAD AI</div>
        <div class="sb-tagline">Dual-Engine Detection</div>
    </div>
    <div class="sb-section-label">Ensemble Architecture</div>
    <div class="sb-chip">Model A (Hunter) <span>Full Image</span></div>
    <div class="sb-chip">Model B (Microscope) <span>Cropped ROI</span></div>
    <div class="sb-chip">Decision Logic <span>Logical OR Gate</span></div>
    <div class="sb-section-label">System Status</div>
    <div class="sb-status">
        <div class="dot {'dot-green' if sys_ready else 'dot-red'}"></div>
        <span style="color:{'#34D399' if sys_ready else '#F87171'};font-weight:600;">{'Dual Pipeline Ready' if sys_ready else 'System Error'}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="smad-hero">
    <div class="smad-hero-inner">
        <div class="smad-icon">🛡️</div>
        <div>
            <div class="smad-title">Advanced Ensemble Detection</div>
            <div class="smad-subtitle">
                Utilizes two parallel neural networks to detect both geometric edge artifacts and deepfake center-face manipulations.
            </div>
        </div>
    </div>
    <div class="smad-pill">S-MAD Ensemble · v3.0</div>
</div>
""", unsafe_allow_html=True)

# ── File Uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Drop a face image here, or click to browse", type=["jpg", "png", "jpeg"])

# ── Main Content ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    # Top Row: Inputs
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown('<div class="panel-label">1. Original Upload</div>', unsafe_allow_html=True)
        img_original = Image.open(uploaded_file).convert('RGB')
        st.image(img_original, use_container_width=True)

    with c2:
        st.markdown('<div class="panel-label">Analysis Panel</div>', unsafe_allow_html=True)
        if st.button("Run Dual-Engine Analysis", use_container_width=True):
            if sys_ready:
                
                # --- STEP 1: Process Full Image (Hunter Model) ---
                tensor_full = process_image(img_original).to(device)
                outputs_full = model_full(tensor_full)
                probs_full = torch.nn.functional.softmax(outputs_full, dim=1)[0]
                prob_fake_full = probs_full[1].item() * 100
                
                # Generate XAI for Full Image
                cam_full_vis = generate_cam(model_full, tensor_full, img_original, 1 if prob_fake_full > 50 else 0)

                # --- STEP 2: Process Cropped Face (Microscope Model) ---
                with st.spinner("Extracting ROI & Running Biometrics..."):
                    face_tensor = mtcnn(img_original)
                    
                    if face_tensor is not None:
                        img_cropped = transforms.ToPILImage()(face_tensor.to(torch.uint8))
                        st.markdown('<div class="panel-label">2. Extracted ROI (Face Only)</div>', unsafe_allow_html=True)
                        st.image(img_cropped, width=180, caption="Passed to Model B")
                        
                        tensor_crop = process_image(img_cropped).to(device)
                        outputs_crop = model_crop(tensor_crop)
                        probs_crop = torch.nn.functional.softmax(outputs_crop, dim=1)[0]
                        prob_fake_crop = probs_crop[1].item() * 100
                        
                        # Generate XAI for Cropped Image
                        cam_crop_vis = generate_cam(model_crop, tensor_crop, img_cropped, 1 if prob_fake_crop > 50 else 0)
                    else:
                        prob_fake_crop = 0.0
                        img_cropped = None
                        st.warning("⚠️ No face detected for Model B. Relying on Model A only.")

                st.divider()

                # --- STEP 3: Ensemble Logic (OR Gate) ---
                # If EITHER model thinks it's a fake (>50%), we flag it!
                is_fake = (prob_fake_full > 50.0) or (prob_fake_crop > 50.0)
                
                if is_fake:
                    # Decide which model was more confident for the display score
                    max_fake_prob = max(prob_fake_full, prob_fake_crop)
                    st.markdown("""
                    <div class="verdict verdict-fake">
                        <em class="v-icon">⚠</em>
                        <div>
                            <div class="v-label">MORPHING ATTACK DETECTED</div>
                            <div class="v-sub">System flagged synthetic artifacts or identity blending.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="Highest Threat Confidence", value=f"{max_fake_prob:.2f}%")
                else:
                    st.markdown("""
                    <div class="verdict verdict-real">
                        <em class="v-icon">✓</em>
                        <div>
                            <div class="v-label">BONA FIDE (AUTHENTIC)</div>
                            <div class="v-sub">Both neural engines confirmed image authenticity.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    max_real_prob = max((100-prob_fake_full), (100-prob_fake_crop))
                    st.metric(label="Overall Authenticity Confidence", value=f"{max_real_prob:.2f}%")

                st.divider()

                # --- STEP 4: Dual XAI Display ---
                st.markdown('<div class="panel-label" style="text-align:center;">Explainable AI (Grad-CAM) Comparison</div>', unsafe_allow_html=True)
                
                xai_col1, xai_col2 = st.columns(2)
                
                with xai_col1:
                    st.markdown(f'<div class="cam-title">Model A: Full Context<br><span style="color:#F87171">Fake Prob: {prob_fake_full:.1f}%</span></div>', unsafe_allow_html=True)
                    st.image(cam_full_vis, use_container_width=True)
                    st.caption("Checks edges, hair, and background for geometric morphing traces.")
                    
                with xai_col2:
                    if img_cropped is not None:
                        st.markdown(f'<div class="cam-title">Model B: Cropped ROI<br><span style="color:#F87171">Fake Prob: {prob_fake_crop:.1f}%</span></div>', unsafe_allow_html=True)
                        st.image(cam_crop_vis, use_container_width=True)
                        st.caption("Analyzes high-frequency noise and blending inside the T-zone.")
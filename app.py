import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from core_engine import load_all_systems, process_image, generate_cam 

# ==========================================
# 1. Page Config & Loading Assets
# ==========================================
st.set_page_config(page_title="S-MAD: Ensemble Fusion", layout="wide", page_icon="🛡️")

# טעינת קובץ העיצוב החיצוני
with open("style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# טעינת המודלים מהמנוע ושמירה בזיכרון מטמון
@st.cache_resource
def init_systems():
    return load_all_systems()

mtcnn, model_full, model_crop, device = init_systems()
sys_ready = mtcnn and model_full and model_crop

# ==========================================
# 2. Sidebar & Header UI
# ==========================================
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
    <div class="sb-chip">Decision Logic <span>Max Rule (Fusion)</span></div>
    <div class="sb-section-label">System Status</div>
    <div class="sb-status">
        <div class="dot {'dot-green' if sys_ready else 'dot-red'}"></div>
        <span style="color:{'#34D399' if sys_ready else '#F87171'};font-weight:600;">{'Dual Pipeline Ready' if sys_ready else 'System Error'}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="smad-hero">
    <div class="smad-hero-inner">
        <div class="smad-icon">🛡️</div>
        <div>
            <div class="smad-title">Morphing Attack Detection (Score-Level Fusion)</div>
            <div class="smad-subtitle">
                Utilizes two parallel neural networks and extracts the maximum threat probability to determine authenticity.
            </div>
        </div>
    </div>
    <div class="smad-pill">S-MAD Ensemble · v4.0</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. Main Application Flow
# ==========================================
uploaded_file = st.file_uploader("Drop a face image here, or click to browse", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown('<div class="panel-label">1. Original Upload</div>', unsafe_allow_html=True)
        img_original = Image.open(uploaded_file).convert('RGB')
        st.image(img_original, use_container_width=True)

    with c2:
        st.markdown('<div class="panel-label">Analysis Panel</div>', unsafe_allow_html=True)
        if st.button("Run Dual-Engine Analysis", use_container_width=True):
            if sys_ready:
                
                # --- STEP A: Model A (Full Image) ---
                tensor_full = process_image(img_original).to(device)
                outputs_full = model_full(tensor_full)
                probs_full = torch.nn.functional.softmax(outputs_full, dim=1)[0]
                prob_fake_full = probs_full[1].item() * 100
                cam_full_vis = generate_cam(model_full, tensor_full, img_original, 1 if prob_fake_full > 50 else 0)

                # --- STEP B: Model B (Cropped ROI) ---
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
                        cam_crop_vis = generate_cam(model_crop, tensor_crop, img_cropped, 1 if prob_fake_crop > 50 else 0)
                    else:
                        prob_fake_crop = 0.0
                        img_cropped = None
                        st.warning("⚠️ No face detected for Model B. Relying on Model A only.")

                st.divider()

                # --- STEP C: Ensemble Logic (Score-Level Fusion / MAX Rule) ---
                if img_cropped is not None:
                    # שיערוך לפי חוק המקסימום: לוקחים את האיום הגבוה ביותר מבין שני המודלים
                    ensemble_fake_prob = max(prob_fake_full, prob_fake_crop)
                else:
                    ensemble_fake_prob = prob_fake_full
                
                # ההכרעה נקבעת לפי האיום המקסימלי
                is_fake = ensemble_fake_prob > 50.0
                
                if is_fake:
                    st.markdown("""
                    <div class="verdict verdict-fake">
                        <em class="v-icon">⚠</em>
                        <div>
                            <div class="v-label">MORPHING ATTACK DETECTED</div>
                            <div class="v-sub">System flagged synthetic artifacts based on combined neural analysis.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="Ensemble Threat Confidence", value=f"{ensemble_fake_prob:.2f}%")
                else:
                    st.markdown("""
                    <div class="verdict verdict-real">
                        <em class="v-icon">✓</em>
                        <div>
                            <div class="v-label">BONA FIDE (AUTHENTIC)</div>
                            <div class="v-sub">Combined analysis confirms high probability of image authenticity.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="Ensemble Authenticity Confidence", value=f"{(100.0 - ensemble_fake_prob):.2f}%")

                st.caption(f"🔍 Breakdown: Model A (Full) = {prob_fake_full:.1f}% Threat | Model B (ROI) = {prob_fake_crop:.1f}% Threat")
                st.divider()

                # --- STEP D: Dual XAI Display ---
                st.markdown('<div class="panel-label" style="text-align:center;">Explainable AI (Grad-CAM) Comparison</div>', unsafe_allow_html=True)
                xai_col1, xai_col2 = st.columns(2)
                with xai_col1:
                    st.markdown(f'<div class="cam-title">Model A: Full Context<br><span style="color:#F87171">Fake Prob: {prob_fake_full:.1f}%</span></div>', unsafe_allow_html=True)
                    st.image(cam_full_vis, use_container_width=True)
                with xai_col2:
                    if img_cropped is not None:
                        st.markdown(f'<div class="cam-title">Model B: Cropped ROI<br><span style="color:#F87171">Fake Prob: {prob_fake_crop:.1f}%</span></div>', unsafe_allow_html=True)
                        st.image(cam_crop_vis, use_container_width=True)
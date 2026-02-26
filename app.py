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
st.set_page_config(page_title="S-MAD: Morphing Attack Detection", layout="wide")

# --- Sidebar ---
st.sidebar.title("System Overview")
st.sidebar.info(
    "Single-Image Morphing Attack Detection (S-MAD) system. "
    "Developed as a software engineering final project to identify "
    "facial morphing artifacts using deep learning."
)
st.sidebar.divider()
st.sidebar.markdown("**Model Architecture:** ResNet50")
st.sidebar.markdown("**Input Size:** 224x224 RGB")
st.sidebar.markdown("**Analysis Method:** Deep Feature Extraction & Grad-CAM Visualization")

# --- Main Header ---
st.title("Face Morphing Attack Detection")
st.markdown("Upload a facial image to analyze its authenticity and detect potential morphing artifacts.")

uploaded_file = st.file_uploader("Upload an image for analysis...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- Columns Layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Analysis Panel")
        if st.button('Analyze Image', use_container_width=True):
            if model:
                with st.spinner('Analyzing facial artifacts and generating activation maps...'):
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
                    
                    cam = GradCAM(model=model, target_layers=target_layers)
                    targets = [ClassifierOutputTarget(predicted_class.item())]
                    
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                    
                    # Resize original image to match model input size and normalize to [0, 1]
                    img_resized = np.array(image.resize((224, 224))) / 255.0
                    
                    # Overlay heatmap on the original image
                    cam_visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
                    
                    st.divider()
                    
                    # --- Results Display ---
                    if predicted_class.item() == 0: 
                        st.success(f"Classification: {label}")
                    else: 
                        st.error(f"Classification: {label}")
                    
                    st.metric(label="Confidence Score", value=f"{score:.2f}%")
                    
                    # --- Display XAI Heatmap ---
                    st.markdown("**Explainable AI (Grad-CAM):**")
                    st.markdown("Heatmap indicates the regions most responsible for the model's decision.")
                    st.image(cam_visualization, use_container_width=True)
                    
                    # --- Expander for Technical Details ---
                    with st.expander("View Probability Distribution"):
                        st.markdown("**Class Probabilities:**")
                        prob_real = probabilities[0][0].item() * 100
                        prob_fake = probabilities[0][1].item() * 100
                        
                        st.progress(int(prob_real), text=f"Bona Fide: {prob_real:.2f}%")
                        st.progress(int(prob_fake), text=f"Morphed: {prob_fake:.2f}%")
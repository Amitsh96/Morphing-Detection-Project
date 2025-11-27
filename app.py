"""
S-MAD (Single-Image Morphing Attack Detection) - Streamlit Web Application
Interactive web interface for face morphing attack detection
"""

import streamlit as st
import torch
from PIL import Image
import os
import sys
import numpy as np
import time

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import MorphAttackDetector


# Page configuration
st.set_page_config(
    page_title="S-MAD: Face Morphing Attack Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """
    Load and cache the MorphDetector model.
    Checks for custom weights in models/checkpoints directory.
    """
    checkpoint_path = None
    models_dir = "models/checkpoints"

    # Look for custom model weights
    if os.path.exists(models_dir):
        checkpoint_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if checkpoint_files:
            # Use the most recent checkpoint
            checkpoint_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(models_dir, x)), reverse=True)
            checkpoint_path = os.path.join(models_dir, checkpoint_files[0])
            st.sidebar.success(f"✅ Using custom weights: {checkpoint_files[0]}")
        else:
            st.sidebar.warning("⚠️ No custom weights found. Using ImageNet pretrained weights for demo.")
    else:
        st.sidebar.warning("⚠️ Models directory not found. Using ImageNet pretrained weights for demo.")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = MorphAttackDetector(model_path=checkpoint_path, device=device)

    st.sidebar.info(f"🖥️ Running on: {device.upper()}")
    return detector


def display_prediction_result(label, confidence, image):
    """
    Display the prediction result with appropriate styling.
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Detection Result")

        if label == "Real":
            st.success(f"🟢 **BONA FIDE (Real Face)**")
            st.success(f"**Confidence: {confidence:.2%}**")
            st.info("This image appears to contain a genuine, unmodified face.")
        else:  # Morphed
            st.error(f"🔴 **MORPHING ATTACK DETECTED**")
            st.error(f"**Confidence: {confidence:.2%}**")
            st.warning("⚠️ This image may contain a morphed face! Exercise caution.")

        # Confidence bar
        st.subheader("Confidence Score")
        progress_color = "green" if label == "Real" else "red"
        st.progress(confidence)
        st.caption(f"Score: {confidence:.4f}")


def display_detailed_results(detailed_results):
    """
    Display detailed prediction results in an expandable section.
    """
    with st.expander("🔬 Detailed Analysis", expanded=False):
        st.subheader("Probability Distribution")

        real_prob = detailed_results['probabilities']['Real']
        morph_prob = detailed_results['probabilities']['Morphed']

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real Face", f"{real_prob:.4f}", f"{real_prob:.2%}")
        with col2:
            st.metric("Morphed Face", f"{morph_prob:.4f}", f"{morph_prob:.2%}")

        # Bar chart for probabilities
        prob_data = {
            'Class': ['Real', 'Morphed'],
            'Probability': [real_prob, morph_prob]
        }
        st.bar_chart(prob_data, x='Class', y='Probability')

        st.subheader("Technical Details")
        st.json(detailed_results)


def main():
    """
    Main application function.
    """
    # Header
    st.title("🔍 S-MAD: Face Morphing Attack Detection")
    st.markdown("""
    **Single-Image Morphing Attack Detection System**

    Upload a face image to detect whether it's genuine (**Bona Fide**) or contains a **morphing attack**.
    Morphing attacks blend two different face images to create a synthetic face that can fool face recognition systems.
    """)

    # Sidebar
    st.sidebar.title("⚙️ System Information")

    # Load model
    with st.spinner("Loading AI model..."):
        try:
            detector = load_model()
            st.sidebar.success("✅ Model loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            st.error("Failed to load the detection model. Please check your setup.")
            return

    # File uploader
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a face image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image containing a face for morphing attack detection."
    )

    if uploaded_file is not None:
        # Display upload info
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Load and display image
        try:
            image = Image.open(uploaded_file)
            image_rgb = image.convert('RGB')

            # Show image preview
            with st.container():
                st.subheader("🖼️ Image Preview")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption=uploaded_file.name, use_column_width=True)

            # Prediction button
            if st.button("🚀 Analyze for Morphing Attacks", type="primary"):
                with st.spinner("Analyzing image for morphing attacks..."):
                    start_time = time.time()

                    try:
                        # Get prediction
                        label, confidence = detector.predict_image(image_rgb)

                        # Get detailed results
                        detailed_results = detector.get_detailed_prediction(image_rgb)

                        analysis_time = time.time() - start_time

                        # Display results
                        st.subheader("📊 Analysis Results")
                        display_prediction_result(label, confidence, image)

                        # Performance info
                        st.caption(f"⏱️ Analysis completed in {analysis_time:.2f} seconds")

                        # Detailed results
                        display_detailed_results(detailed_results)

                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.info("Please try with a different image or check if the image contains a clear face.")

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.info("Please upload a valid image file (JPG, PNG, or JPEG).")

    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a face image to begin analysis")

        with st.expander("ℹ️ How to use this system", expanded=False):
            st.markdown("""
            ### Instructions:
            1. **Upload** a clear face image (JPG, PNG, or JPEG format)
            2. **Click** the "Analyze for Morphing Attacks" button
            3. **Review** the results:
               - 🟢 **Green**: Bona Fide (genuine face)
               - 🔴 **Red**: Morphing attack detected

            ### About Morphing Attacks:
            Face morphing is a technique where two different face images are blended to create a synthetic face.
            These attacks can potentially fool face recognition systems by creating an image that matches
            multiple identities.

            ### Model Information:
            - Based on ResNet50 architecture with transfer learning
            - Trained for binary classification (Real vs. Morphed)
            - Uses ImageNet pretrained features as baseline
            """)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center'>
            <small>S-MAD: Single-Image Morphing Attack Detection<br>
            Powered by PyTorch & Streamlit</small>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
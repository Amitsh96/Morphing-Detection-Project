# S-MAD: Single-Image Morphing Attack Detection

🔍 **A deep learning-based system for detecting face morphing attacks using transfer learning on ResNet50.**

Face morphing attacks blend two different face images to create synthetic faces that can potentially fool face recognition systems. This project provides a complete solution for detecting such attacks with a user-friendly web interface.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project:**
   ```bash
   git clone <your-repository-url>
   cd s-mad
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)

## 🏗️ Project Structure

```
s-mad/
├── app.py                          # Streamlit web application (main entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   ├── raw/                       # Raw dataset storage
│   └── processed/                 # Processed data/tensors
├── models/
│   └── checkpoints/              # Trained model weights (.pth files)
├── src/                          # Core logic modules
│   ├── __init__.py              # Package initialization
│   ├── model.py                 # MorphDetector model class
│   └── inference.py             # Prediction and preprocessing logic
└── notebooks/                    # Jupyter notebooks for experimentation
```

## 🎯 Features

### Core Functionality
- **Binary Classification**: Detects "Real" (Bona Fide) vs. "Morphed" faces
- **Transfer Learning**: Uses ResNet50 pretrained on ImageNet as backbone
- **Real-time Inference**: Fast prediction with confidence scores
- **Flexible Input**: Supports JPG, PNG, and JPEG image formats

### Web Interface
- **Intuitive Design**: Clean Streamlit interface with drag-and-drop upload
- **Visual Feedback**:
  - 🟢 Green alerts for genuine faces
  - 🔴 Red alerts for detected morphing attacks
- **Detailed Analysis**: Expandable section with probability distributions
- **Performance Metrics**: Analysis time and confidence visualization

### Model Features
- **Automatic Weight Loading**: Detects custom weights in `models/checkpoints/`
- **Fallback Mode**: Uses ImageNet weights if no custom weights found
- **Device Detection**: Automatically uses GPU if available
- **Modular Design**: Easy to extend and modify

## 🔧 Usage

### Web Interface

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload an image** through the web interface

3. **Click "Analyze for Morphing Attacks"** to get results

4. **Review the prediction:**
   - **Bona Fide (Real)**: Green success message
   - **Morphing Attack**: Red warning message
   - **Confidence Score**: Numerical confidence (0-1)

### Programmatic Usage

```python
from src.inference import MorphAttackDetector

# Initialize detector
detector = MorphAttackDetector()

# Predict on image file
label, confidence = detector.predict_image("path/to/image.jpg")
print(f"Prediction: {label} (Confidence: {confidence:.2%})")

# Get detailed results
detailed = detector.get_detailed_prediction("path/to/image.jpg")
print(f"Probabilities: {detailed['probabilities']}")
```

### Custom Model Training

To use your own trained weights:

1. **Save your trained model** to `models/checkpoints/your_model.pth`
2. **Restart the application** - it will automatically detect and load your weights
3. The interface will show "✅ Using custom weights: your_model.pth"

## 🧠 Model Architecture

### Base Architecture
- **Backbone**: ResNet50 pretrained on ImageNet
- **Modification**: Final fully connected layer replaced for 2-class output
- **Input**: RGB images resized to 224×224 pixels
- **Output**: Binary classification logits (Real vs. Morphed)

### Preprocessing Pipeline
1. **Resize**: Images scaled to 224×224 pixels
2. **Normalization**: ImageNet mean and standard deviation
3. **Tensor Conversion**: PIL/numpy arrays → PyTorch tensors
4. **Device Transfer**: Automatic GPU/CPU handling

## 📊 Model Performance

The model uses transfer learning from ImageNet-pretrained ResNet50:

- **Input Resolution**: 224×224×3
- **Output Classes**: 2 (Real, Morphed)
- **Inference Speed**: ~0.1-0.5 seconds per image (CPU)
- **Memory Usage**: ~500MB (model + overhead)

**Note**: Performance metrics shown are for the baseline ImageNet model. Train on your specific dataset for optimal results.

## 🛠️ Development

### Adding New Features

1. **Model modifications**: Edit `src/model.py`
2. **Preprocessing changes**: Modify `src/inference.py`
3. **UI enhancements**: Update `app.py`

### Training Your Own Model

```python
from src.model import create_model

# Create model
model = create_model(num_classes=2, pretrained=True)

# Train your model here...
# (Add your training loop)

# Save weights
model.save_weights("models/checkpoints/my_model.pth")
```

### Running Tests

Create test images in `data/raw/` and test with:

```python
from src.inference import MorphAttackDetector

detector = MorphAttackDetector()
results = detector.predict_batch([
    "data/raw/real_face.jpg",
    "data/raw/morphed_face.jpg"
])
```

## 📋 Dependencies

Key libraries used in this project:

- **torch** (≥2.0.0): Deep learning framework
- **torchvision** (≥0.15.0): Computer vision utilities
- **streamlit** (≥1.28.0): Web application framework
- **opencv-python-headless** (≥4.8.0): Image processing
- **Pillow** (≥9.5.0): Image manipulation
- **numpy** (≥1.24.0): Numerical computing
- **scikit-learn** (≥1.3.0): Machine learning utilities

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

2. **CUDA errors**: The application automatically falls back to CPU if GPU is unavailable

3. **Image loading errors**: Ensure images are in supported formats (JPG, PNG, JPEG)

4. **Memory issues**: Large images are automatically resized to 224×224

### Model Loading Issues

- **No custom weights warning**: Normal behavior when no trained weights are available
- **Weight loading errors**: Check file format and model compatibility
- **Device errors**: Application will automatically use CPU if GPU unavailable

## 📝 License

This project is created for educational purposes as part of a software engineering final project.

## 🤝 Contributing

This is a student project. Contributions and suggestions are welcome!

## 📧 Contact

For questions about this project, please refer to the course materials or contact your instructor.

---

**S-MAD (Single-Image Morphing Attack Detection)** - Detecting face morphing attacks with deep learning.
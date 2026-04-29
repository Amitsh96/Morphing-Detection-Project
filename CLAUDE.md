# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Note: `pytorch-grad-cam` is used in `app.py` for Grad-CAM XAI visualization but is not listed in `requirements.txt`. Install it separately if needed:
```bash
pip install grad-cam
```

**Run the web application:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

**Run inference programmatically:**
```python
from src.inference import MorphAttackDetector
detector = MorphAttackDetector(model_path="models/resnet50_morph.pth")
label, confidence = detector.predict_image("path/to/face.jpg")
```

## Architecture

This is **S-MAD (Single-Image Morphing Attack Detection)** — a Streamlit web app that classifies face images as "Bona Fide (Real)" or "Morphed (Fake)" using a fine-tuned ResNet50.

### Two parallel code paths

**`app.py` (production UI):** Loads the model directly using `torch.load('models/resnet50_morph.pth')` and applies `models.resnet50` with a replaced `fc` layer. Runs Grad-CAM on `model.layer4[-1]` to generate explainability heatmaps overlaid on the image. Note: this file does *not* use the `src/` modules.

**`src/` package (programmatic API):**
- `src/model.py` — `MorphDetector(nn.Module)` wraps ResNet50, replacing the final FC layer for 2-class output. `create_model()` is the factory function.
- `src/inference.py` — `ImagePreprocessor` handles PIL/numpy/path inputs with ImageNet normalization. `MorphAttackDetector` wraps the model for `predict_image()`, `predict_batch()`, and `get_detailed_prediction()`.

### Model details
- **Backbone:** ResNet50 pretrained on ImageNet, final FC replaced with `Linear(2048, 2)`
- **Input:** 224×224 RGB, normalized with ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- **Output:** Logits for `[Bona Fide, Morphed]`; softmax gives confidence scores
- **Trained weights:** Expected at `models/resnet50_morph.pth` (flat state dict format)
- **XAI:** Grad-CAM targets `model.layer4[-1]` (last conv block of ResNet50)

### Data layout
```
models/               # Trained .pth weights
data/raw/             # Original face images
data/processed/       # Preprocessed tensors
notebooks/            # Jupyter notebooks for training experiments
```

### Checkpoint formats
`src/model.py` handles three formats: `{'model_state_dict': ...}`, `{'state_dict': ...}`, or a raw state dict. `app.py` assumes a raw state dict only.

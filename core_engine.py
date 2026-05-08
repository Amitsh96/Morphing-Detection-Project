import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_model(path, device):
    """טוען מודל בודד לזיכרון"""
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

def load_all_systems():
    """טוען את חותך הפנים ושני המודלים (השלם והחסין)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(select_largest=True, post_process=False, device=device)
    model_full = get_model('models/best_mad_resnet50.pth', device)
    model_crop = get_model('models/robust_mad_resnet50.pth', device)
    return mtcnn, model_full, model_crop, device

def process_image(image):
    """מכין תמונה להזנה לתוך המודל"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def generate_cam(model, input_tensor, pil_img, predicted_idx):
    """מייצר מפת חום שמראה על מה המודל הסתכל"""
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    img_resized = np.array(pil_img.resize((224, 224))) / 255.0
    return show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
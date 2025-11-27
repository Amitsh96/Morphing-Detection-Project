"""
S-MAD (Single-Image Morphing Attack Detection) Inference
Image preprocessing and prediction logic
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from typing import Tuple, Union, Optional
import os

from .model import MorphDetector, create_model


class ImagePreprocessor:
    """
    Handles image preprocessing for the MorphDetector model.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image preprocessor.

        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet pretrained means
                std=[0.229, 0.224, 0.225]   # ImageNet pretrained stds
            )
        ])

    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for model inference.

        Args:
            image: Input image (file path, numpy array, or PIL Image)

        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            # Load from file path
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.shape[-1] == 3:  # RGB
                pil_image = Image.fromarray(image)
            elif len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Apply transformations
        tensor = self.transform(pil_image)

        # Add batch dimension
        return tensor.unsqueeze(0)


class MorphAttackDetector:
    """
    Main detector class for morphing attack detection.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Initialize the morph attack detector.

        Args:
            model_path: Path to custom model weights (optional)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        # Load model
        self.model = create_model(
            num_classes=2,
            pretrained=True,
            checkpoint_path=model_path,
            device=self.device
        )
        self.model.eval()

        # Class labels
        self.class_labels = ['Real', 'Morphed']

    @torch.no_grad()
    def predict_image(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[str, float]:
        """
        Predict whether an image contains a morphed face or not.

        Args:
            image: Input image (file path, numpy array, or PIL Image)

        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        try:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

            # Get prediction
            confidence, predicted = torch.max(probabilities, 1)
            predicted_label = self.class_labels[predicted.item()]
            confidence_score = confidence.item()

            return predicted_label, confidence_score

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """
        Predict on a batch of images.

        Args:
            images: List of images (file paths, numpy arrays, or PIL Images)

        Returns:
            List of tuples (predicted_label, confidence_score)
        """
        results = []
        for image in images:
            try:
                result = self.predict_image(image)
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(("Error", 0.0))
        return results

    def get_detailed_prediction(self, image: Union[str, np.ndarray, Image.Image]) -> dict:
        """
        Get detailed prediction results including probabilities for both classes.

        Args:
            image: Input image (file path, numpy array, or PIL Image)

        Returns:
            Dictionary with detailed prediction information
        """
        try:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

            # Get detailed results
            probs = probabilities.squeeze().cpu().numpy()
            predicted_class = np.argmax(probs)

            return {
                'predicted_class': self.class_labels[predicted_class],
                'confidence': float(probs[predicted_class]),
                'probabilities': {
                    'Real': float(probs[0]),
                    'Morphed': float(probs[1])
                },
                'raw_logits': outputs.squeeze().cpu().numpy().tolist()
            }

        except Exception as e:
            raise RuntimeError(f"Error during detailed prediction: {str(e)}")


def predict_image(model_path: Optional[str], image_path: str,
                 device: str = 'cpu') -> Tuple[str, float]:
    """
    Standalone function for single image prediction.

    Args:
        model_path: Path to model weights (None for ImageNet pretrained)
        image_path: Path to the image file
        device: Device to run inference on

    Returns:
        Tuple of (predicted_label, confidence_score)
    """
    detector = MorphAttackDetector(model_path=model_path, device=device)
    return detector.predict_image(image_path)


def validate_image_file(file_path: str) -> bool:
    """
    Validate if a file is a supported image format.

    Args:
        file_path: Path to the image file

    Returns:
        True if valid image file, False otherwise
    """
    if not os.path.exists(file_path):
        return False

    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path.lower())[1]

    if file_extension not in supported_formats:
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Example usage
    detector = MorphAttackDetector()

    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    label, confidence = detector.predict_image(dummy_image)

    print(f"Predicted: {label} with confidence: {confidence:.4f}")

    # Get detailed prediction
    detailed = detector.get_detailed_prediction(dummy_image)
    print(f"Detailed prediction: {detailed}")
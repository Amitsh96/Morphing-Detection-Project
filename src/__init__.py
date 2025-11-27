"""
S-MAD (Single-Image Morphing Attack Detection) Package
"""

from .model import MorphDetector, create_model
from .inference import MorphAttackDetector, ImagePreprocessor, predict_image

__all__ = [
    'MorphDetector',
    'create_model',
    'MorphAttackDetector',
    'ImagePreprocessor',
    'predict_image'
]
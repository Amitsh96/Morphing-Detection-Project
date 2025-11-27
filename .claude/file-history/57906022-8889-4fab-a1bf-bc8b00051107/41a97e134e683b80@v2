"""
S-MAD (Single-Image Morphing Attack Detection) Model
Core model implementation using ResNet50 with transfer learning
"""

import torch
import torch.nn as nn
from torchvision import models
import os
from typing import Optional


class MorphDetector(nn.Module):
    """
    Morphing attack detector based on ResNet50 architecture.

    The model uses a pre-trained ResNet50 backbone and replaces the final
    classification layer for binary classification (Real vs. Morphed faces).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the MorphDetector model.

        Args:
            num_classes: Number of output classes (default: 2 for Real/Morphed)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(MorphDetector, self).__init__()

        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Get the number of input features for the final layer
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(num_features, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def load_weights(self, checkpoint_path: str, device: str = 'cpu') -> bool:
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pth checkpoint file
            device: Device to load the model on ('cpu' or 'cuda')

        Returns:
            True if weights loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint file not found at {checkpoint_path}")
                return False

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Load state dict
            self.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            return False

    def save_weights(self, checkpoint_path: str, epoch: Optional[int] = None,
                    optimizer_state: Optional[dict] = None) -> bool:
        """
        Save model weights to a checkpoint file.

        Args:
            checkpoint_path: Path to save the .pth checkpoint file
            epoch: Current epoch number (optional)
            optimizer_state: Optimizer state dict (optional)

        Returns:
            True if weights saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': self.state_dict(),
                'num_classes': self.num_classes
            }

            if epoch is not None:
                checkpoint_data['epoch'] = epoch

            if optimizer_state is not None:
                checkpoint_data['optimizer_state_dict'] = optimizer_state

            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Successfully saved weights to {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error saving weights: {str(e)}")
            return False

    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extractor (backbone without final classification layer).

        Returns:
            Feature extractor module
        """
        feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        return feature_extractor


def create_model(num_classes: int = 2, pretrained: bool = True,
                checkpoint_path: Optional[str] = None, device: str = 'cpu') -> MorphDetector:
    """
    Factory function to create and optionally load a MorphDetector model.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        checkpoint_path: Path to custom checkpoint file (optional)
        device: Device to load the model on

    Returns:
        Initialized MorphDetector model
    """
    model = MorphDetector(num_classes=num_classes, pretrained=pretrained)
    model.to(device)

    if checkpoint_path:
        model.load_weights(checkpoint_path, device)

    return model


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"Model loaded on: {device}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
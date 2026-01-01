"""
Enhanced ASCII Character Selector using Trained ResNet
=======================================================
This module uses the ResNet model trained on 140k ASCII images 
to improve character selection accuracy.

Usage:
    1. Download ascii_model.pth from Colab
    2. Place in models/ascii_model.pth
    3. Import and use EnhancedMapper
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import os

class EnhancedASCIIMapper:
    """Uses trained ResNet to extract ASCII-aware features for better mapping."""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Default model path - use latest Kaggle-trained model
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "ascii_resnet18_final.pth"
        
        self.model_path = Path(model_path)
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Character ramps for mapping
        self.RAMP = " .:-=+*#%@"
        self._load_model()
    
    def _load_model(self):
        """Load the trained ResNet model."""
        if not self.model_path.exists():
            print(f"⚠️ Model not found at {self.model_path}")
            print("   Download ascii_model.pth from Colab and place in models/")
            self.model = None
            return
            
        try:
            # Create model architecture based on filename
            is_vit = "vit" in str(self.model_path).lower()
            
            if is_vit:
                try:
                    # Initialize ViT-B/16 architecture
                    self.model = models.vit_b_16(weights=None)
                    # Modify head for feature extraction (256 dim to match checkpoint)
                    self.model.heads = nn.Linear(768, 256)
                except AttributeError:
                    # Fallback for older torchvision versions
                     self.model = models.vit_b_16(pretrained=False)
                     self.model.heads = nn.Linear(768, 256)
            else:
                # Initialize ResNet18 architecture
                self.model = models.resnet18(weights=None)
                self.model.fc = nn.Linear(512, 128)
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Enhanced ASCII mapper loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract ASCII-aware features from an image."""
        if self.model is None:
            return None
            
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(tensor)
            return features.cpu().numpy().flatten()
    
    def enhanced_char_select(self, tile: Image.Image, brightness: float) -> str:
        """
        Select character using both brightness and trained features.
        
        Args:
            tile: PIL Image of the tile
            brightness: Average brightness (0-255)
            
        Returns:
            Selected ASCII character
        """
        # Base selection from brightness
        brightness_idx = int((brightness / 255) * (len(self.RAMP) - 1))
        base_char = self.RAMP[brightness_idx]
        
        if self.model is None:
            return base_char
            
        # Extract features for refinement
        features = self.extract_features(tile)
        if features is None:
            return base_char
        
        # Use feature magnitude to adjust character density
        feature_magnitude = np.mean(np.abs(features))
        
        # High features = more detail = denser character
        if feature_magnitude > 0.5:
            # Shift towards denser characters
            new_idx = min(brightness_idx + 1, len(self.RAMP) - 1)
        elif feature_magnitude < 0.2:
            # Shift towards sparser characters for smooth areas
            new_idx = max(brightness_idx - 1, 0)
        else:
            new_idx = brightness_idx
            
        return self.RAMP[new_idx]
    
    def is_available(self) -> bool:
        """Check if the enhanced model is loaded."""
        return self.model is not None


# Singleton instance
_enhanced_mapper = None
_current_model_path = None

def get_enhanced_mapper(model_path: str = None) -> EnhancedASCIIMapper:
    """Get or create the enhanced mapper instance with specified model."""
    global _enhanced_mapper, _current_model_path
    
    # If model path changed, reload
    if model_path and model_path != _current_model_path:
        _enhanced_mapper = EnhancedASCIIMapper(model_path=model_path)
        _current_model_path = model_path
    elif _enhanced_mapper is None:
        _enhanced_mapper = EnhancedASCIIMapper(model_path=model_path)
        _current_model_path = model_path
    
    return _enhanced_mapper

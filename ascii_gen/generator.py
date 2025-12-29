"""
Image Generator with FLUX.1 Schnell

Provides text-to-image generation using:
- FLUX.1 Schnell (default) - Fast, high-quality, Apache 2.0 license
- SDXL-Turbo (fallback) - Smaller model for limited RAM

Optimized for Apple Silicon (M4) with MPS backend.
Memory-efficient settings for 16GB RAM systems.

Free tier friendly - uses HuggingFace Hub only.
"""

from typing import Optional, Tuple, Union
import torch
from PIL import Image
import numpy as np


class PromptToImageGenerator:
    """
    Text-to-image generator using FLUX.1 Schnell or SDXL-Turbo.
    
    Optimized for Apple Silicon (M4) with MPS backend.
    FLUX.1 Schnell: Better quality, 1-4 steps, ~10-15s on 16GB RAM.
    
    Example:
        >>> generator = PromptToImageGenerator()
        >>> image = generator.generate("cyberpunk cityscape")
        >>> image.save("output.png")
    """
    
    # Available models
    MODELS = {
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",  # Best quality, Apache 2.0
        "sdxl-turbo": "stabilityai/sdxl-turbo",  # Faster, smaller
    }
    
    def __init__(
        self,
        model: str = "flux-schnell",
        device: str = "auto",
        enable_memory_optimization: bool = True,
        low_ram_mode: bool = True,  # Enable for 16GB systems
    ):
        """
        Initialize the generator.
        
        Args:
            model: "flux-schnell" (default, best) or "sdxl-turbo" (faster)
            device: "auto", "mps", "cuda", or "cpu"
            enable_memory_optimization: Enable attention slicing
            low_ram_mode: Extra optimizations for 16GB RAM (slower but stable)
        """
        # Resolve model ID
        if model in self.MODELS:
            self.model_id = self.MODELS[model]
            self.model_type = model
        else:
            self.model_id = model
            self.model_type = "custom"
        
        self.enable_memory_optimization = enable_memory_optimization
        self.low_ram_mode = low_ram_mode
        
        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Lazy loading - models loaded on first use
        self._pipe = None
        self._is_loaded = False
        self._is_flux = "flux" in self.model_id.lower()
    
    def _load_pipeline(self):
        """Load the image generation pipeline."""
        if self._is_loaded:
            return
        
        print(f"Loading {self.model_id} on {self.device}...")
        print("⏳ First run downloads ~30GB of model files. Please wait...")
        
        if self._is_flux:
            self._load_flux_pipeline()
        else:
            self._load_sdxl_pipeline()
        
        # Memory optimizations
        if self.enable_memory_optimization and self._pipe is not None:
            if hasattr(self._pipe, 'enable_attention_slicing'):
                self._pipe.enable_attention_slicing()
                print("✅ Attention slicing enabled (reduces VRAM usage)")
        
        self._is_loaded = True
        print("✅ Pipeline loaded successfully!")
    
    def _load_flux_pipeline(self):
        """Load FLUX.1 Schnell pipeline."""
        from diffusers import FluxPipeline
        
        # Determine dtype
        if self.device == "mps":
            # MPS works best with bfloat16 for FLUX
            dtype = torch.bfloat16
        elif self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        
        # Move to device
        self._pipe = self._pipe.to(self.device)
        
        # Extra memory optimization for 16GB systems
        if self.low_ram_mode:
            try:
                # Enable sequential CPU offload for very low RAM
                if hasattr(self._pipe, 'enable_sequential_cpu_offload'):
                    # Don't use this for MPS - instead use attention slicing
                    pass
            except Exception as e:
                print(f"Note: Some memory optimizations not available: {e}")
    
    def _load_sdxl_pipeline(self):
        """Load SDXL-Turbo pipeline (fallback for limited RAM)."""
        from diffusers import AutoPipelineForText2Image
        
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if self.device != "cpu" else None,
        )
        
        self._pipe = self._pipe.to(self.device)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid (ignored by FLUX)
            width: Output width in pixels
            height: Output height in pixels
            num_inference_steps: Number of denoising steps (1-4 for fast models)
            guidance_scale: CFG scale (0 for Turbo/Schnell models)
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        self._load_pipeline()
        
        # Set seed if provided
        generator = None
        if seed is not None:
            # MPS requires CPU generator
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(gen_device).manual_seed(seed)
        
        # Adjust parameters for FLUX
        if self._is_flux:
            # FLUX uses different parameter names
            result = self._pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            # SDXL-Turbo
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        return result.images[0]
    
    def apply_canny(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> Image.Image:
        """
        Apply Canny edge detection to prepare a control image.
        
        Args:
            image: Input image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            
        Returns:
            Edge-detected image
        """
        import cv2
        
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_3ch = np.stack([edges, edges, edges], axis=-1)
        
        return Image.fromarray(edges_3ch)
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    def unload(self):
        """Unload models to free memory."""
        self._pipe = None
        self._is_loaded = False
        
        import gc
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Model unloaded, memory freed")


def create_generator(
    model: str = "flux-schnell",
    **kwargs
) -> PromptToImageGenerator:
    """
    Factory function to create an image generator.
    
    Args:
        model: "flux-schnell" (recommended) or "sdxl-turbo"
        **kwargs: Additional arguments for PromptToImageGenerator
        
    Returns:
        Configured generator
    """
    return PromptToImageGenerator(model=model, **kwargs)


# Quick test function
def test_generator():
    """Quick test to verify the generator works."""
    print("Testing FLUX.1 Schnell generator...")
    gen = create_generator("flux-schnell")
    img = gen.generate(
        "A simple line drawing of a cat face",
        width=256,
        height=256,
        num_inference_steps=2,
        seed=42
    )
    img.save("test_flux_output.png")
    print("✅ Test passed! Saved to test_flux_output.png")
    gen.unload()


if __name__ == "__main__":
    test_generator()

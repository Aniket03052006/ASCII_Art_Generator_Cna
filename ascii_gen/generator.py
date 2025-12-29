"""
Stable Diffusion Image Generator

Provides text-to-image generation using Stable Diffusion with:
- SDXL-Turbo for fast inference (4 steps)
- ControlNet for structural guidance
- MPS (Metal) optimization for Apple Silicon

Free tier friendly - uses HuggingFace Hub only.
"""

from typing import Optional, Tuple, Union
import torch
from PIL import Image
import numpy as np


class PromptToImageGenerator:
    """
    Text-to-image generator using Stable Diffusion.
    
    Optimized for Apple Silicon (M4) with MPS backend.
    Uses SDXL-Turbo for fast inference (~4 steps).
    
    Example:
        >>> generator = PromptToImageGenerator()
        >>> image = generator.generate("cyberpunk cityscape")
        >>> image.save("output.png")
    """
    
    def __init__(
        self,
        model: str = "stabilityai/sdxl-turbo",
        controlnet_model: Optional[str] = None,  # "lllyasviel/sd-controlnet-canny"
        device: str = "auto",
        enable_cpu_offload: bool = True,
        compile_model: bool = False,
    ):
        """
        Initialize the generator.
        
        Args:
            model: HuggingFace model ID for Stable Diffusion
            controlnet_model: Optional ControlNet model ID
            device: "auto", "mps", "cuda", or "cpu"
            enable_cpu_offload: Enable memory-efficient CPU offloading
            compile_model: Use torch.compile for faster inference (experimental)
        """
        self.model_id = model
        self.controlnet_id = controlnet_model
        self.enable_cpu_offload = enable_cpu_offload
        self.compile_model = compile_model
        
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
        self._controlnet_pipe = None
        self._is_loaded = False
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline."""
        if self._is_loaded:
            return
        
        from diffusers import AutoPipelineForText2Image
        
        print(f"Loading {self.model_id} on {self.device}...")
        
        # Load base pipeline
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            variant="fp16" if self.device != "cpu" else None,
        )
        
        # Move to device
        if self.device == "mps":
            self._pipe = self._pipe.to("mps")
        elif self.device == "cuda":
            if self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe = self._pipe.to("cuda")
        else:
            self._pipe = self._pipe.to("cpu")
        
        # Compile for faster inference (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            self._pipe.unet = torch.compile(self._pipe.unet, mode="reduce-overhead")
        
        self._is_loaded = True
        print("Pipeline loaded successfully!")
    
    def _load_controlnet_pipeline(self):
        """Load ControlNet pipeline if configured."""
        if not self.controlnet_id or self._controlnet_pipe is not None:
            return
        
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        
        print(f"Loading ControlNet: {self.controlnet_id}...")
        
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        # Create pipeline with ControlNet
        self._controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        if self.device == "mps":
            self._controlnet_pipe = self._controlnet_pipe.to("mps")
        elif self.device == "cuda":
            self._controlnet_pipe.enable_model_cpu_offload()
        
        print("ControlNet loaded!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,  # SDXL-Turbo is 4-step
        guidance_scale: float = 0.0,   # Turbo doesn't need CFG
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            width: Output width in pixels
            height: Output height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (0 for Turbo models)
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        self._load_pipeline()
        
        # Set seed if provided
        generator = None
        if seed is not None:
            if self.device == "mps":
                generator = torch.Generator("cpu").manual_seed(seed)
            else:
                generator = torch.Generator(self.device).manual_seed(seed)
        
        # Generate
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
    
    def generate_with_control(
        self,
        prompt: str,
        control_image: Image.Image,
        control_strength: float = 0.8,
        negative_prompt: str = "blurry, low quality",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image with ControlNet guidance.
        
        Args:
            prompt: Text description
            control_image: Control image (e.g., Canny edges)
            control_strength: How strongly to follow the control (0-1)
            negative_prompt: What to avoid
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            
        Returns:
            Generated PIL Image
        """
        if not self.controlnet_id:
            raise RuntimeError("ControlNet model not configured. Set controlnet_model in __init__.")
        
        self._load_controlnet_pipeline()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device if self.device != "mps" else "cpu").manual_seed(seed)
        
        result = self._controlnet_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=control_strength,
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
            Edge-detected image suitable for ControlNet
        """
        import cv2
        
        # Convert to numpy
        img_array = np.array(image.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to 3-channel for ControlNet
        edges_3ch = np.stack([edges, edges, edges], axis=-1)
        
        return Image.fromarray(edges_3ch)
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    def unload(self):
        """Unload models to free memory."""
        self._pipe = None
        self._controlnet_pipe = None
        self._is_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_generator(
    model: str = "stabilityai/sdxl-turbo",
    use_controlnet: bool = False,
    **kwargs
) -> PromptToImageGenerator:
    """
    Factory function to create an image generator.
    
    Args:
        model: Model ID
        use_controlnet: Whether to enable ControlNet
        **kwargs: Additional arguments for PromptToImageGenerator
        
    Returns:
        Configured generator
    """
    controlnet = "lllyasviel/sd-controlnet-canny" if use_controlnet else None
    return PromptToImageGenerator(model=model, controlnet_model=controlnet, **kwargs)

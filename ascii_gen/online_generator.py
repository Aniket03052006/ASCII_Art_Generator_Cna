"""
Online API-based Image Generator

Uses HuggingFace Inference API for FREE cloud-based image generation.
No 30GB download required!

Free tier: 300 requests/hour (after registration)
"""

import os
import io
import time
import requests
from PIL import Image
from typing import Optional

# Default to FLUX.1-schnell (faster, Apache 2.0 license)
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"
API_URL = "https://router.huggingface.co/hf-inference/models/"


class OnlineGenerator:
    """
    Generate images using HuggingFace Inference API.
    
    No local GPU or 30GB download required!
    
    Example:
        >>> gen = OnlineGenerator(api_key="hf_...")
        >>> image = gen.generate("a mountain landscape")
        >>> image.save("output.png")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize the online generator.
        
        Args:
            api_key: HuggingFace API token (get free at huggingface.co/settings/tokens)
            model: Model ID (default: FLUX.1-schnell)
        """
        # Try to get API key from environment or parameter
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.api_key:
            print("⚠️  No API key found. Get a free one at:")
            print("   https://huggingface.co/settings/tokens")
            print("   Then set: export HF_TOKEN='your_token_here'")
        
        self.model = model
        self.api_url = f"{API_URL}{model}"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
        max_retries: int = 3,
    ) -> Optional[Image.Image]:
        """
        Generate an image from a text prompt using the cloud API.
        
        Args:
            prompt: Text description
            width: Image width
            height: Image height  
            guidance_scale: CFG scale (0 for FLUX Schnell)
            num_inference_steps: Number of steps
            seed: Random seed
            max_retries: Retry count for temporary errors
            
        Returns:
            PIL Image or None on failure
        """
        if not self.api_key:
            raise ValueError("API key required. Set HF_TOKEN environment variable.")
        
        # Use the dedicated prompt engineering module
        from .prompt_engineering import enhance_prompt
        
        result = enhance_prompt(prompt)
        enhanced_prompt = result.enhanced
        
        # Log enhancement info
        print(f"📝 Category: {result.category.value}")
        if result.warnings:
            for warning in result.warnings:
                print(f"⚠️  {warning}")
        print(f"📝 Enhanced: {enhanced_prompt[:80]}...")
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
        }
        
        if seed is not None:
            payload["parameters"]["seed"] = seed
        
        for attempt in range(max_retries):
            try:
                print(f"🌐 Sending request to {self.model}...")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                
                if response.status_code == 200:
                    # Success - return image
                    image = Image.open(io.BytesIO(response.content))
                    print("✅ Image generated successfully!")
                    return image
                
                elif response.status_code == 503:
                    # Model loading
                    wait_time = response.json().get("estimated_time", 20)
                    print(f"⏳ Model loading... waiting {wait_time:.0f}s")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:
                    print("❌ Invalid API key. Check your HF_TOKEN.")
                    return None
                
                elif response.status_code == 429:
                    print("⚠️  Rate limited. Waiting 60s...")
                    time.sleep(60)
                    continue
                
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout (attempt {attempt + 1}/{max_retries})")
                continue
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
        
        print("❌ Max retries exceeded")
        return None


def create_online_generator(api_key: Optional[str] = None) -> OnlineGenerator:
    """Create an online generator with API key."""
    return OnlineGenerator(api_key=api_key)


# Quick test
if __name__ == "__main__":
    print("Testing Online Generator...")
    print("=" * 50)
    
    gen = OnlineGenerator()
    
    if gen.api_key:
        img = gen.generate("a simple line drawing of a cat face")
        if img:
            img.save("test_online_output.png")
            print("Saved to test_online_output.png")
    else:
        print("\nTo test, set your API key:")
        print("  export HF_TOKEN='hf_your_token_here'")
        print("  python online_generator.py")

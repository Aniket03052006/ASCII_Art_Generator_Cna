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
        log_callback: Optional[callable] = None,
    ):
        """
        Initialize the online generator.
        
        Args:
            api_key: HuggingFace API token
            model: Model ID
            log_callback: Function to call with log messages (str)
        """
        # Try to get API key from environment or parameter
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "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"
        self.log_callback = log_callback
        
        if not self.api_key:
            self._log("⚠️  No API key found. Using Pollinations fallback by default if needed.")
        
        self.model = model
        self.api_url = f"{API_URL}{model}"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _log(self, message: str):
        """Emit log message to callback and/or stdout."""
        print(message)
        if self.log_callback:
            self.log_callback(message)
    
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
        max_retries: int = 3,
        skip_preprocessing: bool = False,
        log_callback: Optional[callable] = None,
        model_type: str = "flux",
    ) -> Optional[Image.Image]:
        """
        Generate an image from a text prompt.
        """
        # Use instance callback if none provided here
        old_callback = self.log_callback
        if log_callback:
            self.log_callback = log_callback

        if not self.api_key:
            # raise ValueError("API key required. Set HF_TOKEN environment variable.")
             self._log("⚠️ No API Key set. Switching to Fallback immediately.")
             return self._generate_pollinations(prompt, width, height, seed)
        
        working_prompt = prompt
        
        if not skip_preprocessing:
            # Stage 1: Try LLM-powered rewriting (if available)
            try:
                from .llm_rewriter import llm_rewrite_prompt
                rewritten, was_llm = llm_rewrite_prompt(prompt, model_type=model_type)
                if was_llm:
                    self._log(f"🤖 LLM Rewritten: {rewritten[:60]}...")
                    working_prompt = rewritten
            except ImportError:
                pass
            
            # Stage 2: Apply rule-based enhancement
            try:
                from .prompt_engineering import enhance_prompt
                enhanced_prompt = enhance_prompt(working_prompt)
                self._log(f"📝 Enhanced: {enhanced_prompt[:80]}...")
                working_prompt = enhanced_prompt
            except ImportError:
                pass
        
        payload = {
            "inputs": working_prompt,
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
                masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
                self._log(f"🌐 Sending request to {self.model} (Key: {masked_key})...")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    self._log("✅ Image generated successfully!")
                    self.log_callback = old_callback
                    return image
                
                elif response.status_code == 503:
                    wait_time = response.json().get("estimated_time", 20)
                    self._log(f"⏳ Model loading... waiting {wait_time:.0f}s")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:
                    self._log(f"❌ Invalid API key ({masked_key}).")
                    break 
                
                elif response.status_code == 429:
                    self._log("⚠️  Rate limited (429). Waiting 60s...")
                    time.sleep(60)
                    continue
                
                else:
                    self._log(f"❌ Error {response.status_code}: {response.text[:100]}...")
                    if attempt == max_retries - 1:
                         break
                    continue
                    
            except requests.exceptions.Timeout:
                self._log(f"⏰ Timeout (attempt {attempt + 1}/{max_retries})")
                continue
            except Exception as e:
                self._log(f"❌ Error: {e}")
                if attempt == max_retries - 1:
                     break
                continue
        
        self._log("❌ Primary API failed. Switching to Fallback...")
        
        # FALLBACK: Pollinations.ai
        self._log("\n🔄 Switching to Fallback: Pollinations.ai (Free)...")
        result = self._generate_pollinations(working_prompt, width, height, seed)
        self.log_callback = old_callback
        return result

    def _generate_pollinations(
        self, 
        prompt: str, 
        width: int, 
        height: int, 
        seed: Optional[int] = None
    ) -> Optional[Image.Image]:
        """Generate using Pollinations.ai (Free Fallback) with retry logic."""
        import urllib.parse
        import random
        
        encoded_prompt = urllib.parse.quote(prompt)
        base_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
        if seed is not None:
            base_url += f"&seed={seed}"
        
        # Add model=flux explicitly as it's their best current model
        url = base_url + "&model=flux"
        
        self._log(f"🌐 Requesting Pollinations: {url[:80]}...")
        
        max_retries = 3
        timeout = 120 # Increased timeout
        
        for attempt in range(max_retries):
            try:
                # Add headers to avoid bot detection/throttling
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }

                if attempt > 0:
                     self._log(f"   ⏳ Retry {attempt+1}/{max_retries}...")
                
                response = requests.get(url, timeout=timeout, headers=headers)
                
                if response.status_code == 200:
                    content = response.content
                    if len(content) < 4000:
                        self._log("   ⚠️ Warning: Small file received (possible error placeholder)")
                        
                    image = Image.open(io.BytesIO(content)).convert("RGB")
                    self._log("✅ Fallback Image generated successfully!")
                    return image
                else:
                    self._log(f"   ❌ Fallback Error {response.status_code}")
                    
            except Exception as e:
                self._log(f"   ❌ Fallback Exception: {e}")
            
            # Backoff before retry
            time.sleep(2 + attempt)

        self._log("❌ All Fallback Retries Failed with Flux.")
        
        # Try Pollinations with different model (turbo is faster/more reliable)
        self._log("\n🔄 Trying Pollinations with turbo model...")
        turbo_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&model=turbo"
        
        for attempt in range(2):
            try:
                if attempt > 0:
                    self._log(f"   ⏳ Retry {attempt+1}/2...")
                response = requests.get(turbo_url, timeout=90, headers=headers)
                if response.status_code == 200 and len(response.content) > 4000:
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    self._log("✅ Turbo model succeeded!")
                    return image
                else:
                    self._log(f"   ❌ Turbo Error {response.status_code}")
            except Exception as e:
                self._log(f"   ❌ Turbo Exception: {e}")
            time.sleep(2)
        
        self._log("❌ All image generation attempts failed. Try again in a moment.")
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

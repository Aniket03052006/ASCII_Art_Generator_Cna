"""
FastAPI Backend for ASCII Art Generator React Frontend
Wraps the existing ascii_gen module for HTTP access
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import sys
import base64
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.production_training import ProductionCNNMapper
from ascii_gen.gradient_mapper import (
    GradientMapper, GradientConfig,
    RAMP_STANDARD, RAMP_ULTRA, RAMP_NEAT, RAMP_DETAILED, RAMP_MINIMAL
)
from ascii_gen.perceptual import SSIMMapper
from ascii_gen.llm_rewriter import LLMPromptRewriter

app = FastAPI(title="ASCII Art Generator API", version="2.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (lazy loaded)
online_gen = None
cnn_mapper = None
rewriter = None

def load_models():
    """Load models on first request."""
    global online_gen, cnn_mapper, rewriter
    
    if rewriter is None:
        try:
            rewriter = LLMPromptRewriter()
        except Exception as e:
            print(f"Warning: Could not load rewriter: {e}")
    
    if cnn_mapper is None:
        cnn_mapper = ProductionCNNMapper()
        try:
            cnn_mapper.load("models/production_cnn.pth")
        except:
            cnn_mapper.train(epochs=50)
    
    api_key = os.getenv("HF_TOKEN")
    if api_key and online_gen is None:
        online_gen = OnlineGenerator(api_key=api_key)


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 125
    quality: str = "standard"
    image_model: str = "flux-hf"
    ascii_model: str = "resnet18"
    seed: Optional[int] = None
    invert: bool = False
    auto_routing: bool = False
    semantic_palette: bool = False
    neural_mapper: bool = True
    custom_token: Optional[str] = None


class GenerateResponse(BaseModel):
    ascii: str
    image_url: str = ""
    rewritten_prompt: str = ""


@app.get("/")
async def root():
    return {"status": "ok", "message": "ASCII Art Generator API v2.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate ASCII art from a prompt."""
    load_models()
    
    prompt = request.prompt
    rewritten = prompt
    
    # Step 1: Rewrite prompt with LLM
    llm_success = False
    if rewriter:
        try:
            result = rewriter.rewrite(prompt, model_type="FLUX")
            rewritten = result.rewritten
            llm_success = True
        except Exception as e:
            print(f"Rewrite failed: {e}")
            
    # Fallback to rule-based enhancement if LLM failed or isn't available
    if not llm_success:
        try:
            from ascii_gen.prompt_engineering import enhance_prompt
            rewritten = enhance_prompt(prompt)
        except Exception as e:
            print(f"Rule-based enhancement failed: {e}")
            rewritten = prompt
    
    # Step 2: Generate image based on selected image model
    image = None
    
    # Determine which HF token to use
    hf_token = request.custom_token or os.getenv("HF_TOKEN")
    
    try:
        if request.image_model == "flux-hf":
            # Use HuggingFace FLUX.1-schnell
            if not hf_token:
                raise ValueError("HF_TOKEN required for FLUX.1 HuggingFace model")
            gen = OnlineGenerator(api_key=hf_token)
            image = gen.generate(
                rewritten,
                skip_preprocessing=True,
                seed=request.seed,
            )
        
        elif request.image_model == "pollinations-flux":
            # Use Pollinations with FLUX model (free, no token needed)
            gen = OnlineGenerator(api_key=None)
            image = gen._generate_pollinations(
                rewritten,
                width=512,
                height=512,
                seed=request.seed,
            )
        
        elif request.image_model == "pollinations-turbo":
            # Use Pollinations with turbo model (fastest, free)
            import urllib.parse
            import requests as req
            import io
            from PIL import Image as PILImage
            
            encoded = urllib.parse.quote(rewritten)
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true&model=turbo"
            if request.seed is not None:
                url += f"&seed={request.seed}"
            
            print(f"🌐 Pollinations Turbo: {url[:80]}...")
            resp = req.get(url, timeout=120, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            })
            if resp.status_code == 200 and len(resp.content) > 4000:
                image = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                print("✅ Pollinations Turbo succeeded!")
            else:
                raise ValueError(f"Pollinations Turbo failed: {resp.status_code}")
        
        else:
            # Default fallback to HuggingFace
            if online_gen:
                image = online_gen.generate(rewritten, skip_preprocessing=True, seed=request.seed)
            else:
                raise ValueError(f"Unknown image model: {request.image_model}")
        
        if image is None:
            raise HTTPException(status_code=500, detail="Image generation failed: No image returned from API")
    
    except HTTPException:
        raise
    except Exception as e:
        with open("server_error.log", "a") as f:
            import traceback
            traceback.print_exc(file=f)
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
    
    # Step 3: Convert to ASCII based on selected render mode
    try:
        ascii_art = convert_to_ascii(image, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASCII conversion failed: {str(e)}")
    
    # Step 4: Encode image as base64 for frontend
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    image_url = f"data:image/png;base64,{img_base64}"
    
    return GenerateResponse(
        ascii=ascii_art,
        image_url=image_url,
        rewritten_prompt=rewritten
    )


def convert_to_ascii(image, request: GenerateRequest) -> str:
    """Convert image to ASCII using the selected render mode."""
    mode = request.quality.lower()
    width = request.width
    invert = request.invert
    
    # SSIM mode — uses perceptual structural similarity
    if mode == "ssim":
        mapper = SSIMMapper(width=width)
        return mapper.convert_image(image)
    
    # CNN mode — uses trained neural network for character selection
    if mode == "cnn":
        if cnn_mapper:
            return cnn_mapper.convert_image(image)
        else:
            print("⚠️ CNN mapper not available, falling back to gradient")
            mode = "standard"  # fallback
    
    # Auto mode — try CNN first (best quality), fall back to gradient
    if mode == "auto":
        if cnn_mapper:
            return cnn_mapper.convert_image(image)
        else:
            mode = "standard"  # fallback
    
    # Gradient-based modes — defaults tuned for BOTH boundary clarity AND tonal shading.
    # Richer ramps (DETAILED/ULTRA) give the mapper more characters to express grey gradients.
    ramp_map = {
        "ultra": RAMP_ULTRA,
        "high": RAMP_DETAILED,
        "standard": RAMP_DETAILED,   # was RAMP_STANDARD — DETAILED gives ~40 chars for shading
        "neat": RAMP_NEAT,
        "portrait": RAMP_ULTRA,
    }

    ramp = ramp_map.get(mode, RAMP_DETAILED)

    # Mode-specific config — lower contrast preserves midtones so shading survives
    # to the ASCII output. Edge-blended conversion (convert_with_edges) keeps boundaries sharp.
    contrast = 1.3
    dither = True
    sharpness = 1.4
    edge_weight = 0.35   # blend Canny edges into the brightness map for crisp outlines
    use_edges = True     # default: boundary-clear + shaded output

    if mode == "neat":
        # Pure line-art look: heavy edges, hard contrast, no dithering
        contrast = 2.5
        dither = False
        sharpness = 2.0
        edge_weight = 0.7
    elif mode == "portrait":
        # Faces: low contrast preserves skin gradients, light edge weight
        contrast = 1.15
        sharpness = 1.3
        edge_weight = 0.2
    elif mode == "ultra":
        contrast = 1.3
        sharpness = 1.3
        edge_weight = 0.3
    elif mode == "high":
        contrast = 1.35
        sharpness = 1.4
        edge_weight = 0.35
    elif mode == "standard":
        contrast = 1.3
        sharpness = 1.4
        edge_weight = 0.35

    config = GradientConfig(
        width=width,
        ramp=ramp,
        invert=invert,
        contrast=contrast,
        dither=dither,
        sharpness=sharpness,
        edge_enhance=True,
    )
    mapper = GradientMapper(config)

    if use_edges:
        return mapper.convert_with_edges(image, edge_weight=edge_weight)
    return mapper.convert(image)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

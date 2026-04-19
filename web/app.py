"""
ASCII Art Generator - Web Interface

Modern dark mode UI with:
- Gradio for rapid AI demo deployment
- Claude-inspired design (glassmorphism, purple accents)
- HuggingFace Spaces compatible

Features:
- Prompt-to-ASCII generation (via online API)
- Image upload and conversion
- Multiple mapper options (CNN, RF, AISS)
- Adjustable output width
- Copy/download functionality
"""

import gradio as gr
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.production_training import ProductionCNNMapper
from ascii_gen.gradient_mapper import (
    GradientMapper, GradientConfig,
    RAMP_ULTRA, RAMP_STANDARD, RAMP_DETAILED,
    image_to_gradient_ascii, image_to_boundary_ascii,
)
from ascii_gen.multimodal import CLIPSelector
from ascii_gen.perceptual import create_ssim_mapper
from ascii_gen.diff_render import DiffRenderer
from ascii_gen.exporter import render_ascii_to_image
from ascii_gen.advanced_preprocessing import enhance_face_contrast


# Global models (loaded once)
cnn_mapper = None
cnn_mapper = None
online_gen = None
diff_renderer = None
rewriter = None


def load_models():
    """Load models on startup."""
    global cnn_mapper, online_gen, rewriter
    
    if rewriter is None:
        try:
            from ascii_gen.llm_rewriter import LLMPromptRewriter
            rewriter = LLMPromptRewriter()
        except Exception as e:
            print(f"Warning: Could not load rewriter: {e}")
    
    if cnn_mapper is None:
        cnn_mapper = ProductionCNNMapper()
        try:
            cnn_mapper.load("models/production_cnn.pth")
        except:
            cnn_mapper.train(epochs=50)
    
    # Use env var if set, otherwise use hardcoded token for testing
    api_key = os.getenv("HF_TOKEN", "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")
    if api_key and online_gen is None:
        online_gen = OnlineGenerator(api_key=api_key)


# Helper for HTML preview
def create_html_preview(ascii_text):
    return f"""
    <div style="
        font-family: 'Courier New', monospace; 
        line-height: 1.0; 
        font-size: 5px; 
        letter-spacing: 0px;
        white-space: pre; 
        overflow-x: auto; 
        background: #fff; 
        color: #000; 
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #333;
        width: 100%;
        text-align: center;
    ">
    {ascii_text}
    </div>
    """


def generate_from_prompt(
    prompt: str, 
    width: int, 
    seed: int,
    quality_mode: str,
    invert_ramp: bool = False,
    auto_route: bool = True,
    use_semantic_palette: bool = True,
    gen_source: str = "Default (Auto)",
    custom_token: str = "",
    use_enhanced_mapper: bool = True,
    resnet_model: str = "ascii_model.pth (Kaggle 20 epochs)",
    image_gen_model: str = "FLUX.1 Schnell (HuggingFace) - Best Quality",
    progress=gr.Progress()
):
    """
    Full pipeline: Prompt -> Rewrite (LLM) -> Generate (FLUX) -> Convert (ASCII)
    Yields intermediate steps for the "Thinking Process" UI.
    """
    if not prompt: yield None, "", "Enter a prompt first", "", "Waiting for input...", None
    
    # Logic for Custom Generator
    active_generator = online_gen
    if gen_source == "Custom HF Token" and custom_token.strip():
        # Create a temporary generator with the custom key
        try:
            from ascii_gen.online_generator import OnlineGenerator
            active_generator = OnlineGenerator(api_key=custom_token.strip())
        except Exception as e:
             yield None, "", f"❌ Error creating generator: {e}", "", "Failed", None
             return

    # Log accumulator (Use a list for mutability in closure)
    log_state = { "text": "🚀 Starting generation process...\n" }
    
    def log_msg(msg):
        log_state["text"] += f"{msg}\n"
    
    yield None, "", "Starting...", "", log_state["text"], None
    
    progress(0, "Thinking (LLM)...")
    load_models()
    
    
    if not active_generator:
        log_msg("❌ Set HF_TOKEN environment variable")
        yield None, "", "❌ HF_TOKEN missing", "", log_state["text"], None
        return

    # 1. Thinking / Rewriting Phase
    log_msg("\n🧠 Phase 1: Semantic Understanding & Rewriting")
    yield None, "", "Thinking...", "", log_state["text"], None
    
    rewritten_prompt = prompt
    try:
        if rewriter:
            # Determine prompt strategy based on model
            model_type = "pollinations" if "pollinations" in image_gen_model.lower() else "flux"
            
            result = rewriter.rewrite(prompt, model_type=model_type)
            
            # Append detailed thinking logs
            if result.logs:
                for log_entry in result.logs:
                     log_msg(f"  • {log_entry}")

            rewritten_prompt = result.rewritten
            
            target_model_name = "Pollinations" if model_type == "pollinations" else "FLUX.1"
            log_msg(f"\n✨ Optimized Prompt for {target_model_name}: '{rewritten_prompt}'")
        else:
            log_msg("  ℹ️ Rewriter not initialized, skipping...")
    except Exception as e:
        log_msg(f"  ⚠️ Rewrite Error: {e}")
        print(f"Rewrite error: {e}")

    yield None, "", "Generating Image...", "", log_state["text"], None

    # 2. Image Generation Phase - select model based on user choice
    selected_gen_model = "HuggingFace FLUX"
    if "Pollinations FLUX" in image_gen_model:
        selected_gen_model = "Pollinations FLUX"
    elif "Pollinations Turbo" in image_gen_model:
        selected_gen_model = "Pollinations Turbo"
    
    log_msg(f"\n🎨 Phase 2: Image Synthesis ({selected_gen_model})")
    if gen_source == "Custom HF Token":
        log_msg("  🔑 Using Custom API Key")
    # online_gen logs itself now via callback
    
    yield None, "", "Generating Image...", "", log_state["text"], None

    progress(0.4, "Generating image...")
    # Threaded generation to allow live log streaming
    import threading
    import time
    
    gen_result = {"image": None, "done": False}
    
    def run_generation():
        try:
            if selected_gen_model == "Pollinations FLUX":
                # Direct Pollinations FLUX call
                log_msg("  🌐 Using Pollinations FLUX directly...")
                gen_result["image"] = active_generator._generate_pollinations(
                    rewritten_prompt, width=512, height=384, seed=seed
                )
            elif selected_gen_model == "Pollinations Turbo":
                # Direct Pollinations Turbo call
                log_msg("  🌐 Using Pollinations Turbo directly...")
                import urllib.parse
                import requests
                import io
                from PIL import Image
                encoded = urllib.parse.quote(rewritten_prompt)
                url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=384&nologo=true&model=turbo"
                if seed:
                    url += f"&seed={seed}"
                resp = requests.get(url, timeout=90)
                if resp.status_code == 200:
                    gen_result["image"] = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    log_msg("  ✅ Pollinations Turbo generated!")
                else:
                    log_msg(f"  ❌ Pollinations Turbo Error: {resp.status_code}")
            else:
                # Default: HuggingFace FLUX with fallback chain
                gen_result["image"] = active_generator.generate(
                    rewritten_prompt, 
                    width=512, 
                    height=384, 
                    seed=seed, 
                    skip_preprocessing=True,
                    log_callback=log_msg
                )
        except Exception as e:
            log_msg(f"❌ Thread Error: {e}")
        finally:
            gen_result["done"] = True

    # Start generation thread
    t = threading.Thread(target=run_generation)
    t.start()
    
    # Stream logs while waiting
    while not gen_result["done"]:
        yield None, "", "Generating...", "", log_state["text"], None
        time.sleep(0.1)
        
    image = gen_result["image"]

    if image is None:
        log_msg("❌ Image generation failed.")
        yield None, "", "Failed", "", log_state["text"], None
        return

    log_msg("  ✅ Image generated successfully (512x384)")
    yield image, "", "Converting...", "", log_state["text"], None

    # 3. ASCII Conversion Phase
    progress(0.7, "Converting to ASCII...")
    log_msg(f"\n⚙️  Phase 3: Structural Mapping & ASCII Conversion")

    # Auto-Routing Logic
    if auto_route and rewriter and 'result' in locals() and result.classification:
        cls = result.classification.lower()
        old_mode = quality_mode
        
        if cls == "structure":
            quality_mode = "Deep Structure (SSIM)"
            reason = "Optimized for Grids/Geometry"
        elif cls == "text":
            quality_mode = "Standard (Gradient)"
            reason = "Optimized for Sharp Edges"
            
        # Special Face Handling
        if cls == "face":
            quality_mode = "Portrait (Gradient)"
            reason = "Optimized for Facial Features"
            log_msg(f"\n👤 Face Detected! Applying Adaptive Contrast Enhancement (CLAHE)...")
            # Apply enhancement immediately
            image = enhance_face_contrast(image)
            
        if old_mode != quality_mode:
            log_msg(f"  🔀 Smart Router: Detected {cls.upper()} -> Switching to '{quality_mode}'")
            log_msg(f"     ({reason})")
        else:
            log_msg(f"  ✅ Smart Router: Kept '{quality_mode}' (Matches {cls.upper()})")

    # SEMANTIC PALETTE LOGIC
    custom_charset = None
    if use_semantic_palette:
        # Check for bold keywords first
        bold_keywords = ["bold", "thick", "heavy", "strong", "dark", "outline", "solid"]
        if any(k in prompt.lower() for k in bold_keywords):
            from ascii_gen.charsets import ASCII_HEAVY
            custom_charset = ASCII_HEAVY
            log_msg(f"  🎨 Palette Switch: Using BOLD/HEAVY characters for clarity")
        elif rewriter and 'result' in locals() and result.semantic_palette:
            palette_str = "".join(result.semantic_palette)
            if len(palette_str) > 5:
                custom_charset = palette_str
                log_msg(f"  🎨 Semantic Palette: Custom charset generated ({len(custom_charset)} chars)")
                log_msg(f"  • Subject Texture: {result.classification.upper()}")
                log_msg(f"  • Generated Palette: {palette_str}")
                log_msg(f"  • Forcing 'Deep Structure (SSIM)' to apply palette...")
                quality_mode = "Deep Structure (SSIM)"
            
    # ENHANCED MAPPER LOGIC
    enhanced_mapper_active = False
    if use_enhanced_mapper:
        try:
            from ascii_gen.enhanced_mapper import get_enhanced_mapper
            # Extract model filename from dropdown selection
            model_filename = resnet_model.split(" ")[0]  # e.g. "ascii_vit_final.pth"
            model_path = Path(__file__).parent.parent / "models" / model_filename
            em = get_enhanced_mapper(model_path=str(model_path))
            if em.is_available():
                enhanced_mapper_active = True
                model_type = "ViT" if "vit" in model_filename.lower() else "ResNet"
                log_msg(f"\n🧠 Neural Mapper: {model_type} ({model_filename})")
            else:
                log_msg(f"\n🧠 Neural Mapper: Model not found ({model_filename})")
        except Exception as e:
            log_msg(f"\n🧠 Neural Mapper: Failed to load ({e})")

    status_msg = ""
    ascii_art = ""
    
    # Choose conversion method based on quality mode
    if quality_mode == "AI Auto-Select (Best Quality)":
        progress(0.8, "🤖 AI evaluating variants...")
        log_msg("  • AI Auto-Select: Generating variants to score with CLIP...")
        yield image, "", "AI Scoring...", "", log_state["text"], None
        
        selector = CLIPSelector()
        
        # Define strategies
        mappers = {
            "Boundary (Edge-Focused)": lambda img, w: image_to_boundary_ascii(img, width=w),
            "Neat (Gradient)": lambda img, w: image_to_gradient_ascii(img, width=w, ramp="neat", with_edges=True, edge_weight=0.6),
            "Standard (CNN)": lambda img, w: cnn_mapper.convert_image(img.resize((w*8, int(w*8*img.height/img.width*0.55)))),
            "Ultra (Gradient)": lambda img, w: image_to_gradient_ascii(img, width=w, ramp="ultra", with_edges=True, edge_weight=0.3),
            "Portrait (Gradient)": lambda img, w: image_to_gradient_ascii(img, width=int(w*1.5), ramp="portrait", with_edges=True, edge_weight=0.2),
            "Deep Structure (SSIM)": lambda img, w: create_ssim_mapper(width=w, charset=custom_charset).convert_image(img),
        }
        
        # We need to capture logs from CLIPSelector if we want them, but let's just log result
        ascii_art, strategy_name, score = selector.select_best_ascii(image, prompt, width, mappers)
        
        log_msg(f"  ✅ AI Selected: {strategy_name} (Score: {score:.3f})")
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | 🤖 AI Selected: {strategy_name}"
        
    elif quality_mode == "Deep Structure (SSIM)":
        progress(0.8, "Running Perceptual Optimization...")
        log_msg("  • Optimizing SSIM (Structural Similarity)...")
        yield image, "", "Optimizing...", "", log_state["text"], None
        # Pass custom_charset if available (from Semantic Palette)
        if custom_charset:
            mapper = create_ssim_mapper(width=width, charset=custom_charset)
        else:
            mapper = create_ssim_mapper(width=width)
        ascii_art = mapper.convert_image(image)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "Ultra (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="ultra", with_edges=True, edge_weight=0.3, invert_ramp=invert_ramp)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "High (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="detailed", with_edges=True, edge_weight=0.4, invert_ramp=invert_ramp)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "Standard (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="standard", with_edges=True, edge_weight=0.3, invert_ramp=invert_ramp)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "Neat (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="neat", with_edges=True, edge_weight=0.6, invert_ramp=invert_ramp)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "Boundary (Edge-Focused)":
        ascii_art = image_to_boundary_ascii(image, width=width)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    else:  # CNN (default)
        aspect = image.height / image.width
        new_width = width * 8
        new_height = int(new_width * aspect * 0.55)
        image_resized = image.resize((new_width, new_height))
        ascii_art = cnn_mapper.convert_image(image_resized)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    
    # 4. Final Constraints
    from ascii_gen.grammar_validator import enforce_grammar
    log_msg("  • Enforcing visual grammar constraints (rectilinearity)...")
    ascii_art = enforce_grammar(ascii_art)
    
    
    log_msg("✅ Process Complete!")
    progress(1.0, "Done!")
    
    # 5. Render Output Image
    rendered_img = render_ascii_to_image(ascii_art)
    
    yield image, ascii_art, status_msg, create_html_preview(ascii_art), log_state["text"], rendered_img


def convert_image(image: Image.Image, width: int, quality_mode: str):
    """Convert uploaded image to ASCII art."""
    if image is None: return "Upload an image first", ""
    load_models()
    
    if image is None:
        return "Upload an image first"
    
    # Choose conversion method based on quality mode
    if quality_mode == "Deep Structure (SSIM)":
        ssim_mapper = create_ssim_mapper(width=width)
        ascii_art = ssim_mapper.convert_image(image)
    elif quality_mode == "Ultra (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="ultra", with_edges=True, edge_weight=0.3)
    elif quality_mode == "High (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="detailed", with_edges=True, edge_weight=0.4)
    elif quality_mode == "Standard (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="standard", with_edges=True, edge_weight=0.3)
    elif quality_mode == "Neat (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="neat", with_edges=True, edge_weight=0.6)
    elif quality_mode == "Boundary (Edge-Focused)":
        ascii_art = image_to_boundary_ascii(image, width=width)
    else:  # CNN
        aspect = image.height / image.width
        new_width = width * 8
        new_height = int(new_width * aspect * 0.55)
        image_resized = image.resize((new_width, new_height))
        ascii_art = cnn_mapper.convert_image(image_resized, apply_edge_detection=True)
    
    
    return ascii_art, create_html_preview(ascii_art)


def run_direct_optimization(prompt, width, steps, progress=gr.Progress()):
    """Run differentiable rendering optimization."""
    global diff_renderer, rewriter
    if not prompt: return "Enter prompt first", ""
    
    # 1. Enhance Prompt with LLM (Crucial for multi-subject/action)
    load_models()
    enhanced_prompt = prompt
    if rewriter:
        progress(0.1, "Refining prompt concept...")
        try:
            res = rewriter.rewrite(prompt)
            enhanced_prompt = res.rewritten
            status_msg = f"✨ Optimization Concept: {enhanced_prompt}"
        except:
            pass # Fallback to raw prompt
    
    if diff_renderer is None:
        progress(0.2, "Loading CLIP model (~600MB)...")
        try:
            diff_renderer = DiffRenderer()
        except Exception as e:
            return f"Error loading model: {str(e)}", ""
            
    progress(0.3, f"Optimizing...")
    ascii_art = diff_renderer.optimize(enhanced_prompt, width=width, steps=steps)
    
    return ascii_art, create_html_preview(ascii_art)


# Custom CSS for Claude-inspired dark theme
CUSTOM_CSS = """
/* =============================================
   PAYROT FINTECH GLASSMORPHISM THEME
   Deep Navy + Glass Cards + Electric Blue Accents
   ============================================= */

/* --- Global Body Styles --- */
.gradio-container {
    background: radial-gradient(circle at 50% 0%, #1e3a5f 0%, #0a192f 80%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    min-height: 100vh;
}

/* --- Header with Gradient Text --- */
.main-header {
    text-align: center;
    padding: 2.5rem 2rem;
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    margin-bottom: 2rem;
}

.main-header h1 {
    background: linear-gradient(135deg, #2b7af1, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.main-header p {
    color: rgba(255, 255, 255, 0.6);
    font-size: 1.15rem;
    font-weight: 400;
}

/* --- Transforming Containers into "Glass" Cards --- */
.gr-box, .gr-panel, .gr-form, .gr-group {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

/* --- Input Fields --- */
.gr-input, .gr-textbox textarea, .gr-dropdown {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    font-size: 1rem !important;
}

.gr-input::placeholder, .gr-textbox textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

.gr-input:focus, .gr-textbox textarea:focus {
    border-color: #4facfe !important;
    box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.2) !important;
    outline: none !important;
}

/* --- Labels --- */
.gr-form label, .gr-block label, label {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* --- The Payrot Pill-Button (Primary) --- */
button.primary, .gr-button-primary, .gr-button-lg {
    background: linear-gradient(135deg, #2b7af1 0%, #4facfe 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 16px 40px !important;
    font-size: 1rem !important;
    box-shadow: 0 8px 20px rgba(79, 172, 254, 0.4) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover, .gr-button-primary:hover, .gr-button-lg:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 25px rgba(79, 172, 254, 0.6) !important;
}

/* --- Secondary Buttons (Glass Pills) --- */
button.secondary, .gr-button-secondary, button:not(.primary) {
    background: rgba(255, 255, 255, 0.08) !important;
    color: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover, .gr-button-secondary:hover, button:not(.primary):hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-1px) !important;
}

/* --- ASCII Output: Terminal Style --- */
.ascii-output, .gr-textbox.ascii-output textarea {
    background: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    color: #38bdf8 !important;
    font-family: 'Fira Code', 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 7px !important;
    line-height: 1.0 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    white-space: pre !important;
    overflow-x: auto !important;
    box-shadow: inset 0 4px 20px rgba(0, 0, 0, 0.5) !important;
}

/* --- Thinking Process Log: Semi-transparent Terminal Overlay --- */
.process-log textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    color: rgba(255, 255, 255, 0.8) !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 16px !important;
    backdrop-filter: blur(5px);
}

/* --- Tabs --- */
.gr-tab-nav {
    background: transparent !important;
    border: none !important;
}

.gr-tab-nav button {
    background: transparent !important;
    color: rgba(255, 255, 255, 0.5) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

.gr-tab-nav button.selected, .gr-tab-nav button:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.1) !important;
}

/* --- Accordion --- */
.gr-accordion {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
}

.gr-accordion-title {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
}

/* --- Sliders --- */
.gr-slider input[type="range"] {
    accent-color: #4facfe !important;
}

/* --- Checkboxes --- */
.gr-checkbox input[type="checkbox"]:checked {
    background: #4facfe !important;
    border-color: #4facfe !important;
}

/* --- Dropdown --- */
.gr-dropdown select {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
}

/* --- Image Preview --- */
.gr-image {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

/* --- Markdown Text --- */
.gr-markdown {
    color: rgba(255, 255, 255, 0.85) !important;
}

.gr-markdown h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* --- Example Buttons Row --- */
.example-buttons button {
    background: rgba(255, 255, 255, 0.1) !important;
    color: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 50px !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.2s ease !important;
}

.example-buttons button:hover {
    background: rgba(79, 172, 254, 0.3) !important;
    border-color: rgba(79, 172, 254, 0.5) !important;
    color: white !important;
    transform: translateY(-1px) !important;
}

/* --- Scrollbar Styling --- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #2b7af1, #4facfe);
    border-radius: 4px;
}

/* --- Hide Footer --- */
footer {
    display: none !important;
}

/* --- Status Text --- */
.status-text {
    color: rgba(255, 255, 255, 0.7) !important;
}

/* --- Info Text --- */
.gr-info {
    color: rgba(255, 255, 255, 0.5) !important;
}
"""


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="ASCII Art Generator") as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 ASCII Art Generator</h1>
            <p>Transform text prompts and images into stunning ASCII art using AI</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Prompt to ASCII
            with gr.TabItem("✨ Prompt to ASCII"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Prompt input with examples in same group
                        with gr.Group():
                            prompt_input = gr.Textbox(
                                label="Your Prompt",
                                placeholder="a cute cat sitting on a chair...",
                                lines=3,
                                max_lines=5,
                            )
                            gr.Markdown("**💡 Try these examples:**", elem_classes=["example-label"])
                            with gr.Row():
                                ex1 = gr.Button("🏠 House", size="sm", variant="secondary")
                                ex2 = gr.Button("🐱 Cat on chair", size="sm", variant="secondary")
                                ex3 = gr.Button("⭐ Stars & moon", size="sm", variant="secondary")
                            with gr.Row():
                                ex4 = gr.Button("🏔️ Mountain", size="sm", variant="secondary")
                                ex5 = gr.Button("🌳 Tree", size="sm", variant="secondary")
                                ex6 = gr.Button("❤️ Heart", size="sm", variant="secondary")
                        
                        # Unified Settings Wizard
                        with gr.Group():
                            # Step indicator
                            step_indicator = gr.Markdown("**Step 1 of 3:** Image Source")
                            
                            # === Step 1: Image Generator Source ===
                            with gr.Column(visible=True) as step1_container:
                                gr.Markdown("### 🖼️ Image Generator Source")
                                with gr.Row():
                                    gen_source = gr.Radio(
                                        choices=["Default (Auto)", "Custom HF Token"],
                                        value="Default (Auto)",
                                        label="Choose how to generate images",
                                        interactive=True
                                    )
                                custom_token_input = gr.Textbox(
                                    label="HuggingFace Token (Write Access)",
                                    placeholder="hf_...",
                                    type="password",
                                    visible=False
                                )
                                step1_next = gr.Button("Configure AI Model (Expand) ▼", variant="secondary")
                            
                            # === Step 2: AI Configuration ===
                            with gr.Column(visible=False) as step2_container:
                                gr.Markdown("### 🤖 AI Configuration")
                                with gr.Row():
                                    image_gen_model = gr.Dropdown(
                                        choices=[
                                            "FLUX.1 Schnell (HuggingFace) - Best Quality",
                                            "Pollinations FLUX - Free Fallback",
                                            "Pollinations Turbo - Fast",
                                        ],
                                        value="FLUX.1 Schnell (HuggingFace) - Best Quality",
                                        label="🎨 Image Generation Model",
                                        info="Model used to generate the image from prompt",
                                        interactive=True
                                    )
                                with gr.Row():
                                    resnet_model_selector = gr.Dropdown(
                                        choices=[
                                            "ascii_resnet18_final.pth (ResNet18 - BEST)",
                                            "ascii_vit_final.pth (ViT - Experimental)",
                                            "ascii_model.pth (ResNet18 - Legacy)",
                                        ],
                                        value="ascii_resnet18_final.pth (ResNet18 - BEST)",
                                        label="🧠 ASCII Mapping Model",
                                        info="Neural network for character selection",
                                        interactive=True,
                                        scale=2
                                    )
                                    use_enhanced_mapper = gr.Checkbox(
                                        label="Enable Neural Mapper",
                                        value=True,
                                        info="Use AI for character selection",
                                        scale=1
                                    )
                                with gr.Row():
                                    step2_back = gr.Button("▲ Collapse", variant="secondary")
                                    step2_next = gr.Button("Customize ASCII Output (Expand) ▼", variant="secondary")
                            
                            # === Step 3: ASCII Settings ===
                            with gr.Column(visible=False) as step3_container:
                                gr.Markdown("### ⚙️ ASCII Settings")
                                with gr.Row():
                                    width_slider = gr.Slider(
                                        minimum=30, maximum=150, value=125, step=5,
                                        label="Output Width",
                                        info="Character count per line"
                                    )
                                    quality_selector = gr.Dropdown(
                                        choices=["AI Auto-Select (Best Quality)", "Boundary (Edge-Focused)", "Deep Structure (SSIM)", "Portrait (Gradient)", "Standard (CNN)", "Neat (Gradient)", "Standard (Gradient)", "High (Gradient)", "Ultra (Gradient)"],
                                        value="AI Auto-Select (Best Quality)",
                                        label="Render Mode",
                                        info="Algorithm for converting image to text"
                                    )
                                step3_back = gr.Button("▲ Collapse", variant="secondary")
                            
                            # === Advanced Options (Always visible) ===
                            with gr.Accordion("🛠️ Advanced Options", open=False):
                                with gr.Row():
                                    seed_input = gr.Number(value=42, label="Seed", precision=0)
                                    invert_ramp_checkbox = gr.Checkbox(label="🌙 Dark Mode Invert", value=False)
                                
                                with gr.Row():
                                    auto_route_checkbox = gr.Checkbox(label="🧭 Auto-Routing", value=True, info="Smart algorithm switching")
                                    use_semantic_palette = gr.Checkbox(label="🎨 Semantic Palette", value=True, info="Use bold/themed characters")
                            
                            # Step navigation logic
                            def toggle_token_input(choice):
                                return gr.update(visible=(choice == "Custom HF Token"))
                            
                            def go_to_step2():
                                return (
                                    gr.update(visible=False),  # hide step1
                                    gr.update(visible=True),   # show step2
                                    gr.update(visible=False),  # hide step3
                                    "**Step 2 of 3:** AI Configuration"
                                )
                            
                            def go_to_step1():
                                return (
                                    gr.update(visible=True),   # show step1
                                    gr.update(visible=False),  # hide step2
                                    gr.update(visible=False),  # hide step3
                                    "**Step 1 of 3:** Image Source"
                                )
                            
                            def go_to_step3():
                                return (
                                    gr.update(visible=False),  # hide step1
                                    gr.update(visible=False),  # hide step2
                                    gr.update(visible=True),   # show step3
                                    "**Step 3 of 3:** ASCII Settings ✓"
                                )
                            
                            def go_back_to_step2():
                                return (
                                    gr.update(visible=False),  # hide step1
                                    gr.update(visible=True),   # show step2
                                    gr.update(visible=False),  # hide step3
                                    "**Step 2 of 3:** AI Configuration"
                                )
                            
                            gen_source.change(fn=toggle_token_input, inputs=[gen_source], outputs=[custom_token_input])
                            step1_next.click(fn=go_to_step2, outputs=[step1_container, step2_container, step3_container, step_indicator])
                            step2_back.click(fn=go_to_step1, outputs=[step1_container, step2_container, step3_container, step_indicator])
                            step2_next.click(fn=go_to_step3, outputs=[step1_container, step2_container, step3_container, step_indicator])
                            step3_back.click(fn=go_back_to_step2, outputs=[step1_container, step2_container, step3_container, step_indicator])
                        
                        generate_btn = gr.Button("🚀 Generate ASCII Art", variant="primary", size="lg")
                        
                        # Generated Image moved here (Left Column)
                        preview_image = gr.Image(label="Generated Image", type="pil")
                    
                    with gr.Column(scale=1):
                         # Process Log moved here (Right Column)
                        with gr.Accordion("🧠 Thinking Process (Live Log)", open=True):
                            process_log = gr.Textbox(
                                label="Process Log", 
                                lines=15, 
                                interactive=False,
                                elem_id="process-log"
                            )
                        status_text = gr.Textbox(label="Status", interactive=False)
                
                
                with gr.Accordion("Micro Preview (Zoomed Out)", open=True):
                    preview_html = gr.HTML(label="Micro Preview")

                ascii_output = gr.Textbox(
                    label="ASCII Art Output (Copy here)",
                    lines=25,
                    max_lines=50,
                    elem_classes=["ascii-output"],
                )
                
                with gr.Accordion("📷 Rendered Output (Preview & Download)", open=False):
                     output_render = gr.Image(label="Rendered PNG", type="pil")
                
                with gr.Row():
                    export_btn = gr.Button("💾 Download as PNG (Better Quality)", size="sm")
                    download_file = gr.File(label="Download Image", interactive=False, visible=True)
                
                export_btn.click(
                    fn=lambda x: render_ascii_to_image(x),
                    inputs=[ascii_output],
                    outputs=[download_file]
                )
                
                # Sample prompt click handlers
                ex1.click(lambda: "a simple house with roof, door, and windows", outputs=prompt_input)
                ex2.click(lambda: "a cute cat sitting on a chair, cat has ears head body tail, chair has seat back legs", outputs=prompt_input)
                ex3.click(lambda: "three stars and a crescent moon", outputs=prompt_input)
                ex4.click(lambda: "a mountain with snow peak and pine trees", outputs=prompt_input)
                ex5.click(lambda: "a simple tree with trunk and leafy branches", outputs=prompt_input)
                ex6.click(lambda: "a simple heart shape", outputs=prompt_input)
                
                # Event Handlers
                generate_btn.click(
                    fn=generate_from_prompt,
                    inputs=[prompt_input, width_slider, seed_input, quality_selector, invert_ramp_checkbox, auto_route_checkbox, use_semantic_palette, gen_source, custom_token_input, use_enhanced_mapper, resnet_model_selector, image_gen_model],
                    outputs=[preview_image, ascii_output, status_text, preview_html, process_log, output_render],
                )
            
            # Tab 2: Image to ASCII
            with gr.TabItem("🖼️ Image to ASCII"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image", type="pil")
                        
                        with gr.Row():
                            img_width = gr.Slider(30, 120, 80, step=5, label="Width")
                            img_quality = gr.Dropdown(
                                choices=["Boundary (Edge-Focused)", "Portrait (Gradient)", "Deep Structure (SSIM)", "Standard (CNN)", "Neat (Gradient)", "Standard (Gradient)", "High (Gradient)", "Ultra (Gradient)"],
                                value="Standard (CNN)",
                                label="Quality Mode"
                            )
                        
                        convert_btn = gr.Button("Convert to ASCII", variant="primary")
                    
                    with gr.Column():
                        img_ascii_output = gr.Textbox(
                            label="ASCII Output",
                            lines=30,
                            elem_classes=["ascii-output"],
                        )
                        
                        with gr.Accordion("Micro Preview", open=True):
                            img_preview_html = gr.HTML(label="Micro Preview")
                
                convert_btn.click(
                    fn=convert_image,
                    inputs=[image_input, img_width, img_quality],
                    outputs=[img_ascii_output, img_preview_html],
                )
                
            # Tab 3: Experimental Direct Gen
            with gr.TabItem("🧪 Direct Generation"):
                gr.Markdown("GENERATE ASCII FROM SCRATCH using Differentiable Rendering (CLIPDraw logic). No image generation involved.")
                
                with gr.Row():
                    direct_prompt = gr.Textbox(label="Concept Prompt", placeholder="a mushroom")
                    direct_width = gr.Slider(20, 80, 40, step=5, label="Width (Keep small for speed)")
                    direct_steps = gr.Slider(50, 300, 150, step=50, label="Optimization Steps")
                
                direct_btn = gr.Button("✨ Optimize ASCII (Slow)", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        direct_output = gr.Textbox(label="Optimized ASCII", lines=20, elem_classes=["ascii-output"])
                        with gr.Row():
                            direct_export_btn = gr.Button("💾 Download PNG", size="sm")
                            direct_download = gr.File(label="Download", interactive=False)
                            
                        direct_export_btn.click(
                            fn=lambda x: render_ascii_to_image(x),
                            inputs=[direct_output],
                            outputs=[direct_download]
                        )
                        
                    with gr.Column():
                        direct_html = gr.HTML(label="Preview")
                        
                direct_btn.click(
                    fn=run_direct_optimization,
                    inputs=[direct_prompt, direct_width, direct_steps],
                    outputs=[direct_output, direct_html]
                )
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## ASCII Art Generator
                
                This tool converts text prompts and images into ASCII art using:
                
                - **FLUX.1 Schnell** - Fast, high-quality image generation
                - **Production CNN** - 243K parameter neural network for character mapping
                - **Edge Detection** - Canny algorithm for structure preservation
                - **LLM Prompt Rewriting** - Gemini/Groq for intelligent prompt enhancement
                
                ### How It Works
                
                1. **Text → Image**: Your prompt is enhanced by LLM and sent to FLUX.1 Schnell
                2. **Image → Tiles**: The image is split into small tiles (8x14 pixels)
                3. **Tiles → Characters**: Each tile is classified to the best-matching ASCII character
                4. **Assembly**: Characters are assembled into the final ASCII art
                
                ### Tips for Best Results
                
                - Use descriptive prompts with "line art", "simple", "high contrast"
                - Black and white / silhouette images work best
                - Adjust width based on your display size
                
                ---
                
                Built with 💜 using Gradio + PyTorch + HuggingFace + Gemini
                """)
        
        # Load models on startup
        app.load(load_models)
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        css=CUSTOM_CSS,
    )

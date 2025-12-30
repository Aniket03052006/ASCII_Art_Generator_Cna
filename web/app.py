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

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.production_training import ProductionCNNMapper
from ascii_gen.gradient_mapper import (
    GradientMapper, GradientConfig, 
    RAMP_ULTRA, RAMP_STANDARD, RAMP_DETAILED,
    image_to_gradient_ascii
)
from ascii_gen.multimodal import CLIPSelector
from ascii_gen.perceptual import create_ssim_mapper
from ascii_gen.diff_render import DiffRenderer
from ascii_gen.exporter import render_ascii_to_image


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
        font-size: 4px; 
        letter-spacing: 0px;
        white-space: pre; 
        overflow-x: auto; 
        background: #111; 
        color: #eee; 
        padding: 20px;
        border-radius: 8px;
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
    progress=gr.Progress()
):
    """Generate ASCII art from text prompt."""
    if not prompt: yield None, "", "Enter a prompt first", "", "Waiting for input..."
    
    # Log accumulator
    log_text = "🚀 Starting generation process...\n"
    yield None, "", "Starting...", "", log_text
    
    progress(0, "Thinking (LLM)...")
    load_models()
    
    if not online_gen:
        yield None, "", "❌ Set HF_TOKEN environment variable", "", log_text + "\n❌ HF_TOKEN missing"
        return
    
    # 1. Thinking / Rewriting Phase
    log_text += "\n🧠 Phase 1: Semantic Understanding & Rewriting\n"
    yield None, "", "Thinking...", "", log_text
    
    rewritten_prompt = prompt
    try:
        if rewriter:
            result = rewriter.rewrite(prompt)
            
            # Append detailed thinking logs
            if result.logs:
                for log_entry in result.logs:
                    log_text += f"  • {log_entry}\n"
            
            rewritten_prompt = result.rewritten
            log_text += f"\n✨ Optimized Prompt for FLUX.1: '{rewritten_prompt}'\n"
        else:
            log_text += "  ℹ️ Rewriter not initialized, skipping...\n"
    except Exception as e:
        log_text += f"  ⚠️ Rewrite Error: {e}\n"
        print(f"Rewrite error: {e}")
            
    yield None, "", "Generating Image...", "", log_text

    # 2. Image Generation Phase
    log_text += "\n🎨 Phase 2: Image Synthesis (FLUX.1 Schnell)\n"
    log_text += f"  • Sending request to HuggingFace Inference API...\n"
    log_text += f"  • Model: black-forest-labs/FLUX.1-schnell\n"
    yield None, "", "Generating Image...", "", log_text
    
    progress(0.4, "Generating image...")
    # Skip internal preprocessing since we did it explicitly
    image = online_gen.generate(rewritten_prompt, width=512, height=384, seed=seed, skip_preprocessing=True)
    
    if image is None:
        log_text += "❌ Image generation failed. Check API key or quota.\n"
        yield None, "", "Failed", "", log_text
        return
    
    log_text += "  ✅ Image generated successfully (512x384)\n"
    yield image, "", "Converting...", "", log_text
    
    # 3. ASCII Conversion Phase
    progress(0.7, "Converting to ASCII...")
    log_text += f"\n⚙️  Phase 3: Structural Mapping & ASCII Conversion\n"
    log_text += f"  • Mode: {quality_mode}\n"
    log_text += f"  • Target Width: {width} chars\n"
    yield image, "", "Converting...", "", log_text
    
    # Initialize SSIM mapper lazily if needed
    ssim_mapper = create_ssim_mapper(width=width)
    
    status_msg = ""
    ascii_art = ""
    
    # Choose conversion method based on quality mode
    if quality_mode == "AI Auto-Select (Best Quality)":
        progress(0.8, "🤖 AI evaluating variants...")
        log_text += "  • AI Auto-Select: Generating variants to score with CLIP...\n"
        yield image, "", "AI Scoring...", "", log_text
        
        selector = CLIPSelector()
        
        # Define strategies
        mappers = {
            "Neat (Gradient)": lambda img, w: image_to_gradient_ascii(img, width=w, ramp="neat", with_edges=True, edge_weight=0.6),
            "Standard (CNN)": lambda img, w: cnn_mapper.convert_image(img.resize((w*8, int(w*8*img.height/img.width*0.55)))),
            "Ultra (Gradient)": lambda img, w: image_to_gradient_ascii(img, width=w, ramp="ultra", with_edges=True, edge_weight=0.3),
            "Portrait (Gradient)": lambda img, w: image_to_gradient_ascii(img, width=int(w*1.5), ramp="portrait", with_edges=True, edge_weight=0.2),
            "Deep Structure (SSIM)": lambda img, w: create_ssim_mapper(width=w).convert_image(img),
        }
        
        # We need to capture logs from CLIPSelector if we want them, but let's just log result
        ascii_art, strategy_name, score = selector.select_best_ascii(image, prompt, width, mappers)
        
        log_text += f"  ✅ AI Selected: {strategy_name} (Score: {score:.3f})\n"
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | 🤖 AI Selected: {strategy_name}"
        
    elif quality_mode == "Deep Structure (SSIM)":
        progress(0.8, "Running Perceptual Optimization...")
        log_text += "  • Optimizing SSIM (Structural Similarity)...\n"
        yield image, "", "Optimizing...", "", log_text
        ascii_art = ssim_mapper.convert_image(image)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "Ultra (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="ultra", with_edges=True, edge_weight=0.3, invert_ramp=invert_ramp)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    elif quality_mode == "High (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="detailed", with_edges=True, edge_weight=0.4, invert_ramp=invert_ramp)
    elif quality_mode == "Standard (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="standard", with_edges=True, edge_weight=0.3, invert_ramp=invert_ramp)
    elif quality_mode == "Neat (Gradient)":
        ascii_art = image_to_gradient_ascii(image, width=width, ramp="neat", with_edges=True, edge_weight=0.6, invert_ramp=invert_ramp)
    else:  # CNN (default)
        aspect = image.height / image.width
        new_width = width * 8
        new_height = int(new_width * aspect * 0.55)
        image_resized = image.resize((new_width, new_height))
        ascii_art = cnn_mapper.convert_image(image_resized)
        status_msg = f"Generated {len(ascii_art.split(chr(10)))} lines | Mode: {quality_mode}"
    
    # 4. Final Constraints
    from ascii_gen.grammar_validator import enforce_grammar
    log_text += "  • Enforcing visual grammar constraints (rectilinearity)...\n"
    ascii_art = enforce_grammar(ascii_art)
    
    log_text += "✅ Process Complete!"
    progress(1.0, "Done!")
    
    yield image, ascii_art, status_msg, create_html_preview(ascii_art), log_text


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
:root {
    --primary: #6c5ce7;
    --primary-hover: #5f4dd0;
    --bg-dark: #0f0f1a;
    --bg-card: #1a1a2e;
    --bg-input: #16213e;
    --text-primary: #e4e4e7;
    --text-secondary: #a1a1aa;
    --border: #2d2d44;
}

.gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Inter', system-ui, sans-serif;
}

.main-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--bg-card), var(--bg-dark));
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}

.main-header h1 {
    background: linear-gradient(135deg, #6c5ce7, #a29bfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

.ascii-output {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 10px !important;
    line-height: 1.1 !important;
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    white-space: pre !important;
    overflow-x: auto !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(108, 92, 231, 0.3) !important;
}

.gr-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.gr-input, .gr-textbox textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

.gr-padded {
    padding: 1.5rem !important;
}

footer {
    display: none !important;
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
                        prompt_input = gr.Textbox(
                            label="Your Prompt",
                            placeholder="Describe what you want to generate...",
                            lines=3,
                            max_lines=5,
                        )
                        
                        # Sample prompts (like ChatGPT)
                        gr.Markdown("**Try these examples:**")
                        with gr.Row():
                            ex1 = gr.Button("🏠 House", size="sm", variant="secondary")
                            ex2 = gr.Button("🐱 Cat on chair", size="sm", variant="secondary")
                            ex3 = gr.Button("⭐ Stars & moon", size="sm", variant="secondary")
                        with gr.Row():
                            ex4 = gr.Button("🏔️ Mountain", size="sm", variant="secondary")
                            ex5 = gr.Button("🌳 Tree", size="sm", variant="secondary")
                            ex6 = gr.Button("❤️ Heart", size="sm", variant="secondary")
                        
                        with gr.Row():
                            width_slider = gr.Slider(
                                minimum=30, maximum=120, value=80, step=5,
                                label="Output Width (characters)"
                            )
                            quality_selector = gr.Dropdown(
                                choices=["AI Auto-Select (Best Quality)", "Portrait (Gradient)", "Deep Structure (SSIM)", "Standard (CNN)", "Neat (Gradient)", "Standard (Gradient)", "High (Gradient)", "Ultra (Gradient)"],
                                value="Standard (CNN)",
                                label="Quality Mode",
                                info="AI Auto-Select picks best result. SSIM uses structural optimization (slower)."
                            )
                            seed_input = gr.Number(
                                value=42, label="Seed", precision=0
                            )
                            invert_ramp_checkbox = gr.Checkbox(
                                label="Invert Ramp (Dark BG style)",
                                value=False,
                                info="Swap light/dark mapping for dark background output"
                            )
                        
                        generate_btn = gr.Button("🚀 Generate ASCII Art", variant="primary", size="lg")
                        
                        with gr.Accordion("🧠 Thinking Process (Live Log)", open=True):
                            process_log = gr.Textbox(
                                label="Process Log", 
                                lines=8, 
                                interactive=False,
                                elem_id="process-log"
                            )
                            
                        status_text = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        preview_image = gr.Image(label="Generated Image", type="pil")
                
                
                with gr.Accordion("Micro Preview (Zoomed Out)", open=True):
                    preview_html = gr.HTML(label="Micro Preview")

                ascii_output = gr.Textbox(
                    label="ASCII Art Output (Copy here)",
                    lines=25,
                    max_lines=50,
                    elem_classes=["ascii-output"],
                )
                
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
                
                generate_btn.click(
                    fn=generate_from_prompt,
                    inputs=[prompt_input, width_slider, seed_input, quality_selector, invert_ramp_checkbox],
                    outputs=[preview_image, ascii_output, status_text, preview_html, process_log],
                )
            
            # Tab 2: Image to ASCII
            with gr.TabItem("🖼️ Image to ASCII"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image", type="pil")
                        
                        with gr.Row():
                            img_width = gr.Slider(30, 120, 80, step=5, label="Width")
                            img_quality = gr.Dropdown(
                                choices=["Portrait (Gradient)", "Deep Structure (SSIM)", "Standard (CNN)", "Neat (Gradient)", "Standard (Gradient)", "High (Gradient)", "Ultra (Gradient)"],
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
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )

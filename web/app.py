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


# Global models (loaded once)
cnn_mapper = None
online_gen = None


def load_models():
    """Load models on startup."""
    global cnn_mapper, online_gen
    
    if cnn_mapper is None:
        cnn_mapper = ProductionCNNMapper()
        try:
            cnn_mapper.load("models/production_cnn.pth")
        except:
            cnn_mapper.train(epochs=50)
    
    api_key = os.getenv("HF_TOKEN", "")
    if api_key and online_gen is None:
        online_gen = OnlineGenerator(api_key=api_key)


def generate_from_prompt(prompt: str, width: int, seed: int, progress=gr.Progress()):
    """Generate ASCII art from text prompt."""
    load_models()
    
    if not online_gen:
        return None, "❌ Set HF_TOKEN environment variable for API access", None
    
    progress(0.2, "Generating image...")
    image = online_gen.generate(prompt, width=512, height=384, seed=seed)
    
    if image is None:
        return None, "❌ Image generation failed. Check your API key.", None
    
    progress(0.7, "Converting to ASCII...")
    
    # Resize image based on width
    aspect = image.height / image.width
    new_width = width * 8  # 8 pixels per character
    new_height = int(new_width * aspect * 0.55)  # Adjust for character aspect
    image_resized = image.resize((new_width, new_height))
    
    ascii_art = cnn_mapper.convert_image(image_resized)
    
    progress(1.0, "Done!")
    
    return image, ascii_art, f"Generated {len(ascii_art.split(chr(10)))} lines, {len(ascii_art)} chars"


def convert_image(image: Image.Image, width: int, edge_detection: bool):
    """Convert uploaded image to ASCII art."""
    load_models()
    
    if image is None:
        return "Upload an image first"
    
    # Resize
    aspect = image.height / image.width
    new_width = width * 8
    new_height = int(new_width * aspect * 0.55)
    image_resized = image.resize((new_width, new_height))
    
    ascii_art = cnn_mapper.convert_image(image_resized, apply_edge_detection=edge_detection)
    
    return ascii_art


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
                            placeholder="A majestic mountain with snow peaks and pine trees...",
                            lines=3,
                            max_lines=5,
                        )
                        
                        with gr.Row():
                            width_slider = gr.Slider(
                                minimum=30, maximum=100, value=60, step=5,
                                label="Output Width (characters)"
                            )
                            seed_input = gr.Number(
                                value=42, label="Seed", precision=0
                            )
                        
                        generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                        
                        status_text = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        preview_image = gr.Image(label="Generated Image", type="pil")
                
                ascii_output = gr.Textbox(
                    label="ASCII Art Output",
                    lines=25,
                    max_lines=50,
                    elem_classes=["ascii-output"],
                )
                
                generate_btn.click(
                    fn=generate_from_prompt,
                    inputs=[prompt_input, width_slider, seed_input],
                    outputs=[preview_image, ascii_output, status_text],
                )
            
            # Tab 2: Image to ASCII
            with gr.TabItem("🖼️ Image to ASCII"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image", type="pil")
                        
                        with gr.Row():
                            img_width = gr.Slider(30, 100, 60, step=5, label="Width")
                            edge_toggle = gr.Checkbox(value=True, label="Edge Detection")
                        
                        convert_btn = gr.Button("Convert to ASCII", variant="primary")
                    
                    with gr.Column():
                        img_ascii_output = gr.Textbox(
                            label="ASCII Output",
                            lines=30,
                            elem_classes=["ascii-output"],
                        )
                
                convert_btn.click(
                    fn=convert_image,
                    inputs=[image_input, img_width, edge_toggle],
                    outputs=img_ascii_output,
                )
            
            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## ASCII Art Generator
                
                This tool converts text prompts and images into ASCII art using:
                
                - **FLUX.1 Schnell** - Fast, high-quality image generation
                - **Production CNN** - 243K parameter neural network for character mapping
                - **Edge Detection** - Canny algorithm for structure preservation
                
                ### How It Works
                
                1. **Text → Image**: Your prompt is sent to FLUX.1 Schnell via HuggingFace API
                2. **Image → Tiles**: The image is split into small tiles (8x14 pixels)
                3. **Tiles → Characters**: Each tile is classified to the best-matching ASCII character
                4. **Assembly**: Characters are assembled into the final ASCII art
                
                ### Tips for Best Results
                
                - Use descriptive prompts with "line art", "simple", "high contrast"
                - Black and white / silhouette images work best
                - Adjust width based on your display size
                
                ---
                
                Built with 💜 using Gradio + PyTorch + HuggingFace
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
    )

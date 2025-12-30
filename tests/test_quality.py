
"""
Test ALL Standard Options - Comparing quality across modes
"""
import os
from PIL import Image
from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.gradient_mapper import image_to_gradient_ascii
from ascii_gen.cnn_mapper import CNNMapper
from ascii_gen.exporter import render_ascii_to_image

OUTPUT_DIR = "outputs/quality_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_TOKEN = "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"

def run_comparison():
    prompt = "a simple standing person silhouette"
    print(f"\n🎯 Quality Comparison Test: '{prompt}'")
    
    # 1. Generate image
    print("📷 Generating source image...")
    generator = OnlineGenerator(api_key=HF_TOKEN)
    image = generator.generate(prompt, width=512, height=512)
    
    if not image:
        print("❌ Image generation failed")
        return
    
    image.save(os.path.join(OUTPUT_DIR, "source.png"))
    print("   ✅ Source image saved")
    
    # 2. Test all modes
    modes = {
        "Standard_CNN": lambda: CNNMapper().convert_image(
            image.resize((80*8, int(80*8*image.height/image.width*0.55)))
        ),
        "Standard_Gradient": lambda: image_to_gradient_ascii(image, width=80, ramp="standard", invert_ramp=False),
        "Ultra_Gradient": lambda: image_to_gradient_ascii(image, width=80, ramp="ultra", invert_ramp=False),
        "Neat_Gradient": lambda: image_to_gradient_ascii(image, width=80, ramp="neat", invert_ramp=False),
        # Inverted versions for dark terminal
        "Ultra_Inverted": lambda: image_to_gradient_ascii(image, width=80, ramp="ultra", invert_ramp=True),
    }
    
    for name, converter in modes.items():
        print(f"\n🔄 Testing: {name}")
        try:
            ascii_art = converter()
            
            # Save text
            txt_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
            with open(txt_path, "w") as f:
                f.write(ascii_art)
            
            # Save PNG
            png_path = render_ascii_to_image(ascii_art)
            if png_path:
                final_png = os.path.join(OUTPUT_DIR, f"{name}.png")
                os.rename(png_path, final_png)
                print(f"   ✅ {name}: {txt_path}")
            
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    print("\n🎉 COMPARISON COMPLETE!")
    print(f"Check {OUTPUT_DIR}/ for results.")

if __name__ == "__main__":
    run_comparison()

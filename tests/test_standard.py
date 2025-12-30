
"""
STANDARD Pipeline Test (PROVEN TO WORK)
LLM Rewrite -> HuggingFace Image -> ASCII Conversion
"""
import os
from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.gradient_mapper import image_to_gradient_ascii, GradientConfig, RAMP_ULTRA
from ascii_gen.exporter import render_ascii_to_image

OUTPUT_DIR = "outputs/standard_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_standard_pipeline():
    prompt = "cat chasing a mouse"
    print(f"\n🎯 Standard Pipeline Test: '{prompt}'")
    
    # 1. LLM Rewrite
    print("1️⃣  LLM Rewriting...")
    try:
        rewriter = LLMPromptRewriter()
        res = rewriter.rewrite(prompt)
        rewritten = res.rewritten
        print(f"   ✅ Rewritten: {rewritten[:100]}...")
    except Exception as e:
        print(f"   ⚠️ Skipped: {e}")
        rewritten = "A black and white line drawing of a cat running after a small mouse, side view, simple silhouette"
    
    # 2. Image Generation (HuggingFace)
    print("2️⃣  Generating Image (HuggingFace)...")
    HF_TOKEN = "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"
    generator = OnlineGenerator(api_key=HF_TOKEN)
    image = generator.generate(rewritten, width=512, height=512)
    
    if image:
        image.save(os.path.join(OUTPUT_DIR, "generated_image.png"))
        print(f"   ✅ Image saved!")
        
        # 3. ASCII Conversion (Gradient Mapper - PROVEN)
        print("3️⃣  Converting to ASCII (Gradient)...")
        ascii_art = image_to_gradient_ascii(image, width=80, ramp="ultra")
        
        # Save
        txt_path = os.path.join(OUTPUT_DIR, "cat_chase.txt")
        with open(txt_path, "w") as f:
            f.write(ascii_art)
        print(f"   ✅ ASCII saved to {txt_path}")
        
        # 4. Export to PNG
        png_path = render_ascii_to_image(ascii_art)
        if png_path:
            final_path = os.path.join(OUTPUT_DIR, "cat_chase_standard.png")
            os.rename(png_path, final_path)
            print(f"   ✅ PNG saved to {final_path}")
            
        print("\n🎉 STANDARD PIPELINE COMPLETE!")
        print("Check outputs/standard_test/ for results.")
    else:
        print("   ❌ Image generation failed (check HF_TOKEN)")

if __name__ == "__main__":
    run_standard_pipeline()

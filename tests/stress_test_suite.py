
import os
import time
from PIL import Image
from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.gradient_mapper import image_to_gradient_ascii
from ascii_gen.perceptual import create_ssim_mapper

# Test Prompts from User Request
TEST_CASES = {
    "spatial_nightmare": "small cat sitting on the back of a large chair next to a tall table",
    "ambiguity_stress": "bat on a trunk next to a mouse",
    "complexity_bomb": "cat sitting on a wooden chair in a cozy living room with a fireplace, bookshelf, lamp, rug, and window showing sunset",
    "impossible_request": "photorealistic 3D rotating rainbow-colored holographic cat with flowing fur dancing in slow motion",
    "vague_mystery": "thing on stuff",
    "abstract_challenge": "freedom"
}

# User provided token for testing
HF_TOKEN = "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"

OUTPUT_DIR = "outputs/stress_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_stress_test():
    print("🚀 Starting Comprehensive Stress Test Suite...")
    
    # Initialize components
    # Note: Using hardcoded keys from modules implicitly
    rewriter = LLMPromptRewriter()
    generator = OnlineGenerator(api_key=HF_TOKEN)
    ssim_mapper = create_ssim_mapper(width=100)
    
    # 1. Run "Spatial Nightmare" first as recommended
    run_single_case("spatial_nightmare", TEST_CASES["spatial_nightmare"], rewriter, generator, ssim_mapper)
    
    # 2. Run others
    for name, prompt in TEST_CASES.items():
        if name == "spatial_nightmare": continue
        run_single_case(name, prompt, rewriter, generator, ssim_mapper)
        
def run_single_case(name, prompt, rewriter, generator, ssim_mapper):
    print(f"\n==================================================")
    print(f"🧪 Testing Case: {name}")
    print(f"📝 Input: '{prompt}'")
    
    # A. LLM Rewrite
    print("🤖 (1/3) Rewriting prompt...")
    try:
        result = rewriter.rewrite(prompt)
        rewritten_text = result.rewritten
        print(f"   -> Rewritten: {rewritten_text}")
        
        # Check for refusal/clarification (Vague Mystery)
        if "unable" in rewritten_text.lower() or "clarif" in rewritten_text.lower():
            print(f"   ⚠️ System Refused/Asked Clarification (Expected for vague prompts)")
            with open(f"{OUTPUT_DIR}/{name}_LOG.txt", "w") as f:
                f.write(f"Prompt: {prompt}\nResult: Refused/Clarification\nOutput: {rewritten_text}")
            return
            
    except Exception as e:
        print(f"   ❌ Rewrite Failed: {e}")
        rewritten_text = prompt
        
    # B. Image Generation
    print("🎨 (2/3) Generating image (this make take 10-20s)...")
    image = None
    try:
        if generator.api_key:
            image = generator.generate(rewritten_text, width=512, height=512)
        else:
            print("   ⚠️ No HF_TOKEN found. Using placeholder image for ASCII test.")
            # Create a dummy image with some shapes to test ASCII mapper
            image = Image.new('RGB', (512, 512), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.rectangle([100, 100, 400, 400], outline='black', width=5)
            draw.text((200, 200), name, fill='black')
            
        if not image:
            print("   ❌ Image generation failed")
            return
        image.save(f"{OUTPUT_DIR}/{name}.png")
    except Exception as e:
        print(f"   ❌ Generation Error: {e}")
        return

    # C. ASCII Conversion (Using multiple modes)
    print("🔠 (3/3) Converting to ASCII...")
    
    # Mode 1: Neat (Gradient)
    ascii_neat = image_to_gradient_ascii(image, width=100, ramp="neat", with_edges=True, edge_weight=0.6)
    with open(f"{OUTPUT_DIR}/{name}_NEAT.txt", "w") as f:
        f.write(ascii_neat)
        
    # Mode 2: Deep Structure (SSIM)
    try:
        ascii_ssim = ssim_mapper.convert_image(image)
        with open(f"{OUTPUT_DIR}/{name}_SSIM.txt", "w") as f:
            f.write(ascii_ssim)
    except Exception as e:
        print(f"   ❌ SSIM Failed: {e}")

    print(f"✅ Case {name} Complete! Saved to {OUTPUT_DIR}/")
    time.sleep(2) # Be nice to API

if __name__ == "__main__":
    run_stress_test()

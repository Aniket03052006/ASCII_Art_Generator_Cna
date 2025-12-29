"""
Comprehensive test of all mappers with detailed prompts.
Tests: Random Forest, CNN, and AISS mappers.
"""

from PIL import Image
import time

# Test with detailed prompts
DETAILED_PROMPTS = [
    "A majestic mountain landscape with snow-capped peaks, pine forests, and a serene lake reflecting the sunset",
    "Cyberpunk city at night with neon signs, flying cars, and towering skyscrapers",
    "A detailed portrait of a wise owl perched on an ancient oak branch",
]

def test_image_conversion():
    """Test all mappers on the existing test image."""
    from ascii_gen import PromptToASCII
    
    print("=" * 60)
    print("TESTING ALL MAPPERS ON test_input.png")
    print("=" * 60)
    
    img = Image.open("test_input.png")
    
    # Test Random Forest (fastest)
    print("\n🌲 RANDOM FOREST MAPPER")
    print("-" * 40)
    start = time.time()
    pipeline_rf = PromptToASCII(mapper="random_forest", charset="ascii_structural")
    result_rf = pipeline_rf.from_image(img, char_width=50)
    rf_time = time.time() - start
    result_rf.display()
    print(f"\n⏱️ Time: {rf_time:.2f}s")
    
    # Test CNN (highest quality)
    print("\n🧠 CNN MAPPER (DeepAA-inspired)")
    print("-" * 40)
    start = time.time()
    pipeline_cnn = PromptToASCII(mapper="cnn", charset="ascii_structural")
    result_cnn = pipeline_cnn.from_image(img, char_width=50)
    cnn_time = time.time() - start
    result_cnn.display()
    print(f"\n⏱️ Time: {cnn_time:.2f}s")
    
    # Test AISS (structure-focused)
    print("\n📐 AISS MAPPER (Log-Polar)")
    print("-" * 40)
    start = time.time()
    pipeline_aiss = PromptToASCII(mapper="aiss", charset="ascii_structural")
    result_aiss = pipeline_aiss.from_image(img, char_width=50)
    aiss_time = time.time() - start
    result_aiss.display()
    print(f"\n⏱️ Time: {aiss_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Random Forest: {rf_time:.2f}s")
    print(f"CNN:           {cnn_time:.2f}s")
    print(f"AISS:          {aiss_time:.2f}s")
    
    # Save outputs
    result_rf.save("output_rf.txt")
    result_cnn.save("output_cnn.txt")
    result_aiss.save("output_aiss.txt")
    print("\n✅ Outputs saved to output_rf.txt, output_cnn.txt, output_aiss.txt")

def test_prompt_generation():
    """Test prompt-to-ASCII with detailed prompts (requires FLUX model download)."""
    from ascii_gen import PromptToASCII
    
    print("\n" + "=" * 60)
    print("TESTING PROMPT-TO-ASCII WITH FLUX.1 SCHNELL")
    print("⚠️ First run downloads ~30GB of model files!")
    print("=" * 60)
    
    pipeline = PromptToASCII(mapper="random_forest", charset="ascii_standard")
    
    for i, prompt in enumerate(DETAILED_PROMPTS, 1):
        print(f"\n📝 Prompt {i}: {prompt[:50]}...")
        print("-" * 40)
        
        start = time.time()
        result = pipeline.generate(
            prompt=prompt,
            char_width=60,
            seed=42 + i,  # Reproducible
        )
        total_time = time.time() - start
        
        result.display()
        print(f"\n⏱️ Total time: {total_time:.2f}s")
        result.save(f"output_prompt_{i}.txt")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--prompts":
        # Test with FLUX (requires model download)
        test_prompt_generation()
    else:
        # Quick test on existing image
        test_image_conversion()

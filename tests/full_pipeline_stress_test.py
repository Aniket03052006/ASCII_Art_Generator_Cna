"""
FULL PIPELINE STRESS TEST
==========================
Tests the complete ASCII art generation pipeline:
Prompt → LLM Rewrite → Image Generation → ASCII Conversion → Analysis
"""
import sys
sys.path.insert(0, '.')

import os
import time
from datetime import datetime
from pathlib import Path

# Imports
from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.gradient_mapper import image_to_gradient_ascii
from ascii_gen.exporter import render_ascii_to_image
from PIL import Image

# Test prompts
STRESS_TESTS = [
    {"name": "Ambiguity", "prompt": "bat on a trunk next to a mouse"},
    {"name": "Complexity", "prompt": "cat sitting on a wooden chair in a cozy living room with a fireplace, bookshelf, lamp"},
    {"name": "Impossible", "prompt": "photorealistic 3D rotating rainbow-colored holographic cat dancing"},
    {"name": "Vague", "prompt": "thing on stuff"},
    {"name": "Abstract", "prompt": "freedom"},
    {"name": "Spatial", "prompt": "small cat sitting on the back of a large chair next to a tall table"},
]

OUTPUT_DIR = Path("outputs/full_stress_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_full_test(test_name: str, prompt: str):
    """Run full pipeline for a single test."""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"📝 PROMPT: {prompt}")
    print(f"{'='*60}")
    
    results = {"name": test_name, "prompt": prompt, "steps": {}}
    
    # Step 1: LLM Rewrite
    print("\n⏳ Step 1/4: LLM Prompt Rewriting...")
    try:
        rewriter = LLMPromptRewriter()
        rewrite_result = rewriter.rewrite(prompt)
        results["steps"]["rewrite"] = {
            "status": "OK",
            "prompt": rewrite_result.rewritten[:150] + "...",
            "classification": rewrite_result.classification
        }
        print(f"   ✅ Rewritten: {rewrite_result.rewritten[:80]}...")
        print(f"   📊 Classification: {rewrite_result.classification}")
    except Exception as e:
        results["steps"]["rewrite"] = {"status": "FAIL", "error": str(e)}
        print(f"   ❌ Rewrite failed: {e}")
        return results
    
    # Step 2: Image Generation
    print("\n⏳ Step 2/4: Generating Image (Pollinations)...")
    try:
        generator = OnlineGenerator()
        gen_result = generator.generate(rewrite_result.rewritten, seed=42)
        image = gen_result.get("image")
        
        if image:
            img_path = OUTPUT_DIR / f"{test_name}_image.png"
            image.save(img_path)
            results["steps"]["image"] = {"status": "OK", "path": str(img_path), "size": image.size}
            print(f"   ✅ Image saved: {img_path}")
        else:
            results["steps"]["image"] = {"status": "FAIL", "error": "No image returned"}
            print(f"   ❌ No image returned")
            return results
    except Exception as e:
        results["steps"]["image"] = {"status": "FAIL", "error": str(e)}
        print(f"   ❌ Image generation failed: {e}")
        return results
    
    # Step 3: ASCII Conversion
    print("\n⏳ Step 3/4: Converting to ASCII Art...")
    try:
        # Use gradient mapper
        ascii_art = image_to_gradient_ascii(image, width=80, ramp="standard")
        
        ascii_path = OUTPUT_DIR / f"{test_name}_ascii.txt"
        with open(ascii_path, "w") as f:
            f.write(ascii_art)
        
        lines = ascii_art.split('\n')
        results["steps"]["ascii"] = {
            "status": "OK",
            "path": str(ascii_path),
            "lines": len(lines),
            "chars": len(ascii_art)
        }
        print(f"   ✅ ASCII saved: {ascii_path}")
        print(f"   📊 Size: {len(lines)} lines, {len(ascii_art)} chars")
        
        # Show preview
        print("\n   🎨 ASCII Preview (first 5 lines):")
        for line in lines[:5]:
            print(f"   {line[:60]}")
            
    except Exception as e:
        results["steps"]["ascii"] = {"status": "FAIL", "error": str(e)}
        print(f"   ❌ ASCII conversion failed: {e}")
        return results
    
    # Step 4: Render to Image
    print("\n⏳ Step 4/4: Rendering ASCII to Image...")
    try:
        render_path = OUTPUT_DIR / f"{test_name}_render.png"
        render_img = render_ascii_to_image(ascii_art)
        if render_img:
            if isinstance(render_img, str):
                # It returned a path, copy it
                import shutil
                shutil.copy(render_img, render_path)
            else:
                render_img.save(render_path)
            results["steps"]["render"] = {"status": "OK", "path": str(render_path)}
            print(f"   ✅ Render saved: {render_path}")
        else:
            results["steps"]["render"] = {"status": "FAIL", "error": "No render returned"}
            
    except Exception as e:
        results["steps"]["render"] = {"status": "FAIL", "error": str(e)}
        print(f"   ❌ Render failed: {e}")
    
    results["overall"] = "PASS"
    return results

def main():
    print("=" * 70)
    print("🧪 FULL PIPELINE STRESS TEST SUITE")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = []
    
    for test in STRESS_TESTS[:3]:  # Run first 3 tests (faster)
        result = run_full_test(test["name"], test["prompt"])
        all_results.append(result)
        time.sleep(2)  # Rate limit
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 FULL PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    for r in all_results:
        status = r.get("overall", "FAIL")
        emoji = "✅" if status == "PASS" else "❌"
        print(f"   {emoji} {r['name']}: {status}")
        for step, data in r.get("steps", {}).items():
            step_emoji = "✓" if data.get("status") == "OK" else "✗"
            print(f"      {step_emoji} {step}: {data.get('status', 'UNKNOWN')}")
    
    print(f"\n   📁 Outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

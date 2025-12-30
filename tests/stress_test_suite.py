
import os
import sys
import time
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.production_training import ProductionCNNMapper
from PIL import Image

# Hardcoded token for testing if env var missing
HF_TOKEN = os.getenv("HF_TOKEN", "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")

STRESS_PROMPTS = [
    {
        "name": "Spatial Nightmare",
        "prompt": "small cat sitting on the back of a large chair next to a tall table",
        "why": "Tests spatial relationship handling and relative sizing."
    },
    {
        "name": "Ambiguity Stress Test",
        "prompt": "bat on a trunk next to a mouse",
        "why": "Tests ambiguous terms (animal vs sports, elephant vs car)."
    },
    {
        "name": "Complexity Bomb",
        "prompt": "cat sitting on a wooden chair in a cozy living room with a fireplace, bookshelf, lamp, rug, and window showing sunset",
        "why": "Tests complexity detection (>7 objects)."
    },
    {
        "name": "Impossible Request",
        "prompt": "photorealistic 3D rotating rainbow-colored holographic cat with flowing fur dancing in slow motion",
        "why": "Tests impossible constraints (motion, color) -> simplfication."
    },
    {
        "name": "Vague Mystery",
        "prompt": "thing on stuff",
        "why": "Tests vague input checks."
    },
    {
        "name": "Abstract Challenge",
        "prompt": "freedom",
        "why": "Tests abstract concept handling."
    }
]

def run_stress_tests():
    print("🔥 Starting Comprehensive Stress Test Suite...")
    os.makedirs("outputs/stress_tests", exist_ok=True)
    
    # Initialize Models
    print("Initialize Models...")
    try:
        rewriter = LLMPromptRewriter()
    except Exception as e:
        print(f"Failed to load rewriter: {e}")
        return

    try:
        gen = OnlineGenerator(api_key=HF_TOKEN)
    except Exception as e:
        print(f"Failed to load generator: {e}")
        return
        
    cnn_mapper = ProductionCNNMapper()
    try:
        cnn_mapper.load("models/production_cnn.pth")
    except:
        print("Model not found, using untained fallback (ok for flow test)")

    report = []

    for i, item in enumerate(STRESS_PROMPTS):
        name = item['name']
        prompt = item['prompt']
        why = item['why']
        
        print(f"\n[{i+1}/6] Testing: {name}")
        print(f"📝 Prompt: '{prompt}'")
        print(f"❓ Goal: {why}")
        
        report.append(f"## Test {i+1}: {name}")
        report.append(f"**Original**: `{prompt}`")
        report.append(f"**Goal**: {why}")
        
        # 1. Rewrite
        start_t = time.time()
        res = rewriter.rewrite(prompt)
        report.append(f"**Complexity**: {res.complexity_score:.2f}")
        report.append(f"**Classification**: `{res.classification.upper()}`")  # Log Classification
        report.append(f"**Rewritten**: `{res.rewritten}`")
        if res.logs:
             report.append("**Thinking Logs**:")
             for log in res.logs:
                 report.append(f"- {log}")
        
        print(f"   🧠 Classified as: {res.classification.upper()}") # Console feedback
        
        # 2. Generate
        print("   🎨 Generating...")
        img = gen.generate(res.rewritten, width=512, height=384, skip_preprocessing=True)
        
        if img:
            img_path = f"outputs/stress_tests/test_{i+1}_{name.replace(' ','_').lower()}.png"
            img.save(img_path)
            report.append(f"**Image**: Saved to `{img_path}`")
            
            # 3. ASCII
            print("   ⚙️  Converting...")
            ascii_art = cnn_mapper.convert_image(img.resize((640, 360))) # rough resize
            txt_path = img_path.replace('.png', '.txt')
            with open(txt_path, 'w') as f:
                f.write(ascii_art)
            report.append(f"**ASCII**: Saved to `{txt_path}`")
            report.append(f"\n```\n{ascii_art[:500]}...\n```\n")
        else:
            report.append("**Image**: ❌ GENERATION FAILED")
            
        print(f"   ✅ Done in {time.time() - start_t:.1f}s")
        report.append("\n---\n")

    # Save Report
    with open("outputs/stress_tests/FINAL_REPORT.md", "w") as f:
        f.write("\n".join(report))
        
    print("\n✅ All tests completed. Report saved to outputs/stress_tests/FINAL_REPORT.md")

if __name__ == "__main__":
    run_stress_tests()

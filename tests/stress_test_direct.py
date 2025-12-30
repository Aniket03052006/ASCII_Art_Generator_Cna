
import os
import time
from ascii_gen.diff_render import DiffRenderer

# Same stress cases as the main suite
TEST_CASES = {
    "abstract_freedom": "freedom",
    "vague_mystery": "thing on stuff",
    "impossible_cat": "photorealistic 3D rotating rainbow-colored holographic cat"
}

OUTPUT_DIR = "outputs/stress_direct"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_direct_stress():
    print("🚀 Starting Direct Generation (DiffRender) Stress Test...")
    
    try:
        renderer = DiffRenderer(device="cpu") # Force CPU to start simple
    except Exception as e:
        print(f"❌ Failed to init renderer: {e}")
        return

    for case_name, prompt in TEST_CASES.items():
        print(f"\n🧪 Testing Case: {case_name}")
        print(f"📝 Prompt: '{prompt}'")
        
        start_t = time.time()
        # Use small width for speed during test
        ascii_art = renderer.optimize(prompt, width=40, steps=100)
        dur = time.time() - start_t
        
        print(f"✅ Generated in {dur:.2f}s")
        
        # Save output
        out_path = os.path.join(OUTPUT_DIR, f"{case_name}.txt")
        with open(out_path, "w") as f:
            f.write(ascii_art)
        
        # Also save a preview HTML for inspection (simplistic)
        html_path = os.path.join(OUTPUT_DIR, f"{case_name}.html")
        with open(html_path, "w") as f:
            f.write(f"<pre style='font-family: monospace; line-height:0.6; font-size: 8px; background: black; color: white;'>{ascii_art}</pre>")
            
    print(f"\n🎉 All Direct Tests Complete! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    run_direct_stress()

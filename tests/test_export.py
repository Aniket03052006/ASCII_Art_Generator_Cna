
import os
import time
from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.diff_render import DiffRenderer
from ascii_gen.exporter import render_ascii_to_image

OUTPUT_DIR = "outputs/end_to_end"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_full_pipeline_test():
    prompt = "cat chasing a mouse" # User requested "mouse", close enough to rat
    print(f"🚀 Starting End-to-End Test for: '{prompt}'")
    
    # 1. Pipeline: Rewrite -> Generate
    print("1️⃣  Enhancing prompt with LLM...")
    try:
        rewriter = LLMPromptRewriter()
        res = rewriter.rewrite(prompt)
        rewritten = res.rewritten
        print(f"   Shape Logic: {rewritten}")
    except Exception as e:
        print(f"   ⚠️ Rewrite skipped: {e}")
        rewritten = prompt
        
    print("2️⃣  Generating ASCII (soft rasterizer)...")
    renderer = DiffRenderer(device="cpu")
    start_t = time.time()
    ascii_art = renderer.optimize(rewritten) # Use new defaults (width=50, steps=200)
    dur = time.time() - start_t
    print(f"   Generated in {dur:.2f}s")
    
    # Save ASCII text
    txt_path = os.path.join(OUTPUT_DIR, "cat_chase.txt")
    with open(txt_path, "w") as f:
        f.write(ascii_art)
        
    # 2. Export to PNG (Clean Professional style: black on white)
    print("3️⃣  Exporting to PNG...")
    png_path = render_ascii_to_image(ascii_art) # Use upgraded defaults
    
    if png_path and os.path.exists(png_path):
        print(f"✅ Export Success! Image: {png_path}")
        # Move to a clear location
        final_path = os.path.join(OUTPUT_DIR, "cat_chase_export.png")
        os.rename(png_path, final_path)
        print(f"   Saved to: {final_path}")
    else:
        print("❌ Export Failed.")

if __name__ == "__main__":
    run_full_pipeline_test()


import os
import time
from ascii_gen.llm_rewriter import LLMPromptRewriter
from ascii_gen.diff_render import DiffRenderer

OUTPUT_DIR = "outputs/test_cat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_test():
    prompt = "cat chasing a rat"
    print(f"🧪 Testing prompt: '{prompt}'")
    
    # 1. Test LLM Rewriting
    print("🧠 Rewriting with LLM...")
    try:
        rewriter = LLMPromptRewriter()
        res = rewriter.rewrite(prompt)
        rewritten = res.rewritten
        print(f"✅ Rewritten: {rewritten}")
        
        # Save Rewrite for inspection
        with open(os.path.join(OUTPUT_DIR, "rewrite.txt"), "w") as f:
            f.write(f"Original: {prompt}\nRewritten: {rewritten}")
            
    except Exception as e:
        print(f"❌ Rewrite failed: {e}")
        rewritten = prompt # Fallback

    # 2. Test Direct Generation
    print("\n🎨 Generating ASCII (DiffRender)...")
    try:
        renderer = DiffRenderer(device="cpu")
        width = 50
        steps = 150
        
        start_t = time.time()
        ascii_art = renderer.optimize(rewritten, width=width, steps=steps)
        dur = time.time() - start_t
        
        print(f"✅ Generated in {dur:.2f}s")
        
        # Save output
        out_path = os.path.join(OUTPUT_DIR, "cat_chase_ascii.txt")
        with open(out_path, "w") as f:
            f.write(ascii_art)
            
        print(f"🎉 Saved to {out_path}")
        print("\nASCII Preview:\n" + ascii_art)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    run_test()

#!/usr/bin/env python3
"""
Interactive Prompt-to-ASCII Art Generator

A command-line interface for converting text prompts to ASCII art.
Supports both single prompts and interactive REPL mode.

Usage:
    python prompt_pipeline.py                    # Interactive mode
    python prompt_pipeline.py "your prompt"     # Single prompt
    python prompt_pipeline.py --help            # Help

NOTE: First run downloads FLUX.1 Schnell (~30GB)
"""

import argparse
import os
import sys
import time
from PIL import Image

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_pipeline(mapper_type="cnn"):
    """Load the ASCII art generation pipeline."""
    from ascii_gen import PromptToASCII
    from ascii_gen.production_training import ProductionCNNMapper
    
    print("🚀 Initializing Prompt-to-ASCII Pipeline...")
    print("   Model: FLUX.1 Schnell")
    print("   Mapper:", "Production CNN" if mapper_type == "cnn" else "Random Forest")
    print()
    
    # Initialize main pipeline
    pipeline = PromptToASCII(
        mapper="random_forest",
        sd_model="flux-schnell",
        charset="ascii_structural",
    )
    
    # Load production CNN if available
    cnn = None
    if mapper_type == "cnn":
        cnn = ProductionCNNMapper()
        try:
            cnn.load("models/production_cnn.pth")
            print("✅ Loaded pre-trained CNN model")
        except:
            print("📊 Training CNN model (first time only)...")
            cnn.train(epochs=50)
            cnn.save("models/production_cnn.pth")
    
    return pipeline, cnn


def generate_ascii(pipeline, cnn, prompt: str, width: int = 60, seed: int = None):
    """Generate ASCII art from a text prompt."""
    print(f"\n🎨 Generating: {prompt}")
    print("⏳ This may take 10-30 seconds...")
    
    start = time.time()
    
    # Generate image with FLUX
    result = pipeline.generate(
        prompt=prompt,
        char_width=width,
        seed=seed or int(time.time()) % 10000,
    )
    
    gen_time = time.time() - start
    
    # If CNN mapper available, use it for better quality
    if cnn and result.source_image:
        ascii_text = cnn.convert_image(result.source_image)
    else:
        ascii_text = result.text
    
    print(f"\n✅ Generated in {gen_time:.1f}s")
    print("=" * 60)
    print(ascii_text)
    print("=" * 60)
    
    return ascii_text, result.source_image


def interactive_mode(pipeline, cnn):
    """Interactive REPL mode for continuous prompt input."""
    print("\n" + "=" * 60)
    print("   INTERACTIVE PROMPT-TO-ASCII GENERATOR")
    print("=" * 60)
    print()
    print("Commands:")
    print("  Type any prompt to generate ASCII art")
    print("  'save <filename>' - Save last output")
    print("  'width <N>' - Set output width (default: 60)")
    print("  'quit' or 'exit' - Exit the program")
    print()
    
    last_ascii = None
    last_image = None
    width = 60
    
    os.makedirs("outputs", exist_ok=True)
    
    while True:
        try:
            prompt = input("\n📝 Enter prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("👋 Goodbye!")
                break
            
            if prompt.lower().startswith('save '):
                filename = prompt[5:].strip() or f"output_{int(time.time())}"
                if last_ascii:
                    # Save ASCII
                    txt_path = f"outputs/{filename}.txt"
                    with open(txt_path, 'w') as f:
                        f.write(last_ascii)
                    print(f"✅ Saved ASCII to {txt_path}")
                    
                    # Save image if available
                    if last_image:
                        img_path = f"outputs/{filename}.png"
                        last_image.save(img_path)
                        print(f"✅ Saved image to {img_path}")
                else:
                    print("❌ No output to save yet!")
                continue
            
            if prompt.lower().startswith('width '):
                try:
                    width = int(prompt[6:].strip())
                    print(f"✅ Width set to {width}")
                except:
                    print("❌ Invalid width. Usage: width 80")
                continue
            
            # Generate ASCII art
            last_ascii, last_image = generate_ascii(pipeline, cnn, prompt, width)
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert text prompts to ASCII art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prompt_pipeline.py
    Start interactive mode
    
  python prompt_pipeline.py "a cat sitting on a moon"
    Generate ASCII art for a single prompt
    
  python prompt_pipeline.py --mapper rf "a mountain landscape"
    Use Random Forest mapper (faster)
"""
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text prompt (omit for interactive mode)"
    )
    
    parser.add_argument(
        "--mapper", "-m",
        choices=["cnn", "rf"],
        default="cnn",
        help="Mapper type: cnn (better) or rf (faster)"
    )
    
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=60,
        help="Output width in characters (default: 60)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Load pipeline
    pipeline, cnn = load_pipeline(args.mapper)
    
    if args.prompt:
        # Single prompt mode
        ascii_text, image = generate_ascii(
            pipeline, cnn, args.prompt, args.width, args.seed
        )
        
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, 'w') as f:
                f.write(ascii_text)
            print(f"✅ Saved to {args.output}")
    else:
        # Interactive mode
        interactive_mode(pipeline, cnn)


if __name__ == "__main__":
    main()

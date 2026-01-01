
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ascii_gen.prompt_engineering import enhance_prompt, enhancer

def test_composition():
    test_prompts = [
        "elephant on a car",
        "dog next to a tree",
    ]

    print("=" * 70)
    print("DEBUGGING COMPOSITION")
    print("=" * 70)

    # Check if detector works
    for prompt in test_prompts:
        match = enhancer.composition_handler.detect_composition(prompt)
        print(f"Prompt: '{prompt}' -> Match: {match}")
        
    print("-" * 30)

    for prompt in test_prompts:
        print(f"\nINPUT: '{prompt}'")
        enhanced = enhance_prompt(prompt)
        print(f"OUTPUT: {enhanced}")
        print("-" * 30)

if __name__ == "__main__":
    test_composition()

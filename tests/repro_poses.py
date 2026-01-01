
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ascii_gen.prompt_engineering import enhance_prompt

def test_poses():
    test_prompts = [
        "a cat",
        "a beetle",
        "a chair",
        "a curled up cat",
        "a sleeping dog",
        "a spider"
    ]

    print("=" * 70)
    print("POSE ENFORCEMENT TEST")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\nINPUT: '{prompt}'")
        enhanced = enhance_prompt(prompt)
        print(f"OUTPUT: {enhanced}")
        print("-" * 30)

if __name__ == "__main__":
    test_poses()

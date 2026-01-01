from datasets import load_dataset
try:
    print("Loading dataset...")
    ds = load_dataset("mrzjy/ascii_art_generation_140k", split="train")
    print(f"Dataset type: {type(ds)}")
    print(f"Dataset columns: {ds.column_names}")
    print(f"First item keys: {ds[0].keys()}")
    print(f"First item: {ds[0]}")
except Exception as e:
    print(f"Error: {e}")

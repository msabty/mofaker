import os
from datasets import load_dataset

def download_dataset():
    print("Downloading openai/gsm8k dataset...")
    # Load the training and testing splits
    dataset_train = load_dataset("openai/gsm8k", "main", split="train")
    dataset_test = load_dataset("openai/gsm8k", "main", split="test")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save as JSONL
    train_path = "data/gsm8k_train.jsonl"
    test_path = "data/gsm8k_test.jsonl"
    
    print(f"Saving {len(dataset_train)} training examples to {train_path}...")
    dataset_train.to_json(train_path, orient="records", lines=True)
    
    print(f"Saving {len(dataset_test)} testing examples to {test_path}...")
    dataset_test.to_json(test_path, orient="records", lines=True)
    
    print("Download and save complete!")

if __name__ == "__main__":
    download_dataset()

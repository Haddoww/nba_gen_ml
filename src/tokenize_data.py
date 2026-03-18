import os
import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer

BASE_PATH = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "games.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "tokenized")
MAX_LENGTH = 512  # cap sequence length, adjust based on Colab memory

# ── Step 1: Load processed games ─────────────────────────────────────────────

def load_games(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    games = content.split("\n[NEW GAME]\n")
    games = [g.strip() for g in games if g.strip()]  # remove empty strings
    return games
    

# ── Step 2: Load tokenizer ────────────────────────────────────────────────────

def load_tokenizer():
    # TODO: load GPT2Tokenizer from pretrained 'gpt2'
    # GPT-2 doesn't have a padding token by default — set it to the eos token
    # return tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token 
    
    return tokenizer

# ── Step 3: Tokenize ──────────────────────────────────────────────────────────

def tokenize(games, tokenizer):
    # TODO: tokenize each game string
    # truncate to MAX_LENGTH
    # pad to MAX_LENGTH so all sequences are the same length
    # return tokenized output
    encoded_inputs = tokenizer(
        games,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )
    return encoded_inputs

# ── Step 4: Build HuggingFace Dataset ────────────────────────────────────────

def build_dataset(tokenized):
    # TODO: convert tokenized output into a HuggingFace Dataset object
    # it expects a dictionary of lists e.g. {'input_ids': [...], 'attention_mask': [...]}
    # return dataset
    dataset = Dataset.from_dict(tokenized)

    return dataset

# ── Step 5: Split ─────────────────────────────────────────────────────────────

def split_dataset(dataset):
    # TODO: split into train and validation sets
    # look into dataset.train_test_split()
    # use a 90/10 split
    # return train and val datasets
    train_validate = dataset.train_test_split(test_size=0.1, seed=42)

    return train_validate["train"], train_validate["test"]

    

# ── Step 6: Save ──────────────────────────────────────────────────────────────

def save_dataset(train, val):
    # TODO: save both splits to OUTPUT_PATH
    # look into dataset.save_to_disk()
    # save train to OUTPUT_PATH/train
    # save val to OUTPUT_PATH/val

    val_out = OUTPUT_PATH + "/val"
    train_out = OUTPUT_PATH + "/train"

    train.save_to_disk(train_out)
    val.save_to_disk(val_out)
    return 

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading games...")
    games = load_games(INPUT_PATH)
    print(f"{len(games)} games loaded")


    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Tokenizing...")
    tokenized = tokenize(games, tokenizer)

    print("Building dataset...")
    dataset = build_dataset(tokenized)

    print("Splitting...")
    train, val = split_dataset(dataset)
    print(f"Train: {len(train)} | Val: {len(val)}")

    print("Saving...")
    save_dataset(train, val)
    print("Done.")

if __name__ == "__main__":
    main()
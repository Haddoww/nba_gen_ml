import os
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

BASE_PATH = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "tokenized", "train")
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "tokenized", "val")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "models", "gpt2-nba")

# ── Step 1: Load datasets ─────────────────────────────────────────────────────

def load_data():
    # TODO: load train and val datasets from disk
    # return both

    train = load_from_disk("data/processed/tokenized/train")
    val = load_from_disk("data/processed/tokenized/val")
    return train, val

# ── Step 2: Load model ────────────────────────────────────────────────────────

def load_model():
    # TODO: load GPT2LMHeadModel from pretrained 'gpt2'
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # return model
    return model

# ── Step 3: Apply LoRA ────────────────────────────────────────────────────────

def apply_lora(model):

    # Low-Rank Adaptation (LoRA) is a PEFT method that decomposes a large matrix
    # into two smaller low-rank matrices in the attention layers. T
    # his drastically reduces the number of parameters that need to be fine-tuned.
    # TODO: create a LoraConfig with:

    #   task_type = TaskType.CAUSAL_LM
    #   r = 8  (rank, controls how many parameters LoRA adds)
    #   lora_alpha = 32  (scaling factor)
    #   lora_dropout = 0.1
    #   target_modules = ["c_attn"]  (which layers to apply LoRA to)
    # wrap model with get_peft_model()
    # print trainable parameters to see how few LoRA uses
    # return lora model

    config = LoraConfig(
        task_type= TaskType.CAUSAL_LM,
        r=8, #how many parameters lora adds
        lora_alpha=32, #scaling factor
        lora_dropout=0.1,
        target_modules=["c_attn"]       #layers tp apply the decompression 
    )

    lora_model = get_peft_model(model, config)
    
    return lora_model

# ── Step 4: Training arguments ────────────────────────────────────────────────

def get_training_args():
    # TODO: create TrainingArguments with:
    #   output_dir = OUTPUT_PATH
    #   num_train_epochs = 3
    #   per_device_train_batch_size = 8
    #   per_device_eval_batch_size = 8
    #   evaluation_strategy = "epoch"
    #   save_strategy = "epoch"
    #   logging_steps = 50
    #   learning_rate = 2e-4
    #   fp16 = True  (mixed precision, faster on Colab GPU)
    #   report_to = "none"  (swap to "wandb" when you set that up)

    args = TrainingArguments(
        output_dir = OUTPUT_PATH
        num_train_epochs = 3
        per_device_train_batch_size = 8
        per_device_eval_batch_size = 8
        evaluation_strategy = "epoch"
        save_strategy = "epoch"
        logging_steps = 50
        learning_rate = 2e-4
        fp16 = True # (mixed precision, faster on Colab GPU)
        report_to = "none" # (swap to "wandb" when you set that up)
    )
    return args
# ── Step 5: Data collator ─────────────────────────────────────────────────────

def get_collator(tokenizer):
    # TODO: create DataCollatorForLanguageModeling
    #   mlm = False  (we're doing causal LM not masked LM)
    # return collator

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,        
    )
    return collator

# ── Step 6: Train ─────────────────────────────────────────────────────────────

def train(model, train_data, val_data, training_args, collator):
    # TODO: create Trainer with model, args, datasets, and collator
    # call trainer.train()
    # save the final model to OUTPUT_PATH

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=val_data
    )

    trainer.train()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    train_data, val_data = load_data()
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    print("Loading model...")
    model = load_model()

    print("Applying LoRA...")
    model = apply_lora(model)

    print("Configuring training...")
    training_args = get_training_args()
    
    print("Loading tokenizer and collator...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = get_collator(tokenizer)

    print("Training...")
    train(model, train_data, val_data, training_args, collator)
    print("Done.")

if __name__ == "__main__":
    main()
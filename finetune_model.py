#!/usr/bin/env python3
"""
Fine-tune a smaller language model for HTML-to-JSON conversion.

This script uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA
to efficiently fine-tune smaller models like Phi-2, Mistral-7B, etc.
"""

import json
import torch
from torch.utils.data import Dataset
import argparse
import os
import sys
from typing import List, Dict
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required!")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade to Python 3.10+")
        print("   Visit: https://www.python.org/downloads/")
        sys.exit(1)

# Check Python version at import time
check_python_version()

class HTMLToJSONDataset(Dataset):
    """Dataset for HTML-to-JSON fine-tuning"""

    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format the conversation
        messages = example['messages']

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback formatting
            text = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                text += f"### {role.title()}: {content}\n"
            text += "### Assistant:"

        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        # Create labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()

        return tokenized

def find_linear_layers(model):
    """Find linear layers for LoRA"""
    linear_layers = set()

    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            linear_layers.add(names[-1])

    return list(linear_layers)

def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Setup model and tokenizer with quantization"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb.QuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    return model, tokenizer

def setup_lora_config(model, rank: int = 64, alpha: int = 128):
    """Setup LoRA configuration"""

    # Find target modules
    target_modules = find_linear_layers(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return lora_config

def train_model(
    model_name: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    use_4bit: bool = True,
    lora_rank: int = 64,
    lora_alpha: int = 128
):
    """Main training function"""

    print(f"🚀 Starting fine-tuning of {model_name}")
    print(f"📁 Training data: {train_file}")
    print(f"📁 Validation data: {val_file}")
    print(f"📂 Output directory: {output_dir}")

    # Setup model and tokenizer
    print("🔧 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, use_4bit)

    # Setup LoRA
    print("🎯 Setting up LoRA...")
    lora_config = setup_lora_config(model, lora_rank, lora_alpha)
    model = get_peft_model(model, lora_config)

    print(f"📊 Trainable parameters: {model.print_trainable_parameters()}")

    # Create datasets
    print("📚 Loading datasets...")
    train_dataset = HTMLToJSONDataset(train_file, tokenizer, max_length)
    val_dataset = HTMLToJSONDataset(val_file, tokenizer, max_length)

    print(f"📈 Training examples: {len(train_dataset)}")
    print(f"📉 Validation examples: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train
    print("🏃 Starting training...")
    trainer.train()

    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Fine-tuning completed! Model saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model for HTML-to-JSON conversion")

    # Model selection
    parser.add_argument("--model", default="microsoft/phi-2",
                       choices=["microsoft/phi-2", "microsoft/DialoGPT-medium",
                               "distilgpt2", "gpt2-medium"],
                       help="Base model to fine-tune")

    # Data files
    parser.add_argument("--train-file", default="train.jsonl",
                       help="Training data file")
    parser.add_argument("--val-file", default="val.jsonl",
                       help="Validation data file")

    # Training parameters
    parser.add_argument("--output-dir", default="fine_tuned_model",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")

    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                       help="LoRA alpha")

    # Hardware options
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization (uses more VRAM)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    train_model(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_4bit=not args.no_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

if __name__ == "__main__":
    main()

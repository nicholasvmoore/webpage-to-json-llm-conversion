#!/usr/bin/env python3
"""
Prepare training data for fine-tuning HTML-to-JSON conversion model.

Converts raw examples into proper instruction-tuning format.
"""

import json
import re
import argparse
import sys
from typing import Dict, List

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

def format_instruction_example(input_text: str, output_json: str) -> Dict:
    """Format a single example for instruction tuning"""

    system_message = """You are an AI assistant specialized in extracting structured specifications from HTML content.
Your task is to analyze the given text content and convert it into a well-organized JSON format with proper data types."""

    user_message = f"""Please extract the specifications from the following cleaned HTML text and return them as structured JSON:

{input_text}

Guidelines:
- Organize specifications into logical sections (e.g., "CPU Specifications", "Memory Specifications")
- Use appropriate data types: numbers for numeric values, arrays for comma-separated values, booleans for yes/no
- Keep the original specification names as keys
- Return only valid JSON, no additional text"""

    assistant_message = output_json

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }

def clean_json_output(json_str: str) -> str:
    """Clean and validate JSON output"""
    try:
        # Try to parse and re-format
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        # If parsing fails, try to extract JSON portion
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return json.dumps(parsed, indent=2)
            except:
                pass
        # Return as-is if all parsing attempts fail
        return json_str

def process_training_file(input_file: str, output_file: str):
    """Process raw training data into instruction format"""
    formatted_examples = []

    print(f"Processing {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())

                # Extract and clean data
                input_text = example.get('input', '')
                output_json = example.get('output', '')

                if not input_text or not output_json:
                    print(f"Warning: Skipping line {line_num} - missing input or output")
                    continue

                # Clean the JSON output
                cleaned_output = clean_json_output(output_json)

                # Format as instruction example
                formatted = format_instruction_example(input_text, cleaned_output)
                formatted_examples.append(formatted)

                if line_num % 10 == 0:
                    print(f"Processed {line_num} examples...")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Save formatted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Successfully processed {len(formatted_examples)} examples")
    print(f"📁 Saved to {output_file}")

    # Generate statistics
    total_tokens = sum(len(ex['messages'][0]['content'].split()) +
                      len(ex['messages'][1]['content'].split()) +
                      len(ex['messages'][2]['content'].split())
                      for ex in formatted_examples)

    print("📊 Dataset Statistics:")
    print(f"   Examples: {len(formatted_examples)}")
    print(f"   Estimated tokens: {total_tokens}")
    print(f"   Approximate size: {total_tokens * 4 / (1024**3):.2f} GB (4 bytes per token estimate)")
def split_dataset(input_file: str, train_file: str, val_file: str, val_split: float = 0.1):
    """Split dataset into training and validation sets"""
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    val_size = int(len(examples) * val_split)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_examples)} examples ({train_file})")
    print(f"   Validation: {len(val_examples)} examples ({val_file})")

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument("--input", default="training_data.jsonl",
                       help="Input raw training data file")
    parser.add_argument("--output", default="formatted_training_data.jsonl",
                       help="Output formatted training data file")
    parser.add_argument("--train-output", default="train.jsonl",
                       help="Training split output file")
    parser.add_argument("--val-output", default="val.jsonl",
                       help="Validation split output file")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split ratio")

    args = parser.parse_args()

    # Format the training data
    process_training_file(args.input, args.output)

    # Split into train/val
    if args.train_output and args.val_output:
        split_dataset(args.output, args.train_output, args.val_output, args.val_split)

if __name__ == "__main__":
    main()

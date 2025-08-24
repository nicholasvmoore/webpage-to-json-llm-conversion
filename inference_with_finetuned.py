#!/usr/bin/env python3
"""
Inference script using fine-tuned model for HTML-to-JSON conversion.

This replaces the large LLM with your fine-tuned smaller model.
"""

import torch
import json
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import requests
from bs4 import BeautifulSoup
import os

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

class FineTunedHTMLConverter:
    """HTML to JSON converter using fine-tuned model"""

    def __init__(self, model_path: str, device: str = "auto"):
        print(f"🔧 Loading fine-tuned model from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True
        )

        print("✅ Model loaded successfully")

    def convert_html_to_json(self, html_content: str) -> dict:
        """Convert HTML content to structured JSON"""

        # Prepare the prompt in the same format as training
        system_message = """You are an AI assistant specialized in extracting structured specifications from HTML content.
Your task is to analyze the given text content and convert it into a well-organized JSON format with proper data types."""

        user_message = f"""Please extract the specifications from the following cleaned HTML text and return them as structured JSON:

{html_content}

Guidelines:
- Organize specifications into logical sections (e.g., "CPU Specifications", "Memory Specifications")
- Use appropriate data types: numbers for numeric values, arrays for comma-separated values, booleans for yes/no
- Keep the original specification names as keys
- Return only valid JSON, no additional text"""

        # Format for model
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"### System: {system_message}\n### User: {user_message}\n### Assistant:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return {"error": "No JSON found in response", "raw_response": response}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response", "raw_response": response}

def fetch_and_clean_html(url: str) -> str:
    """Fetch and clean HTML (same as original script)"""
    print(f"🌐 Fetching HTML from: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()

        # Extract text
        text_content = soup.get_text()

        # Clean up
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for phrase in lines if phrase.strip())
        cleaned_text = '\n'.join(chunks)

        print("✅ HTML content fetched and cleaned")
        return cleaned_text

    except Exception as e:
        print(f"❌ Error fetching HTML: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert HTML to JSON using fine-tuned model")
    parser.add_argument("--model-path", required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--url",
                       default="https://www.intel.com/content/www/us/en/products/sku/236848/intel-core-ultra-5-processor-125h-18m-cache-up-to-4-50-ghz/specifications.html",
                       help="URL to fetch HTML from")
    parser.add_argument("--output", default="output_finetuned.json",
                       help="Output JSON file")
    parser.add_argument("--device", default="auto",
                       help="Device to run model on (auto, cpu, cuda)")

    args = parser.parse_args()

    # Initialize converter
    converter = FineTunedHTMLConverter(args.model_path, args.device)

    # Fetch HTML content
    html_content = fetch_and_clean_html(args.url)

    if not html_content:
        print("❌ Failed to fetch HTML content")
        return

    print(f"📝 Content length: {len(html_content)} characters")

    # Convert to JSON
    print("🔄 Converting HTML to JSON...")
    result = converter.convert_html_to_json(html_content)

    # Save result
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved to {args.output}")

        # Print preview
        print("\n📊 Preview of results:")
        if "error" not in result:
            print(json.dumps(result, indent=2)[:500] + "...")
        else:
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()

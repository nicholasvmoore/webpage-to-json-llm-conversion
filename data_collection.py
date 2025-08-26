#!/usr/bin/env python3
"""
Data Collection Pipeline for Fine-tuning HTML-to-JSON Model

This script helps collect training data by:
1. Scraping multiple specification pages
2. Using your existing LLM to generate ground truth
3. Formatting data for fine-tuning
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time
import sys
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import argparse

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

class DataCollector:
    def __init__(self, llm_url: str = "http://localhost:8080/completion"):
        self.llm_url = llm_url
        self.session = requests.Session()

    def fetch_and_clean_html(self, url: str) -> Optional[str]:
        """Fetch and clean HTML content (same as your main script)"""
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

            return cleaned_text

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def generate_training_example(self, url: str) -> Optional[Dict]:
        """Generate a single training example"""
        print(f"Processing: {url}")

        # Get cleaned HTML
        cleaned_text = self.fetch_and_clean_html(url)
        if not cleaned_text:
            return None

        # Use existing LLM to generate ground truth
        prompt = f"""
Extract the specifications from the following text and return them in JSON format.
The JSON should organize the specifications into their sections each with a list of their specifications.
The Specification Name is the key and the Specification Value is the value.
The value should represent the data type.
If the value is separated by a comma, then it should be an array.
If the value is a number, then it should be a number.
If the value is a string, then it should be a string.
If the value is yes or no, then it should be a boolean.

Specifications:
{cleaned_text}

Return only valid JSON.
Return only the JSON, no other text.
Remove escape characters.
"""

        try:
            data = {
                "prompt": prompt,
                "seed": 42,
                "n_predict": 32768,
                "temperature": 0.1
            }

            response = self.session.post(self.llm_url, json=data, timeout=300)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get('content', '')

            # Extract JSON
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_output = json.loads(json_str)

                return {
                    "input": cleaned_text,
                    "output": json.dumps(json_output, indent=2),
                    "url": url,
                    "timestamp": time.time()
                }

        except Exception as e:
            print(f"Error generating example for {url}: {e}")

        return None

    def collect_from_urls(self, urls: List[str], output_file: str = "training_data.jsonl"):
        """Collect training data from multiple URLs"""
        training_examples = []

        for i, url in enumerate(urls):
            print(f"Processing {i+1}/{len(urls)}: {url}")
            example = self.generate_training_example(url)

            if example:
                training_examples.append(example)
                print(f"✓ Successfully generated example {i+1}")

                # Save progress periodically
                if len(training_examples) % 5 == 0:
                    self.save_examples(training_examples, output_file)
                    print(f"📁 Saved progress: {len(training_examples)} examples")
            else:
                print(f"✗ Failed to generate example {i+1}")

            # Rate limiting
            time.sleep(2)

        # Final save
        self.save_examples(training_examples, output_file)
        print(f"🎉 Collected {len(training_examples)} training examples")

        return training_examples

    def save_examples(self, examples: List[Dict], filename: str):
        """Save examples in JSONL format for training"""
        with open(filename, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

# Example usage and URL sources
def get_example_urls() -> List[str]:
    """Get example URLs for different types of specification pages"""
    return [
        "https://www.intel.com/content/www/us/en/products/sku/236848/intel-core-ultra-5-processor-125h-18m-cache-up-to-4-50-ghz/specifications.html",
        "https://www.intel.com/content/www/us/en/products/sku/232086/intel-core-i7-processor-12700k-25m-cache-up-to-5-00-ghz/specifications.html",
        "https://www.intel.com/content/www/us/en/products/sku/230500/intel-core-i9-processor-12900k-30m-cache-up-to-5-20-ghz/specifications.html",
        # Add more URLs from different manufacturers/products
        # NVIDIA GPUs, AMD processors, etc.
    ]

def main():
    parser = argparse.ArgumentParser(description="Collect training data for HTML-to-JSON model")
    parser.add_argument("--llm-url", default="http://localhost:8080/completion",
                       help="LLM server URL")
    parser.add_argument("--output", default="training_data.jsonl",
                       help="Output file for training data")
    parser.add_argument("--urls-file", help="File containing URLs (one per line)")

    args = parser.parse_args()

    collector = DataCollector(args.llm_url)

    if args.urls_file and os.path.exists(args.urls_file):
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = get_example_urls()
        print("Using default URL list. Create a urls.txt file for custom URLs.")

    print(f"Starting data collection with {len(urls)} URLs...")
    collector.collect_from_urls(urls, args.output)

if __name__ == "__main__":
    main()

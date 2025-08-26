# Generated with qwen3-coder
# Fine tuned by Nicholas Moore

import json
from html_fetcher import fetch_and_clean_html
from llm_connector import send_to_llm
import os

# --- 1. CONFIGURATION ---
# IMPORTANT: Replace this URL with the actual address of your llama.cpp server.
# The server must be running in API mode (e.g., ./server -m <model_path>).
LLM_SERVER_URL = "http://localhost:8080/completion"
# Read URLs from environment variables or fallback to defaults
LLM_SERVER_URL = os.getenv(
    "LLM_SERVER_URL", "http://localhost:8080/completion")
HTML_PAGE_URL = os.getenv(
    "HTML_PAGE_URL", "https://www.intel.com/content/www/us/en/products/sku/236848/intel-core-ultra-5-processor-125h-18m-cache-up-to-4-50-ghz/specifications.html")


def main():
    """Main function to orchestrate the process"""
    print("Fetching specifications from Intel webpage...")

    # Step 1: Fetch specifications
    specs = fetch_and_clean_html(HTML_PAGE_URL)

    if not specs:
        print("Failed to fetch specifications")
        return

    print(f"Successfully fetched specifications (found {len(specs)} sections)")

    # Step 2: Send to LLM
    print("Sending specifications to LLM...")
    llm_output = send_to_llm(specs)

    # Step 3: Write to JSON file
    output_file = "output/output.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        print(f"Successfully wrote results to {output_file}")

        # Print a preview of the output
        print("\nPreview of output:")
        print(json.dumps(llm_output, indent=2)[:500] + "...")

    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    main()

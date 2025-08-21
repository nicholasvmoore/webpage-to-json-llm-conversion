# Generated with qwen3-coder
# Fine tuned by Nicholas Moore

import requests
from bs4 import BeautifulSoup
import json
import re
import argparse
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


# --- 2. HTML PAGE FETCHING AND CLEANING ---
def fetch_and_clean_html(url):
    """
    Fetches the HTML content from a URL and cleans it up.

    Returns a string containing the text content of the page.
    """
    print(f"Fetching HTML from: {url}")
    try:
        # Use requests to get the HTML content.
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Use BeautifulSoup to parse the HTML.
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements like scripts, styles, and headers to get clean text.
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()

        # Extract the text and strip whitespace.
        text_content = soup.get_text()

        # Clean up multiple newlines and spaces.
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for phrase in lines if phrase.strip())
        cleaned_text = '\n'.join(chunks)

        print("HTML content successfully fetched and cleaned.")
        return cleaned_text

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the HTML: {e}")
        return None


def send_to_llm(specs):
    """Send specifications to LLM and get JSON response"""

    llm_url = LLM_SERVER_URL  # Use the environment variable or default
    # Create a session
    session = requests.Session()  # Create a session for persistent connections

    # Prepare the prompt for the LLM
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
    {json.dumps(specs, indent=2)}

    Return only valid JSON.
    Return only the JSON, no other text.
    Remove escape characters.
    """

    try:
        # Prepare request data
        data = {
            "prompt": prompt,
            "seed": 42,
            "n_predict": 32768,  # Increased tokens for response
            "temperature": 0.1  # Low temperature for more deterministic output
        }

        # Send request to LLM
        response = session.post(llm_url, json=data)  # Removed timeout
        response.raise_for_status()

        # Parse the response
        result = response.json()
        generated_text = result.get('content', '')

        # Try to find JSON pattern in the response
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text as JSON
                return {"llm_response": generated_text}
        else:
            # Return the entire response as JSON
            return {"llm_response": generated_text}

    except requests.RequestException as e:
        print(f"Error communicating with LLM: {e}")
        return {"error": f"LLM communication error: {e}"}


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

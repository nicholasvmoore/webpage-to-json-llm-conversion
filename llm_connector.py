import requests
import json
import re
import os

# --- 1. CONFIGURATION ---
# IMPORTANT: Replace this URL with the actual address of your llama.cpp server.
# The server must be running in API mode (e.g., ./server -m <model_path>).
LLM_SERVER_URL = "http://localhost:8080/completion"
# Read URLs from environment variables or fallback to defaults
LLM_SERVER_URL = os.getenv(
    "LLM_SERVER_URL", "http://localhost:8080/completion")


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

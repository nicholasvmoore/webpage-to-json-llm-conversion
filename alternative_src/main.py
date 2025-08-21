#!/usr/bin/env python3
"""
Webpage Content Extractor with LLM Integration

This script:
1. Takes a webpage URL as input
2. Extracts key content from the webpage
3. Sends the content to a local LLM (llama.cpp)
4. Generates structured JSON output based on LLM response

Usage:
    python src/main.py https://example.com
"""

import argparse
import json
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time


def extract_webpage_content(url):
    """
    Extract key content from a webpage.
    
    Args:
        url (str): The URL of the webpage to extract content from
        
    Returns:
        dict: Dictionary containing extracted content with metadata
    """
    try:
        # Send GET request to the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else ''
        
        # Extract headings
        headings = {}
        for i in range(1, 7):
            headings[f'h{i}'] = [h.get_text().strip() for h in soup.find_all(f'h{i}') if h.get_text().strip()]
        
        # Extract paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            link_text = link.get_text().strip()
            if link_text or absolute_url:
                links.append({
                    'url': absolute_url,
                    'text': link_text
                })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            absolute_url = urljoin(url, img['src'])
            alt_text = img.get('alt', '')
            images.append({
                'url': absolute_url,
                'alt': alt_text
            })
        
        # Return structured content
        return {
            'url': url,
            'title': title_text,
            'description': description,
            'headings': headings,
            'paragraphs': paragraphs,
            'links': links,
            'images': images,
            'extracted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch webpage: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to extract content: {str(e)}")


def send_to_llm(content, llm_url="http://localhost:8080/completion"):
    """
    Send webpage content to a local LLM via llama.cpp.
    
    Args:
        content (dict): The extracted webpage content
        llm_url (str): The URL of the llama.cpp server endpoint
        
    Returns:
        dict: The structured JSON response from the LLM
    """
    try:
        # Prepare the prompt for the LLM
        prompt = f"""
        You are a content analysis expert. Analyze the following webpage content and generate a structured JSON response with these fields:
        
        - title: The page title
        - summary: A brief summary of the main content
        - key_topics: List of main topics or themes
        - sentiment: Overall sentiment (positive, negative, neutral)
        - main_entities: List of important entities mentioned
        - action_items: List of actions or recommendations if any
        
        Webpage Content:
        {json.dumps(content, indent=2, ensure_ascii=False)}
        
        Please respond with ONLY valid JSON. Do not include any explanations or additional text.
        """
        
        # Prepare request payload
        payload = {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": ["</s>"]
        }
        
        # Send request to LLM
        response = requests.post(
            llm_url, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        # Parse LLM response
        llm_response = response.json()
        generated_text = llm_response.get("content", "").strip()
        
        # Attempt to parse JSON from LLM response
        try:
            # Extract JSON from response
            start = generated_text.find('{')
            end = generated_text.rfind('}') + 1
            
            if start != -1 and end != -1:
                json_str = generated_text[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in LLM response")
                
        except json.JSONDecodeError as e:
            raise Exception(f"LLM returned invalid JSON: {str(e)}")
            
    except requests.RequestException as e:
        raise Exception(f"Failed to communicate with LLM: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to process LLM response: {str(e)}")


def save_output(data, output_file):
    """
    Save structured data to a JSON file.
    
    Args:
        data (dict): The structured data to save
        output_file (str): The path to the output file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Save data to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Output saved to {output_file}")
        
    except Exception as e:
        raise Exception(f"Failed to save output: {str(e)}")


def main():
    """
    Main function to orchestrate the webpage content extraction and LLM processing.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract webpage content and analyze with local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py https://example.com
  python src/main.py --url https://example.com --output results.json
        """
    )
    
    parser.add_argument(
        'url', 
        nargs='?', 
        help='URL of the webpage to analyze (can also use --url flag)'
    )
    
    parser.add_argument(
        '--url',
        dest='url_arg',
        help='URL of the webpage to analyze (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--output',
        dest='output_file',
        default='output.json',
        help='Output JSON file path (default: output.json)'
    )
    
    parser.add_argument(
        '--llm-url',
        dest='llm_url',
        default='http://localhost:8080/completion',
        help='URL of the llama.cpp server endpoint (default: http://localhost:8080/completion)'
    )
    
    args = parser.parse_args()
    
    # Determine URL from arguments
    url = args.url or args.url_arg
    if not url:
        print("Error: No URL provided")
        print("Usage: python src/main.py <url> [--output <file>] [--llm-url <url>]")
        return 1
    
    # Validate URL format
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError("Invalid URL format")
    except Exception as e:
        print(f"Error: Invalid URL provided - {str(e)}")
        return 1
    
    try:
        print(f"Extracting content from: {url}")
        
        # Step 1: Extract webpage content
        content = extract_webpage_content(url)
        print(f"Content extracted successfully")
        
        # Step 2: Send content to LLM
        print("Sending content to LLM...")
        llm_response = send_to_llm(content, args.llm_url)
        print("LLM analysis completed")
        
        # Step 3: Save output
        output_data = {
            "input_url": url,
            "extraction_timestamp": content['extracted_at'],
            "analysis": llm_response
        }
        
        save_output(output_data, args.output_file)
        
        print("\nAnalysis complete!")
        print(f"Output saved to: {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
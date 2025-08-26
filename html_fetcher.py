import requests
from bs4 import BeautifulSoup


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
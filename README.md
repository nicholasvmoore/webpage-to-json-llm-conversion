# webpage-to-json-llm-conversion
A Python application that extracts content from webpages and analyzes it using a local LLM (llama.cpp).

## Features

- Extracts key content from webpages (title, headings, paragraphs, links, images)
- Sends content to a local LLM for analysis
- Generates structured JSON output with analysis results
- Configurable via command-line arguments or environment variables

## Requirements

- Python 3.7+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) server running locally
- /nothink LLM Models

### Tested Models with decent output

- **Qwen3-Coder 30B-A3B Instruct**
- GPT OSS 20B
- Gemma 3 12B

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:nicholasvmoore/webpage-to-json-llm-conversion.git
   cd webpage-to-json-llm-conversion
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the llama.cpp server:
   ```bash
   # Example command (adjust based on your setup)
   ./llama-server -m ~/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf --ctx-size 2048
   ```

## Usage

### Basic usage:
```bash
python main.py https://example.com
```

### With custom output file:
```bash
python main.py https://example.com --output analysis.json
```

### With custom LLM endpoint:
```bash
python main.py https://example.com --llm-url http://localhost:8080/api/completion
```

## Output Format

The application generates a structured JSON output containing:

1. Input URL and extraction timestamp
2. Analysis results from the LLM including:
   - Title
   - Summary
   - Key topics
   - Sentiment
   - Main entities
   - Action items

## Configuration

You can configure the application using command-line arguments or environment variables. The following options are available:

- `--url` or positional argument: The URL to analyze
- `--output`: Output JSON file path (default: output.json)
- `--llm-url`: URL of the llama.cpp server endpoint (default: http://localhost:8080/completion)

## Development

### Setting up pyenv (recommended for Python version management)

1. Install pyenv:
   ```bash
   # On macOS/Linux
   curl https://pyenv.run | bash
   ```

2. Add pyenv to your shell:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PYENV_ROOT="$HOME/.pyenv"
   command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"
   ```

3. Install Python version:
   ```bash
   pyenv install 3.10.0
   pyenv global 3.10.0
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Contributing

1. Clone the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

MIT License

## Author

Nicholas Moore

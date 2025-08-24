# Fine-tuning Guide: HTML-to-JSON Conversion

This guide explains how to fine-tune a smaller language model to replace your large LLM for converting HTML content to structured JSON.

## 🎯 Why Fine-tune?

Your current setup uses a large, slow LLM. Fine-tuning a smaller model offers:
- ⚡ **10-50x faster inference**
- 💰 **Lower computational costs**
- 🔒 **Better privacy** (no external API calls)
- 🎛️ **Full control** over the model

## 📋 Prerequisites

1. **Hardware Requirements:**
   - NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
   - 16GB+ RAM
   - 50GB+ free disk space

2. **Software:**
   - Python 3.10+
   - CUDA 11.6+ (if using GPU)
   - Git

## 🚀 Step-by-Step Fine-tuning Process

### Step 1: Data Collection

Collect training examples by running your existing LLM on various HTML pages:

```bash
# 1. Check compatibility (Python 3.10+ required)
python setup_and_run.py check

# 2. Install dependencies
pip install -e ".[finetune]"

# Or install from requirements file
pip install -r requirements_finetune.txt

# 3. Collect training data (requires running LLM server)
python data_collection.py --output training_data.jsonl

# Or use custom URLs
echo "https://example.com/specs1.html" > urls.txt
echo "https://example.com/specs2.html" >> urls.txt
python data_collection.py --urls-file urls.txt
```

**Expected Output:** `training_data.jsonl` with input/output pairs

### Step 2: Data Preparation

Format the raw data for fine-tuning:

```bash
# Prepare and format training data
python prepare_training_data.py

# This creates:
# - formatted_training_data.jsonl (instruction format)
# - train.jsonl (90% of data)
# - val.jsonl (10% of data)
```

### Step 3: Model Selection

Choose your base model based on your needs:

| Model | Size | VRAM Required | Speed | Quality |
|-------|------|---------------|-------|---------|
| `microsoft/phi-2` | 2.7B | 8GB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| `distilgpt2` | 82M | 2GB | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| `gpt2-medium` | 355M | 4GB | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| `microsoft/DialoGPT-medium` | 355M | 4GB | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Step 4: Fine-tuning

Fine-tune your chosen model:

```bash
# Fine-tune Phi-2 (recommended)
python finetune_model.py \
    --model microsoft/phi-2 \
    --train-file train.jsonl \
    --val-file val.jsonl \
    --output-dir fine_tuned_phi2 \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4

# Fine-tune smaller model (faster, less VRAM)
python finetune_model.py \
    --model distilgpt2 \
    --train-file train.jsonl \
    --val-file val.jsonl \
    --output-dir fine_tuned_distilgpt2 \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 5e-4
```

**Training Time Estimates:**
- Phi-2: ~2-4 hours on A100, ~6-12 hours on RTX 3090
- DistilGPT-2: ~30-60 minutes on RTX 3090

### Step 5: Inference

Use your fine-tuned model to replace the large LLM:

```bash
# Replace your original script
python inference_with_finetuned.py \
    --model-path fine_tuned_phi2 \
    --url "https://www.intel.com/content/www/us/en/products/sku/236848/intel-core-ultra-5-processor-125h-18m-cache-up-to-4-50-ghz/specifications.html" \
    --output output_finetuned.json
```

## 🔧 Configuration Options

### Memory Optimization

If you have limited VRAM, use these options:

```bash
# Use 8-bit quantization instead of 4-bit
python finetune_model.py --no-4bit --batch-size 2

# Reduce max sequence length
python finetune_model.py --max-length 1024

# Use gradient checkpointing (automatic)
```

### Training Optimization

```bash
# Increase LoRA rank for better performance
python finetune_model.py --lora-rank 128 --lora-alpha 256

# Adjust learning rate
python finetune_model.py --learning-rate 1e-4

# More epochs for smaller models
python finetune_model.py --epochs 10
```

## 📊 Performance Comparison

| Metric | Large LLM | Fine-tuned Phi-2 | Fine-tuned DistilGPT-2 |
|--------|-----------|------------------|----------------------|
| Speed | 10-30s | 1-3s | 0.5-1s |
| Cost | API fees | Free after training | Free after training |
| Privacy | External API | Local | Local |
| Customization | Limited | Full control | Full control |

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Reduce batch size: `--batch-size 2`
   - Use 8-bit: `--no-4bit`
   - Reduce sequence length: `--max-length 1024`

2. **Poor Results:**
   - Increase training epochs: `--epochs 5`
   - Collect more diverse training data
   - Try different base model

3. **Training Loss Not Decreasing:**
   - Increase learning rate: `--learning-rate 5e-4`
   - Increase LoRA rank: `--lora-rank 128`

### Validation

Test your model quality:

```bash
# Compare outputs
python inference_with_finetuned.py --model-path fine_tuned_model --url "your_test_url"
# Compare with your original LLM output
```

## 🚀 Production Deployment

### CPU Deployment

```bash
# Fine-tune with CPU (slower training, but CPU inference)
python finetune_model.py --device cpu --no-4bit

# Inference on CPU
python inference_with_finetuned.py --model-path fine_tuned_model --device cpu
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY requirements_finetune.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_finetune.txt

# Copy application code
COPY fine_tuned_model /app/model
COPY inference_with_finetuned.py /app/

WORKDIR /app
CMD ["python", "inference_with_finetuned.py", "--model-path", "model"]
```

## 📚 Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Fine-tuning](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🔄 Integration with Your Existing Code

Replace the `send_to_llm` function in your `main.py`:

```python
# Instead of:
# llm_output = send_to_llm(specs)

# Use:
from inference_with_finetuned import FineTunedHTMLConverter

converter = FineTunedHTMLConverter("fine_tuned_model")
llm_output = converter.convert_html_to_json(specs)
```

## 📈 Monitoring and Improvement

1. **Track Metrics:**
   - JSON validity rate
   - Inference speed
   - Output quality vs. original LLM

2. **Iterative Improvement:**
   - Collect more diverse training data
   - Fine-tune on domain-specific examples
   - Experiment with different base models

## 🤝 Contributing

Feel free to improve these scripts and share your results!

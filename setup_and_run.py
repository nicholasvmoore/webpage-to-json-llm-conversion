#!/usr/bin/env python3
"""
Quick setup and execution script for HTML-to-JSON fine-tuning pipeline.

This script provides an easy way to run the entire fine-tuning process.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required!")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade to Python 3.10+")
        print("   Visit: https://www.python.org/downloads/")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

# Check Python version at import time
check_python_version()

def run_command(cmd, description):
    """Run a command with error handling"""
    print(f"\n🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   ✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def setup_environment():
    """Setup Python environment for fine-tuning"""
    print("🚀 Setting up fine-tuning environment...")

    # Install requirements
    if not run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements_finetune.txt"
    ], "Installing fine-tuning dependencies"):
        return False

    print("✅ Environment setup complete!")
    return True

def collect_data(urls_file=None, output_file="training_data.jsonl"):
    """Step 1: Collect training data"""
    print("\n📊 Step 1: Data Collection")

    if not urls_file or not os.path.exists(urls_file):
        print("   Using default Intel URLs for data collection")
        urls_file = None

    cmd = [sys.executable, "data_collection.py", "--output", output_file]
    if urls_file:
        cmd.extend(["--urls-file", urls_file])

    return run_command(cmd, "Collecting training data")

def prepare_data(input_file="training_data.jsonl"):
    """Step 2: Prepare training data"""
    print("\n🔄 Step 2: Data Preparation")

    if not os.path.exists(input_file):
        print(f"   ❌ Training data file {input_file} not found!")
        return False

    return run_command([
        sys.executable, "prepare_training_data.py",
        "--input", input_file
    ], "Preparing training data")

def fine_tune_model(model_name="microsoft/phi-2", output_dir="fine_tuned_model"):
    """Step 3: Fine-tune the model"""
    print("\n🎯 Step 3: Model Fine-tuning")

    if not os.path.exists("train.jsonl") or not os.path.exists("val.jsonl"):
        print("   ❌ Training/validation files not found!")
        return False

    return run_command([
        sys.executable, "finetune_model.py",
        "--model", model_name,
        "--train-file", "train.jsonl",
        "--val-file", "val.jsonl",
        "--output-dir", output_dir,
        "--epochs", "3",
        "--batch-size", "4"
    ], f"Fine-tuning {model_name}")

def test_model(model_path="fine_tuned_model"):
    """Step 4: Test the fine-tuned model"""
    print("\n🧪 Step 4: Testing Fine-tuned Model")

    if not os.path.exists(model_path):
        print(f"   ❌ Model directory {model_path} not found!")
        return False

    return run_command([
        sys.executable, "inference_with_finetuned.py",
        "--model-path", model_path,
        "--output", "test_output.json"
    ], "Testing fine-tuned model")

def full_pipeline(urls_file=None, model_name="microsoft/phi-2", skip_setup=False):
    """Run the complete fine-tuning pipeline"""
    print("🚀 Starting Complete HTML-to-JSON Fine-tuning Pipeline")
    print("=" * 60)
    print("🔍 Running compatibility check...")

    # Run compatibility check
    result = subprocess.run([sys.executable, "check_compatibility.py"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Compatibility check failed!")
        print(result.stdout)
        return False

    print("✅ Compatibility check passed!")

    # Setup environment
    if not skip_setup:
        if not setup_environment():
            return False

    # Step 1: Data Collection
    if not collect_data(urls_file):
        return False

    # Step 2: Data Preparation
    if not prepare_data():
        return False

    # Step 3: Fine-tuning
    if not fine_tune_model(model_name):
        return False

    # Step 4: Testing
    if not test_model():
        return False

    print("\n🎉 Pipeline completed successfully!")
    print("\n📊 Summary:")
    print("   - Training data: training_data.jsonl")
    print("   - Fine-tuned model: fine_tuned_model/")
    print("   - Test results: test_output.json")
    print("\n🚀 You can now use your fine-tuned model!")

    return True

def quick_test(model_path="fine_tuned_model"):
    """Quick test of an existing fine-tuned model"""
    print("🧪 Quick Test of Fine-tuned Model")

    if not os.path.exists(model_path):
        print(f"❌ Model directory {model_path} not found!")
        print("   Please run the full pipeline first or specify the correct path")
        return False

    return test_model(model_path)

def main():
    parser = argparse.ArgumentParser(description="HTML-to-JSON Fine-tuning Pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run complete pipeline")
    full_parser.add_argument("--urls-file", help="Custom URLs file")
    full_parser.add_argument("--model", default="microsoft/phi-2",
                           choices=["microsoft/phi-2", "distilgpt2", "gpt2-medium"],
                           help="Base model to fine-tune")
    full_parser.add_argument("--skip-setup", action="store_true",
                           help="Skip environment setup")

    # Individual step commands
    subparsers.add_parser("setup", help="Setup environment only")
    subparsers.add_parser("collect", help="Data collection only")
    subparsers.add_parser("prepare", help="Data preparation only")

    tune_parser = subparsers.add_parser("tune", help="Fine-tuning only")
    tune_parser.add_argument("--model", default="microsoft/phi-2",
                           help="Base model to fine-tune")

    test_parser = subparsers.add_parser("test", help="Test model only")
    test_parser.add_argument("--model-path", default="fine_tuned_model",
                           help="Path to fine-tuned model")

    # Quick commands
    subparsers.add_parser("quick-test", help="Quick test existing model")
    subparsers.add_parser("check", help="Run compatibility check")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "full":
        success = full_pipeline(
            urls_file=getattr(args, 'urls_file', None),
            model_name=getattr(args, 'model', 'microsoft/phi-2'),
            skip_setup=getattr(args, 'skip_setup', False)
        )

    elif args.command == "setup":
        success = setup_environment()

    elif args.command == "collect":
        success = collect_data()

    elif args.command == "prepare":
        success = prepare_data()

    elif args.command == "tune":
        model_name = getattr(args, 'model', 'microsoft/phi-2')
        success = fine_tune_model(model_name)

    elif args.command == "test":
        model_path = getattr(args, 'model_path', 'fine_tuned_model')
        success = test_model(model_path)

    elif args.command == "quick-test":
        success = quick_test()

    elif args.command == "check":
        success = run_command([sys.executable, "check_compatibility.py"], "Running compatibility check")

    if success:
        print("\n✅ Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

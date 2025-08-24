#!/usr/bin/env python3
"""
Python compatibility checker for webpage-to-json-llm-conversion.

This script verifies that your Python environment meets the requirements
for running the fine-tuning pipeline.
"""

import sys
import platform
import subprocess
import importlib.util

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check")
    print(f"   Current version: {sys.version}")

    if sys.version_info >= (3, 10):
        print("   ✅ Compatible (Python 3.10+)")
        return True
    else:
        print("   ❌ Incompatible (Python 3.10+ required)")
        print("   Please upgrade to Python 3.10 or higher:")
        print("   https://www.python.org/downloads/")
        return False

def check_pip_packages():
    """Check if required packages can be imported"""
    print("\n📦 Package Compatibility Check")

    required_packages = {
        "torch": "PyTorch for model training",
        "transformers": "Hugging Face transformers",
        "beautifulsoup4": "HTML parsing",
        "requests": "HTTP requests",
        "numpy": "Numerical computations",
        "json": "JSON handling (built-in)",
        "os": "OS interface (built-in)",
        "sys": "System interface (built-in)"
    }

    missing_packages = []

    for package, description in required_packages.items():
        try:
            if package in ["json", "os", "sys"]:
                importlib.import_module(package)
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}: {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}: {description}")

    return len(missing_packages) == 0

def check_system_info():
    """Display system information"""
    print("\n💻 System Information")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Platform: {platform.platform()}")

    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   CUDA: Available (Devices: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   CUDA: Not available")
    except ImportError:
        print("   CUDA: PyTorch not installed")

def check_memory():
    """Check available memory (basic check)"""
    print("\n🧠 Memory Check")

    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print(".1f")

        if available_gb < 8:
            print("   ⚠️  Consider upgrading RAM for better performance")
        else:
            print("   ✅ Sufficient memory available")
    except ImportError:
        print("   ⚠️  psutil not available - install with: pip install psutil")

def main():
    """Main compatibility check"""
    print("🔍 Compatibility Check for webpage-to-json-llm-conversion")
    print("=" * 60)

    all_good = True

    # Check Python version
    if not check_python_version():
        all_good = False

    # Check packages
    if not check_pip_packages():
        all_good = False

    # System info
    check_system_info()
    check_memory()

    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Your system is compatible!")
        print("\nNext steps:")
        print("1. Run: python setup_and_run.py full")
        print("2. Or follow the fine-tuning guide in FINETUNING_README.md")
    else:
        print("❌ Compatibility issues found!")
        print("   Please resolve the issues above before proceeding.")

    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main())

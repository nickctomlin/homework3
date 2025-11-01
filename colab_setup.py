"""
Helper script to set up the project in Google Colab.
Run this in a Colab cell to install dependencies and set up the environment.
"""
import sys
import subprocess
import os

def setup_colab():
    """Install dependencies and set up the project structure for Colab"""
    print("Installing dependencies...")
    
    # Install required packages
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "fire>=0.7.0",
        "lightning>=2.5.0", 
        "torch>=2.6.0",
        "transformers==4.52.4",
        "numpy>=2.2.1",
        "Pillow>=11.1.0",
        "tqdm>=4.66.0",
        "accelerate>=0.26.0",
        "tensorboard>=2.15.0",
        "peft>=0.15.0",
    ])
    
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Upload your homework folder to Colab (or use git clone)")
    print("2. Change directory: %cd /content/homework3_v3")
    print("3. Run your training: !python -m homework.sft train")

if __name__ == "__main__":
    setup_colab()


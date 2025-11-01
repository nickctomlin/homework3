# Google Colab Setup Guide

This guide shows you how to use Google Colab's free GPU while coding in your local editor (Cursor).

## Option 1: Google Drive Sync (Recommended)

This is the easiest approach - sync your code via Google Drive.

### Step 1: Upload Project to Google Drive

1. Zip your `homework3_v3` folder
2. Upload to Google Drive (e.g., `MyDrive/homework3_v3.zip`)
3. Or upload the folder directly

### Step 2: Setup Colab Notebook

Create a new Colab notebook and run this in the first cell:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip and navigate to project
import zipfile
import os

# If you uploaded a zip file
zip_path = '/content/drive/MyDrive/homework3_v3.zip'
extract_path = '/content'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Or if you uploaded a folder
project_path = '/content/drive/MyDrive/homework3_v3'

# Change to project directory
import os
os.chdir('/content/homework3_v3')  # Adjust path as needed

# Install dependencies
!pip install -q fire>=0.7.0 lightning>=2.5.0 torch>=2.6.0 transformers==4.52.4 numpy>=2.2.1 Pillow>=11.1.0 tqdm>=4.66.0 accelerate>=0.26.0 tensorboard>=2.15.0 peft>=0.15.0

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 3: Run Training

After setup, run your training commands:

```python
# For supervised fine-tuning
!python -m homework.sft train

# For data generation
!python -m homework.datagen generate_dataset data/rft.json

# For RFT training
!python -m homework.rft train
```

### Step 4: Sync Changes Back

1. Edit code locally in Cursor
2. Re-upload changed files to Drive (or use Google Drive sync)
3. In Colab, re-run the setup cell if needed, or just import your updated modules

**Tip**: You can also use `git` in Colab to sync from a repository:

```python
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo
```

## Option 2: GitHub Sync (Best for Version Control)

### Step 1: Push to GitHub

1. Initialize git repo locally:
```bash
git init
git add .
git commit -m "Initial commit"
```

2. Create a GitHub repo and push:
```bash
git remote add origin https://github.com/yourusername/homework3.git
git push -u origin main
```

### Step 2: Clone in Colab

```python
# Clone your repo
!git clone https://github.com/yourusername/homework3.git
%cd homework3

# Install dependencies
!pip install -q fire>=0.7.0 lightning>=2.5.0 torch>=2.6.0 transformers==4.52.4 numpy>=2.2.1 Pillow>=11.1.0 tqdm>=4.66.0 accelerate>=0.26.0 tensorboard>=2.15.0 peft>=0.15.0

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Step 3: Workflow

1. **Develop locally** in Cursor
2. **Commit and push** changes:
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```
3. **Pull in Colab**:
   ```python
   !git pull
   ```
4. **Run training** in Colab

## Option 3: Hybrid Approach (Local Testing + Colab Training)

### For Parts 1-2 (No GPU needed):
- Test `base_llm.py` and `cot.py` locally on CPU
- These can run without GPU

### For Parts 3-4 (GPU needed):
- Develop code locally
- Test structure with small data locally
- Upload to Colab for full training

## Colab Workflow Template

Here's a complete Colab notebook template:

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project
import os
os.chdir('/content/drive/MyDrive/homework3_v3')  # Adjust path

# Install dependencies
!pip install -q -r requirements.txt

# Verify setup
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 2: Run training
!python -m homework.sft train

# Cell 3: Check results
!ls -lh homework/sft_model/

# Cell 4: Download results (if needed)
from google.colab import files
!zip -r results.zip homework/sft_model/
files.download('results.zip')
```

## Tips

1. **Save Checkpoints**: Colab sessions can disconnect. Make sure your code saves checkpoints regularly
2. **Session Timeout**: Free Colab sessions timeout after ~12 hours. Consider saving intermediate results
3. **Download Models**: After training, download your LoRA adapters:
   ```python
   from google.colab import files
   files.download('homework/sft_model/adapter_config.json')
   files.download('homework/sft_model/adapter_model.bin')
   ```
4. **Monitor Training**: Use TensorBoard:
   ```python
   !tensorboard --logdir=output_dir --port=6006
   # Then use Colab's ngrok or port forwarding to view
   ```

## Alternative: VS Code Remote with Colab

This is more complex but allows direct integration:

1. Install "Remote - SSH" extension in VS Code/Cursor
2. Set up Colab SSH (requires Colab Pro/Pro+)
3. Connect directly to Colab runtime

However, the GitHub/Drive sync method is much simpler and works well.

## Quick Start Commands

```python
# One-time setup
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/homework3_v3
!pip install -q -r requirements.txt

# Training commands
!python -m homework.sft train
!python -m homework.datagen generate_dataset data/rft.json  
!python -m homework.rft train

# Testing
!python -m homework.base_llm test
!python -m homework.cot test
```


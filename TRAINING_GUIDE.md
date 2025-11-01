# Training and Testing Guide

This guide explains how to train and test all parts of the homework.

## Quick Overview

- **Parts 1-2** (base_llm, cot): Can test locally (CPU is fine)
- **Parts 3-4** (sft, rft, datagen): Need GPU - use Google Colab

---

## Part 1: Test Base LLM (Local - No GPU needed)

Test that your `batched_generate` implementation works:

```bash
python -m homework.base_llm test
```

**Expected output**: Should print some generated text (may be nonsensical, but should not crash).

---

## Part 2: Test Chain-of-Thought (Local - No GPU needed)

Test that your CoT prompt formatting works:

```bash
python -m homework.cot test
```

**Expected output**: Should show accuracy and answer_rate. 
- Target: `accuracy >= 0.5` and `answer_rate >= 0.85`
- If lower, try adjusting the example or instructions in `format_prompt()`

---

## Part 3: Supervised Fine-Tuning (GPU Required - Use Colab)

### Step 1: Set up Colab
1. Open [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → GPU (T4)**
3. Mount Drive and clone your repo (see Colab setup below)

### Step 2: Train the model
```bash
python -m homework.sft train
```

**What it does**:
- Loads the train dataset
- Creates LoRA adapter (r=8, ~20MB)
- Trains for 3 epochs
- Saves model to `homework/sft_model/`

**Expected time**: ~10-30 minutes depending on GPU

### Step 3: Test the trained model
```bash
python -m homework.sft test homework/sft_model
```

**Expected output**: Should show improved accuracy compared to Part 2.

---

## Part 4: RFT Dataset Generation + Training (GPU Required - Use Colab)

### Step 1: Generate RFT Dataset
First, generate the dataset with chain-of-thought reasoning:

```bash
python -m homework.datagen generate_dataset data/rft.json oversample=10 temperature=0.6
```

**What it does**:
- Uses CoTModel to generate 10 different completions per question
- Filters to keep only correct answers
- Saves to `data/rft.json`

**Expected time**: ~30-60 minutes (depends on dataset size)
**Expected result**: Should generate 90%+ of examples successfully

### Step 2: Train RFT Model
After generating the dataset, train the RFT model:

```bash
python -m homework.rft train
```

**What it does**:
- Loads the RFT dataset from `data/rft.json`
- Creates LoRA adapter (r=16, slightly larger for better performance)
- Trains on question + reasoning pairs
- Saves model to `homework/rft_model/`

**Expected time**: ~10-30 minutes

### Step 3: Test the trained model
```bash
python -m homework.rft test homework/rft_model
```

---

## Colab Setup (Quick Steps)

### Option 1: Google Drive + Upload
1. **Upload your project** to Google Drive (zip or folder)
2. **In Colab**, run:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/homework3_v3')  # Update path

!pip install -q -r requirements.txt
```

### Option 2: GitHub (Recommended)
1. **Push your code to GitHub**:
```bash
git init
git add .
git commit -m "Homework 3 implementation"
# Create repo on GitHub and push
```

2. **In Colab**, run:
```python
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo
!pip install -q -r requirements.txt
```

### Verify GPU in Colab
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## Complete Training Workflow (Colab)

Here's a complete notebook workflow:

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/homework3_v3')  # UPDATE PATH

!pip install -q -r requirements.txt

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 2: Test Part 1 (optional)
!python -m homework.base_llm test

# Cell 3: Test Part 2 (optional)
!python -m homework.cot test

# Cell 4: Train Part 3 - SFT
!python -m homework.sft train

# Cell 5: Test SFT model
!python -m homework.sft test homework/sft_model

# Cell 6: Generate RFT Dataset
!python -m homework.datagen generate_dataset data/rft.json oversample=10 temperature=0.6

# Cell 7: Train Part 4 - RFT
!python -m homework.rft train

# Cell 8: Test RFT model
!python -m homework.rft test homework/rft_model

# Cell 9: Download models (optional)
from google.colab import files
import zipfile
with zipfile.ZipFile('models.zip', 'w') as z:
    z.write('homework/sft_model')
    z.write('homework/rft_model')
files.download('models.zip')
```

---

## Testing Locally (Parts 1-2 Only)

If you want to test Parts 1-2 locally without GPU:

```bash
# Make sure you're in the project directory
cd homework3_v3

# Activate conda environment
conda activate advances_in_deeplearning

# Test Part 1
python -m homework.base_llm test

# Test Part 2
python -m homework.cot test
```

**Note**: Parts 3-4 require GPU. Use Colab for those.

---

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `per_device_train_batch_size` in training args (e.g., 16 instead of 32)
- Reduce LoRA rank `r` (e.g., 4 instead of 8)

### Low Accuracy
- For Part 2 (CoT): Adjust the example in `format_prompt()` to be more helpful
- For Part 3 (SFT): Train for more epochs (but max 5 as per README)
- For Part 4 (RFT): Increase `oversample` in datagen to get better reasoning examples

### Dataset Generation Takes Too Long
- Reduce `oversample` parameter (e.g., 5 instead of 10)
- This may reduce quality but will be faster

### Model Not Saving
- Check that `homework/sft_model/` and `homework/rft_model/` directories exist
- The training code should create them automatically

---

## File Structure After Training

After completing all training, you should have:

```
homework3_v3/
├── homework/
│   ├── sft_model/          # LoRA adapter (Part 3)
│   │   ├── adapter_config.json
│   │   └── adapter_model.bin
│   └── rft_model/          # LoRA adapter (Part 4)
│       ├── adapter_config.json
│       └── adapter_model.bin
├── data/
│   └── rft.json            # Generated RFT dataset
└── ...
```

---

## Submission Checklist

Before submitting:
1. ✅ All code implemented
2. ✅ Models trained and saved
3. ✅ Models can be loaded (test with `test` command)
4. ✅ Delete any extra checkpoints to keep size < 50MB
5. ✅ Run grader: `python3 -m grader [YOUR_UT_ID].zip`

Good luck! 🚀


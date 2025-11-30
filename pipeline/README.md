# SLM-ArchBench — Quick Start Guide

## Prerequisites

* Python 3.10+
* **NVIDIA GPU with FlashAttention support (required)**
  - Recommended: RTX 3090, A100, H100, or newer
  - Minimum: GPU with compute capability 8.0+ (Ampere architecture or newer)
  - CUDA 11.8+ drivers installed
* A valid Hugging Face token with **READ permission**

> ⚠️ **Important**: This pipeline requires FlashAttention for Phi-3 Small model. Running without a compatible NVIDIA GPU will result in errors.

---

## Installation

### 1. Copy Dataset Folder

**Copy the `datasets` folder into the `pipeline` folder:**

```
project/
├── datasets/           ← Copy this folder
└── pipeline/
    ├── datasets/       ← Paste here (same level as pipeline.py)
    ├── pipeline.py
    └── requirements.txt
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Set Hugging Face Token

**Your token must have READ permission.** Get it from [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens).

### macOS / Linux
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Windows PowerShell
```powershell
$env:HF_TOKEN="your_huggingface_token_here"
```

---

## Set Gemini API Key

Get it from [Google AI Studio → Create API Key](https://aistudio.google.com/api-keys).

### macOS / Linux
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Windows PowerShell
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## Run the Pipeline

```bash
python pipeline.py
```

Or run installation + execution together:
```bash
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
python pipeline.py
```

---

## Running on Lightning AI

1. Upload your project.
2. Add `HF_TOKEN` in Environment Variables (with READ permission).
3. Add `GEMINI_API_KEY` in Environment Variables.
3. **Select GPU runtime** (required for FlashAttention).
4. Run:
```bash
pip install -r requirements.txt && python pipeline.py
```

---

That's it — clean and fast to follow.
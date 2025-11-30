# SLM-ArchBench — Quick Start Guide

## Prerequisites

* Python 3.10+
* GPU recommended (CUDA drivers installed if applicable)
* A valid Hugging Face token (`HF_TOKEN`)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Set Hugging Face Token

### macOS / Linux

```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Windows PowerShell

```powershell
$env:HF_TOKEN="your_huggingface_token_here"
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
python pipeline.py
```

---

## Running on Lightning AI

1. Upload your project.
2. Add `HF_TOKEN` in Environment Variables.
3. (Optional) Select GPU runtime.
4. Run:

```bash
pip install -r requirements.txt && python pipeline.py
```

---

That’s it — clean and fast to follow.

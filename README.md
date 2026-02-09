# SLM-ArchBench — Quick Start Guide

## Prerequisites

* Python 3.10+
* NVIDIA GPU (recommended for fast fine-tuning)
* A valid Gemini API key for LLM-as-a-Judge evaluation
* A valid Hugging Face token with READ permission

---

## Installation

### Navigate to Pipeline Directory

```bash
cd pipeline
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter an `externally-managed-environment` error, it's recommended to use a virtual environment:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate      # On Linux/Mac
> # or
> .venv\Scripts\activate         # On Windows
> pip install -r requirements.txt
> ```

---

## Environment Variables

### Required Variables

#### 1. Hugging Face Token

**Your token must have READ permission.** Get it from [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens).

| OS | Command |
|---|---|
| macOS / Linux | `export HF_TOKEN="your_huggingface_token_here"` |
| Windows PowerShell | `$env:HF_TOKEN="your_huggingface_token_here"` |

#### 2. Gemini API Key

Get it from [Google AI Studio → Create API Key](https://aistudio.google.com/apikey).

> 🚨 **CRITICAL: Paid API Key Required for Full Evaluation**
> 
> Google has significantly reduced free-tier limits:
> - ~~**250 requests/day**~~ → **20 requests/day** per key
> - ~~**10 requests/minute**~~ → **5 requests/minute** per key
> 
> **Full evaluation requires ~570 Gemini API calls**, which is **impossible with free tier** (would take 29+ days with 1 key).
> 
> ⚠️ **Free-tier limitations:**
> - Even with 4 API keys rotating (as previous version): only **80 requests/day** → requires **8+ days** to complete
> - **Quota may be reduced unexpectedly** by Google without notice
> - **Rate limits will cause incomplete evaluations**, resulting in missing or corrupted metrics
> - If interrupted, you must re-run the entire evaluation, wasting time and compute
> 
> **✅ Solution: Use a paid Gemini API key** (~$5-15 per full run, completes in 10-15 minutes)

**Free-tier setup (NOT recommended - takes 8+ days with 4 keys):**

The free tier limits each API key to ~~**250 requests/day**~~ **20 requests/day** and ~~**10 requests/minute**~~ **5 requests/minute**. To increase throughput, you can use up to 4 API keys that rotate automatically.

| OS | Commands |
|---|---|
| macOS / Linux | `export GEMINI_API_KEY_1="your_first_api_key"`<br>`export GEMINI_API_KEY_2="your_second_api_key"`<br>`export GEMINI_API_KEY_3="your_third_api_key"`<br>`export GEMINI_API_KEY_4="your_fourth_api_key"` |
| Windows PowerShell | `$env:GEMINI_API_KEY_1="your_first_api_key"`<br>`$env:GEMINI_API_KEY_2="your_second_api_key"`<br>`$env:GEMINI_API_KEY_3="your_third_api_key"`<br>`$env:GEMINI_API_KEY_4="your_fourth_api_key"` |

> **Note:** You only need to set the keys you have.  The system will automatically use all available keys.

**Paid API setup (recommended):**

| OS | Command |
|---|---|
| macOS / Linux | `export GEMINI_API_KEY_1="your_paid_api_key"` |
| Windows PowerShell | `$env:GEMINI_API_KEY_1="your_paid_api_key"` |

### Optional Variables

| Variable | Description | Default | Note |
|---|---|---|---|
| `FEW_SHOT_K` | Number of examples for few-shot evaluation | `2` | Min value |

**Example:**
```bash
# macOS / Linux
export FEW_SHOT_K=5

# Windows PowerShell
$env:FEW_SHOT_K=5
```

---

## Configuration

You can customize the pipeline by editing `config.py`:

### Evaluation Modes

```python
# Run all 3 modes sequentially (default)
EVAL_MODES = ["zero_shot", "few_shot", "fine_tune"]

# Run only specific modes
EVAL_MODES = ["zero_shot"]  # Zero-shot only
EVAL_MODES = ["fine_tune"]  # Fine-tuning only
```

### Model Selection

```python
MODEL_CANDIDATES = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    # Add or remove models as needed
}
```

### Training Hyperparameters

```python
TRAINING_ARGS = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "learning_rate":  2e-4,
    # ...  see config.py for full options
}
```

---

## Run the Pipeline

```bash
python pipeline.py
```

### One-liner Setup & Run

**macOS / Linux:**
```bash
cd pipeline && \
pip install -r requirements.txt && \
export HF_TOKEN="your_huggingface_token_here" && \
export GEMINI_API_KEY_1="your_gemini_api_key_here" && \
python pipeline.py
```

**Windows PowerShell:**
```powershell
cd pipeline; `
pip install -r requirements.txt; `
$env:HF_TOKEN="your_huggingface_token_here"; `
$env:GEMINI_API_KEY_1="your_gemini_api_key_here"; `
python pipeline.py
```

---

## Output Structure

Results are organized by model and evaluation mode:

```
results/
├── llama-3.2-1b/
│   ├── zeroshot_archai-adr_llama-3.2-1b_detailed.json
│   ├── zeroshot_archai-adr_llama-3.2-1b_summary.json
│   ├── fewshot_archai-adr_llama-3.2-1b_detailed.json
│   ├── fewshot_archai-adr_llama-3.2-1b_summary.json
│   ├── finetuned_archai-adr_llama-3.2-1b_detailed.json
│   └── finetuned_archai-adr_llama-3.2-1b_summary.json
├── phi-3-mini/
│   └── ... 
├── all_runs_summary.csv
├── zero_shot_summary.csv
├── few_shot_summary.csv
└── fine_tune_summary.csv
```

---

## Running on Lightning AI

1. Upload your project.
2. Add environment variables:
   - `HF_TOKEN` (with READ permission)
   - `GEMINI_API_KEY_1` (paid key recommended)
   - (Optional) `FEW_SHOT_K`
3. **Select GPU runtime** (recommended:  A10G or higher).
4. Run:
```bash
cd pipeline && pip install -r requirements.txt && python pipeline.py
```

---

## Evaluation Modes

| Mode | Description | Training Required |
|---|---|---|
| `zero_shot` | Evaluate pre-trained model without examples | ❌ No |
| `few_shot` | Evaluate with K examples prepended to prompt | ❌ No |
| `fine_tune` | Fine-tune with LoRA, then evaluate | ✅ Yes |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Reduce `per_device_train_batch_size` in `config.py` |
| `Rate limit exceeded` (Gemini) | Use a paid API key, or wait and retry |
| `Quota exceeded unexpectedly` | Switch to a paid Gemini API key |
| `Model access denied` | Ensure HF_TOKEN has READ permission and model access is granted |
| `Context length exceeded` | Reduce `FEW_SHOT_K` for few-shot mode |
| `Incomplete evaluation results` | Check if Gemini quota was exhausted mid-run; re-run with paid key |

---

## Cost Estimation

### Gemini API (for Compliance Score)

| Setup | Estimated Cost | Reliability | Time to Complete |
|---|---|---|---|
| Free tier (4 keys) | $0 | ❌ Not viable | **8+ days** |
| Pay-as-you-go | ~$5-15 per full run | ✅ Reliable | **10-15 minutes** |

> **Important:** A full evaluation run (10 models × 3 modes × 19 samples) requires approximately **570 Gemini API calls**. With ~~free-tier limits of 250 requests/day~~ **new limits of 20 requests/day** per key, you would need ~~at least 3 days~~ **29 days with 1 key or 8 days with 4 keys** to complete using free keys—assuming no quota reductions occur. **Free tier is no longer practical for this benchmark.**

---

## Result Variability

Due to LLM-based judging, scores may vary slightly between runs.  Variations are typically small and do not affect overall conclusions. 
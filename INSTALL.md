# SLM-ArchBench — Install & Run

This benchmark can take over 1 hour when running the full pipeline. For artifact evaluation, use the `example-pipeline`.

## Prerequisites

* Python 3.10+
* NVIDIA GPU L4 or higher (recommended for fast fine-tuning)
* A valid Gemini API key for LLM-as-a-Judge evaluation
* A valid Hugging Face token with READ permission

---

## Recommended: Example Pipeline (Artifact Evaluation)

### 1) Install dependencies

```bash
cd example-pipeline
pip install -r requirements.txt
```

> **Note:** If you hit an `externally-managed-environment` error, use a virtual environment:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate      # On Linux/Mac
> # or
> .venv\Scripts\activate         # On Windows
> pip install -r requirements.txt
> ```

### 2) Set environment variables

**Hugging Face Token:**

| OS | Command |
|---|---|
| macOS / Linux | `export HF_TOKEN="your_huggingface_token_here"` |
| Windows PowerShell | `$env:HF_TOKEN="your_huggingface_token_here"` |

**Gemini API Key:**

Since the example pipeline needs only ~3 API calls, a single key is sufficient:

| OS | Command |
|---|---|
| macOS / Linux | `export GEMINI_API_KEY_1="your_first_api_key"` <br>`export GEMINI_API_KEY_2="your_second_api_key"`<br>`export GEMINI_API_KEY_3="your_third_api_key"` |
| Windows PowerShell | `$env:GEMINI_API_KEY_1="your_first_api_key"` `$env:GEMINI_API_KEY_2="your_second_api_key"`<br>`$env:GEMINI_API_KEY_3="your_third_api_key"` |

### 3) Run

```bash
python pipeline.py
```

---

## Full Pipeline (Longer Runtime)

Use this only if you need full-scale results. Runtime can exceed 1 hour depending on model count and GPU.

```bash
cd pipeline
pip install -r requirements.txt
```

Set the same `HF_TOKEN` and `GEMINI_API_KEY_1` (or multiple keys for faster throughput) environment variables as above, then run:
    
```bash
python pipeline.py
```
> ⚠️ **Note:** Full pipeline requires ~570 Gemini API calls. See [Cost Estimation](#cost-estimation) for paid key recommendation.

### Configuration (Optional)

**Evaluation Modes:** Customize the pipeline by editing `config.py`:
```python
# Run all 3 modes (default)
EVAL_MODES = ["zero_shot", "few_shot", "fine_tune"]

# Or run specific modes
EVAL_MODES = ["zero_shot"]  # Zero-shot only
```

**Few-shot Examples:**
| Variable | Description | Default | Note |
|---|---|---|---|
| `FEW_SHOT_K` | Number of examples for few-shot evaluation | `2` | Min value |

***Example:***
```bash
# macOS / Linux
export FEW_SHOT_K=5

# Windows PowerShell
$env:FEW_SHOT_K=5
```

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
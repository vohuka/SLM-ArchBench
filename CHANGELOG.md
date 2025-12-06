# Changelog

All notable changes to this project will be documented in this file.

---

## [3.1] - 2025-12-05

### Highlights
- Added **smart resume** for fine-tuning mode: skip training if model already exists, only re-run validation.
- Fixed **METEOR metric** always returning 0 due to missing NLTK data. 
- Added **multi API key rotation** for Gemini to handle rate limits. 
- Added **rate limit delay** between compliance score requests. 

### Details
- If `models/` exists but `results/` is missing → load saved model and validate only (no retraining). 
- Added `load_finetuned_model()` function to support loading LoRA adapters. 
- Model is now saved before validation to enable resume if validation fails. 
- Added `_ensure_nltk_data()` function to auto-download required NLTK resources: `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`. 
- Fixed METEOR computation to use proper tokenization with `word_tokenize()` instead of raw strings.
- Added fallback to simple `split()` if tokenization fails.
- Suppressed BERTScore warnings about uninitialized RobertaModel weights. 
- Support multiple Gemini API keys via `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc. 
- Rotate API keys automatically when rate limit is hit.
- Added `GEMINI_RATE_LIMIT_DELAY = 5s` between requests to respect 10 RPM limit.
---
## [3.0] - 2025-12-05

### Highlights
- Added support for **three evaluation modes**: Zero-shot, Few-shot, and Fine-tuning, all running sequentially in a single pipeline execution. 
- Restructured output folder organization: results are now grouped by model name for better organization.
- Added mode-specific prefixes to output files (`zeroshot_`, `fewshot_`, `finetuned_`) for clear identification.
- Optimized evaluation pipeline to skip fine-tuning when running zero-shot or few-shot modes. 

### Details

#### New Features
- **Multi-mode evaluation pipeline**: The pipeline now automatically runs all three evaluation modes sequentially:
  1. `zero_shot`: Evaluates the pre-trained model without any training or examples
  2. `few_shot`: Evaluates with K examples prepended to each prompt (configurable via `FEW_SHOT_K`)
  3. `fine_tune`: Performs LoRA fine-tuning before evaluation
- **Smart mode handling**: Zero-shot and few-shot modes skip the training phase entirely, saving computational resources. 
- **Resume capability**: Each mode checks for existing results and skips if already completed.

#### Output Structure Changes
- Results are now organized by model in subdirectories:

---

## [2.1] - 2025-12-04
Highlights
- Replaced model `phi-3-small` with `owen-2.5-3b`.
- Switched from OpenAI GPT-4 to Gemini 2.5 Flash to balance speed, accuracy, and cost
- Refactored repository folder structure to minimize configuration steps required for execution.
- Added this changelog file to keep track of releases and important changes.

Details
- Model swap: `phi-3-small` → `owen-2.5-3b`. This change was made to avoid environment-specific dependencies (e.g., FlashAttention) and to provide a consistently usable model across setups.
- API Swap: OpenAI GPT-4 → Google Gemini 2.5 Flash. This provides a better trade-off between inference speed, output quality, and operational cost.
- Integrated the datasets folder into the project structure to allow the pipeline to directly access data.
- Documentation: initial changelog added and README notes updated to reflect new model and folder layout.

---

## [2.0] - 2025-12-01
Highlights
- Refactored the original single-file pipeline into multiple, smaller Python modules for each task.

Details
- Broke the monolithic `pipeline.py` into focused modules:
  - `pipeline.py` (orchestrator)
  - `models.py` (model & tokenizer loading)
  - `training.py` (training loop and Trainer setup)
  - `evaluation.py` (evaluation and metric collection)
  - `preprocessing.py` (dataset preprocessing)
  - `metrics.py`, `utils.py` (metrics and helpers)
- Benefits:
  - Easier maintenance and testing
  - Clearer responsibilities per file
  - Faster onboarding for contributors
- Migration notes:
  - Ensure imports/paths are updated if you run older scripts that expect the one-file pipeline.

---

## [1.0] - 2025-11-30
Initial release
- The pipeline was implemented as a single Python file to provide a minimal, working end-to-end baseline.
- Included basic training, evaluation, and generation flow for architecture ADR tasks.
- Served as the foundation for modular refactors in later versions.

---

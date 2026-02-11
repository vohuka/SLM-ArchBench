# Changelog

All notable changes to this project will be documented in this file.

---
## [3.4] - 2026-02-11

### Highlights
- **Added visualization scripts**: Created Python scripts to generate bar charts for compliance, diversity, and F1 scores.
- **Added consolidated results**: Added `all_runs_summary.csv` containing aggregated evaluation results across all models and evaluation modes.
- **Code cleanup**: Removed deprecated pipeline files from both `pipeline/` and `example-pipeline/` directories.

### Details
- **result_visualization/** directory:
  - Added `draw_barchart_compilance.py` to visualize compliance scores across models
  - Added `draw_barchart_diversity.py` to visualize diversity scores across models  
  - Added `draw_barchart_f1.py` to visualize F1 scores across models
  - All scripts now read from `../result/all_runs_summary.csv` using relative paths
- **result/** directory:
  - Added `all_runs_summary.csv` with comprehensive metrics for all 10 models across 3 evaluation modes (zero-shot, few-shot, fine-tune)
  - Includes ROUGE, BLEU, METEOR, BERTScore, diversity, compliance, and Ripple metrics
- **Deprecated file removal**: Deleted `[deprecated]pipeline.py` files that were replaced in version 2.0 modular refactor.

---
## [3.3] - 2026-02-09

### Critical Updates
- **Updated README for new Gemini API free-tier limits**: Google reduced free tier from 250 req/day to **20 req/day** and from 10 req/min to **5 req/min**.
- **Added strong warning**: Free tier is no longer viable for full evaluation (requires 8+ days with 4 keys vs 10-15 minutes with paid key).
- **Renamed README.md → INSTALL.md**: Separated installation/setup instructions into dedicated file. Future README.md will focus on showcasing results and project overview.

### Details
- **INSTALL.md** (previously README.md): 
  - Renamed to better reflect its purpose as installation and setup guide
  - Updated GPU requirement to specify L4 or higher
  - Updated Gemini API section with strikethrough for old limits and prominent warnings about impracticality of free tier
- **Documentation restructure**: README.md will be recreated later to showcase benchmark results, methodology, and project highlights.

---
## [3.2] - 2025-12-07

### Highlights
- **Fixed few-shot evaluation**: Now uses `MODEL_MAX_TOKEN` per model instead of hardcoded 1024, preventing prompt truncation. 
- **Configurable few-shot examples**: Added `FEW_SHOT_FIXED_INDICES` to specify exact training samples for few-shot prompts.
- **Reduced few-shot K**: Changed default `FEW_SHOT_K` from 3 to 2 for faster inference. 

### Details

#### Few-shot Prompt Fix
- **evaluation.py**: Changed `max_length` in tokenizer from hardcoded `1024` to `MODEL_MAX_TOKEN[model_key]` to support full few-shot prompts without truncation. 

#### Configurable Few-shot Examples
- **config. py**: Added `FEW_SHOT_FIXED_INDICES = [50, 72]` to allow specifying which training samples to use as few-shot examples. 
- **evaluation.py**: Refactored `build_few_shot_prefix()` function:
  - Uses fixed indices from config for the first K examples.
  - If `k > len(fixed_indices)`: adds random samples for the rest.
  - Validates indices exist in dataframe before accessing.

#### Performance Optimizations
- **config.py**: Changed `FEW_SHOT_K` default from 3 to 2 for faster evaluation. 
- **evaluation.py**: Changed `num_return_sequences` from 5 to 3 to reduce generation time while still computing diversity score. 

#### Code Quality
- **evaluation.py**: Added import for `FEW_SHOT_FIXED_INDICES` from config.
- **evaluation.py**: Added logging to show few-shot configuration: `[INFO] Built few-shot prefix with k={k}, fixed_indices={indices}`.
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


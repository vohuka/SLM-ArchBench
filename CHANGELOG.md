# Changelog

All notable changes to this project will be documented in this file.

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

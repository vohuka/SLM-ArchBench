"""
Configuration for models, datasets, and hyperparameters.
"""
import os

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CANDIDATES = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct"
}

MODEL_MAX_TOKEN = {
    "llama-3.2-1b": 131072
}
# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASETS_CONFIG = [
    {"name": "ArchAI-ADR", "path": "datasets/ArchAI-ADR.csv"},
]

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

USE_4BIT = True
#Based on model config
MAX_LENGTH = 1024
TEST_SIZE = 0.2
RANDOM_SEED = 42

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": "all-linear",
}

# Training Arguments
TRAINING_ARGS = {
    "num_train_epochs": 10,           
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,               
    "weight_decay": 0.01,
    "logging_steps": 5,               
    "eval_strategy": "epoch",          
    "save_strategy": "epoch",          
    "save_total_limit": 3,             
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss"
}

# Generation Configuration
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.92,
    "top_k": 50,
    "num_return_sequences": 5,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
    "early_stopping": False 
}

# Few-shot configuration
FEW_SHOT_K = int(os.environ.get("FEW_SHOT_K", 2))

# Path to few-shot prompt prefix file (contains golden examples)
FEW_SHOT_PROMPT_PATH = "few_shot_prompt.txt"

# Number of golden examples in the prompt file
FEW_SHOT_GOLDEN_COUNT = 2

# Run all 3 modes sequentially (default)
EVAL_MODES = ["zero_shot", "few_shot", "fine_tune"]

# API Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", None)
JUDGE_MODEL = "gemini-2.5-flash" 

GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1", None),
    os.environ.get("GEMINI_API_KEY_2", None),
    os.environ.get("GEMINI_API_KEY_3", None),
    os.environ.get("GEMINI_API_KEY_4", None)
]

# Filter None
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]
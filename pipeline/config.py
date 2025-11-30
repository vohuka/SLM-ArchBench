"""
Configuration for models, datasets, and hyperparameters.
"""
import os

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CANDIDATES = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "olmo-2-1b": "allenai/OLMo-2-0425-1B-SFT",
    "olmo-2-7b": "allenai/OLMo-2-1124-7B-SFT",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "vaultgemma-1b": "google/vaultgemma-1b",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
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
    "metric_for_best_model": "loss",
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

# API Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
JUDGE_MODEL = "gpt-4o-mini"
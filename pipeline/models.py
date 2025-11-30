"""
Model and tokenizer loading with LoRA.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from config import USE_4BIT, LORA_CONFIG, MAX_LENGTH


def load_tokenizer_and_model(model_id: str):
    """Load tokenizer and model with LoRA for 4-bit training."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        use_fast=True, 
        padding_side="right",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
    model = None

    if USE_4BIT:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map=device_map, 
                quantization_config=bnb_config,
                trust_remote_code=True
            )
            
            # Prepare model for LoRA
            model = prepare_model_for_kbit_training(model)
            
            peft_config = LoraConfig(
                r=LORA_CONFIG["r"],
                lora_alpha=LORA_CONFIG["lora_alpha"],
                lora_dropout=LORA_CONFIG["lora_dropout"],
                bias=LORA_CONFIG["bias"],
                task_type=TaskType.CAUSAL_LM,
                target_modules=LORA_CONFIG["target_modules"],
            )
            model = get_peft_model(model, peft_config)
            print("[INFO] LoRA configuration applied:")
            model.print_trainable_parameters()
            
        except ImportError:
            print("[WARN] bitsandbytes not available, loading in float16.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map=device_map, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device_map, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    return tokenizer, model


def build_tokenize_function(tokenizer):
    """Return a tokenize function bound to a specific tokenizer."""
    def tokenize_function(example):
        full_text = example["prompt"] + "\n" + example["target"]
        tokenized = tokenizer(
            full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    return tokenize_function
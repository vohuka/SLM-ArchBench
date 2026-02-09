"""
Training loop logic.
Supports three modes: zero_shot, few_shot, fine_tune
"""
import os
import json
import time
import torch
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftModel
from config import TRAINING_ARGS, TEST_SIZE, RANDOM_SEED
from models import build_tokenize_function, load_tokenizer_and_model
from utils import count_parameters
from evaluation import evaluate_all_metrics, get_mode_prefix, get_results_dir
from preprocessing import preprocess_archai_adr
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, 'get_usable_length'):
    def get_usable_length(self, layer_idx: int = 0, seq_length: int = None):
        """Compatibility method for Phi-3"""
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length

if not hasattr(DynamicCache, 'seen_tokens'):
    original_init = DynamicCache.__init__
    
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.seen_tokens = 0
    
    DynamicCache.__init__ = patched_init


def preprocess_by_name(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch preprocessing by dataset name."""
    name = dataset_name.lower().strip()
    if name == "archai-adr":
        print("[INFO] Using preprocessing for ArchAI-ADR dataset.")
        return preprocess_archai_adr(df)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_finetuned_model(save_dir: str, model_id: str):
    """Load a previously fine-tuned model from disk."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print(f"[INFO] Loading fine-tuned model from {save_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    
    # Check if this is a LoRA model (adapter_config.json exists)
    adapter_config_path = os.path.join(save_dir, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        # Load base model first, then apply LoRA adapter
        from transformers import BitsAndBytesConfig
        from config import USE_4BIT
        
        if USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, save_dir)
        print(f"[INFO] Loaded LoRA adapter from {save_dir}")
    else:
        # Full model saved
        model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    return tokenizer, model


def evaluate_only(
    model, tokenizer, model_key: str, model_id: str,
    dataset_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
    mode: str
) -> dict:
    """
    Evaluate model without training (for zero_shot and few_shot modes).
    """
    run_id = f"{dataset_name.lower().replace(' ', '_')}_{model_key}"
    mode_prefix = get_mode_prefix(mode)
    results_dir = get_results_dir(model_key)
    
    # Check if already completed
    results_file = os.path.join(results_dir, f"{mode_prefix}_{run_id}_detailed.json")
    summary_file = os.path.join(results_dir, f"{mode_prefix}_{run_id}_summary.json")
    
    if os.path.exists(results_file) and os.path.exists(summary_file):
        print(f"\n{'='*60}")
        print(f"[SKIP] {mode_prefix} - {model_key} on {dataset_name}")
        print(f"       Already completed (found saved results)")
        print(f"       Delete '{results_file}' to re-evaluate")
        print(f"{'='*60}\n")
        
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except:
            pass
        return None
    
    print(f"\n{'='*60}")
    print(f"[START] {mode_prefix.upper()} Evaluation: {model_key} on {dataset_name}")
    print(f"{'='*60}\n")
    
    # Evaluate
    print(f"[INFO] Evaluating all metrics...(mode={mode})")
    all_metrics = evaluate_all_metrics(
        model, tokenizer, val_df, run_id,
        model_key=model_key,
        support_df=train_df,
        mode=mode
    )
    
    # Compile results
    param_info = count_parameters(model)
    
    result_record = {
        "eval_mode": mode,
        "model_key": model_key,
        "model_id": model_id,
        "dataset_name": dataset_name,
        "num_train_samples": len(train_df),
        "num_test_samples": len(val_df),
        "training_time_sec": 0.0,
        "final_train_loss": None
    }
    result_record.update(param_info)
    result_record.update(all_metrics)
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(result_record, f, indent=2)
    print(f"[INFO] Saved summary to {summary_file}")
    
    print(f"\n[SUCCESS] Completed {mode_prefix} evaluation: {model_key} on {dataset_name}\n")
    
    return result_record


def train_and_evaluate_model(
    model, tokenizer, model_key: str, model_id: str,
    dataset_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
    tokenized_train, tokenized_val
) -> dict:
    """
    Train and evaluate a single model on a dataset (fine_tune mode only).
    If model is already fine-tuned but results are missing, only re-evaluate.
    """
    
    mode = "fine_tune"
    mode_prefix = get_mode_prefix(mode)
    
    # Setup paths
    run_id = f"{dataset_name.lower().replace(' ', '_')}_{model_key}"
    save_dir = os.path.join("models", run_id)
    results_dir = get_results_dir(model_key)
    results_file = os.path.join(results_dir, f"{mode_prefix}_{run_id}_detailed.json")
    summary_file = os.path.join(results_dir, f"{mode_prefix}_{run_id}_summary.json")
    
    # Check if BOTH model AND results exist -> fully completed
    if os.path.exists(save_dir) and os.path.exists(results_file) and os.path.exists(summary_file):
        print(f"\n{'='*60}")
        print(f"[SKIP] {mode_prefix} - {model_key} on {dataset_name}")
        print(f"       Already completed (found saved model and results)")
        print(f"       Delete '{save_dir}' to retrain")
        print(f"       Delete '{results_file}' to re-evaluate only")
        print(f"{'='*60}\n")
        
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except:
            pass
        return None
    
    # Check if model exists but results are missing -> re-evaluate only
    if os.path.exists(save_dir) and (not os.path.exists(results_file) or not os.path.exists(summary_file)):
        print(f"\n{'='*60}")
        print(f"[RE-EVALUATE] {mode_prefix} - {model_key} on {dataset_name}")
        print(f"              Found saved model but missing results")
        print(f"              Loading model and re-evaluating...")
        print(f"{'='*60}\n")
        
        # Load the fine-tuned model
        try:
            ft_tokenizer, ft_model = load_finetuned_model(save_dir, model_id)
        except Exception as e:
            print(f"[ERROR] Failed to load fine-tuned model: {e}")
            print(f"[INFO] Will retrain from scratch...")
            # Fall through to training
        else:
            # Evaluate the loaded model
            print(f"[INFO] Evaluating all metrics...(mode={mode})")
            all_metrics = evaluate_all_metrics(
                ft_model, ft_tokenizer, val_df, run_id,
                model_key=model_key,
                support_df=train_df,
                mode=mode
            )
            
            # Compile results (no training time since we loaded)
            param_info = count_parameters(ft_model)
            
            result_record = {
                "eval_mode": mode,
                "model_key": model_key,
                "model_id": model_id,
                "dataset_name": dataset_name,
                "num_train_samples": len(train_df),
                "num_test_samples": len(val_df),
                "training_time_sec": 0.0,  # Loaded from disk
                "final_train_loss": None,
                "note": "Re-evaluated from saved model"
            }
            result_record.update(param_info)
            result_record.update(all_metrics)
            
            # Save summary
            with open(summary_file, 'w') as f:
                json.dump(result_record, f, indent=2)
            print(f"[INFO] Saved summary to {summary_file}")
            
            # Cleanup
            del ft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n[SUCCESS] Re-evaluated {model_key} on {dataset_name}\n")
            return result_record
    
    # Model doesn't exist -> need to train
    print(f"\n{'='*60}")
    print(f"[START] FINE-TUNING: {model_key} on {dataset_name}")
    print(f"{'='*60}\n")
    
    # Setup training
    output_dir = os.path.join("runs", run_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 8,
        **TRAINING_ARGS,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Train
    print(f"[INFO] Training {model_key} on {dataset_name}...")
    train_start_time = time.time()
    trainer.train()
    train_time = time.time() - train_start_time
    print(f"[INFO] Training finished in {train_time:.2f} sec.")

    # Get training loss
    train_loss_history = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    final_train_loss = train_loss_history[-1] if train_loss_history else None

    # Save model BEFORE evaluation (so we can re-evaluate if validation fails)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Saved model to {save_dir}")

    # Standard evaluation
    print(f"[INFO] Evaluating standard metrics...")
    eval_results = trainer.evaluate()
    
    try:
        perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    except:
        perplexity = float('inf')
    
    eval_results["perplexity"] = perplexity

    # Research metrics
    print(f"[INFO] Evaluating all metrics...(mode={mode})")
    all_metrics = evaluate_all_metrics(
        model, tokenizer, val_df, run_id,
        model_key=model_key,
        support_df=train_df,
        mode=mode
    )

    # Compile results
    param_info = count_parameters(model)
    
    result_record = {
        "eval_mode": mode,
        "model_key": model_key,
        "model_id": model_id,
        "dataset_name": dataset_name,
        "num_train_samples": len(train_df),
        "num_test_samples": len(val_df),
        "training_time_sec": train_time,
        "final_train_loss": final_train_loss,
    }
    result_record.update(param_info)
    result_record.update(eval_results)
    result_record.update(all_metrics)
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(result_record, f, indent=2)
    print(f"[INFO] Saved summary to {summary_file}")
    
    print(f"\n[SUCCESS] Completed fine-tuning: {model_key} on {dataset_name}\n")
    
    return result_record


def run_all_modes_for_model(
    model, tokenizer, model_key: str, model_id: str,
    dataset_name: str, data_path: str, modes: list
) -> list:
    """
    Run all evaluation modes for a single model.
    Modes: zero_shot, few_shot run without training.
    fine_tune mode trains then evaluates.
    """
    results = []
    
    if not os.path.exists(data_path):
        print(f"[WARN] File not found: {data_path}")
        return results

    df_raw = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset from {data_path}")
    print(f"[INFO] Columns: {df_raw.columns.tolist()}")
    print(f"[INFO] Samples: {len(df_raw)}")

    try:
        df_processed = preprocess_by_name(dataset_name, df_raw)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return results

    if len(df_processed) == 0:
        print(f"[WARN] Empty dataset, skipping.")
        return results

    # Split data
    train_df, val_df = train_test_split(
        df_processed, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Tokenize (needed for fine_tune mode)
    train_ds = HFDataset.from_pandas(train_df)
    val_ds = HFDataset.from_pandas(val_df)
    tokenize_function = build_tokenize_function(tokenizer)
    tokenized_train = train_ds.map(tokenize_function, batched=False)
    tokenized_val = val_ds.map(tokenize_function, batched=False)

    # Run each mode
    for mode in modes:
        print("\n" + "=" * 80)
        print(f"### MODE: {mode.upper()} | MODEL: {model_key} | DATASET: {dataset_name}")
        print("=" * 80)
        
        if mode in ["zero_shot", "few_shot"]:
            # No training, just evaluation
            result = evaluate_only(
                model, tokenizer, model_key, model_id,
                dataset_name, train_df, val_df, mode
            )
        elif mode == "fine_tune":
            # Train then evaluate (or re-evaluate if model exists)
            result = train_and_evaluate_model(
                model, tokenizer, model_key, model_id,
                dataset_name, train_df, val_df,
                tokenized_train, tokenized_val
            )
        else:
            print(f"[WARN] Unknown mode: {mode}, skipping.")
            continue
        
        if result:
            results.append(result)
    
    return results
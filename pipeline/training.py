"""
Training loop logic.
"""
import os
import time
import torch
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import TRAINING_ARGS, TEST_SIZE, RANDOM_SEED
from models import build_tokenize_function
from utils import count_parameters
from evaluation import evaluate_all_metrics
from preprocessing import preprocess_archai_adr


def preprocess_by_name(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch preprocessing by dataset name."""
    name = dataset_name.lower().strip()
    if name == "archai-adr":
        print("[INFO] Using preprocessing for ArchAI-ADR dataset.")
        return preprocess_archai_adr(df)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def train_and_evaluate_model(
    model, tokenizer, model_key: str, model_id: str,
    dataset_name: str, data_path: str
):
    """Train and evaluate a single model on a dataset."""
    
    if not os.path.exists(data_path):
        print(f"[WARN] File not found: {data_path}")
        return None

    df_raw = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset from {data_path}")
    print(f"[INFO] Columns: {df_raw.columns.tolist()}")
    print(f"[INFO] Samples: {len(df_raw)}")

    try:
        df_processed = preprocess_by_name(dataset_name, df_raw)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None

    if len(df_processed) == 0:
        print(f"[WARN] Empty dataset, skipping.")
        return None

    # Split data
    train_df, val_df = train_test_split(
        df_processed, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    train_ds = HFDataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = HFDataset.from_pandas(val_df.reset_index(drop=True))

    # Tokenize
    tokenize_function = build_tokenize_function(tokenizer)
    tokenized_train = train_ds.map(tokenize_function, batched=False)
    tokenized_val = val_ds.map(tokenize_function, batched=False)

    # Setup training
    run_id = f"{dataset_name.lower().replace(' ', '_')}_{model_key}"
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

    # Standard evaluation
    print(f"[INFO] Evaluating standard metrics...")
    eval_results = trainer.evaluate()
    
    try:
        perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    except:
        perplexity = float('inf')
    
    eval_results["perplexity"] = perplexity

    # Research metrics
    print(f"[INFO] Evaluating all metrics...")
    all_metrics = evaluate_all_metrics(model, tokenizer, val_df, run_id)

    # Save model
    save_dir = os.path.join("models", run_id)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Saved model to {save_dir}")

    # Compile results
    param_info = count_parameters(model)
    
    result_record = {
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
    
    return result_record
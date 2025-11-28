"""
Main pipeline orchestrator.
"""
import random
import torch
import pandas as pd
from huggingface_hub import login
from config import MODEL_CANDIDATES, DATASETS_CONFIG, HF_TOKEN, RANDOM_SEED
from models import load_tokenizer_and_model
from training import train_and_evaluate_model


def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 80)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    print("=" * 80)

    if HF_TOKEN:
        print("[INFO] Logging in to Hugging Face...")
        login(token=HF_TOKEN)

    # Set seeds
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    results = []

    # Main loop
    for model_key, model_id in MODEL_CANDIDATES.items():
        print("\n" + "=" * 80)
        print(f"### LOADING MODEL: {model_key} -> {model_id}")
        print("=" * 80)

        tokenizer, model = load_tokenizer_and_model(model_id)

        for ds_cfg in DATASETS_CONFIG:
            dataset_name = ds_cfg["name"]
            data_path = ds_cfg["path"]

            print("\n" + "-" * 80)
            print(f"### START: {model_key} on {dataset_name}")
            print("-" * 80)

            result = train_and_evaluate_model(
                model, tokenizer, model_key, model_id,
                dataset_name, data_path
            )

            if result:
                results.append(result)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("=== ALL RUNS SUMMARY ===")
        print("=" * 80)
        print(results_df)
        
        import os
        os.makedirs("results", exist_ok=True)
        
        csv_path = "results/all_runs_summary.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved to {csv_path}")
        
        try:
            excel_path = "results/all_runs_summary.xlsx"
            results_df.to_excel(excel_path, index=False)
            print(f"[INFO] Saved to {excel_path}")
        except:
            pass
    else:
        print("[WARN] No successful runs.")


if __name__ == "__main__":
    main()
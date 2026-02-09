import os
import json
import random
from typing import List, Dict
import time
import re

import pandas as pd
import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType
)
from preprocessing import preprocess_archai_adr
from huggingface_hub import login

# ============================================================
# 1. BASIC SETUP
# ============================================================

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

if "HF_TOKEN" in os.environ:
    print("Logging in to Hugging Face using Token...")
    login(token=os.environ["HF_TOKEN"])

# For reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# 2. MODEL CANDIDATES
# ============================================================

model_candidates = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "olmo-2-1b": "allenai/OLMo-2-0425-1B-SFT",
    "olmo-2-7b": "allenai/OLMo-2-7B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "vaultgemma-1b": "google/vaultgemma-1b",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
}

use_4bit = True


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "trainable_percentage": round(100 * trainable_params / total_params, 2)
    }


def load_tokenizer_and_model(model_id: str):
    """Load tokenizer and model with LoRA for 4-bit training."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
    model = None

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, quantization_config=bnb_config
            )
            
            # Prepare model for LoRA
            model = prepare_model_for_kbit_training(model)
            
            peft_config = LoraConfig(
                r=16,       # Rank
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear"  # Important for Llama/Mistral
            )
            model = get_peft_model(model, peft_config)
            print("[INFO] LoRA configuration applied:")
            model.print_trainable_parameters()
            
        except ImportError:
            print("bitsandbytes not available, loading in float16.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, torch_dtype=torch.float16
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.float16
        )

    return tokenizer, model


# ============================================================
# 3. DATASETS CONFIG
# ============================================================

datasets_config = [
    {"name": "ArchAI-ADR", "path": "dataset/archAI-ADR.csv"},
]


def preprocess_by_name(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch to the correct preprocessing function."""
    name = dataset_name.lower().strip()
    if name == "archai-adr":
        print("Using preprocessing for ArchAI-ADR dataset.")
        return preprocess_archai_adr(df)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# ============================================================
# 4. TOKENIZATION FUNCTION
# ============================================================

max_length = 1024


def build_tokenize_function(tokenizer):
    """Return a tokenize function bound to a specific tokenizer."""

    def tokenize_function(example):
        full_text = example["prompt"] + "\n" + example["target"]
        tokenized = tokenizer(
            full_text, truncation=True, max_length=max_length, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return tokenize_function


# ============================================================
# 5. TRADITIONAL NLG METRICS (ROUGE, BLEU, METEOR, BERTScore)
# ============================================================

def compute_nlg_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE, BLEU, METEOR, BERTScore for generation tasks.
    """
    metrics = {}
    
    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        metrics['rouge1'] = np.mean(rouge_scores['rouge1'])
        metrics['rouge2'] = np.mean(rouge_scores['rouge2'])
        metrics['rougeL'] = np.mean(rouge_scores['rougeL'])
    except ImportError:
        print("[WARN] rouge-score not installed. Skipping ROUGE.")
        metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
    
    # BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(score)
        
        metrics['bleu'] = np.mean(bleu_scores)
    except ImportError:
        print("[WARN] nltk not installed. Skipping BLEU.")
        metrics['bleu'] = 0.0
    
    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            score = meteor_score([ref.split()], pred.split())
            meteor_scores.append(score)
        
        metrics['meteor'] = np.mean(meteor_scores)
    except Exception as e:
        print(f"[WARN] METEOR scoring failed: {e}. Skipping.")
        metrics['meteor'] = 0.0
    
    # BERTScore
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        metrics['bertscore_precision'] = P.mean().item()
        metrics['bertscore_recall'] = R.mean().item()
        metrics['bertscore_f1'] = F1.mean().item()
    except ImportError:
        print("[WARN] bert-score not installed. Skipping BERTScore.")
        metrics['bertscore_precision'] = 0.0
        metrics['bertscore_recall'] = 0.0
        metrics['bertscore_f1'] = 0.0
    
    return metrics


# ============================================================
# 6. RESEARCH METRICS FUNCTIONS
# ============================================================

def compute_diversity_score(responses: List[str]) -> float:
    """
    Diversity Score: Measure semantic diversity using Sentence-BERT embeddings.
    Formula: Avg(1 - CosineSimilarity(Option_i, Option_j))
    """
    if len(responses) < 2:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(responses)

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(1 - sim)

        return float(np.mean(similarities)) if similarities else 0.0
    except ImportError:
        print("[WARN] sentence-transformers not installed. Skipping Diversity Score.")
        return 0.0


def compute_compliance_score(
    model_answer: str, ground_truth: str, judge_model: str = "gpt-4o-mini"
) -> float:
    """
    Architectural Pattern Compliance: Use LLM-as-a-Judge (GPT-4o-mini) to score compliance.
    Returns: Compliance Rate (0-100%).
    Note: Using gpt-4o-mini for cost efficiency (15x cheaper than gpt-4).
    """
    try:
        import openai

        if "OPENAI_API_KEY" not in os.environ:
            print("[WARN] OPENAI_API_KEY not set. Skipping Compliance Score.")
            return 0.0

        openai.api_key = os.environ["OPENAI_API_KEY"]

        prompt = f"""
        You are an expert software architecture evaluator.
        Given a model's generated architectural decision and the ground truth, rate the compliance of the model's answer with standard architectural patterns (e.g., MVC separation, layered architecture).

        Model Answer:
        {model_answer}

        Ground Truth:
        {ground_truth}

        Rate the compliance from 0 to 100 (0 = not compliant, 100 = fully compliant). Respond ONLY with a number.
        """

        response = openai.ChatCompletion.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        score_str = response["choices"][0]["message"]["content"].strip()
        return float(score_str)
    except Exception as e:
        print(f"[ERROR] Compliance scoring failed: {e}")
        return 0.0


def extract_components(text: str) -> set:
    """
    Extract architectural components (file names, class names, modules) using regex.
    Improved version from simple whitespace tokenization.
    """
    components = set()
    
    # Pattern 1: File names (e.g., UserService.java, app.py, config.yaml)
    file_pattern = r'\b\w+\.(java|py|ts|tsx|jsx|js|xml|yaml|yml|json|properties|conf)\b'
    files = re.findall(file_pattern, text, re.IGNORECASE)
    components.update(files)
    
    # Pattern 2: Class/Service names (e.g., UserService, PaymentController, DataRepository)
    class_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:Service|Controller|Repository|Model|Manager|Handler|Provider|Factory|Builder|Adapter|Facade|Strategy)\b'
    classes = re.findall(class_pattern, text)
    components.update(classes)
    
    # Pattern 3: Package/Module names (e.g., com.example.payment, user.auth.service)
    package_pattern = r'\b[a-z]+(?:\.[a-z]+){2,}\b'
    packages = re.findall(package_pattern, text)
    components.update(packages)
    
    # Pattern 4: Architecture layer keywords
    layer_keywords = r'\b(controller|service|repository|model|view|database|api|frontend|backend|middleware|gateway|proxy)\b'
    layers = re.findall(layer_keywords, text, re.IGNORECASE)
    components.update([l.lower() for l in layers])
    
    return components


def compute_ripple_effect_recall(
    model_answer: str, ground_truth: str
) -> Dict[str, float]:
    """
    Ripple Effect Prediction Recall: Extract components/files mentioned and compute Recall/Precision.
    Now using improved regex-based component extraction.
    """
    model_components = extract_components(model_answer)
    truth_components = extract_components(ground_truth)

    if len(truth_components) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    tp = len(model_components & truth_components)
    recall = tp / len(truth_components) if len(truth_components) > 0 else 0.0
    precision = tp / len(model_components) if len(model_components) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return {
        "recall": recall, 
        "precision": precision, 
        "f1": f1,
        "tp": tp,
        "model_components_count": len(model_components),
        "truth_components_count": len(truth_components)
    }


def evaluate_all_metrics(
    model, tokenizer, val_df: pd.DataFrame, run_id: str
) -> Dict[str, float]:
    """
    Evaluate metrics with proper Diversity Logic (Generating multiple alternatives).
    Now generates 5 alternatives per sample for better diversity measurement.
    """
    detailed_results = []
    
    # Lists to store scores for averaging
    sample_diversity_scores = []
    sample_compliance_scores = []
    sample_ripple_recalls = []
    sample_ripple_precisions = []
    sample_ripple_f1s = []
    
    # Performance tracking
    inference_times = []

    # NLG Calculation Lists
    all_main_predictions = []
    all_references = []

    # Collect ALL alternatives for global diversity
    all_alternatives = []

    print(f"[INFO] Evaluating on {len(val_df)} samples...")

    for idx, row in val_df.iterrows():
        prompt = row["prompt"]
        ground_truth = row["target"]
        
        # Format input for model
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        start_time = time.time()
        
        # Generate 5 alternatives for better diversity measurement
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=True,      # Enable sampling for diversity
                temperature=0.7,     # Balance between creativity and coherence
                top_p=0.9,
                num_return_sequences=5,  # Generate 5 alternatives (increased from 3)
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Decode all 5 alternatives
        alternatives = []
        for output in outputs:
            text = tokenizer.decode(output[input_ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            alternatives.append(text)

        # Add to global alternatives pool
        all_alternatives.extend(alternatives)

        # 1. Compute Diversity Score (within this sample's 5 alternatives)
        div_score = compute_diversity_score(alternatives)
        sample_diversity_scores.append(div_score)

        # 2. Use first alternative as main answer for Compliance/NLG evaluation
        main_answer = alternatives[0]
        all_main_predictions.append(main_answer)
        all_references.append(ground_truth)

        # 3. Compute Compliance (only on main answer to save API costs)
        comp_score = compute_compliance_score(main_answer, ground_truth)
        sample_compliance_scores.append(comp_score)

        # 4. Compute Ripple Effect Recall (with improved component extraction)
        ripple = compute_ripple_effect_recall(main_answer, ground_truth)
        sample_ripple_recalls.append(ripple["recall"])
        sample_ripple_precisions.append(ripple["precision"])
        sample_ripple_f1s.append(ripple["f1"])

        # Store detailed result
        detailed_results.append({
            "sample_id": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "main_answer": main_answer,
            "alternatives": alternatives,  # All 5 alternatives
            "inference_time_sec": inference_time,
            "diversity_score": div_score,
            "compliance_score": comp_score,
            "ripple_recall": ripple["recall"],
            "ripple_precision": ripple["precision"],
            "ripple_f1": ripple["f1"],
            "ripple_tp": ripple["tp"],
            "ripple_model_components": ripple["model_components_count"],
            "ripple_truth_components": ripple["truth_components_count"],
        })

    # Save detailed results to JSON
    os.makedirs("results", exist_ok=True)
    detailed_path = os.path.join("results", f"{run_id}_detailed.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved detailed results to {detailed_path}")

    # Compute Standard NLG metrics based on main predictions
    nlg_metrics = compute_nlg_metrics(all_main_predictions, all_references)

    # Compute global diversity across ALL alternatives from all samples
    global_diversity = compute_diversity_score(all_alternatives)

    # Aggregate metrics
    return {
        **nlg_metrics,
        "diversity_score_per_sample_avg": np.mean(sample_diversity_scores) if sample_diversity_scores else 0.0,
        "diversity_score_global": global_diversity,
        "compliance_avg": np.mean(sample_compliance_scores) if sample_compliance_scores else 0.0,
        "ripple_recall_avg": np.mean(sample_ripple_recalls) if sample_ripple_recalls else 0.0,
        "ripple_precision_avg": np.mean(sample_ripple_precisions) if sample_ripple_precisions else 0.0,
        "ripple_f1_avg": np.mean(sample_ripple_f1s) if sample_ripple_f1s else 0.0,
        "avg_inference_time_sec": np.mean(inference_times) if inference_times else 0.0,
        "total_inference_time_sec": np.sum(inference_times) if inference_times else 0.0,
    }


# ============================================================
# 7. TRAINING LOOP (MODEL × DATASET)
# ============================================================

def main():
    results = []

    for model_key, model_id in model_candidates.items():
        print("=" * 80)
        print(f"### LOADING MODEL: {model_key} -> {model_id}")
        print("=" * 80)

        tokenizer, model = load_tokenizer_and_model(model_id)

        # Count parameters
        param_info = count_parameters(model)
        print(f"[INFO] Model parameters: {param_info}")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        tokenize_function = build_tokenize_function(tokenizer)

        for ds_cfg in datasets_config:
            dataset_name = ds_cfg["name"]
            data_path = ds_cfg["path"]

            print("\n" + "-" * 80)
            print(f"### START RUN: MODEL = {model_key}, DATASET = {dataset_name}")
            print("-" * 80)

            if not os.path.exists(data_path):
                print(f"[WARN] File not found: {data_path}")
                continue

            df_raw = pd.read_csv(data_path)
            print(f"[INFO] Loaded dataset from {data_path}")
            print("Columns:", df_raw.columns.tolist())
            print("Number of samples:", len(df_raw))

            try:
                df_processed = preprocess_by_name(dataset_name, df_raw)
            except Exception as e:
                print(f"[ERROR] Preprocessing failed: {e}")
                continue

            if len(df_processed) == 0:
                print(f"[WARN] Empty dataset, skipping.")
                continue

            # Train/val split (20% test as recommended)
            train_df, val_df = train_test_split(
                df_processed, test_size=0.2, random_state=42
            )

            train_ds = HFDataset.from_pandas(train_df.reset_index(drop=True))
            val_ds = HFDataset.from_pandas(val_df.reset_index(drop=True))

            tokenized_train = train_ds.map(tokenize_function, batched=False)
            tokenized_val = val_ds.map(tokenize_function, batched=False)

            run_id = f"{dataset_name.lower().replace(' ', '_')}_{model_key}"
            output_dir = os.path.join("runs", run_id)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=5,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,  # Higher LR for LoRA
                warmup_ratio=0.03,
                weight_decay=0.01,
                logging_steps=10,
                logging_dir=os.path.join(output_dir, "logs"),
                evaluation_strategy="steps",
                eval_steps=50,
                save_steps=200,
                save_total_limit=2,
                load_best_model_at_end=True,
                bf16=torch.cuda.is_available(),
                fp16=not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 8,
                report_to="tensorboard",
            )

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
            train_result = trainer.train()
            train_time = time.time() - train_start_time
            print(f"[INFO] Training finished in {train_time:.2f} seconds.")

            # Extract training loss from log history
            train_loss_history = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
            final_train_loss = train_loss_history[-1] if train_loss_history else None

            # Standard Evaluation
            print(f"[INFO] Evaluating standard metrics...")
            eval_results = trainer.evaluate()
            
            # Compute perplexity from loss
            try:
                perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
            except:
                perplexity = float('inf')
            
            eval_results["perplexity"] = perplexity
            print("Standard Eval Results:", eval_results)

            # All Metrics Evaluation (NLG + Research)
            print(f"[INFO] Evaluating all metrics on test set...")
            all_metrics = evaluate_all_metrics(model, tokenizer, val_df, run_id)
            print("All Metrics:", all_metrics)

            # Save model (LoRA adapters only)
            save_dir = os.path.join("models", run_id)
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[INFO] Saved model (LoRA adapters) to {save_dir}")

            # Store comprehensive results
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
            results.append(result_record)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== ALL RUNS SUMMARY ===")
        print(results_df)
        
        os.makedirs("results", exist_ok=True)
        results_path = os.path.join("results", "all_runs_summary.csv")
        results_df.to_csv(results_path, index=False)
        print(f"[INFO] Saved summary results to {results_path}")
        
        # Also save as Excel for easier analysis
        try:
            excel_path = os.path.join("results", "all_runs_summary.xlsx")
            results_df.to_excel(excel_path, index=False)
            print(f"[INFO] Saved Excel summary to {excel_path}")
        except ImportError:
            print("[WARN] openpyxl not installed. Skipping Excel export.")
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()
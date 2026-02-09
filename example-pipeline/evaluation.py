"""
Evaluation pipeline for all metrics.

Supports three evaluation modes:
- "fine_tune": use standard prompts after fine-tuning
- "zero_shot": evaluate pre-trained model without examples
- "few_shot": prepend FEW_SHOT_K examples from support_df (train set) to each prompt
"""
import os
import json
import time
import random
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional
from metrics import (
    compute_nlg_metrics,
    compute_diversity_score,
    compute_compliance_score,
    compute_ripple_effect_recall,
)
from config import GENERATION_CONFIG, FEW_SHOT_K, RANDOM_SEED, MODEL_MAX_TOKEN, FEW_SHOT_GOLDEN_COUNT, FEW_SHOT_PROMPT_PATH

# Rate limit delay for Gemini API (free tier: 15 RPM)
GEMINI_RATE_LIMIT_DELAY = 5 


def get_mode_prefix(mode: str) -> str:
    """Get the file prefix for a given evaluation mode."""
    mode_prefixes = {
        "zero_shot": "zeroshot",
        "few_shot": "fewshot",
        "fine_tune": "finetuned"
    }
    return mode_prefixes.get(mode, mode)


def get_results_dir(model_key: str) -> str:
    """Get the results directory for a specific model."""
    results_dir = os.path.join("results", model_key)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def build_few_shot_prefix(support_df: pd.DataFrame, k: int = 2, seed: int = 42) -> str:
    """
    Build few-shot examples prefix. 
    
    - K < 2: Raise error (minimum 2 golden examples required)
    - K == 2: Use only golden examples from prompt file
    - K > 2: Use golden examples + (K-2) random samples from support_df
    
    Args:
        support_df: DataFrame containing training examples (for additional random samples)
        k: Number of few-shot examples to use (minimum 2)
        seed: Random seed for additional sampling
    
    Returns:
        String containing formatted few-shot prefix
    """
    # Validate K
    if k < FEW_SHOT_GOLDEN_COUNT:
        raise ValueError(
            f"FEW_SHOT_K must be >= {FEW_SHOT_GOLDEN_COUNT} (number of golden examples).  "
            f"Got K={k}"
        )
    
    # Load golden examples from file
    try:
        with open(FEW_SHOT_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prefix = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Few-shot prompt file not found: {FEW_SHOT_PROMPT_PATH}. "
            f"Please create this file with {FEW_SHOT_GOLDEN_COUNT} golden examples."
        )
    
    # If K == 2, return only golden examples
    if k == FEW_SHOT_GOLDEN_COUNT:
        print(f"[INFO] Using {k} golden examples from {FEW_SHOT_PROMPT_PATH}")
        return prefix
    
    # If K > 2, add random samples from support_df
    num_random = k - FEW_SHOT_GOLDEN_COUNT
    
    if support_df is None or len(support_df) == 0:
        print(f"[WARN] support_df is empty, using only {FEW_SHOT_GOLDEN_COUNT} golden examples")
        return prefix
    
    # Random sample from support_df
    num_random = min(num_random, len(support_df))
    random_samples = support_df.sample(n=num_random, random_state=seed)
    
    # Build additional examples
    additional_examples = []
    for _, r in random_samples.iterrows():
        ctx = r.get("prompt", "")
        tgt = r.get("target", "")
        # Extract context from prompt (remove instruction prefix)
        if "Context:" in ctx:
            ctx_content = ctx.split("Context:")[-1].split("Decision:")[0].strip()
        else:
            ctx_content = ctx
        example = f"Context:\n{ctx_content}\n\nDecision:\n{tgt}\n\n---\n\n"
        additional_examples. append(example)
    
    print(f"[INFO] Using {FEW_SHOT_GOLDEN_COUNT} golden examples + {num_random} random samples (total K={k})")
    
    return prefix + "".join(additional_examples)

def evaluate_all_metrics(model, tokenizer, val_df: pd.DataFrame, run_id: str,
                         model_key: str,
                         support_df: Optional[pd.DataFrame] = None,
                         mode: str = "fine_tune") -> Dict[str, float]:
    """Evaluate all metrics on validation set with multiple alternatives."""
    detailed_results = []

    sample_diversity_scores = []
    sample_compliance_scores = []
    sample_ripple_recalls = []
    sample_ripple_precisions = []
    sample_ripple_f1s = []
    inference_times = []

    all_main_predictions = []
    all_references = []
    all_alternatives = []

    print(f"[INFO] Evaluating on {len(val_df)} samples...(mode={mode})")

    # build few-shot prefix if requested
    few_shot_prefix = ""
    if mode == "few_shot":
        few_shot_prefix = build_few_shot_prefix(support_df, k=FEW_SHOT_K, seed=RANDOM_SEED)
        if not few_shot_prefix:
            print("[WARN] few_shot mode requested but support_df empty - falling back to zero-shot")

    single_sequence_cfg = dict(GENERATION_CONFIG)
    single_sequence_cfg["num_return_sequences"] = 3

    for idx, row in val_df.iterrows():
        base_prompt = row["prompt"]
        ground_truth = row["target"]

        if mode == "few_shot" and few_shot_prefix:
            prompt = few_shot_prefix + base_prompt
        else:
            prompt = base_prompt

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MODEL_MAX_TOKEN[model_key]
        ).to(model.device)

        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=single_sequence_cfg.get("max_new_tokens", GENERATION_CONFIG["max_new_tokens"]),
                do_sample=single_sequence_cfg.get("do_sample", GENERATION_CONFIG["do_sample"]),
                temperature=single_sequence_cfg.get("temperature", GENERATION_CONFIG["temperature"]),
                top_p=single_sequence_cfg.get("top_p", GENERATION_CONFIG["top_p"]),
                num_return_sequences=single_sequence_cfg.get("num_return_sequences", 3),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        alternatives = []
        prompt_len = input_ids["input_ids"].shape[-1]
        for output in outputs:
            if output.shape[-1] <= prompt_len:
                text = ""
            else:
                text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            alternatives.append(text)

        if not alternatives:
            alternatives = [""]

        all_alternatives.extend(alternatives)

        div_score = compute_diversity_score(alternatives) if len(alternatives) > 1 else 0.0
        sample_diversity_scores.append(div_score)

        main_answer = alternatives[0]
        all_main_predictions.append(main_answer)
        all_references.append(ground_truth)

        # ============================================================
        # COMPLIANCE SCORE WITH RATE LIMITING
        # ============================================================
        print(f"[INFO] Computing compliance score for sample {idx + 1}/{len(val_df)}...")
        comp_score = compute_compliance_score(main_answer, ground_truth)
        sample_compliance_scores.append(comp_score)
        
        # Add delay to respect Gemini rate limit (15 RPM = 1 request per 4 seconds)
        time.sleep(GEMINI_RATE_LIMIT_DELAY)
        # ============================================================

        ripple = compute_ripple_effect_recall(main_answer, ground_truth)
        if ripple["truth_components_count"] == 0:
            sample_ripple_recalls.append(None)
            sample_ripple_precisions.append(None)
            sample_ripple_f1s.append(None)
        else:
            sample_ripple_recalls.append(ripple["recall"])
            sample_ripple_precisions.append(ripple["precision"])
            sample_ripple_f1s.append(ripple["f1"])

        detailed_results.append({
            "sample_id": int(idx),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "main_answer": main_answer,
            "alternatives": alternatives,
            "inference_time_sec": inference_time,
            "diversity_score": div_score,
            "compliance_score": comp_score,
            "ripple_recall": ripple["recall"] if ripple["truth_components_count"] > 0 else None,
            "ripple_precision": ripple["precision"] if ripple["truth_components_count"] > 0 else None,
            "ripple_f1": ripple["f1"] if ripple["truth_components_count"] > 0 else None,
            "ripple_tp": ripple["tp"],
            "ripple_model_components": ripple["model_components_count"],
            "ripple_truth_components": ripple["truth_components_count"],
        })

    # Get mode prefix and results directory
    mode_prefix = get_mode_prefix(mode)
    results_dir = get_results_dir(model_key)

    # Save detailed results with mode prefix
    detailed_filename = f"{mode_prefix}_{run_id}_detailed.json"
    detailed_path = os.path.join(results_dir, detailed_filename)
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved detailed results to {detailed_path}")

    # Compute metrics
    nlg_metrics = compute_nlg_metrics(all_main_predictions, all_references)
    global_diversity = compute_diversity_score(all_alternatives)

    ripple_recalls_clean = [r for r in sample_ripple_recalls if r is not None]
    ripple_precisions_clean = [r for r in sample_ripple_precisions if r is not None]
    ripple_f1s_clean = [r for r in sample_ripple_f1s if r is not None]

    return {
        **nlg_metrics,
        "diversity_score_per_sample_avg": np.mean(sample_diversity_scores) if sample_diversity_scores else 0.0,
        "diversity_score_global": global_diversity,
        "compliance_avg": np.mean(sample_compliance_scores) if sample_compliance_scores else 0.0,
        "ripple_recall_avg": np.mean(ripple_recalls_clean) if ripple_recalls_clean else 0.0,
        "ripple_precision_avg": np.mean(ripple_precisions_clean) if ripple_precisions_clean else 0.0,
        "ripple_f1_avg": np.mean(ripple_f1s_clean) if ripple_f1s_clean else 0.0,
        "avg_inference_time_sec": np.mean(inference_times) if inference_times else 0.0,
        "total_inference_time_sec": np.sum(inference_times) if inference_times else 0.0,
    }
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
from config import GENERATION_CONFIG, FEW_SHOT_K, RANDOM_SEED


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
    os. makedirs(results_dir, exist_ok=True)
    return results_dir


# Helper to build few-shot prefix
def build_few_shot_prefix(support_df: pd.DataFrame, k: int = 3, seed: int = 42) -> str:
    """
    Build few-shot examples prefix from support_df.
    support_df is expected to have columns 'prompt' and 'target'.
    Returns a string containing k examples formatted as:
    <EXAMPLE>
    Context:
    ... 
    Decision:
    ... 
    </EXAMPLE>
    """
    if support_df is None or len(support_df) == 0:
        return ""
    k = min(k, len(support_df))
    # deterministic sampling
    sampled = support_df.sample(n=k, random_state=seed) if hasattr(support_df, "sample") else support_df[:k]
    example_texts = []
    for _, r in sampled.iterrows():
        ctx = r. get("prompt", "")
        tgt = r.get("target", "")
        # Some prompt builders already include "Context:" and "Decision:" - ensure consistent formatting
        # We include the prompt (which already contains the "Context:" header) and then the target decision. 
        example = f"{ctx}\n{tgt}\n"
        example_texts.append(example)
    # join with separator
    return "\n---\n".join(example_texts) + "\n\n"


def evaluate_all_metrics(model, tokenizer, val_df: pd.DataFrame, run_id: str,
                         model_key: str,
                         support_df: Optional[pd.DataFrame] = None, 
                         mode: str = "fine_tune") -> Dict[str, float]:
    """Evaluate all metrics on validation set with multiple alternatives. 

    Args:
        model: the model (transformers)
        tokenizer: tokenizer
        val_df: dataframe with columns 'prompt' and 'target'
        run_id: run id for saving details
        model_key: model key for organizing results
        support_df: optional dataframe (train) used for few-shot examples
        mode: 'fine_tune' | 'zero_shot' | 'few_shot'
    """
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

    print(f"[INFO] Evaluating on {len(val_df)} samples...  (mode={mode})")

    # build few-shot prefix if requested
    few_shot_prefix = ""
    if mode == "few_shot":
        few_shot_prefix = build_few_shot_prefix(support_df, k=FEW_SHOT_K, seed=RANDOM_SEED)
        if not few_shot_prefix:
            print("[WARN] few_shot mode requested but support_df empty - falling back to zero-shot")

    # For performance, use 1 sequence for primary metrics.  If you need diversity, set GENERATION_CONFIG accordingly.
    single_sequence_cfg = dict(GENERATION_CONFIG)
    single_sequence_cfg["num_return_sequences"] = 1

    for idx, row in val_df.iterrows():
        base_prompt = row["prompt"]
        ground_truth = row["target"]

        # Prepend few-shot examples if requested
        if mode == "few_shot" and few_shot_prefix:
            prompt = few_shot_prefix + base_prompt
        else:
            # zero-shot and fine_tune use the prompt as-is
            prompt = base_prompt

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        start_time = time. time()
        
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=single_sequence_cfg. get("max_new_tokens", GENERATION_CONFIG["max_new_tokens"]),
                do_sample=single_sequence_cfg.get("do_sample", GENERATION_CONFIG["do_sample"]),
                temperature=single_sequence_cfg.get("temperature", GENERATION_CONFIG["temperature"]),
                top_p=single_sequence_cfg.get("top_p", GENERATION_CONFIG["top_p"]),
                num_return_sequences=single_sequence_cfg.get("num_return_sequences", 1),
                eos_token_id=tokenizer. eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        inference_time = time. time() - start_time
        inference_times.append(inference_time)

        # Decode alternatives (robustly slice off the prompt tokens)
        alternatives = []
        prompt_len = input_ids["input_ids"].shape[-1]
        for output in outputs:
            if output.shape[-1] <= prompt_len:
                text = ""
            else:
                text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True). strip()
            alternatives. append(text)

        if not alternatives:
            alternatives = [""]

        all_alternatives.extend(alternatives)

        # Compute metrics
        div_score = compute_diversity_score(alternatives) if len(alternatives) > 1 else 0.0
        sample_diversity_scores. append(div_score)

        main_answer = alternatives[0]
        all_main_predictions. append(main_answer)
        all_references. append(ground_truth)

        comp_score = compute_compliance_score(main_answer, ground_truth)
        sample_compliance_scores. append(comp_score)

        ripple = compute_ripple_effect_recall(main_answer, ground_truth)
        # handle truth_components == 0 gracefully (do not pollute averages)
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
    detailed_path = os. path.join(results_dir, detailed_filename)
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved detailed results to {detailed_path}")

    # Compute metrics
    nlg_metrics = compute_nlg_metrics(all_main_predictions, all_references)
    global_diversity = compute_diversity_score(all_alternatives)

    # Clean ripple lists (ignore None)
    ripple_recalls_clean = [r for r in sample_ripple_recalls if r is not None]
    ripple_precisions_clean = [r for r in sample_ripple_precisions if r is not None]
    ripple_f1s_clean = [r for r in sample_ripple_f1s if r is not None]

    return {
        **nlg_metrics,
        "diversity_score_per_sample_avg": np.mean(sample_diversity_scores) if sample_diversity_scores else 0.0,
        "diversity_score_global": global_diversity,
        "compliance_avg": np. mean(sample_compliance_scores) if sample_compliance_scores else 0.0,
        "ripple_recall_avg": np.mean(ripple_recalls_clean) if ripple_recalls_clean else 0.0,
        "ripple_precision_avg": np.mean(ripple_precisions_clean) if ripple_precisions_clean else 0.0,
        "ripple_f1_avg": np.mean(ripple_f1s_clean) if ripple_f1s_clean else 0.0,
        "avg_inference_time_sec": np.mean(inference_times) if inference_times else 0.0,
        "total_inference_time_sec": np.sum(inference_times) if inference_times else 0.0,
    }
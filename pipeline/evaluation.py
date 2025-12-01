"""
Evaluation pipeline for all metrics.
"""
import os
import json
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict
from metrics import (
    compute_nlg_metrics,
    compute_diversity_score,
    compute_compliance_score,
    compute_ripple_effect_recall,
)
from config import GENERATION_CONFIG


def evaluate_all_metrics(model, tokenizer, val_df: pd.DataFrame, run_id: str) -> Dict[str, float]:
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

    print(f"[INFO] Evaluating on {len(val_df)} samples...")

    for idx, row in val_df.iterrows():
        prompt = row["prompt"]
        ground_truth = row["target"]
        
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                do_sample=GENERATION_CONFIG["do_sample"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                num_return_sequences=GENERATION_CONFIG["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Decode alternatives
        alternatives = []
        for output in outputs:
            text = tokenizer.decode(
                output[input_ids["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            alternatives.append(text)

        all_alternatives.extend(alternatives)

        # Compute metrics
        div_score = compute_diversity_score(alternatives)
        sample_diversity_scores.append(div_score)

        main_answer = alternatives[0]
        all_main_predictions.append(main_answer)
        all_references.append(ground_truth)

        comp_score = compute_compliance_score(main_answer, ground_truth)
        sample_compliance_scores.append(comp_score)

        ripple = compute_ripple_effect_recall(main_answer, ground_truth)
        sample_ripple_recalls.append(ripple["recall"])
        sample_ripple_precisions.append(ripple["precision"])
        sample_ripple_f1s.append(ripple["f1"])

        detailed_results.append({
            "sample_id": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "main_answer": main_answer,
            "alternatives": alternatives,
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

    # Save detailed results
    os.makedirs("results", exist_ok=True)
    detailed_path = os.path.join("results", f"{run_id}_detailed.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved detailed results to {detailed_path}")

    # Compute metrics
    nlg_metrics = compute_nlg_metrics(all_main_predictions, all_references)
    global_diversity = compute_diversity_score(all_alternatives)

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
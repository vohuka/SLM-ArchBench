"""
All metric computation functions: NLG + Research metrics.
"""
import os
import numpy as np
from typing import List, Dict
from utils import extract_components
from config import JUDGE_MODEL


# ============================================================
# NLG METRICS
# ============================================================

def compute_nlg_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE, BLEU, METEOR, BERTScore."""
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
# RESEARCH METRICS
# ============================================================

def compute_diversity_score(responses: List[str]) -> float:
    """Diversity Score using Sentence-BERT."""
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


def compute_compliance_score(model_answer: str, ground_truth: str) -> float:
    """Compliance Score using LLM-as-a-Judge."""
    try:
        import openai

        if "OPENAI_API_KEY" not in os.environ:
            print("[WARN] OPENAI_API_KEY not set. Skipping Compliance Score.")
            return 0.0

        openai.api_key = os.environ["OPENAI_API_KEY"]

        #IMPORTANT
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
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        score_str = response["choices"][0]["message"]["content"].strip()
        return float(score_str)
    except Exception as e:
        print(f"[ERROR] Compliance scoring failed: {e}")
        return 0.0


def compute_ripple_effect_recall(model_answer: str, ground_truth: str) -> Dict[str, float]:
    """Ripple Effect Recall with regex-based component extraction."""
    model_components = extract_components(model_answer)
    truth_components = extract_components(ground_truth)

    if len(truth_components) == 0:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0,
            "model_components_count": 0, "truth_components_count": 0
        }

    tp = len(model_components & truth_components)
    recall = tp / len(truth_components) if len(truth_components) > 0 else 0.0
    precision = tp / len(model_components) if len(model_components) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "model_components_count": len(model_components),
        "truth_components_count": len(truth_components)
    }
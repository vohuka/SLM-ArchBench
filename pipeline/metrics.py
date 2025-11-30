"""
All metric computation functions: NLG + Research metrics. 
"""
import numpy as np
import time
from typing import List, Dict
from utils import extract_components
from config import JUDGE_MODEL, GEMINI_API_KEY


# ============================================================
# NLG METRICS (giữ nguyên)
# ============================================================

def compute_nlg_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE, BLEU, METEOR, BERTScore."""
    metrics = {}
    
    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer. RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1']. fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        metrics['rouge1'] = np.mean(rouge_scores['rouge1'])
        metrics['rouge2'] = np.mean(rouge_scores['rouge2'])
        metrics['rougeL'] = np.mean(rouge_scores['rougeL'])
    except ImportError:
        print("[WARN] rouge-score not installed.  Skipping ROUGE.")
        metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
    
    # BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        
        bleu_scores = []
        smoothing = SmoothingFunction(). method1
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
            score = meteor_score([ref. split()], pred. split())
            meteor_scores. append(score)
        
        metrics['meteor'] = np.mean(meteor_scores)
    except Exception as e:
        print(f"[WARN] METEOR scoring failed: {e}. Skipping.")
        metrics['meteor'] = 0.0
    
    # BERTScore
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        metrics['bertscore_precision'] = P. mean(). item()
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
    """
    Compliance Score using Gemini 2.0 Flash as LLM-as-a-Judge.
    Free tier: 15 RPM, 1M TPM, 1500 RPD. 
    """
    # Check API key
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set. Skipping Compliance Score.")
        return 0.0
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize model with safety settings disabled
        model = genai.GenerativeModel(
            JUDGE_MODEL,
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
        
        # Build prompt
        prompt = f"""You are an expert software architecture evaluator.  

        Given a model's generated architectural decision and the ground truth, rate how well the model's answer aligns with standard architectural patterns and the ground truth decision.

        Consider:
        - Correctness of architectural approach
        - Alignment with the ground truth rationale
        - Compliance with architectural best practices (e.g., MVC, microservices, layered architecture)

        Model Answer:
        {model_answer}

        Ground Truth:
        {ground_truth}

        Rate the compliance from 0 to 100:
        - 0-20: Completely wrong or irrelevant
        - 21-40: Partially correct but major issues
        - 41-60: Somewhat aligned but significant gaps
        - 61-80: Good alignment with minor differences
        - 81-100: Excellent alignment, equivalent or better

        Respond with ONLY a number between 0 and 100.  No explanation."""

        # Generate with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        max_output_tokens=10,
                    )
                )
                
                # Check if response was blocked
                if not response.candidates:
                    print(f"[WARN] Gemini response blocked (no candidates)")
                    return 0.0
                
                candidate = response.candidates[0]
                
                # Check finish reason
                # 0=FINISH_REASON_UNSPECIFIED, 1=STOP (normal), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                if candidate.finish_reason == 3:  # SAFETY
                    print(f"[WARN] Gemini blocked response due to safety filters")
                    return 0.0
                elif candidate.finish_reason not in [1, 2]:  # Not STOP or MAX_TOKENS
                    print(f"[WARN] Unexpected finish_reason: {candidate.finish_reason}")
                    return 0.0
                
                # Extract score
                if not response.text:
                    print(f"[WARN] Empty response text")
                    return 0.0
                    
                score_text = response.text.strip()
                
                # Parse score (handle various formats)
                import re
                numbers = re.findall(r'\d+', score_text)
                if numbers:
                    score = float(numbers[0])
                    # Clamp to 0-100
                    score = max(0.0, min(100.0, score))
                    return score
                else:
                    print(f"[WARN] Could not parse score from: '{score_text}'")
                    return 0.0
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg. lower() or "resource_exhausted" in error_msg. lower():
                    # Rate limit hit
                    if attempt < max_retries - 1:
                        print(f"[WARN] Rate limit hit, retrying in {retry_delay}s...  (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"[ERROR] Rate limit exceeded after {max_retries} attempts")
                        return 0.0
                else:
                    raise
        
        return 0.0
        
    except ImportError:
        print("[ERROR] google-generativeai not installed. Run: pip install google-generativeai")
        return 0.0
    except Exception as e:
        print(f"[ERROR] Compliance scoring with Gemini failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def compute_ripple_effect_recall(model_answer: str, ground_truth: str) -> Dict[str, float]:
    """Ripple Effect Recall with improved regex-based component extraction."""
    model_components = extract_components(model_answer)
    truth_components = extract_components(ground_truth)

    if len(truth_components) == 0:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0,
            "model_components_count": len(model_components), 
            "truth_components_count": 0
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
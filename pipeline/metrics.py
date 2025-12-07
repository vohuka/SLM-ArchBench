"""
All metric computation functions: NLG + Research metrics.
Improved: corpus-level BLEU (sacrebleu), robust METEOR handling,
batched BERTScore with device selection, and better fallbacks.
"""
import warnings
import logging
import numpy as np
import time
import torch
from typing import List, Dict, Iterable

from utils import extract_components
from config import JUDGE_MODEL, GEMINI_API_KEYS
import itertools

# Suppress BERTScore warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
logging.getLogger("transformers").setLevel(logging.ERROR)
_api_key_cycle = itertools.cycle(GEMINI_API_KEYS) if GEMINI_API_KEYS else None

# Pre-download NLTK data for METEOR
def _ensure_nltk_data():
    """Ensure NLTK data is downloaded for METEOR."""
    try:
        import nltk
        resources = ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}' if resource in ['wordnet', 'omw-1.4'] else f'tokenizers/{resource}')
            except LookupError:
                print(f"[INFO] Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
    except ImportError:
        pass

# Run at import time
_ensure_nltk_data()


def _batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def compute_nlg_metrics(predictions: List[str], references: List[str], bert_batch_size: int = 64) -> Dict[str, float]:
    """Compute ROUGE, BLEU, METEOR, BERTScore.Returns a dict of metrics."""
    metrics = {}

    # Basic checks
    if len(predictions) != len(references):
        print("[WARN] compute_nlg_metrics: predictions and references length mismatch.")
    if len(predictions) == 0:
        print("[WARN] compute_nlg_metrics: empty predictions.Returning zeros.")
        return {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'bleu': 0.0, 'meteor': 0.0,
            'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0
        }

    # -------------------------
    # ROUGE (using rouge_score)
    # -------------------------
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        metrics['rouge1'] = float(np.mean(rouge1_scores)) if rouge1_scores else 0.0
        metrics['rouge2'] = float(np.mean(rouge2_scores)) if rouge2_scores else 0.0
        metrics['rougeL'] = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0
    except ImportError:
        print("[WARN] rouge-score not installed.Skipping ROUGE.")
        metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0

    # -------------------------
    # BLEU
    # -------------------------
    try:
        import sacrebleu
        bleu_score = sacrebleu.corpus_bleu(predictions, [references])
        metrics['bleu'] = float(bleu_score.score / 100.0)
    except Exception as e:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            metrics['bleu'] = float(np.mean(bleu_scores)) if bleu_scores else 0.0
            print("[WARN] sacrebleu not available or failed, used nltk sentence_bleu fallback.")
        except ImportError:
            print("[WARN] nltk not installed.Skipping BLEU.")
            metrics['bleu'] = 0.0

    # -------------------------
    # METEOR (nltk) - FIXED
    # -------------------------
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        
        # Ensure all required NLTK data is available
        required_resources = [
            ('corpora', 'wordnet'),
            ('corpora', 'omw-1.4'),
            ('tokenizers', 'punkt'),
            ('tokenizers', 'punkt_tab'),
        ]
        
        for category, resource in required_resources:
            try:
                nltk.data.find(f'{category}/{resource}')
            except LookupError:
                print(f"[INFO] Downloading NLTK {resource}...")
                nltk.download(resource, quiet=True)

        meteor_scores = []
        for pred, ref in zip(predictions, references):
            try:
                # Tokenize both reference and hypothesis
                ref_tokens = word_tokenize(ref.lower())
                pred_tokens = word_tokenize(pred.lower())
                
                # meteor_score expects: references (list of tokenized refs), hypothesis (tokenized)
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            except Exception as e:
                # If tokenization fails, try simple split
                try:
                    ref_tokens = ref.lower().split()
                    pred_tokens = pred.lower().split()
                    score = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores.append(score)
                except:
                    meteor_scores.append(0.0)
        
        metrics['meteor'] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        print(f"[INFO] METEOR computed successfully: {metrics['meteor']:.4f}")
        
    except ImportError as e:
        print(f"[WARN] nltk not installed.Skipping METEOR.Error: {e}")
        metrics['meteor'] = 0.0
    except Exception as e:
        print(f"[WARN] METEOR scoring failed: {e}.Skipping METEOR.")
        metrics['meteor'] = 0.0

    # -------------------------
    # BERTScore (batched, device-aware)
    # -------------------------
    try:
        from bert_score import score as bert_score
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        P_vals = []
        R_vals = []
        F_vals = []
        
        for pred_batch, ref_batch in zip(_batch_iter(predictions, bert_batch_size),
                                         _batch_iter(references, bert_batch_size)):
            P, R, F1 = bert_score(list(pred_batch), list(ref_batch),
                                  lang='en',
                                  device=device,
                                  batch_size=min(bert_batch_size, 64),
                                  rescale_with_baseline=False,
                                  verbose=False)
            P_vals.extend(P.detach().cpu().tolist())
            R_vals.extend(R.detach().cpu().tolist())
            F_vals.extend(F1.detach().cpu().tolist())

        n = min(len(predictions), len(P_vals))
        if n == 0:
            metrics['bertscore_precision'] = metrics['bertscore_recall'] = metrics['bertscore_f1'] = 0.0
        else:
            metrics['bertscore_precision'] = float(np.mean(P_vals[:n]))
            metrics['bertscore_recall'] = float(np.mean(R_vals[:n]))
            metrics['bertscore_f1'] = float(np.mean(F_vals[:n]))
    except ImportError:
        print("[WARN] bert-score not installed.Skipping BERTScore.")
        metrics['bertscore_precision'] = metrics['bertscore_recall'] = metrics['bertscore_f1'] = 0.0
    except Exception as e:
        print(f"[WARN] BERTScore computation failed: {e}.Skipping BERTScore.")
        metrics['bertscore_precision'] = metrics['bertscore_recall'] = metrics['bertscore_f1'] = 0.0

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
        print("[WARN] sentence-transformers not installed.Skipping Diversity Score.")
        return 0.0


def compute_compliance_score(model_answer: str, ground_truth: str) -> float:
    """
    Compliance Score using Gemini 2.5 Flash as LLM-as-a-Judge.
    Rotates between multiple API keys to increase rate limit.
    """
    global _api_key_cycle
    
    if not GEMINI_API_KEYS:
        print("[WARN] No GEMINI_API_KEY set.Skipping Compliance Score.")
        return 0.0
    
    try:
        import google.generativeai as genai
        
        current_api_key = next(_api_key_cycle)
        genai.configure(api_key=current_api_key)
        
        model = genai.GenerativeModel(JUDGE_MODEL)
        
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

        Respond with ONLY a number between 0 and 100.No explanation."""

        max_retries = 5
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        max_output_tokens=65536,
                    )
                )
                
                if not response.candidates:
                    print(f"[WARN] Gemini response blocked (no candidates)")
                    return 0.0
                
                candidate = response.candidates[0]
                
                if candidate.finish_reason == 3:
                    print(f"[WARN] Gemini blocked response due to safety filters")
                    return 0.0
                
                score_text = ""
                
                if candidate.finish_reason == 1:
                    try:
                        score_text = response.text.strip()
                    except:
                        pass
                
                if not score_text:
                    try:
                        if candidate.content and candidate.content.parts:
                            parts_text = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    parts_text.append(part.text)
                            score_text = "".join(parts_text).strip()
                    except Exception as e:
                        print(f"[WARN] Failed to extract from parts: {e}")
                
                if score_text:
                    import re
                    numbers = re.findall(r'\d+', score_text)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.0, min(100.0, score))
                        return score
                    else:
                        print(f"[WARN] Could not parse score from: '{score_text}'")
                        return 0.0
                else:
                    print(f"[WARN] Empty response (finish_reason={candidate.finish_reason})")
                    return 0.0
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
                    if attempt < max_retries - 1:
                        current_api_key = next(_api_key_cycle)
                        genai.configure(api_key=current_api_key)
                        print(f"[WARN] Rate limit hit, switching API key and retrying in {retry_delay}s...(attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[ERROR] Rate limit exceeded after {max_retries} attempts")
                        return 0.0
                else:
                    print(f"[WARN] Unexpected error in compliance scoring: {e}")
                    return 0.0
        
        return 0.0
        
    except ImportError:
        print("[ERROR] google-generativeai not installed.Run: pip install google-generativeai")
        return 0.0
    except Exception as e:
        print(f"[ERROR] Compliance scoring with Gemini failed: {e}")
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
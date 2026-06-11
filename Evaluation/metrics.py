"""
metrics.py — Evaluation metrics for Medical QA
Based on methodology from:
  - BioMistral (Labrak et al., 2024): ROUGE, BERTScore on medical QA benchmarks
  - Meditron (Chen et al., 2023): Exact Match, F1-token overlap
  - RAG medical QA (Scientific Reports, 2025): ROUGE-L F1, BERTScore-F1 as primary

Research notes:
  • Exact-Match is near-zero for free-form medical QA (answers rarely match word
    for word). It is reported because it is standard, but Token-F1, ROUGE-L and
    BERTScore-F1 are the informative metrics for open-ended answers.
  • Normalisation follows the SQuAD convention (lowercase, drop articles,
    strip punctuation, collapse whitespace).
  • BERTScore uses microsoft/deberta-xlarge-mnli (best human correlation for
    English; recommended in the BERTScore paper) with an automatic fallback to
    roberta-large if the larger model cannot be loaded. The SAME model is used
    for every system so the comparison stays fair.
"""

import os
import re
import string
from collections import Counter
from rouge_score import rouge_scorer


def _safe_mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ─── Token-level F1 (standard in SQuAD / MedQA) ─────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, extra whitespace (SQuAD-style)."""
    s = (s or "").lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ─── ROUGE scores ─────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L F1.
    Primary metric per (Scientific Reports, 2025): ROUGE-L F1.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref or "", pred or "")
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
    return {
        'ROUGE-1': round(_safe_mean(r1_scores), 4),
        'ROUGE-2': round(_safe_mean(r2_scores), 4),
        'ROUGE-L': round(_safe_mean(rl_scores), 4),
    }


# ─── BERTScore (requires bert_score package + torch) ─────────────────────────

# Primary model + ordered fallbacks. Same model is applied to every system.
_BERTSCORE_MODELS = [
    os.environ.get('BERTSCORE_MODEL', 'microsoft/deberta-xlarge-mnli'),
    'roberta-large',
]


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """
    BERTScore-F1 — the most informative metric for medical QA, capturing
    semantic similarity / paraphrase that ROUGE misses
    (Labrak et al., 2024; biomedical RAG, 2025).
    """
    try:
        from bert_score import score as bs_score
    except ImportError:
        print("  [WARN] bert_score not installed — skipping BERTScore. "
              "Install with: pip install bert-score")
        return {'BERTScore-P': None, 'BERTScore-R': None, 'BERTScore-F1': None}

    last_err = None
    for model_type in _BERTSCORE_MODELS:
        try:
            P, R, F1 = bs_score(
                predictions, references,
                model_type=model_type,
                lang='en', verbose=False,
            )
            return {
                'BERTScore-P': round(P.mean().item(), 4),
                'BERTScore-R': round(R.mean().item(), 4),
                'BERTScore-F1': round(F1.mean().item(), 4),
                'BERTScore-model': model_type,
            }
        except Exception as e:
            last_err = e
            print(f"  [WARN] BERTScore with '{model_type}' failed ({e}); trying next.")
    print(f"  [WARN] BERTScore failed on all models: {last_err}")
    return {'BERTScore-P': None, 'BERTScore-R': None, 'BERTScore-F1': None}


# ─── Aggregate all metrics ────────────────────────────────────────────────────

def evaluate_all(predictions: list[str], references: list[str],
                 use_bertscore: bool = True) -> dict:
    """
    Returns a flat dict of all metrics for a model's outputs.
    Standard in medical LLM evaluation papers (BioMistral, Meditron, RAG surveys).
    """
    if not predictions or not references:
        return {'Exact-Match': 0.0, 'Token-F1': 0.0,
                'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    rouge = compute_rouge(predictions, references)
    metrics = {
        'Exact-Match': round(_safe_mean(em_scores), 4),
        'Token-F1':    round(_safe_mean(f1_scores), 4),
        **rouge,
    }
    if use_bertscore:
        bs = compute_bertscore(predictions, references)
        bs.pop('BERTScore-model', None)   # keep the metrics table clean
        metrics.update(bs)
    return metrics

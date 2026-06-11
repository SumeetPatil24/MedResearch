"""
evaluate_kg_v1.py
-----------------
Standalone evaluation script for medical_kg.json (built by KG.py)
using kg_pipeline.py (KGRetriever + GroqKGGenerator + PureKGFormatter).

Evaluates two systems:
  1. KG + Llama-3.1 (Groq)  — KGRetriever → GroqKGGenerator
  2. Pure KG                 — KGRetriever → PureKGFormatter (no LLM)

Metrics:
  Automatic : Exact-Match, Token-F1, ROUGE-1, ROUGE-2, ROUGE-L
  LLM Judge : Escalation-Accuracy, Red-Flag-Recall, Unsafe-Rate,
              Hallucination-Rate, Guideline-Adherence, Explainability,
              Logical-Consistency  (GPT-4o via OpenAI API)

Kept completely separate from evaluate_pipeline3.py / kg_pipeline3.py.
No imports from those files.

Usage (Kaggle / terminal):
    # Automatic metrics only (free, no API key needed)
    python evaluate_kg_v1.py \\
        --kg medical_kg.json \\
        --questions firstaidqa_relevant_500.json \\
        --models kg_llm kg_pure \\
        --output_dir results_kgv1

    # With LLM judge
    python evaluate_kg_v1.py \\
        --kg medical_kg.json \\
        --questions firstaidqa_relevant_500.json \\
        --models kg_llm kg_pure \\
        --llm_judge \\
        --output_dir results_kgv1_judge

    # Quick pilot run (50 samples)
    python evaluate_kg_v1.py \\
        --kg medical_kg.json \\
        --questions firstaidqa_relevant_500.json \\
        --models kg_llm kg_pure \\
        --n_samples 50 \\
        --output_dir results_kgv1_pilot

Notebook cell:
    from evaluate_kg_v1 import run
    run(kg_path='medical_kg.json',
        questions_path='firstaidqa_relevant_500.json',
        models=['kg_llm', 'kg_pure'],
        llm_judge=True,
        output_dir='results_kgv1')
"""

import os
import json
import re
import csv
import string
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# 1. Metric helpers  (self-contained — no dependency on metrics.py)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lowercase, drop articles, strip punctuation."""
    s = (s or "").lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common  = Counter(pred_tokens) & Counter(gt_tokens)
    n_same  = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall    = n_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_rouge(predictions: list, references: list) -> dict:
    try:
        from rouge_score import rouge_scorer as rs
    except ImportError:
        print("  [WARN] rouge-score not installed. pip install rouge-score")
        return {'ROUGE-1': None, 'ROUGE-2': None, 'ROUGE-L': None}
    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref or "", pred or "")
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    return {
        'ROUGE-1': round(_safe_mean(r1), 4),
        'ROUGE-2': round(_safe_mean(r2), 4),
        'ROUGE-L': round(_safe_mean(rl), 4),
    }

def compute_auto_metrics(predictions: list, references: list) -> dict:
    """All automatic metrics — no API needed."""
    predictions = [p if p and p.strip() else " " for p in predictions]
    references  = [r if r and r.strip() else " " for r in references]
    em  = [exact_match(p, r)  for p, r in zip(predictions, references)]
    f1s = [token_f1(p, r)     for p, r in zip(predictions, references)]
    acc = [float(token_f1(p, r) > 0.0) for p, r in zip(predictions, references)]
    rec = [
        sum((Counter(normalize_answer(p).split()) &
             Counter(normalize_answer(r).split())).values()) /
        max(len(normalize_answer(r).split()), 1)
        for p, r in zip(predictions, references)
    ]
    rouge = compute_rouge(predictions, references)
    return {
        'Exact-Match': round(_safe_mean(em),  4),
        'Token-F1':    round(_safe_mean(f1s), 4),
        'Accuracy':    round(_safe_mean(acc), 4),
        'Recall':      round(_safe_mean(rec), 4),
        **rouge,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. LLM Judge  (imported from llm_judge2.py — single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

def _get_judge(openai_key: str = ''):
    """
    Returns the shared judge from llm_judge2.py.
    Uses the same prompt, model, and scoring guide as evaluate_pipeline3.py
    so results are directly comparable across both KG stacks.
    """
    import os
    # Resolve key — explicit arg > env var
    key = openai_key or os.environ.get('OPENAI_API_KEY', '')
    if key:
        os.environ['OPENAI_API_KEY'] = key  # ensure it's in env for llm_judge2
    from llm_judge2 import get_judge
    return get_judge(openai_key=key)  # pass explicitly, don't rely on module-level var

def _call_judge(judge, question: str, reference: str,
                prediction: str, sample_idx: int = -1) -> dict:
    if sample_idx > 0:
        print(f"    Judging [{sample_idx}]...")
    raw = judge.judge_sample(question, prediction, reference, sample_idx=sample_idx)
    return raw

def _aggregate_judge(judge_results: list) -> dict:
    """Average judge scores across all samples."""
    keys = [
        'escalation_accuracy', 'red_flag_recall', 'unsafe_rate',
        'hallucination_rate',  'guideline_adherence',
        'explainability',      'logical_consistency',
    ]
    agg = {}
    for k in keys:
        vals = [r[k] for r in judge_results if isinstance(r.get(k), (int, float))]
        agg[k] = round(_safe_mean(vals), 4) if vals else None
    return agg


# 3. Dataset loader
# ══════════════════════════════════════════════════════════════════════════════

def load_questions(path: str, n_samples: int = None) -> list[dict]:
    """
    Load pre-saved question set. Accepts:
      - List of {"question": ..., "answer": ...} dicts
      - HuggingFace-style {"train": [...]} export
    """
    print(f"Loading questions from {path} ...")
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        key = next(iter(raw))
        raw = raw[key]
        print(f"  Detected dict format — using split '{key}'")
    normalised = []
    for row in raw:
        q_key = next((k for k in row if k.lower() == 'question'), None)
        a_key = next((k for k in row if k.lower() in ('answer', 'answers')), None)
        if q_key and a_key:
            normalised.append({'question': row[q_key], 'answer': str(row[a_key])})
    if n_samples:
        normalised = normalised[:n_samples]
    print(f"  Loaded {len(normalised)} questions.")
    return normalised


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model wrappers — thin wrappers over kg_pipeline.py classes
# ══════════════════════════════════════════════════════════════════════════════

def build_models(model_keys: list, kg_path: str,
                 groq_api_key: str = '') -> dict:
    """
    Instantiate the requested pipeline objects from kg_pipeline.py.
    Returns dict of model_key → pipeline object.
    """
    import sys
    sys.path.insert(0, '.')   # ensure kg_pipeline.py is importable

    pipelines = {}

    for key in model_keys:
        if key == 'kg_llm':
            print(f"\n  Building KG + BioMistral (local) pipeline...")
            from kg_pipeline import KGPipeline
            pipelines[key] = KGPipeline(
                kg_path=kg_path,
                groq_api_key=groq_api_key,
                prefer_local=True,   # uses biomistral_backend singleton
            )
            print(f"  ✅ KG + BioMistral pipeline ready.")

        elif key == 'kg_pure':
            print(f"\n  Building Pure KG pipeline...")
            from kg_pipeline import PureKGPipeline
            pipelines[key] = PureKGPipeline(kg_path=kg_path)
            print(f"  ✅ Pure KG pipeline ready.")

        else:
            print(f"  [WARN] Unknown model key '{key}' — skipping.")

    return pipelines


# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_LABELS = {
    'kg_llm':  'KG + BioMistral-7B (local)',
    'kg_pure': 'Pure KG (no LLM)',
}


def evaluate_model(model_key: str, pipeline, questions: list,
                   judge=None) -> dict:
    """
    Run a single pipeline over all questions.
    Returns dict with predictions list, references list, and all metric scores.
    """
    label = SYSTEM_LABELS.get(model_key, model_key)
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}  ({len(questions)} questions)")
    print(f"{'='*60}")

    predictions  = []
    references   = []
    sample_rows  = []
    judge_raw    = []

    for i, row in enumerate(questions):
        q   = row['question']
        ref = row['answer']

        print(f"  [{i+1}/{len(questions)}] {q[:70]}...")

        # Generate prediction
        try:
            pred = pipeline.generate_answer_only(q)
        except Exception as e:
            print(f"  [WARN] Generation failed for sample {i+1}: {e}")
            pred = ""

        predictions.append(pred)
        references.append(ref)

        # Per-sample metrics
        em = exact_match(pred, ref)
        f1 = token_f1(pred, ref)

        sample_row = {
            'sample_id':  i + 1,
            'model':      model_key,
            'question':   q,
            'reference':  ref,
            'prediction': pred,
            'exact_match': em,
            'token_f1':   f1,
        }

        # LLM judge (per sample)
        if judge:
            scores = _call_judge(judge, q, ref, pred, sample_idx=i+1)
            sample_row.update(scores)
            judge_raw.append(scores)

        sample_rows.append(sample_row)

    # Aggregate automatic metrics
    auto_metrics = compute_auto_metrics(predictions, references)

    # Aggregate judge metrics
    judge_metrics = {}
    if judge and judge_raw:
        judge_metrics = _aggregate_judge(judge_raw)

    results = {
        'model_key':   model_key,
        'label':       label,
        'n_samples':   len(questions),
        'predictions': predictions,
        'references':  references,
        'sample_rows': sample_rows,
        'metrics': {**auto_metrics, **judge_metrics},
    }

    # Print results
    print(f"\nResults for {label}:")
    for metric, val in results['metrics'].items():
        display = f"{val:.4f}" if isinstance(val, float) else str(val)
        # Format metric name nicely
        name = metric.replace('_', '-').title()
        print(f"    {name}: {display}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_results(all_results: list[dict], output_dir: str):
    """Save per-sample CSV, summary JSON, and human-readable results table."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Per-sample CSV ────────────────────────────────────────────────────────
    all_rows = []
    for r in all_results:
        all_rows.extend(r['sample_rows'])

    if all_rows:
        csv_path = os.path.join(output_dir, f'eval_results_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-sample CSV saved: {csv_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        'timestamp':  timestamp,
        'n_questions': all_results[0]['n_samples'] if all_results else 0,
        'systems': {
            r['model_key']: {
                'label':   r['label'],
                'metrics': r['metrics'],
            }
            for r in all_results
        }
    }
    json_path = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved:    {json_path}")

    # ── Human-readable results table ──────────────────────────────────────────
    lines = [
        f"KG v1 Evaluation Results",
        f"{'='*70}",
        f"Timestamp : {timestamp}",
        f"Questions : {summary['n_questions']}",
        f"{'='*70}",
        "",
    ]

    # Collect all metric names across all systems
    all_metrics = []
    for r in all_results:
        for k in r['metrics']:
            if k not in all_metrics:
                all_metrics.append(k)

    # Header row
    col_w = 28
    header = f"{'Metric':<28}" + "".join(
        f"{SYSTEM_LABELS.get(r['model_key'], r['model_key']):<22}"
        for r in all_results
    )
    lines.append(header)
    lines.append("-" * (28 + 22 * len(all_results)))

    for metric in all_metrics:
        name = metric.replace('_', '-').title()
        row = f"{name:<28}"
        for r in all_results:
            val = r['metrics'].get(metric)
            row += f"{val:<22.4f}" if isinstance(val, float) else f"{'N/A':<22}"
        lines.append(row)

    table_path = os.path.join(output_dir, f'results_table_{timestamp}.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Results table saved:   {table_path}")
    print('\n' + '\n'.join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Entry points
# ══════════════════════════════════════════════════════════════════════════════

def run(kg_path: str = 'medical_kg.json',
        questions_path: str = 'firstaidqa_relevant_500.json',
        models: list = None,
        n_samples: int = None,
        llm_judge: bool = False,
        output_dir: str = 'results_kgv1',
        groq_api_key: str = '',
        openai_api_key: str = '') -> dict:
    """
    Notebook-friendly entry point.

    Example
    -------
    from evaluate_kg_v1 import run
    results = run(
        kg_path='medical_kg.json',
        questions_path='firstaidqa_relevant_500.json',
        models=['kg_llm', 'kg_pure'],
        llm_judge=True,
        output_dir='results_kgv1',
        groq_api_key='gsk_...',
        openai_api_key='sk-...',
    )
    """
    if models is None:
        models = ['kg_llm', 'kg_pure']

    # Resolve API keys from env if not passed directly
    groq_key   = groq_api_key   or os.environ.get('GROQ_API_KEY', '')
    openai_key = openai_api_key or os.environ.get('OPENAI_API_KEY', '')

    print("\nKG v1 Evaluation Pipeline")
    print("=" * 40)
    print(f"KG path       : {kg_path}")
    print(f"Questions     : {questions_path}")
    print(f"Models        : {models}")
    print(f"Samples       : {n_samples or 'all'}")
    print(f"LLM Judge     : {'enabled (GPT-4o)' if llm_judge else 'disabled'}")
    print(f"Output dir    : {output_dir}")
    if llm_judge and not openai_key:
        print("  ⚠️  --llm_judge set but OPENAI_API_KEY not found — judge will be skipped.")
        llm_judge = False
    print()

    # Load questions
    questions = load_questions(questions_path, n_samples=n_samples)

    # Build pipelines
    pipelines = build_models(models, kg_path=kg_path, groq_api_key=groq_key)

    # Set up judge
    judge = _get_judge(openai_key) if llm_judge else None
    if llm_judge and not judge:
        print("  ⚠️  Judge could not be initialised — check OPENAI_API_KEY.")

    # Evaluate each model
    all_results = []
    for key in models:
        if key not in pipelines:
            print(f"  [WARN] No pipeline built for '{key}' — skipping.")
            continue
        result = evaluate_model(key, pipelines[key], questions, judge=judge)
        all_results.append(result)

    # Save outputs
    if all_results:
        save_results(all_results, output_dir)

    return {r['model_key']: r['metrics'] for r in all_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kg',          default='medical_kg.json',
                        help='Path to KG JSON file (default: medical_kg.json)')
    parser.add_argument('--questions',   default='firstaidqa_relevant_500.json',
                        help='Path to pre-saved questions JSON')
    parser.add_argument('--models',      nargs='+',
                        choices=['kg_llm', 'kg_pure'],
                        default=['kg_llm', 'kg_pure'],
                        help='Systems to evaluate (default: both)')
    parser.add_argument('--n_samples',   type=int, default=None,
                        help='Limit to first N samples (default: all)')
    parser.add_argument('--llm_judge',   action='store_true',
                        help='Enable GPT-4o LLM judge (needs OPENAI_API_KEY)')
    parser.add_argument('--output_dir',  default='results_kgv1',
                        help='Directory to save results (default: results_kgv1)')
    args = parser.parse_args()

    run(
        kg_path=args.kg,
        questions_path=args.questions,
        models=args.models,
        n_samples=args.n_samples,
        llm_judge=args.llm_judge,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()

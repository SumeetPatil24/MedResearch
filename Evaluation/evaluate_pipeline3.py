"""
evaluate_pipeline.py — Unified Evaluation Runner for FirstAidQA
================================================================
Evaluates the following systems on the FirstAidQA dataset:
  1. Generic LLM:     Llama-3 (Meta-Llama-3-8B-Instruct)
  2. Medical LLM:     Meditron-7B  (Chen et al., 2023, EPFL)
  3. Lightweight Med: BioMistral-7B (Labrak et al., 2024)
  4. Vanilla RAG:     Llama-3 + PubMed retrieval

Metrics (per BioMistral & Meditron evaluation standards):
  - Exact Match (EM)
  - Token-F1
  - ROUGE-1 / ROUGE-2 / ROUGE-L F1
  - BERTScore-F1 (primary metric for open-ended QA)

Usage:
    # With HuggingFace Inference API (recommended for research):
    python evaluate_pipeline.py --backend hf_api --hf_token hf_xxx --n_samples 50

    # With Ollama (local, no internet needed after model pull):
    python evaluate_pipeline.py --backend ollama --n_samples 100

    # Dry run with mock models (test the pipeline without any API):
    python evaluate_pipeline.py --dry_run --n_samples 5
    
    # Skip BERTScore (faster, no torch needed):
    python evaluate_pipeline.py --dry_run --no_bertscore

Output:
    results/eval_results.csv     — per-sample predictions and scores
    results/summary.json         — aggregated metrics table (for paper)
    results/summary_table.txt    — formatted ASCII table for quick review
"""

import os
import json
import time
import argparse
import csv
from datetime import datetime
from typing import Optional

from metrics import evaluate_all, token_f1, exact_match, compute_rouge


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on FirstAidQA')
    parser.add_argument('--backend', choices=['local', 'groq', 'hf_api', 'ollama', 'mock'],
                        default='local', help='Inference backend. '
                        "'local' downloads BioMistral-7B and runs it on-device "
                        "(unlimited calls); falls back to API if no GPU.")
    parser.add_argument('--groq_api_key', type=str, default='',
                        help='Groq API key (or set GROQ_API_KEY env var). Get free key: https://console.groq.com')
    parser.add_argument('--openai_api_key', type=str, default='',
                        help='OpenAI API key for GPT-4o-mini (KG generator + judge). Set OPENAI_API_KEY env var or paste in kg_pipeline.py/llm_judge.py')
    parser.add_argument('--hf_token', type=str, default='',
                        help='HuggingFace token (needed for meditron via HF API, or set HF_TOKEN env var)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Limit evaluation to N samples (default: full dataset)')
    parser.add_argument('--models', nargs='+',
                        choices=['llama3', 'llama', 'meditron', 'biomistral', 'rag', 'kg', 'kg_pure'],
                        default=['biomistral', 'kg', 'kg_pure'],
                        help='Which systems to evaluate. Default focuses on '
                             'BioMistral baseline + KG+BioMistral + Pure KG. '
                             "Add 'llama' for local Llama-3.1-8B-Instruct standalone baseline. "
                             "Add 'llama3'/'meditron'/'rag' for more baselines "
                             '(meditron needs --hf_token; rag needs rag_pipeline.py).')
    parser.add_argument('--top_k_rag', type=int, default=3,
                        help='Number of PubMed docs to retrieve for RAG')
    parser.add_argument('--ncbi_api_key', type=str, default=None,
                        help='NCBI API key for higher PubMed rate limits (optional)')
    parser.add_argument('--kg_path', type=str, default='medical_kg_v3.json',
                        help='Path to medical_kg_v3.json for KG pipeline')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='Path to a pre-saved JSON file of questions to evaluate on. '
                             'Expected format: list of {"question": ..., "answer": ...} dicts '
                             '(i.e. the output of your 500-question selection script). '
                             'When provided, --n_samples is ignored.')
    parser.add_argument('--no_bertscore', action='store_true',
                        help='Skip BERTScore computation (faster, no torch needed)')
    parser.add_argument('--llm_judge', action='store_true',
                        help='Run LLM-as-judge metrics (Escalation, Safety, Hallucination etc.)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Use mock models (no API calls) for pipeline testing')
    return parser.parse_args()


# ─── Mock model for dry-run testing ───────────────────────────────────────────

class MockModel:
    """Returns deterministic fake answers for pipeline testing."""
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # Return a slightly varied answer to simulate model output
        responses = {
            'llama3':    "Apply direct pressure to stop bleeding and call emergency services.",
            'meditron':  "Perform basic life support. Ensure airway is clear. Monitor vital signs.",
            'biomistral':"Administer appropriate first aid and seek medical attention immediately.",
        }
        for key in responses:
            if key in self.name.lower():
                return responses[key]
        if 'kg_pure' in self.name.lower() or self.name == 'kg_pure':
            return "CPR: 1. Check responsiveness. 2. Call emergency services. 3. Give 30 chest compressions. 4. Give 2 rescue breaths. Repeat."
        if 'kg' in self.name.lower():
            return "Apply pressure to the wound. Perform CPR if unresponsive. Call emergency services immediately."
        return "Seek immediate medical attention and call emergency services."

    def generate_answer_only(self, q: str, **kw) -> str:
        return self.generate(q)


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_firstaid_qa(n_samples: Optional[int] = None) -> list[dict]:
    """
    Load FirstAidQA from HuggingFace datasets hub.
    Dataset: i-am-mushfiq/FirstAidQA
    Falls back to a built-in sample set if Hub is unreachable.
    """
    try:
        from datasets import load_dataset
        print("Loading i-am-mushfiq/FirstAidQA from HuggingFace Hub...")
        ds = load_dataset("i-am-mushfiq/FirstAidQA")
        # Detect split: use 'test' if available, else 'train'
        split = 'test' if 'test' in ds else list(ds.keys())[0]
        data = list(ds[split])
        print(f"  Loaded {len(data)} samples from split '{split}'")
    except Exception as e:
        print(f"  [WARN] Could not load from Hub ({e}). Using built-in sample data.")
        # Built-in sample — replace with actual dataset when online
        data = SAMPLE_DATA

    # Normalise column names — dataset uses 'question'/'answer' or 'Question'/'Answer'
    normalised = []
    for row in data:
        q_key = next((k for k in row if k.lower() == 'question'), None)
        a_key = next((k for k in row if k.lower() in ('answer', 'answers')), None)
        if q_key and a_key:
            normalised.append({'question': row[q_key], 'answer': str(row[a_key])})

    if n_samples:
        normalised = normalised[:n_samples]
    print(f"  Using {len(normalised)} samples for evaluation.\n")
    return normalised


def load_questions_from_file(path: str) -> list[dict]:
    """
    Load a pre-saved question set from a JSON file.

    Accepts two formats:
      1. List of {"question": ..., "answer": ...} dicts  ← direct output of
         your 500-question selection script; no conversion needed.
      2. HuggingFace-style export with arbitrary key casing
         (e.g. "Question"/"Answer" or "question"/"answers") — normalised
         the same way as load_firstaid_qa does it.

    --n_samples is intentionally ignored when --questions_file is used:
    the file *is* your fixed evaluation set and should not be truncated.
    """
    print(f"Loading pre-saved question set from: {path}")
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    # raw can be a flat list or a HuggingFace dataset export dict {"train": [...]}
    if isinstance(raw, dict):
        key = next(iter(raw))
        raw = raw[key]
        print(f"  Detected dict format — using split key '{key}'")

    normalised = []
    for row in raw:
        q_key = next((k for k in row if k.lower() == 'question'), None)
        a_key = next((k for k in row if k.lower() in ('answer', 'answers')), None)
        if q_key and a_key:
            normalised.append({'question': row[q_key], 'answer': str(row[a_key])})
        else:
            print(f"  [WARN] Skipping row with unrecognised keys: {list(row.keys())}")

    if not normalised:
        raise ValueError(
            f"No valid question/answer pairs found in {path}. "
            "Each entry must have 'question' and 'answer' (or 'answers') keys."
        )

    print(f"  Loaded {len(normalised)} questions from file.\n")
    return normalised


# Built-in fallback sample (representative first-aid questions)
SAMPLE_DATA = [
    {'question': 'What should you do if someone is choking?',
     'answer': 'Perform the Heimlich maneuver. Stand behind the person, wrap arms around waist, make a fist above navel, and give quick upward thrusts until dislodged. For infants, use back blows and chest thrusts.'},
    {'question': 'How do you treat a minor burn?',
     'answer': 'Cool the burn under cool running water for at least 10 minutes. Do not use ice, butter, or toothpaste. Cover loosely with a sterile non-stick bandage.'},
    {'question': 'What are the signs of a heart attack?',
     'answer': 'Chest pain or pressure, shortness of breath, pain radiating to arm or jaw, cold sweats, nausea, lightheadedness. Call emergency services (911) immediately. Give aspirin if not allergic.'},
    {'question': 'How do you stop severe bleeding?',
     'answer': 'Apply firm direct pressure with a clean cloth. Maintain pressure for at least 15 minutes. Do not remove cloth; add more on top. Elevate the limb if possible. Apply tourniquet for life-threatening limb bleeding.'},
    {'question': 'What is the recovery position?',
     'answer': 'Place unconscious breathing casualty on their side. Lower arm extended forward, upper knee bent to stabilize. Tilt head back gently to keep airway open. Check breathing regularly.'},
    {'question': 'How do you treat a sprained ankle?',
     'answer': 'Follow RICE: Rest the ankle, Ice for 20 minutes every 2 hours, Compress with bandage, Elevate above heart level. Avoid weight bearing. See doctor if severe.'},
    {'question': 'What should you do if someone has a seizure?',
     'answer': 'Protect from injury by clearing hard objects. Place something soft under head. Do not restrain or put anything in mouth. Time the seizure. Call 911 if it lasts >5 minutes or person does not regain consciousness.'},
    {'question': 'How do you perform CPR on an adult?',
     'answer': 'Call 911. Place heel of hand on center of chest, interlock fingers. Push down 2-2.4 inches at 100-120 compressions per minute. Give 2 rescue breaths after every 30 compressions if trained. Use AED as soon as available.'},
    {'question': 'What is the treatment for anaphylaxis?',
     'answer': 'Administer epinephrine (EpiPen) immediately into outer thigh. Call 911. Lay person flat with legs elevated unless breathing difficulties. Be ready to perform CPR. Give second EpiPen after 5-15 minutes if no improvement.'},
    {'question': 'How do you treat a broken bone before emergency services arrive?',
     'answer': 'Immobilize the injured area without trying to straighten it. Splint using a rigid material padded with cloth. Apply ice pack wrapped in cloth. Elevate if possible. Treat for shock. Call emergency services.'},
]


# ─── Evaluation loop ──────────────────────────────────────────────────────────

def run_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load dataset — pre-saved file takes priority over HuggingFace Hub
    if args.questions_file:
        dataset = load_questions_from_file(args.questions_file)
    else:
        dataset = load_firstaid_qa(args.n_samples)
    questions = [row['question'] for row in dataset]
    references = [row['answer']  for row in dataset]

    # Build model registry
    models_to_run = {}

    # Shared resolved keys for API calls
    groq_key = args.groq_api_key or os.environ.get('GROQ_API_KEY', '')
    hf_token  = args.hf_token    or os.environ.get('HF_TOKEN', '')

    # Keys handled by get_model (standalone LLM baselines only)
    _LLM_BASELINES = {'llama3', 'llama', 'meditron', 'biomistral'}

    if args.dry_run or args.backend == 'mock':
        print("=== DRY RUN MODE (mock models) ===\n")
        for key in args.models:
            models_to_run[key] = MockModel(key)
    else:
        from models import get_model

        # Standalone LLM baselines
        for key in args.models:
            if key not in _LLM_BASELINES:
                continue
            print(f"Initializing {key} ({args.backend})...")
            try:
                models_to_run[key] = get_model(
                    key,
                    backend=args.backend,
                    groq_api_key=groq_key,
                    hf_token=hf_token,
                )
            except ValueError as e:
                print(f"  [SKIP] {key}: {e}")

        # RAG pipeline
        if 'rag' in args.models:
            try:
                from rag_pipeline import VanillaRAGPipeline
                print("Initializing VanillaRAG (Llama-3 + PubMed)...")
                rag = VanillaRAGPipeline(
                    generator_backend=args.backend,
                    groq_api_key=groq_key,
                    hf_token=hf_token,
                    top_k=args.top_k_rag,
                    ncbi_api_key=args.ncbi_api_key,
                )
                models_to_run['rag'] = rag
            except ImportError:
                print("  [SKIP] rag: rag_pipeline.py not found. "
                      "Place rag_pipeline.py in the same directory to enable RAG.")

        # KG + BioMistral pipeline
        if 'kg' in args.models:
            from kg_pipeline3 import KGPipeline
            print("Initializing KG + BioMistral Pipeline "
                  f"(backend={args.backend})...")
            kg_pipe = KGPipeline(
                kg_path=args.kg_path,
                groq_api_key=groq_key,
                use_biomistral=True,
                prefer_local=(args.backend == 'local'),
            )
            models_to_run['kg'] = kg_pipe

        # Pure KG pipeline (no LLM — answer directly from KG graph)
        if 'kg_pure' in args.models:
            from kg_pipeline3 import PureKGPipeline
            print("Initializing Pure KG Pipeline (no LLM, RedCross KG only)...")
            models_to_run['kg_pure'] = PureKGPipeline(kg_path=args.kg_path)

    # ── Per-model inference and scoring ──────────────────────────────────────
    all_results = {}       # model_key -> list of per-sample dicts
    all_predictions = {}   # model_key -> list of predicted strings

    system_labels = {
        'llama3':    'Generic LLM (Llama-3)',
        'llama':     'Standalone Llama-3.1-8B-Instruct',
        'meditron':  'Medical LLM (Meditron-7B)',
        'biomistral':'Standalone BioMistral-7B',
        'rag':       'Vanilla RAG (Llama-3 + PubMed)',
        'kg':        'Your Method (RedCross KG + BioMistral-7B)',
        'kg_pure':   'Pure KG (RedCross KG, no LLM)',
    }

    for model_key, model in models_to_run.items():
        label = system_labels.get(model_key, model_key)
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        predictions = []
        sample_rows = []

        for i, (q, ref) in enumerate(zip(questions, references)):
            print(f"  [{i+1}/{len(questions)}] {q[:60]}...")

            # Get prediction — routing logic:
            #   1. KG pipelines / Pure KG: use generate_answer_only (no build_prompt)
            #   2. Local BioMistral baseline: model.wants_raw_question = True
            #      → pass raw question so the backend applies its own chat template
            #   3. Groq/HF baseline models: use build_prompt → model.generate(prompt)
            if hasattr(model, 'generate_answer_only'):
                pred = model.generate_answer_only(q)
            elif getattr(model, 'wants_raw_question', False):
                # Local BioMistral baseline — skip build_prompt to avoid double-wrapping
                pred = model.generate(q)
            else:
                from models import build_prompt
                prompt = build_prompt(q, model_key)
                pred = model.generate(prompt)

            # Per-sample metrics
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)

            predictions.append(pred)
            sample_rows.append({
                'model':       model_key,
                'question':    q,
                'reference':   ref,
                'prediction':  pred,
                'exact_match': em,
                'token_f1':    f1,
            })

            # Rate-limit sleep for HF API backend only
            if args.backend == 'hf_api' and not args.dry_run:
                time.sleep(1.2)

        all_predictions[model_key] = predictions
        all_results[model_key] = sample_rows

        # Compute aggregate metrics for this model
        use_bs = not args.no_bertscore
        agg = evaluate_all(predictions, references, use_bertscore=use_bs)

        # LLM-as-judge metrics (optional, uses Groq)
        if args.llm_judge and not args.dry_run:
            print(f"  Running LLM judge for {label}...")
            from llm_judge2 import evaluate_with_judge
            groq_key = args.groq_api_key or os.environ.get('GROQ_API_KEY', '')
            openai_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY', '')
            judge_scores = evaluate_with_judge(
                questions, predictions, references,
                openai_api_key=openai_key,
                groq_api_key=groq_key, verbose=True
            )
            agg.update(judge_scores)

        print(f"\n  Results for {label}:")
        for metric, val in agg.items():
            print(f"    {metric}: {val}")

        # Attach aggregated scores to sample rows for reference
        for row in sample_rows:
            row.update({f'agg_{k}': v for k, v in agg.items()})

    # ── Write per-sample CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f'eval_results_{timestamp}.csv')
    all_rows = [row for rows in all_results.values() for row in rows]
    if all_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-sample results saved: {csv_path}")

    # ── Compute and write aggregated summary ──────────────────────────────────
    summary = {}
    use_bs = not args.no_bertscore
    for model_key, preds in all_predictions.items():
        label = system_labels.get(model_key, model_key)
        summary[label] = evaluate_all(preds, references, use_bertscore=use_bs)

    summary_path = os.path.join(args.output_dir, f'summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved: {summary_path}")

    # ── Print formatted results table ─────────────────────────────────────────
    print_results_table(summary, timestamp, args.output_dir)

    return summary


def print_results_table(summary: dict, timestamp: str, output_dir: str):
    """Prints and saves a publication-style results table."""
    metrics = ['Exact-Match', 'Token-F1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1']
    metrics = [m for m in metrics if any(
        summary[s].get(m) is not None for s in summary
    )]

    # Build table
    col_w = 34
    m_w   = 14
    header = f"{'System':<{col_w}}" + "".join(f"{m:>{m_w}}" for m in metrics)
    sep    = "-" * len(header)

    lines = [
        "\n\n" + "=" * len(header),
        "  EVALUATION RESULTS — FirstAidQA",
        "=" * len(header),
        header,
        sep,
    ]
    for system, scores in summary.items():
        row = f"{system:<{col_w}}"
        for m in metrics:
            val = scores.get(m)
            row += f"{val if val is not None else 'N/A':>{m_w}}"
        lines.append(row)
    lines += [sep, ""]

    table_str = "\n".join(lines)
    print(table_str)

    table_path = os.path.join(output_dir, f'summary_table_{timestamp}.txt')
    with open(table_path, 'w') as f:
        f.write(table_str)
    print(f"Results table saved: {table_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    print("\nFirstAidQA Evaluation Pipeline")
    print("================================")
    print(f"Backend:    {args.backend if not args.dry_run else 'mock (dry run)'}")
    print(f"Models:     {args.models}")
    if args.questions_file:
        print(f"Questions:  {args.questions_file}  (pre-saved file — --n_samples ignored)")
    else:
        print(f"Samples:    {args.n_samples or 'all'}")
    print(f"BERTScore:  {'disabled' if args.no_bertscore else 'enabled'}")
    print(f"LLM Judge:  {'enabled' if args.llm_judge else 'disabled'}")
    print()

    run_evaluation(args)
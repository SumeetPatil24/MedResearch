"""
fetch_firstaidqa.py
-------------------
Fetches ALL rows from the i-am-mushfiq/FirstAidQA dataset on HuggingFace,
filters to the 500 most relevant to the Indian Red Cross Society First Aid Manual,
and saves the result as a CSV and JSON file.

Requirements:
    pip install datasets pandas
    pip install pdfplumber   # optional — only needed if you supply --pdf

── Running as a script (terminal / Colab) ────────────────────────────────────
    python fetch_firstaidqa.py
    python fetch_firstaidqa.py --pdf RedCrossSocietyManual.pdf
    python fetch_firstaidqa.py --top 500 --out my_eval_set

── Running inside a Kaggle / Colab notebook cell ─────────────────────────────
    from fetch_firstaidqa import run
    rows = run()                        # uses built-in keywords, saves to /kaggle/working/
    rows = run(pdf="/kaggle/input/your-dataset/RedCrossSocietyManual.pdf")

Output lands in /kaggle/working/ (shown in the Kaggle output panel on the right).

If you don't have the PDF locally you can skip --pdf and it will use built-in
keyword matching based on the manual's known topic list.
"""

import argparse
import json
import re
import math
import sys
from pathlib import Path
from collections import Counter

# ── 1. Fetch dataset ──────────────────────────────────────────────────────────

def load_dataset_rows() -> list[dict]:
    """Load all rows from i-am-mushfiq/FirstAidQA via the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install the datasets library first:  pip install datasets")

    print("Loading i-am-mushfiq/FirstAidQA from HuggingFace …")
    ds = load_dataset("i-am-mushfiq/FirstAidQA", split="train")
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    print(f"  → {len(rows)} rows loaded.")
    return rows


# ── 2. Extract PDF keywords (optional) ───────────────────────────────────────

def extract_pdf_keywords(pdf_path: str) -> set[str]:
    """Return a set of lowercase content words from the PDF."""
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed – using built-in keyword list instead.")
        return set()

    print(f"Extracting keywords from {pdf_path} …")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "

    # Tokenise: lowercase words ≥ 4 chars, strip numbers/punctuation
    words = re.findall(r"[a-z]{4,}", text.lower())
    freq = Counter(words)

    # Remove very common English stop-words
    STOPWORDS = {
        "this","that","with","from","have","will","when","what","should",
        "person","refer","healthcare","facility","first","also","their",
        "which","they","been","more","than","into","each","some","then",
        "after","before","while","your","about","must","both","call",
        "help","make","take","give","keep","seek","need","does","does",
        "where","there","these","those","other","only","such","away",
        "well","area","body","case","care","pain","side","time","place",
        "able","part","same","like","even","move","back",
    }
    keywords = {w for w, c in freq.items() if c >= 2 and w not in STOPWORDS}
    print(f"  → {len(keywords)} unique content keywords extracted.")
    return keywords


# ── 3. Built-in fallback keyword list ────────────────────────────────────────

MANUAL_TOPICS = {
    # Core first aid concepts
    "first aid", "first aider", "resuscitation", "cpr", "recovery position",
    "unconscious", "unconsciousness", "breathing", "airway", "pulse",
    # Respiratory
    "choking", "drowning", "asthma", "suffocation", "strangulation",
    "hanging", "smoke inhalation", "carbon monoxide",
    # Cardiac / bleeding / shock
    "chest pain", "heart attack", "bleeding", "haemorrhage", "hemorrhage",
    "wound", "shock", "amputation", "crush injury", "internal bleeding",
    "varicose veins", "nose bleed", "nosebleed",
    # Musculoskeletal
    "fracture", "broken bone", "dislocation", "sprain", "strain",
    "spinal injury", "spine", "neck injury", "collarbone", "pelvis",
    "skull fracture", "rib fracture",
    # Nervous system
    "stroke", "seizure", "convulsion", "epilepsy", "concussion",
    "head injury", "cerebral compression",
    # Skin / burns / temperature
    "burn", "scald", "chemical burn", "electrical burn", "sunburn",
    "heat exhaustion", "heatstroke", "hypothermia", "frostbite", "fever",
    # GI / metabolic
    "diarrhoea", "diarrhea", "food poisoning", "diabetes", "hypoglycaemia",
    "hypoglycemia", "hyperglycaemia", "hyperglycemia", "diabetic",
    # Poisoning / bites
    "poisoning", "poison", "snake bite", "snakebite", "animal bite",
    "insect sting", "bee sting", "wasp sting", "scorpion",
    # Eyes / ears / foreign body
    "foreign body", "foreign object", "eye injury", "ear foreign body",
    "nose foreign body", "swallowed object",
    # Reproduction / childbirth
    "emergency childbirth", "labour", "delivery", "pregnant", "pregnancy",
    # Psychological
    "psychological first aid", "trauma", "traumatic",
    # Emergency / disaster
    "triage", "road accident", "traffic accident", "disaster", "evacuation",
    "stretcher", "transport", "bandage", "dressing", "sling",
    "tourniquet", "splint",
    # Legal
    "good samaritan", "consent", "negligence", "duty of care",
    # Prevention / hygiene
    "hand washing", "hygiene", "infection", "prevention",
}


# ── 4. Relevance scoring ──────────────────────────────────────────────────────

def build_tfidf_index(rows: list[dict]) -> list[list[str]]:
    """Tokenise each QA pair's question + answer into a word list."""
    tokenised = []
    for r in rows:
        combined = (r["question"] + " " + r["answer"]).lower()
        words = re.findall(r"[a-z]{3,}", combined)
        tokenised.append(words)
    return tokenised


def score_row(
    words: list[str],
    pdf_keywords: set[str],
    topic_tokens: set[str],
) -> float:
    """
    Score = (PDF keyword hits × 2) + (manual topic hits × 3)

    We upweight topic_tokens because they are hand-curated from the manual's
    own table of contents and are very high-precision.
    """
    word_set = set(words)
    pdf_score = sum(2 for w in word_set if w in pdf_keywords)
    topic_score = sum(3 for phrase in topic_tokens for w in phrase.split() if w in word_set)
    return pdf_score + topic_score


def rank_rows(
    rows: list[dict],
    pdf_keywords: set[str],
    top_n: int = 500,
) -> list[dict]:
    """Return the top_n rows ranked by relevance to the manual."""
    topic_tokens = {p.lower() for p in MANUAL_TOPICS}
    tokenised = build_tfidf_index(rows)

    scored = []
    for i, (row, words) in enumerate(zip(rows, tokenised)):
        s = score_row(words, pdf_keywords, topic_tokens)
        scored.append((s, i, row))

    scored.sort(key=lambda x: -x[0])
    top = [row for _, _, row in scored[:top_n]]
    print(f"  → Selected {len(top)} rows (min score: {scored[top_n-1][0]:.1f}, "
          f"max score: {scored[0][0]:.1f})")
    return top


# ── 5. Deduplicate ────────────────────────────────────────────────────────────

def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove near-duplicate questions (same first 60 chars after normalising)."""
    seen = set()
    out = []
    for r in rows:
        key = re.sub(r"\s+", " ", r["question"].lower().strip())[:60]
        if key not in seen:
            seen.add(key)
            out.append(r)
    print(f"  → {len(out)} rows after deduplication.")
    return out


# ── 6. Save output ────────────────────────────────────────────────────────────

def save(rows: list[dict], out_stem: str = "firstaidqa_relevant_500"):
    import pandas as pd

    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    df.index.name = "id"

    csv_path = f"{out_stem}.csv"
    json_path = f"{out_stem}.json"

    df.to_csv(csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n✅  Saved {len(rows)} Q&A pairs:")
    print(f"    CSV  → {csv_path}")
    print(f"    JSON → {json_path}")
    print("\nSample (first 3):")
    for r in rows[:3]:
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:120]}…\n")


# ── 7. Notebook-friendly entry point ─────────────────────────────────────────

def run(
    pdf: str = None,
    top: int = 500,
    out: str = "firstaidqa_relevant_500",
) -> list[dict]:
    """
    Call this directly from a Kaggle or Colab notebook cell instead of using
    CLI args.  Returns the final list of dicts so you can inspect it inline.

    Example
    -------
    from fetch_firstaidqa import run
    rows = run()
    # rows is now a list of {"question": ..., "answer": ...} dicts
    # Files saved to /kaggle/working/firstaidqa_relevant_500.{csv,json}
    """
    # Step 1: load all dataset rows
    rows = load_dataset_rows()

    # Step 2: extract PDF keywords (optional)
    pdf_keywords: set[str] = set()
    if pdf:
        pdf_keywords = extract_pdf_keywords(pdf)
    else:
        print("No PDF supplied; using built-in manual topic keywords only.")

    # Step 3: score & rank (fetch 2× target to give dedup room)
    print("Scoring relevance …")
    top_rows = rank_rows(rows, pdf_keywords, top_n=min(top * 2, len(rows)))

    # Step 4: deduplicate, then trim to exact target
    top_rows = deduplicate(top_rows)
    top_rows = top_rows[:top]
    print(f"  → Final count: {len(top_rows)}")

    # Step 5: save
    save(top_rows, out_stem=out)

    return top_rows


# ── 8. CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to RedCrossSocietyManual.pdf (optional – improves matching)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=500,
        help="Number of Q&A pairs to select (default: 500)",
    )
    parser.add_argument(
        "--out",
        default="firstaidqa_relevant_500",
        help="Output file stem (default: firstaidqa_relevant_500)",
    )
    args = parser.parse_args()
    run(pdf=args.pdf, top=args.top, out=args.out)


if __name__ == "__main__":
    main()

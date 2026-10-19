from __future__ import annotations

import argparse
import json
import os
import time
import re
import yaml
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


CANDIDATE_CATEGORIES = {
    "Standards specifications",
    "Standards overview",
    "Lexicon",
}

COVERAGE_THRESHOLD = 0.5

MAX_OPTIONS = 4

# Data structures
@dataclass
class TeleQnAQuestion:
    qid: str                         
    category: str
    question: str
    options: dict[str, str]          
    answer_key: str                 
    answer_text: str                 
    explanation: str


@dataclass
class RetrievalResult:
    qid: str
    top_chunks: list[dict]            
    max_score: float
    supported: bool                   # max_score >= COVERAGE_THRESHOLD


@dataclass
class AnswerResult:
    qid: str
    condition: str                   
    predicted_key: str             
    correct_key: str
    correct: bool
    raw_response: str
    elapsed: float


@dataclass
class TrackDResults:
    candidate_questions: int
    supported_questions: int
    coverage_rate: float
    baseline: dict
    kpi_rag: dict
    per_question: list[dict]
    metadata: dict = field(default_factory=dict)


def load_teleqna(max_per_category: int = 0) -> list[TeleQnAQuestion]:
    """
    Load TeleQnA records handling gated repo authentication & local caching.
    Bypasses PyArrow nested dataset schema issues.
    """
    print("Loading TeleQnA benchmark...")
    cache_file = Path("data/TeleQnA.json")
    records = []

    # 1. Load local cache if available
    if cache_file.exists():
        print(f"  Loading cached TeleQnA benchmark from {cache_file}...")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = list(raw.values()) if isinstance(raw, dict) else raw
        except Exception as e:
            print(f"  Cache read warning: {e}")

    # 2. Download via huggingface_hub 
    if not records:
        try:
            from huggingface_hub import hf_hub_download
            print("  Downloading TeleQnA.json via huggingface_hub...")
            file_path = hf_hub_download(
                repo_id="netop/TeleQnA",
                filename="TeleQnA.json",
                repo_type="dataset"
            )
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = list(raw.values()) if isinstance(raw, dict) else raw

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            print(f"  Cached TeleQnA dataset to {cache_file} ({len(records)} questions)")
        except Exception as e:
            print(f"  Error downloading via huggingface_hub: {e}")

    # 3. Direct HTTP with HF Token Fallback
    if not records:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                token = token_path.read_text().strip()

        url = "https://huggingface.co/datasets/netop/TeleQnA/raw/main/TeleQnA.json"
        headers = {"User-Agent": "Mozilla/5.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            records = list(raw.values()) if isinstance(raw, dict) else raw

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            print(f"  Cached TeleQnA dataset to {cache_file} ({len(records)} questions)")
        except Exception as err:
            print(f"  Direct HTTP download failed: {err}")

    # 4. Fallback to HF datasets library
    if not records:
        try:
            from datasets import load_dataset
            ds = load_dataset("netop/TeleQnA")
            split = ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds

            first_item = split[0]
            if "question" in first_item and ("category" in first_item or "subject" in first_item):
                records = list(split)
            else:
                for col_name in split.column_names:
                    col_val = first_item[col_name]
                    if isinstance(col_val, dict) and "question" in col_val:
                        records.append(col_val)
        except Exception as err:
            print(f"  HF datasets fallback failed: {err}")

    questions = []
    per_cat: dict[str, int] = {}

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        raw_cat = str(rec.get("category") or rec.get("subject", "")).strip()

        matched_cat = None
        for target_cat in CANDIDATE_CATEGORIES:
            if raw_cat.lower() == target_cat.lower():
                matched_cat = target_cat
                break

        if not matched_cat:
            continue

        per_cat[matched_cat] = per_cat.get(matched_cat, 0) + 1
        if max_per_category and per_cat[matched_cat] > max_per_category:
            continue
       
        # TeleQnA stores choices as a list and answer as a 1-based index.
        raw_choices = rec.get("choices") or []
        options = {
            f"option {k}": str(choice)
            for k, choice in enumerate(raw_choices, start=1)
            if choice is not None and str(choice).strip()
        }

        raw_answer = str(rec.get("answer", "")).strip()
        answer_key = ""

       
        # TeleQnA numeric answers are zero-based choice indices.
        if raw_answer.isdigit():
            answer_index = int(raw_answer) + 1
            candidate_key = f"option {answer_index}"
            if candidate_key in options:
                answer_key = candidate_key
        else:
            # Preserve compatibility with textual answers such as "option 2".
            m = re.match(
                r"(option \d+)\s*[:\-]?\s*(.*)",
                raw_answer,
                re.IGNORECASE,
            )
        
            if m:
                candidate_key = m.group(1).lower()
                raw_answer_text = m.group(2).strip()
        
                if candidate_key in options:
                    answer_key = candidate_key
            else:
                raw_answer_text = raw_answer
        
            # If the answer is text, match it to a choice.
            if not answer_key:
                for key, choice in options.items():
                    if choice.strip().casefold() == raw_answer_text.casefold():
                        answer_key = key
                        break


        answer_text = (
            options.get(answer_key, "")
            if answer_key
            else raw_answer
        )

        questions.append(TeleQnAQuestion(
            qid         = f"q{i:05d}",
            category    = matched_cat,
            question    = str(rec.get("question", "")),
            options     = options,
            answer_key  = answer_key,
            answer_text = answer_text,
            explanation = str(rec.get("explanation") or rec.get("explaination") or ""),
        ))

    print(f"  Candidate questions after category filter: {len(questions)}")
    for cat in sorted(CANDIDATE_CATEGORIES):
        kept = sum(1 for q in questions if q.category == cat)
        print(f"    {cat:25s}: {kept} kept")

    return questions


def load_retriever(config_path: str):
    import sys, yaml
    project_root = Path(config_path).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from src.rag_query import retrieve
        from src.kg_indexer import get_collection

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        collection = get_collection(cfg)   # loads telecomts_tickets
        model_name = cfg["rag"]["embedding_model"]
        top_k      = cfg["rag"].get("top_k", 5)
        threshold  = cfg["rag"].get("cosine_threshold", COVERAGE_THRESHOLD)
        model      = SentenceTransformer(model_name)

        print(f"  Retriever loaded: {collection.count()} tickets "
              f"(collection: {cfg['rag']['collection_name']})")

        class _WrappedRetriever:
            def retrieve(self, query: str, top_k: int = top_k) -> list[dict]:
                tickets, _ = retrieve(
                    query      = query,
                    collection = collection,
                    model_name = model_name,
                    k          = top_k,
                    threshold  = threshold,
                )
                return [
                    {
                        "text":   t.content,
                        "score":  t.similarity_score,
                        "source": t.ticket_id,
                    }
                    for t in tickets
                ]

        return _WrappedRetriever()

    except Exception as e:
        print(f"  WARNING: could not load retriever ({e}) — using stub")
        return _StubRetriever()
        
class _StubRetriever:
    """
    Fallback when the real retriever can't be imported.
    Returns empty results (all questions will be 'unsupported').
    Used for skeleton testing only.
    """
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return []


def run_coverage_audit(
    questions: list[TeleQnAQuestion],
    retriever,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Stage 1: run every question through the retriever and classify as
    supported / unsupported based on COVERAGE_THRESHOLD.
    """
    results = []
    n = len(questions)
    print(f"\nStage 1: Evidence-coverage audit ({n} questions)...")

    for i, q in enumerate(questions):
        chunks = retriever.retrieve(q.question, top_k=top_k)
        normalised = []
        for c in chunks:
            normalised.append({
                "text"  : c.get("text") or c.get("content") or c.get("chunk", ""),
                "score" : float(c.get("score") or c.get("similarity") or 0.0),
                "source": c.get("source") or c.get("metadata", {}).get("source", ""),
            })
        max_score = max((c["score"] for c in normalised), default=0.0)
        supported = max_score >= COVERAGE_THRESHOLD

        results.append(RetrievalResult(
            qid        = q.qid,
            top_chunks = normalised,
            max_score  = round(max_score, 4),
            supported  = supported,
        ))

        if (i + 1) % 50 == 0 or (i + 1) == n:
            n_sup = sum(1 for r in results if r.supported)
            print(f"  [{i+1:4d}/{n}]  supported so far: {n_sup} ({100*n_sup/(i+1):.1f}%)")

    n_supported = sum(1 for r in results if r.supported)
    print(f"\n  Coverage audit complete: {n_supported}/{n} supported ({100*n_supported/n:.1f}%)")
    return results


# LLM answer generation
ANSWER_PROMPT = """You are answering a multiple-choice telecommunications question.
{context_block}
QUESTION: {question}

OPTIONS:
{options_block}

Instructions:
- Select the single best answer.
- Reply with ONLY the option label, e.g.: option 2
- Do not explain. Do not add punctuation after the label.

Your answer:"""

SYSTEM_MSG = (
    "You are an expert telecommunications engineer. "
    "Answer multiple-choice questions by selecting the single best option. "
    "Reply with only the option label (e.g. 'option 2')."
)


def build_prompt(q: TeleQnAQuestion, chunks: list[dict]) -> str:
    options_block = "\n".join(
        f"  {k}: {v}" for k, v in sorted(q.options.items())
    )
    if chunks:
        context_lines = []
        for j, c in enumerate(chunks[:3], 1):
            text = c["text"][:400].strip()  # cap each chunk at 400 chars
            context_lines.append(f"[Evidence {j}] {text}")
        context_block = "CONTEXT (from KPI-RAG knowledge base):\n" + "\n\n".join(context_lines) + "\n\n"
    else:
        context_block = ""

    return ANSWER_PROMPT.format(
        context_block = context_block,
        question      = q.question,
        options_block = options_block,
    )


def extract_option(raw: str) -> str:
    """Parse model output -> 'option N' key."""
    raw = raw.strip().lower()
    m = re.search(r'option\s*(\d+)', raw)
    if m:
        return f"option {m.group(1)}"
    m2 = re.match(r'^([a-d])[)\.\s]', raw)
    if m2:
        return f"option {ord(m2.group(1)) - ord('a') + 1}"
    return raw



def answer_question(
    q: TeleQnAQuestion,
    chunks: list[dict],
    condition: str,
    client,
    model_name: str,
    max_retries: int = 4,
    delay: float = 7.5,
) -> AnswerResult:
    prompt = build_prompt(q, chunks)
    t0 = time.time()
    raw = ""
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = (resp.choices[0].message.content or "").strip()

            if raw:
                print(
                    f"[OK] qid={q.qid} condition={condition} "
                    f"raw_response={raw!r}"
                )
                break

            print(
                f"[EMPTY RESPONSE] qid={q.qid} "
                f"condition={condition} attempt={attempt + 1}"
            )

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            err = str(e)

            if "429" in err or "rate_limit" in err.lower():
                m = re.search(r"try again in ([\d.]+)s", err)
                wait = (
                    float(m.group(1)) + 1.0
                    if m
                    else min(delay * (2 ** attempt), 60)
                )

                print(
                    f"[RATE LIMIT] qid={q.qid} condition={condition} "
                    f"attempt={attempt + 1}/{max_retries} "
                    f"wait={wait:.1f}s error={last_error}"
                )
                time.sleep(wait)
            else:
                print(
                    f"[API ERROR] qid={q.qid} condition={condition} "
                    f"error={last_error}"
                )
                raw = f"ERROR: {last_error}"
                break

    if not raw:
        print(
            f"[FAILED AFTER RETRIES] qid={q.qid} "
            f"condition={condition} last_error={last_error!r}"
        )

    elapsed = time.time() - t0
    predicted = (
        extract_option(raw)
        if raw and not raw.startswith("ERROR")
        else "error"
    )
    correct = predicted == q.answer_key

    time.sleep(delay)

    return AnswerResult(
        qid=q.qid,
        condition=condition,
        predicted_key=predicted,
        correct_key=q.answer_key,
        correct=correct,
        raw_response=raw,
        elapsed=round(elapsed, 2),
    )



# Evaluation orchestration
def run_evaluation(
    questions: list[TeleQnAQuestion],
    retrieval_results: list[RetrievalResult],
    client,
    model_name: str,
    delay: float = 7.5,
) -> list[dict]:
    """
    Stage 2: answer every question under both conditions.
    Returns list of per-question result dicts.
    """
    ret_map = {r.qid: r for r in retrieval_results}
    n = len(questions)
    print(f"\nStage 2: LLM evaluation ({n} questions x 2 conditions = {2*n} calls)...")
    print(f"  Estimated time at {delay}s/call: ~{2*n*delay/60:.0f} minutes\n")

    per_question = []
    for i, q in enumerate(questions):
        ret = ret_map[q.qid]

        # Condition A: baseline (no context)
        base = answer_question(q, [], "baseline", client, model_name, delay=delay)

        # Condition B: KPI-RAG (with retrieved chunks)
        rag = answer_question(q, ret.top_chunks, "kpi_rag", client, model_name, delay=delay)

        status = f"{'✓' if base.correct else '✗'}base  {'✓' if rag.correct else '✗'}rag"
        print(f"  [{i+1:3d}/{n}] {q.category[:20]:20s} | {q.qid} | {status} | sup={'Y' if ret.supported else 'N'}")

        per_question.append({
            "qid"             : q.qid,
            "category"        : q.category,
            "supported"       : ret.supported,
            "retrieval_score" : ret.max_score,
            "correct_key"     : q.answer_key,
            "baseline_pred"   : base.predicted_key,
            "kpi_rag_pred"    : rag.predicted_key,
            "baseline_correct": base.correct,
            "kpi_rag_correct" : rag.correct,
        })

    return per_question


# Aggregation & reporting
def aggregate(per_question: list[dict], label: str) -> dict:
    """Compute accuracy stats for a subset of questions."""
    if not per_question:
        return {"n": 0, "baseline_acc": None, "kpi_rag_acc": None}

    n       = len(per_question)
    base_n  = sum(1 for q in per_question if q["baseline_correct"])
    rag_n   = sum(1 for q in per_question if q["kpi_rag_correct"])
    delta   = rag_n - base_n

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")
    print(f"  Baseline accuracy:  {base_n}/{n} = {100*base_n/n:.1f}%")
    print(f"  KPI-RAG accuracy:   {rag_n}/{n}  = {100*rag_n/n:.1f}%")
    print(f"  Delta:              {delta:+d} questions  ({100*delta/n:+.1f}pp)")

    cats = sorted(set(q["category"] for q in per_question))
    print(f"\n  By category:")
    for cat in cats:
        cq = [q for q in per_question if q["category"] == cat]
        cn = len(cq)
        cb = sum(1 for q in cq if q["baseline_correct"])
        cr = sum(1 for q in cq if q["kpi_rag_correct"])
        print(f"    {cat[:35]:35s}  base={100*cb/cn:.0f}%  rag={100*cr/cn:.0f}%  (n={cn})")

    return {
        "n"                  : n,
        "baseline_correct"   : base_n,
        "kpi_rag_correct"    : rag_n,
        "baseline_accuracy"  : round(base_n / n, 4),
        "kpi_rag_accuracy"   : round(rag_n / n, 4),
        "delta_pp"           : round((rag_n - base_n) / n * 100, 2),
        "by_category"        : {
            cat: {
                "n"                 : len(cq := [q for q in per_question if q["category"] == cat]),
                "baseline_accuracy" : round(sum(q["baseline_correct"] for q in cq) / len(cq), 4),
                "kpi_rag_accuracy"  : round(sum(q["kpi_rag_correct"]  for q in cq) / len(cq), 4),
            }
            for cat in cats
        },
    }


def build_results(per_question: list[dict], model_name: str) -> TrackDResults:
    n_total     = len(per_question)
    n_supported = sum(1 for q in per_question if q["supported"])

    supported_qs = [q for q in per_question if q["supported"]]
    all_qs       = per_question

    print("\n" + "="*60)
    print("TRACK D RESULTS — TeleQnA (KPI-RAG filtered subset)")
    print("="*60)
    print(f"  Total candidate questions : {n_total}")
    print(f"  Supported (score>={COVERAGE_THRESHOLD}): {n_supported} ({100*n_supported/n_total:.1f}%)")

    coverage_stats = aggregate(supported_qs, "COVERAGE-SET (supported questions only)")
    all_stats      = aggregate(all_qs,       "ALL-CANDIDATE (full filtered set)")

    return TrackDResults(
        candidate_questions = n_total,
        supported_questions = n_supported,
        coverage_rate       = round(n_supported / n_total, 4) if n_total else 0,
        baseline            = all_stats,
        kpi_rag             = {"coverage_set": coverage_stats, "all_candidate": all_stats},
        per_question        = per_question,
        metadata            = {
            "judge_model"         : model_name,
            "coverage_threshold" : COVERAGE_THRESHOLD,
            "candidate_categories": sorted(CANDIDATE_CATEGORIES),
            "composite_note"      : (
                "coverage_set = questions where retriever max_score >= threshold; "
                "all_candidate = every question in selected categories"
            ),
        },
    )


# CLI
def main():
    parser = argparse.ArgumentParser(description="Track D — TeleQnA evaluation")
    parser.add_argument("--config",   required=True, help="Path to configs/config.yaml")
    parser.add_argument("--output",   default="data/processed/track_d_teleqna.json")
    parser.add_argument("--model",    default="gemini-3.5-flash-lite",
                        help="LLM for answer generation (Groq or Ollama)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (auto-set from GROQ_API_KEY if absent)")
    parser.add_argument("--api-key",  default="local")
    parser.add_argument("--max-per-category", type=int, default=200,
                        help="Max questions per category (0=all). Default 200.")
    parser.add_argument("--top-k",    type=int,   default=5,
                        help="Chunks retrieved per question (default 5)")
    parser.add_argument("--delay",    type=float, default=7.5,
                        help="Seconds between LLM calls (default 7.5 for Groq free tier)")
    parser.add_argument("--audit-only", action="store_true",
                        help="Run coverage audit only, skip LLM evaluation")
    args = parser.parse_args()

    #  Resolve API credentials 
    api_key  = args.api_key
    base_url = args.base_url

    if api_key == "local":
        groq_key   = os.environ.get("GROQ_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if groq_key:
            api_key  = groq_key
            base_url = base_url or "https://api.groq.com/openai/v1"
            print(f"Using Groq cloud API with model: {args.model}")
        elif openai_key:
            api_key = openai_key
            print(f"Using OpenAI cloud API with model: {args.model}")
        elif base_url is None:
            raise ValueError(
                "Set GROQ_API_KEY or OPENAI_API_KEY, "
                "or pass --base-url for a local server."
            )

    # Load dataset
    questions = load_teleqna(max_per_category=args.max_per_category)
    if not questions:
        print("No questions loaded — check HuggingFace login and dataset access.")
        return

    # Load retriever 
    print("\nLoading KPI-RAG retriever...")
    retriever = load_retriever(args.config)

    # Stage 1: Coverage audit 
    retrieval_results = run_coverage_audit(questions, retriever, top_k=args.top_k)

    # Save audit results immediately
    audit_path = Path(args.output).with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_data = [asdict(r) for r in retrieval_results]
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"  Audit saved -> {audit_path}")

    if args.audit_only:
        print("\n--audit-only flag set, stopping before LLM evaluation.")
        return

    # Stage 2: LLM evaluation 
    import httpx
    from openai import OpenAI
    client = OpenAI(
        api_key  = api_key,
        base_url = base_url,
        timeout  = httpx.Timeout(60.0, connect=10.0),
    )

    per_question = run_evaluation(
        questions, retrieval_results, client, args.model, delay=args.delay
    )

    # Aggregate & save 
    results = build_results(per_question, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
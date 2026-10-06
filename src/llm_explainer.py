import json
import logging
import os
import random
import time
from groq import RateLimitError
import re
import requests

try:
    from openai import OpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]
    OpenAIRateLimitError = RateLimitError  # type: ignore[assignment,misc]

# groq is called through the openai client, so its 429 is openai.RateLimitError
_RATE_LIMIT_ERRORS = (RateLimitError, OpenAIRateLimitError)

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
    from google.genai.errors import ClientError as GeminiClientError
except ImportError:
    google_genai = None
    google_genai_types = None
    GeminiClientError = Exception

from src.schema import ClassifierOutput, RetrievedTicket, LLMExplanation
from src.utils import validate_3gpp_ref

logger = logging.getLogger(__name__)

def load_alignment_table(path: str) -> dict[str, dict]:
    """Load alignment_table.json as {fault_type: row}.

    Each row keeps its original fields and gains ``3gpp_ts`` (primary
    standard), ``valid_refs`` (every TS/TR the row cites, primary first),
    ``clause``, ``evidence_span`` and ``oran_component``, parsed out of the
    free-text ``3gpp_reference`` when not given explicitly.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alignment = {}
    entries_list = data.get("entries", data.get("rows", []))
    for entry in entries_list:
        fault_type = entry.get("fault_type", entry.get("telecomts_fault"))
        if not fault_type:
            continue
        ts = entry.get("3gpp_ts")
        if not ts and "3gpp_reference" in entry:
            match = re.search(r'T[SR]\s+(2[1-9]|3[0-8])\.\d{3}(?:-\d+)?', entry["3gpp_reference"])
            if match:
                ts = match.group(0)
        clause = entry.get("clause")
        if not clause and "3gpp_reference" in entry:
            match = re.search(r'§(\d+(?:\.\d+)*)', entry["3gpp_reference"])
            if match:
                clause = match.group(1)
        evidence = entry.get("evidence_span")
        if evidence is None:
            if "clause_text" in entry:
                evidence = entry["clause_text"][:300]
            else:
                evidence = ""
        # some rows cite two standards (CCI Severe: TS 38.141-1 + TS 38.104)
        cited = re.findall(r'T[SR]\s+(?:2[1-9]|3[0-8])\.\d{3}(?:-\d+)?', entry.get("3gpp_reference", ""))
        normalized = dict(entry)
        normalized["3gpp_ts"] = ts if ts else ""
        normalized["valid_refs"] = list(dict.fromkeys(([ts] if ts else []) + cited))
        normalized["clause"] = clause if clause else ""
        normalized["evidence_span"] = evidence
        normalized["oran_component"] = entry.get("oran_component", "")
        alignment[fault_type] = normalized
    logger.info("Loaded %d entries from alignment table", len(alignment))
    return alignment

def build_prompt(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket],
    alignment: dict[str, dict],
    rca_context: str = "",
) -> str:
    """Full (C3) prompt: label + SHAP, optional RCA evidence block, top-3 tickets, alignment row."""
    shap_lines = []
    for x in payload.shap_top3:
        direction = "above" if "above" in x.feature_vs_normal else "below"
        shap_lines.append(f"  {x.channel}: {direction} normal (SHAP={x.shap_value:+.2f})")
    shap_summary = "\n".join(shap_lines)
    if tickets:
        ticket_lines = []
        for i, ticket in enumerate(tickets[:3]):
            ticket_lines.append(f"  [{i+1}] {ticket.content[:200]}...")
        tickets_summary = "\n".join(ticket_lines)
    else:
        tickets_summary = "  No similar incidents retrieved."
    entry = alignment.get(payload.anomaly_type.value, {})
    gpp_ts = entry.get("3gpp_ts", "")
    clause = entry.get("clause", "")
    evidence_span = entry.get("evidence_span", "")
    oran_component = entry.get("oran_component", "")

    rca_block = ""
    if rca_context:
        rca_block = f"\n[EVIDENCE FROM RCA PIPELINE]\n{rca_context}\n"

    # The same alignment row is used by validate_citation(), so C3's citation
    # score measures whether the LLM keeps the standard it was given, not
    # whether it knew it. C1 (no context) is the baseline for that.
    prompt = f"""You are a 5G network fault diagnosis expert.

[FAULT DETECTED]
Fault detected: {payload.anomaly_type.value}
Confidence: {payload.confidence:.0%}

Top contributing KPIs (SHAP):
{shap_summary}
{rca_block}
[RETRIEVED INCIDENTS]
{tickets_summary}

[STANDARDS REFERENCE]
3GPP {gpp_ts} clause {clause}: {evidence_span}
O-RAN component: {oran_component}

Use TR (Technical Report) instead of TS when the standard is a TR -- for example, channel models use TR 38.901.
Ground your explanation in the specific KPI values and SHAP evidence provided above.

Return ONLY a JSON object with exactly these fields:
{{
  "root_cause": "one sentence physical explanation referencing specific KPI values",
  "3gpp_reference": "TS/TR XX.XXX (e.g., TS 38.321 or TR 38.901)",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain citing specific KPI values from the evidence above"
}}
"""
    return prompt


class RateLimiter:
    """Keeps at least min_interval seconds between LLM calls (module-level singleton below)."""

    def __init__(self, min_interval: float = 2.0):
        self.min_interval = float(min_interval)
        self.last_call_time = 0.0

    def wait(self) -> None:
        now = time.time()
        if self.last_call_time > 0:
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


_default_rate_limiter = RateLimiter(min_interval=2.0)


def _extract_retry_after(e: Exception) -> float | None:
    """Retry-After seconds from the error's headers or message ('retry in 12s'), else None."""
    headers = None
    if hasattr(e, "response") and hasattr(e.response, "headers"):
        headers = e.response.headers
    elif hasattr(e, "headers"):
        headers = e.headers

    if headers:
        for k, v in headers.items():
            if k.lower() == "retry-after":
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass

    err_str = str(e)
    match = re.search(r"retry\s+(?:after|in)\s+(\d+(?:\.\d+)?)\s*s?", err_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def call_gemini(prompt: str, cfg: dict) -> str:
    """One Gemini call via google-genai."""
    if google_genai is None:
        raise RuntimeError("google-genai package is not installed")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    client = google_genai.Client(api_key=api_key)
    model_name = cfg["llm"].get("gemini_model", "gemini-3.5-flash-lite")
    temperature = cfg["llm"].get("temperature", 0.1)
    # One call; retries live in _run_with_retry().
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=google_genai_types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    result = response.text or ""
    logger.info("Called gemini (%s), response len: %d", model_name, len(result))
    return result


def _is_rate_limit(e: Exception) -> bool:
    """429 / quota error from any backend?"""
    return (
        isinstance(e, _RATE_LIMIT_ERRORS)
        or getattr(e, "status_code", None) == 429
        or "429" in str(e)
        or "RESOURCE_EXHAUSTED" in str(e)
    )


def _backoff_wait(e: Exception, attempt: int, cfg: dict) -> float:
    """Server's Retry-After if it sent one, else exponential backoff with jitter."""
    retry_after = _extract_retry_after(e)
    if retry_after is not None:
        return retry_after
    base = cfg.get("llm", {}).get("backoff_base_s", 5)
    max_wait = cfg.get("llm", {}).get("backoff_max_s", 60)
    return min(base * (2 ** attempt) + random.uniform(0, 2), max_wait)


def call_llm(prompt: str, cfg: dict) -> str:
    """Send one prompt to the configured backend (ollama / groq / gemini) and return the raw text."""
    # Shared rate limiter, so every backend respects min_request_interval_s.
    _default_rate_limiter.min_interval = cfg.get("llm", {}).get(
        "min_request_interval_s", 2.0
    )
    _default_rate_limiter.wait()

    backend = cfg["llm"]["backend"]
    if backend == "ollama":
        base_url = cfg["llm"].get("ollama_base_url", "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/generate"
        payload = {
            "model": cfg["llm"]["ollama_model"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": cfg["llm"]["temperature"]},   # top-level temperature is ignored by Ollama
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json().get("response", "")
        logger.info("Called ollama, response len: %d", len(result))
        return result
    elif backend == "groq":
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required for the groq backend. "
                "Install it with: pip install 'openai>=1.0'"
            )
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", "")
        )
        completion = client.chat.completions.create(
            model=cfg["llm"]["groq_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["llm"]["temperature"]
        )
        result = completion.choices[0].message.content or ""
        logger.info("Called groq, response len: %d", len(result))
        return result
    elif backend == "gemini":
        return call_gemini(prompt, cfg)
    else:
        raise RuntimeError(f"Unknown LLM backend: {backend}")

def parse_response(raw: str) -> dict:
    """Parse the LLM's JSON (bare or inside a ```json fence).

    Renames ``3gpp_reference`` to ``gpp_reference`` (not a valid Python
    field name) and normalises it to ``TS/TR XX.XXX``. Raises ValueError if
    there is no JSON or any of the five fields is missing.
    """
    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = raw
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON found in response")
    required_keys = {"root_cause", "3gpp_reference", "oran_component", "recommended_action", "reasoning_trace"}
    if not required_keys.issubset(parsed.keys()):
        raise ValueError("Missing required keys in JSON response")
    ref = parsed.get("3gpp_reference", "")
    ts_match = re.search(r'T[SR]\s+\d{2}\.\d{3}(?:-\d+)?', ref)
    if ts_match:
        parsed["3gpp_reference"] = ts_match.group()
    if "3gpp_reference" in parsed:
        parsed["gpp_reference"] = parsed.pop("3gpp_reference")
    return parsed

def validate_citation(ref: str, alignment: dict[str, dict], fault_type: str = None) -> bool:
    """True if ``ref`` is a well-formed TS/TR number AND is cited in the fault's row.

    With ``fault_type`` the row's ``valid_refs`` are used (primary + secondary
    standards); without it, any row's primary standard counts.
    """
    try:
        check1 = bool(validate_3gpp_ref(ref))
    except Exception:
        check1 = False
    if fault_type and fault_type in alignment:
        entry = alignment[fault_type]
        expected = entry.get("valid_refs") or [entry.get("3gpp_ts", "")]
        check2 = bool(ref) and ref in expected
    else:
        all_ts = {entry.get("3gpp_ts") for entry in alignment.values() if entry.get("3gpp_ts")}
        check2 = ref in all_ts
    if not check1:
        logger.warning("Citation validation failed Check 1 (format regex): %s", ref)
    if not check2:
        logger.warning("Citation validation failed Check 2 (alignment table lookup): %s", ref)
    return check1 and check2


def _template_fallback(payload: ClassifierOutput, alignment: dict[str, dict]) -> LLMExplanation:
    entry = alignment.get(payload.anomaly_type.value, {})
    return LLMExplanation(
        root_cause=f"{payload.anomaly_type.value} detected via KPI deviation",
        gpp_reference=entry.get("3gpp_ts", ""),
        oran_component=entry.get("oran_component", ""),
        recommended_action="Refer to alignment table for diagnostic steps",
        reasoning_trace=f"Template fallback. Top KPI: {payload.shap_top3[0].channel}",
        reference_valid=False,
        template_generated=True,
    )


def _run_with_retry(
    prompt: str,
    cfg: dict,
    payload: ClassifierOutput,
    alignment: dict[str, dict],
) -> LLMExplanation:
    """Up to max_retries LLM calls (waiting on 429s), then the template fallback."""
    max_retries = cfg["llm"]["max_retries"]
    for attempt in range(max_retries):
        try:
            parsed = parse_response(call_llm(prompt, cfg))
            ref = parsed["gpp_reference"]
            return LLMExplanation(
                root_cause=parsed["root_cause"],
                gpp_reference=ref,
                oran_component=parsed["oran_component"],
                recommended_action=parsed["recommended_action"],
                reasoning_trace=parsed["reasoning_trace"],
                reference_valid=validate_citation(ref, alignment, fault_type=payload.anomaly_type.value),
                template_generated=False,
            )
        except Exception as e:
            logger.warning("LLM attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = _backoff_wait(e, attempt, cfg)
                logger.warning("Rate limited, waiting %.1fs before retry", wait)
                time.sleep(wait)
    logger.error("All %d LLM attempts failed, using template fallback", max_retries)
    return _template_fallback(payload, alignment)


def explain_from_rca(
    window_index: int,
    fault_type: str,
    condition: str | int,
    cfg: dict,
    rca_context: str = "",
) -> LLMExplanation:
    """Explain one rca_evidence window end to end.

    Builds the ClassifierOutput from the record, retrieves tickets (C2/C3),
    loads the alignment table and calls explain_condition(). ``condition``
    may be 1/2/3 or "C1"/"C2"/"C3"; ``rca_context`` is only used for C3.
    """
    _cond_map = {"C1": 1, "C2": 2, "C3": 3}
    if isinstance(condition, str):
        cond_int = _cond_map.get(condition.upper(), int(condition.lstrip("Cc")))
    else:
        cond_int = int(condition)

    from src.rca_loader import RCALoader
    rca_evidence_path = cfg.get("data", {}).get(
        "rca_evidence_path", "data/processed/rca_evidence.json"
    )
    loader = RCALoader(rca_evidence_path)
    record = loader.get(window_index)
    if record is None:
        raise ValueError(f"window_index {window_index} not found in rca_evidence")

    from src.schema import AnomalyType, ClassifierOutput, SHAPEntry
    layer_b = record.get("layer_b_model_attribution", [])
    top3_raw = sorted(layer_b, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    while len(top3_raw) < 3:   # schema wants exactly three
        top3_raw.append(
            {"channel": "N/A", "feature": "N/A", "shap_value": 0.0, "feature_vs_normal": "above_normal_mean"}
        )
    shap_top3 = [
        SHAPEntry(
            channel=e.get("channel", e.get("feature", "N/A")),
            feature=e.get("feature", ""),
            shap_value=float(e.get("shap_value", 0.0)),
            feature_vs_normal=e.get("feature_vs_normal", "above_normal_mean"),
        )
        for e in top3_raw
    ]
    signal_statistics: dict[str, float] = {
        k: float(v)
        for k, v in record.get("layer_a_observational", {}).items()
        if isinstance(v, (int, float))
    }

    try:
        anomaly_type = AnomalyType(fault_type)
    except ValueError:
        matched = next(
            (at for at in AnomalyType if at.value.lower() == fault_type.lower()), None
        )
        if matched is None:
            raise ValueError(f"Unknown fault_type: {fault_type!r}")
        anomaly_type = matched

    payload = ClassifierOutput(
        anomaly_type=anomaly_type,
        confidence=float(record.get("confidence", 0.0)),
        shap_top3=shap_top3,
        signal_statistics=signal_statistics,
    )

    tickets: list[RetrievedTicket] = []
    if cond_int in (2, 3):
        try:
            from src.kg_indexer import get_collection
            from src.rag_query import query_from_classifier_output
            collection = get_collection(cfg)
            tickets, _ = query_from_classifier_output(payload, collection, cfg)
        except Exception as exc:
            logger.warning("Ticket retrieval failed (window %d): %s", window_index, exc)

    alignment = load_alignment_table("configs/alignment_table.json")

    return explain_condition(
        payload, tickets, cfg, alignment,
        condition=cond_int,
        rca_context=rca_context if cond_int == 3 else "",
    )


def explain(
    payload: "ClassifierOutput | None" = None,
    tickets: "list[RetrievedTicket] | None" = None,
    cfg: dict | None = None,
    alignment: "dict[str, dict] | None" = None,
    *,
    window_index: int | None = None,
    fault_type: str | None = None,
    condition: "str | int | None" = None,
    rca_context: str = "",
) -> LLMExplanation:
    """Generate an explanation.

    Two ways to call it:
        explain(payload, tickets, cfg, alignment)            # full C3 prompt
        explain(window_index=5, fault_type=..., condition="C3", cfg=cfg)
    The second form builds payload and tickets itself via explain_from_rca().
    """
    if window_index is not None:
        if cfg is None:
            raise ValueError("cfg is required when using window_index")
        if fault_type is None:
            raise ValueError("fault_type is required when using window_index")
        if condition is None:
            raise ValueError("condition is required when using window_index")
        return explain_from_rca(
            window_index=window_index,
            fault_type=fault_type,
            condition=condition,
            cfg=cfg,
            rca_context=rca_context,
        )

    if payload is None or tickets is None or cfg is None or alignment is None:
        raise ValueError(
            "explain() requires either (payload, tickets, cfg, alignment) "
            "or (window_index, fault_type, condition, cfg)"
        )
    prompt = build_prompt(payload, tickets, alignment, rca_context=rca_context)
    return _run_with_retry(prompt, cfg, payload, alignment)


def _build_shap_summary(payload: ClassifierOutput) -> str:
    """SHAP lines for the prompts: 'RSRP: below normal (SHAP=-0.42)'."""
    shap_lines = []
    for x in payload.shap_top3:
        direction = "above" if "above" in x.feature_vs_normal else "below"
        shap_lines.append(f"  {x.channel}: {direction} normal (SHAP={x.shap_value:+.2f})")
    return "\n".join(shap_lines)


def build_prompt_condition1(payload: ClassifierOutput) -> str:
    """Condition 1: label + SHAP only -- no tickets, no alignment table."""
    shap_summary = _build_shap_summary(payload)
    prompt = f"""You are a 5G network fault diagnosis expert.

Fault detected: {payload.anomaly_type.value}
Confidence: {payload.confidence:.0%}

Top contributing KPIs (SHAP):
{shap_summary}

Return ONLY a JSON object with exactly these fields:
{{
  "root_cause": "one sentence physical explanation",
  "3gpp_reference": "TS XX.XXX or TR XX.XXX",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain"
}}
"""
    return prompt


def build_prompt_condition2(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket]
) -> str:
    """Condition 2: label + SHAP + tickets -- no alignment table."""
    shap_summary = _build_shap_summary(payload)
    if tickets:
        ticket_lines = []
        for i, ticket in enumerate(tickets[:3]):
            ticket_lines.append(f"  [{i+1}] {ticket.content[:200]}...")
        tickets_summary = "\n".join(ticket_lines)
    else:
        tickets_summary = "  No similar incidents retrieved."
    prompt = f"""You are a 5G network fault diagnosis expert.

Fault detected: {payload.anomaly_type.value}
Confidence: {payload.confidence:.0%}

Top contributing KPIs (SHAP):
{shap_summary}

Retrieved similar incidents:
{tickets_summary}

Return ONLY a JSON object with exactly these fields:
{{
  "root_cause": "one sentence physical explanation",
  "3gpp_reference": "TS XX.XXX or TR XX.XXX",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain"
}}
"""
    return prompt


def explain_condition(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket],
    cfg: dict,
    alignment: dict[str, dict],
    condition: int,
    rca_context: str = "",
) -> LLMExplanation:
    """Track C ablation: 1 = label + SHAP, 2 = + tickets, 3 = + alignment row + rca_context."""
    if condition not in (1, 2, 3):
        raise ValueError(f"condition must be 1, 2, or 3, got {condition}")
    if condition == 1:
        prompt = build_prompt_condition1(payload)
    elif condition == 2:
        prompt = build_prompt_condition2(payload, tickets)
    else:
        prompt = build_prompt(payload, tickets, alignment, rca_context=rca_context)
    return _run_with_retry(prompt, cfg, payload, alignment)

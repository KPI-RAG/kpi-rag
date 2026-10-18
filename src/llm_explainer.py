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

from pydantic import ValidationError

from src.schema import ClassifierOutput, RetrievedTicket, LLMExplanation
from src.utils import validate_3gpp_ref

logger = logging.getLogger(__name__)

DEFAULT_ALIGNMENT_PATH = "data/alignment_table.json"


class LLMResponseError(ValueError):
    """The LLM replied, but not with the expected JSON — worth retrying."""


def alignment_path(cfg: dict) -> str:
    """The alignment table path from cfg["data"], or the default."""
    return cfg.get("data", {}).get("alignment_table_path", DEFAULT_ALIGNMENT_PATH)


def load_alignment_table(path: str = DEFAULT_ALIGNMENT_PATH) -> dict[str, dict]:
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
        # Jamming has no direct 3GPP mapping (relation_type: not-applicable); handled via ticket retrieval only
        if entry.get("relation_type") == "not-applicable" or fault_type == "Jamming":
            continue
        raw_ref = entry.get("3gpp_reference") or ""
        ts = entry.get("3gpp_ts")
        if not ts and raw_ref:
            match = re.search(r'T[SR]\s+(2[1-9]|3[0-8])\.\d{3}(?:-\d+)?', raw_ref)
            if match:
                ts = match.group(0)
        clause = entry.get("clause")
        if not clause and raw_ref:
            match = re.search(r'§(\d+(?:\.\d+)*)', raw_ref)
            if match:
                clause = match.group(1)
        evidence = entry.get("evidence_span")
        if evidence is None:
            raw_clause = entry.get("clause_text") or ""
            evidence = raw_clause[:300]
        # some rows cite two standards (CCI Severe: TS 38.141-1 + TS 38.104)
        cited = re.findall(r'T[SR]\s+(?:2[1-9]|3[0-8])\.\d{3}(?:-\d+)?', raw_ref)
        normalized = dict(entry)
        normalized["3gpp_ts"] = ts if ts else ""
        normalized["valid_refs"] = list(dict.fromkeys(([ts] if ts else []) + cited))
        normalized["clause"] = clause if clause else ""
        normalized["evidence_span"] = evidence
        normalized["oran_component"] = entry.get("oran_component") or ""
        alignment[fault_type] = normalized
    logger.info("Loaded %d entries from alignment table", len(alignment))
    return alignment

def build_prompt(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket],
    alignment: dict[str, dict],
    rca_context: str = "",
) -> str:
    """Full (C3) prompt: label + SHAP, optional RCA evidence block, top-3 tickets, alignment row.

    The same alignment row is used by validate_citation(), so C3's citation score
    measures whether the LLM keeps the standard it was given, not whether it knew it;
    C1 and C2 are the baselines without this context.
    """
    rca_block = f"\n[EVIDENCE FROM RCA PIPELINE]\n{rca_context}\n" if rca_context else ""
    return (
        _prompt_header(payload) + rca_block + _tickets_block(tickets)
        + _standards_block(payload, alignment) + _RESPONSE_INSTRUCTIONS
    )


# One instruction block for all three Track C conditions, so C1/C2/C3 differ only in
# the context they carry. No example standard numbers: those would hint the answer.
_RESPONSE_INSTRUCTIONS = """
Ground your explanation in the specific KPI values and SHAP evidence provided above.

Return ONLY a JSON object with exactly these fields:
{
  "root_cause": "one sentence physical explanation referencing specific KPI values",
  "3gpp_reference": "TS XX.XXX or TR XX.XXX",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain citing specific KPI values from the evidence above"
}
"""


def _prompt_header(payload: ClassifierOutput) -> str:
    return (
        "You are a 5G network fault diagnosis expert.\n\n"
        f"Fault detected: {payload.anomaly_type.value}\n"
        f"Confidence: {payload.confidence:.0%}\n\n"
        f"Top contributing KPIs (SHAP):\n{_build_shap_summary(payload)}\n"
    )


def _tickets_block(tickets: list[RetrievedTicket]) -> str:
    # content is the past incident's KPI narrative; 300 chars cover its radio-layer sentences
    lines = [f"  [{i+1}] (similarity {t.similarity_score:.2f}) {t.content[:300]}..." for i, t in enumerate(tickets[:3])]
    return "\nRetrieved similar incidents:\n" + ("\n".join(lines) or "  No similar incidents retrieved.") + "\n"


def _standards_block(payload: ClassifierOutput, alignment: dict[str, dict]) -> str:
    entry = alignment.get(payload.anomaly_type.value)
    if not entry:   # Jamming: not-applicable row
        return "\n[STANDARDS REFERENCE]\nNo 3GPP clause applies to this fault type.\n"
    return (
        f"\n[STANDARDS REFERENCE]\n3GPP {entry['3gpp_ts']} clause {entry['clause']}: {entry['evidence_span']}\n"
        f"O-RAN component: {entry['oran_component']}\n"
    )


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
    """429 / quota error or connection error from any backend?"""
    return (
        isinstance(e, _RATE_LIMIT_ERRORS)
        or getattr(e, "status_code", None) == 429
        or "429" in str(e)
        or "RESOURCE_EXHAUSTED" in str(e)
        or "getaddrinfo" in str(e)
    )


def _is_non_retryable(e: Exception) -> bool:
    """Config or auth errors (missing key, unknown backend, 401/403) — retrying cannot help.

    A malformed LLM reply (LLMResponseError, pydantic ValidationError) is retried.
    """
    if isinstance(e, (LLMResponseError, ValidationError)):
        return False
    return (
        isinstance(e, (ValueError, RuntimeError))
        or getattr(e, "status_code", None) in (401, 403)
        or getattr(e, "code", None) in (401, 403)   # google-genai ClientError
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
        raise LLMResponseError("No valid JSON found in response")
    required_keys = {"root_cause", "3gpp_reference", "oran_component", "recommended_action", "reasoning_trace"}
    if not isinstance(parsed, dict) or not required_keys.issubset(parsed.keys()):
        raise LLMResponseError("Missing required keys in JSON response")
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
    standards); without it, any row's primary standard counts. A fault with no
    row (Jamming: no 3GPP clause applies) can never have a valid citation.
    """
    if fault_type and fault_type not in alignment:
        logger.warning("No alignment row for %s — citation %r cannot be valid", fault_type, ref)
        return False
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
            logger.warning(
                "LLM attempt %d/%d failed [%s]: %s",
                attempt + 1, max_retries, type(e).__name__, e
            )
            if _is_non_retryable(e):
                logger.error("Non-retryable error (%s) — stopping retries", type(e).__name__)
                break
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = _backoff_wait(e, attempt, cfg)
                logger.warning("Rate-limited, waiting %.1fs", wait)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = min(2 ** attempt, 10)
                logger.warning("Transient error, waiting %.1fs", wait)
                time.sleep(wait)
    logger.error("LLM attempts exhausted or stopped, using template fallback")
    return _template_fallback(payload, alignment)


def explain(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket],
    cfg: dict,
    alignment: dict[str, dict],
    rca_context: str = "",
) -> LLMExplanation:
    """Generate an explanation with the full (C3) prompt."""
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
    return _prompt_header(payload) + _RESPONSE_INSTRUCTIONS


def build_prompt_condition2(payload: ClassifierOutput, tickets: list[RetrievedTicket]) -> str:
    """Condition 2: label + SHAP + tickets -- no alignment table."""
    return _prompt_header(payload) + _tickets_block(tickets) + _RESPONSE_INSTRUCTIONS


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

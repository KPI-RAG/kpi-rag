import json
import logging
import os
import time
from groq import RateLimitError
import re
import requests

try:
    from openai import OpenAI
except ImportError:
    pass

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
        normalized = dict(entry)
        normalized["3gpp_ts"] = ts if ts else ""
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
    """Build the full C3 prompt.

    Parameters
    ----------
    rca_context : str
        Optional pre-formatted string from RCALoader.get_prompt_context().
        When non-empty it is injected between the SHAP summary and the
        retrieved tickets, giving the LLM window-specific KPI evidence.
        Pass "" (default) to reproduce the original C3 prompt exactly.
    """
    shap_lines = []
    for x in payload.shap_top3:
        direction = "above" if x.shap_value > 0 else "below"
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

    # RCA evidence block — only present when rca_context is provided (C3 with evidence)
    rca_block = ""
    if rca_context:
        rca_block = f"\n[EVIDENCE FROM RCA PIPELINE]\n{rca_context}\n"

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


def call_gemini(prompt: str, cfg: dict) -> str:
    """Call Gemini via the modern google-genai SDK (v2.x)."""
    if google_genai is None:
        raise RuntimeError("google-genai package is not installed")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    client = google_genai.Client(api_key=api_key)
    model_name = cfg["llm"].get("gemini_model", "gemini-3.5-flash-lite")
    temperature = cfg["llm"].get("temperature", 0.1)
    max_retries = cfg["llm"].get("max_retries", 3)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=google_genai_types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )
            result = response.text or ""
            logger.info("Called gemini (%s), response len: %d", model_name, len(result))
            return result
        except Exception as e:
            is_rate_limit = (
                (hasattr(e, "status_code") and e.status_code == 429)
                or "429" in str(e)
                or "RESOURCE_EXHAUSTED" in str(e)
            )
            if is_rate_limit:
                if attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                    logger.warning(
                        "Gemini rate limit (attempt %d/%d), waiting %ds...",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Gemini rate limit: all retries exhausted")
                    raise
            else:
                logger.error("Gemini call failed (attempt %d): %s", attempt + 1, e)
                raise
    raise RuntimeError("call_gemini: exhausted all retries without returning")


def call_llm(prompt: str, cfg: dict) -> str:
    backend = cfg["llm"]["backend"]
    if backend == "ollama":
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": cfg["llm"]["ollama_model"],
            "prompt": prompt,
            "stream": False,
            "temperature": cfg["llm"]["temperature"]
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json().get("response", "")
        logger.info("Called ollama, response len: %d", len(result))
        return result
    elif backend == "groq":
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
    try:
        check1 = bool(validate_3gpp_ref(ref))
    except Exception:
        check1 = False
    if fault_type and fault_type in alignment:
        expected = alignment[fault_type].get("3gpp_ts", "")
        check2 = bool(expected) and ref == expected
    else:
        all_ts = {entry.get("3gpp_ts") for entry in alignment.values() if entry.get("3gpp_ts")}
        check2 = ref in all_ts
    if not check1:
        logger.warning("Citation validation failed Check 1 (format regex): %s", ref)
    if not check2:
        logger.warning("Citation validation failed Check 2 (alignment table lookup): %s", ref)
    return check1 and check2

def explain_from_rca(
    window_index: int,
    fault_type: str,
    condition: str | int,
    cfg: dict,
    rca_context: str = "",
) -> LLMExplanation:
    """High-level convenience wrapper for demo / evaluation scripts.

    Builds a ClassifierOutput from the rca_evidence record for *window_index*,
    retrieves ChromaDB tickets, loads the alignment table, then delegates to
    explain_condition().

    Parameters
    ----------
    window_index : int
        Window to explain.
    fault_type : str
        Predicted fault label (from rca_evidence['predicted_fault']).
    condition : str | int
        Ablation condition — 'C1'/'C2'/'C3' or 1/2/3.
    cfg : dict
        Loaded config dict.
    rca_context : str
        Pre-formatted RCA evidence string from RCALoader.get_prompt_context().
        Only injected for condition 3/C3.
    """
    # Normalise condition to int
    _cond_map = {"C1": 1, "C2": 2, "C3": 3}
    if isinstance(condition, str):
        cond_int = _cond_map.get(condition.upper(), int(condition.lstrip("Cc")))
    else:
        cond_int = int(condition)

    # Build ClassifierOutput from rca_evidence data
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
    # Top-3 SHAP entries (sorted by |shap_value| desc) → SHAPEntry objects
    top3_raw = sorted(layer_b, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    # Pad to exactly 3 if fewer features are present
    while len(top3_raw) < 3:
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
        # Fallback: match case-insensitively
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

    # Retrieve ChromaDB tickets (used for C2/C3)
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
    """Flexible entry-point for LLM explanation generation.

    Supports two calling conventions:

    **Legacy (positional) — used by existing tests and scripts:**
        explain(payload, tickets, cfg, alignment)

    **New-style (keyword) — used by Step 6 demo and evaluation scripts:**
        explain(window_index=5, fault_type="Jamming", condition="C3",
                cfg=cfg, rca_context=rca_context)

    When ``window_index`` is provided the call is delegated to
    ``explain_from_rca()`` which builds the full pipeline internally.
    """
    if window_index is not None:
        # New-style call: delegate to explain_from_rca
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

    # Legacy positional call
    if payload is None or tickets is None or cfg is None or alignment is None:
        raise ValueError(
            "explain() requires either (payload, tickets, cfg, alignment) "
            "or (window_index, fault_type, condition, cfg)"
        )
    prompt = build_prompt(payload, tickets, alignment)
    max_retries = cfg["llm"]["max_retries"]
    parsed = None
    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt, cfg)
            parsed = parse_response(raw)
            break
        except RateLimitError:
            wait = 60 * (attempt + 1)
            logger.warning("Rate limit hit (attempt %d/%d), waiting %ds before retry...", attempt + 1, max_retries, wait)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                logger.error("Rate limit exceeded after all retries, using template")
        except Exception as e:
            logger.error("Attempt %d failed: %s", attempt + 1, e)
    if parsed is None:
        entry = alignment.get(payload.anomaly_type.value, {})
        return LLMExplanation(
            root_cause=f"{payload.anomaly_type.value} detected via KPI deviation",
            gpp_reference=entry.get("3gpp_ts", ""),
            oran_component=entry.get("oran_component", ""),
            recommended_action="Refer to alignment table for diagnostic steps",
            reasoning_trace=f"Template fallback. Top KPI: {payload.shap_top3[0].channel}",
            reference_valid=False,
            template_generated=True
        )
    return LLMExplanation(
        root_cause=parsed["root_cause"],
        gpp_reference=parsed["gpp_reference"],
        oran_component=parsed["oran_component"],
        recommended_action=parsed["recommended_action"],
        reasoning_trace=parsed["reasoning_trace"],
        reference_valid=validate_citation(parsed["gpp_reference"], alignment, fault_type=payload.anomaly_type.value),
        template_generated=False
    )


def _build_shap_summary(payload: ClassifierOutput) -> str:
    """Shared helper to format SHAP lines for prompts."""
    shap_lines = []
    for x in payload.shap_top3:
        direction = "above" if x.shap_value > 0 else "below"
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
    """Generate explanation using one of 3 ablation conditions.

    condition 1 -> label + SHAP only (no tickets, no alignment table)
    condition 2 -> label + SHAP + tickets (no alignment table)
    condition 3 -> full system (tickets + alignment table + rca_context)

    Parameters
    ----------
    rca_context : str
        Window-specific RCA evidence string from RCALoader.get_prompt_context().
        Only used for condition 3; ignored for conditions 1 and 2.

    Raises ValueError if condition not in {1, 2, 3}.
    """
    if condition not in (1, 2, 3):
        raise ValueError(f"condition must be 1, 2, or 3, got {condition}")
    if condition == 1:
        prompt = build_prompt_condition1(payload)
    elif condition == 2:
        prompt = build_prompt_condition2(payload, tickets)
    else:
        # C3: full system including RCA evidence when available
        prompt = build_prompt(payload, tickets, alignment, rca_context=rca_context)
    max_retries = cfg["llm"]["max_retries"]
    parsed = None
    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt, cfg)
            parsed = parse_response(raw)
            break
        except RateLimitError:
            wait = 60 * (attempt + 1)
            logger.warning("Rate limit hit (attempt %d/%d), waiting %ds before retry...", attempt + 1, max_retries, wait)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                logger.error("Rate limit exceeded after all retries, using template")
        except Exception as e:
            logger.error("Attempt %d (condition %d) failed: %s", attempt + 1, condition, e)
    if parsed is None:
        entry = alignment.get(payload.anomaly_type.value, {})
        return LLMExplanation(
            root_cause=f"{payload.anomaly_type.value} detected via KPI deviation",
            gpp_reference=entry.get("3gpp_ts", ""),
            oran_component=entry.get("oran_component", ""),
            recommended_action="Refer to alignment table for diagnostic steps",
            reasoning_trace=f"Template fallback. Top KPI: {payload.shap_top3[0].channel}",
            reference_valid=False,
            template_generated=True
        )
    return LLMExplanation(
        root_cause=parsed["root_cause"],
        gpp_reference=parsed["gpp_reference"],
        oran_component=parsed["oran_component"],
        recommended_action=parsed["recommended_action"],
        reasoning_trace=parsed["reasoning_trace"],
        reference_valid=validate_citation(parsed["gpp_reference"], alignment, fault_type=payload.anomaly_type.value),
        template_generated=False
    )

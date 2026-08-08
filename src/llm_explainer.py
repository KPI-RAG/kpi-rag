import json
import logging
import os
import re
import requests

try:
    from openai import OpenAI
except ImportError:
    pass

from src.schema import ClassifierOutput, RetrievedTicket, LLMExplanation
from src.utils import validate_3gpp_ref

logger = logging.getLogger(__name__)

def load_alignment_table(path: str) -> dict[str, dict]:
    with open(path, "r") as f:
        data = json.load(f)
    alignment = {}
    for entry in data.get("entries", []):
        alignment[entry["fault_type"]] = entry
    logger.info("Loaded %d entries from alignment table", len(alignment))
    return alignment

def build_prompt(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket],
    alignment: dict[str, dict]
) -> str:
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
    
    prompt = f"""You are a 5G network fault diagnosis expert.

Fault detected: {payload.anomaly_type.value}
Confidence: {payload.confidence:.0%}

Top contributing KPIs (SHAP):
{shap_summary}

Retrieved similar incidents:
{tickets_summary}

Standards reference for this fault type:
3GPP {gpp_ts} clause {clause}: {evidence_span}
O-RAN component: {oran_component}

Return ONLY a JSON object with exactly these fields:
{{
  "root_cause": "one sentence physical explanation",
  "3gpp_reference": "TS XX.XXX",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain"
}}
"""
    return prompt

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
        
    return parsed

def validate_citation(ref: str, alignment: dict[str, dict]) -> bool:
    try:
        check1 = bool(validate_3gpp_ref(ref))
    except Exception:
        check1 = False
    
    all_ts = {entry.get("3gpp_ts") for entry in alignment.values() if entry.get("3gpp_ts")}
    check2 = ref in all_ts
    
    if not check1:
        logger.warning("Citation validation failed Check 1 (format regex): %s", ref)
    if not check2:
        logger.warning("Citation validation failed Check 2 (alignment table lookup): %s", ref)
        
    return check1 and check2

def explain(
    payload: ClassifierOutput,
    tickets: list[RetrievedTicket],
    cfg: dict,
    alignment: dict[str, dict]
) -> LLMExplanation:
    prompt = build_prompt(payload, tickets, alignment)
    max_retries = cfg["llm"]["max_retries"]
    
    parsed = None
    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt, cfg)
            parsed = parse_response(raw)
            break
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
        gpp_reference=parsed["3gpp_reference"],
        oran_component=parsed["oran_component"],
        recommended_action=parsed["recommended_action"],
        reasoning_trace=parsed["reasoning_trace"],
        reference_valid=validate_citation(parsed["3gpp_reference"], alignment),
        template_generated=False
    )

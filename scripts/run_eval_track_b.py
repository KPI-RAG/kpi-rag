from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI


# Composite metric weights
# Primary overall-quality metric: treats all four dimensions equally.
EQUAL_WEIGHTS = {
    "citation_validity": 0.25,
    "fault_specificity": 0.25,
    "actionability": 0.25,
    "causal_soundness": 0.25,
}

# Secondary sensitivity metric: gives extra importance to standards grounding.
CITATION_WEIGHTED_WEIGHTS = {
    "citation_validity": 0.40,
    "fault_specificity": 0.20,
    "actionability": 0.20,
    "causal_soundness": 0.20,
}


def calculate_weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Calculate a weighted mean from the four G-Eval dimension scores."""
    return sum(weights[dimension] * scores[dimension] for dimension in weights)


# G-Eval rubric
GEVAL_JUDGE_PROMPT = """
Score this 5G telecommunications fault explanation against the supplied
standards-grounded alignment and evidence.

Evaluate technical correctness, not writing style. Score each dimension
independently using integers from 1 to 5.

GROUND TRUTH
Fault type: {fault_type}
Expected 3GPP reference: {expected_3gpp_ref}
Expected O-RAN component: {expected_oran_component}
Expected causal mechanism: {causal_mechanism}

GENERATED EXPLANATION
Root cause: {root_cause}
Cited 3GPP reference: {gpp_reference}
O-RAN component: {generated_oran_component}
Recommended action: {recommended_action}
Reasoning trace: {reasoning_trace}

RUBRICS

CV — Citation Validity
5 = Exact or highly specific specification and clause supporting the claim.
4 = Correct specification, but the clause is broader or less specific.
3 = Related specification, but support for the claim is indirect or weak.
2 = Incorrect or substantially mismatched specification.
1 = No usable citation, fabricated citation, or irrelevant reference.

FS — Fault Specificity
5 = Specific technical mechanism and associated symptom are correctly
    identified.
4 = Correct fault mechanism/family with a minor detail missing.
3 = Relevant fault, but the mechanism is ambiguous.
2 = Incorrect or substantially mismatched mechanism.
1 = Generic statement without a meaningful technical mechanism.

AC — Actionability
5 = Concrete remediation with a specific parameter, threshold, configuration,
    or measurable operational step.
4 = Clear, technically appropriate subsystem-level action, without a specific
    threshold or parameter.
3 = Relevant subsystem or action area, but the recommendation is vague.
2 = Generic advice with limited practical value.
1 = No meaningful or relevant action.

CS — Causal Soundness
5 = Technically coherent cause -> mechanism -> symptom -> KPI chain.
4 = Correct cause-to-KPI relationship, but an intermediate link is incomplete.
3 = Plausible relationship, but the causal mechanism or evidence link is weak.
2 = Little meaningful connection between the proposed cause and KPI behavior.
1 = Absent, contradictory, or technically inconsistent causal explanation.

EVALUATION RULES

1. Compare the generated reference with the expected reference. Do not assume
   a citation is correct simply because it looks plausible.
2. Equivalent technically correct wording should not be penalized.
3. Use the SHAP KPI evidence when assessing consistency with observed evidence.
4. Do not invent missing evidence or assume unsupported technical facts.
5. Assess the generated O-RAN component against the expected component.
6. Score each dimension independently.
7. Return an integer score from 1 to 5 for every dimension.
8. Use the key "justification" for each score's brief technical explanation.
9. Return ONLY a valid JSON object, without markdown fences or extra text.

OUTPUT FORMAT
{{
  "citation_validity": {{
    "score": 1,
    "justification": "Brief technical justification."
  }},
  "fault_specificity": {{
    "score": 1,
    "justification": "Brief technical justification."
  }},
  "actionability": {{
    "score": 1,
    "justification": "Brief technical justification."
  }},
  "causal_soundness": {{
    "score": 1,
    "justification": "Brief technical justification."
  }}
}}
""".strip()



# Data structures
@dataclass
class DimensionScore:
    score: float
    justification: str
    raw_logprobs: Optional[dict] = None


@dataclass
class GEvalScore:
    window_id: str
    fault_type: str
    citation_validity: DimensionScore
    fault_specificity: DimensionScore
    actionability: DimensionScore
    causal_soundness: DimensionScore

    def dimension_scores(self) -> dict[str, float]:
        """Return the four scores under the canonical dimension names."""
        return {
            "citation_validity": self.citation_validity.score,
            "fault_specificity": self.fault_specificity.score,
            "actionability": self.actionability.score,
            "causal_soundness": self.causal_soundness.score,
        }

    @property
    def equal_weight_composite(self) -> float:
        """Primary neutral composite: 25% per evaluation dimension."""
        return calculate_weighted_score(
            self.dimension_scores(), EQUAL_WEIGHTS
        )

    @property
    def citation_weighted_composite(self) -> float:
        """Secondary composite: CV=40%, FS=20%, AC=20%, CS=20%."""
        return calculate_weighted_score(
            self.dimension_scores(), CITATION_WEIGHTED_WEIGHTS
        )

    @property
    def composite(self) -> float:
        """Backward-compatible alias for the equal-weight composite."""
        return self.equal_weight_composite

    @property
    def citation_valid(self) -> bool:
        """A citation score of at least 4 counts as valid."""
        return self.citation_validity.score >= 4.0


@dataclass
class TrackBResults:
    per_sample: list[dict]
    summary: dict
    missing_explanations: list[dict]
    inter_rater: Optional[dict] = None


# Input and alignment helpers
def load_jsonl(path: str) -> list[dict]:
    """Load JSONL records, reporting malformed lines instead of hiding them."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_no} in {path}."
                )

            records.append(record)

    return records


def load_alignment_table(path: str) -> list[dict]:
    """Load alignment rows and normalise common project field names."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        rows = data

    elif isinstance(data, dict):
        rows = None

        for key in ("rows", "alignment_table"):
            if isinstance(data.get(key), list):
                rows = data[key]
                print(
                    f"Alignment table loaded from '{key}' "
                    f"({len(rows)} faults)"
                )
                break

        if rows is None:
            for key, value in data.items():
                if (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], dict)
                ):
                    rows = value
                    print(
                        f"Alignment table loaded from '{key}' "
                        f"({len(rows)} faults)"
                    )
                    break

        if rows is None:
            raise ValueError(
                "Could not find fault rows in the alignment table."
            )

    else:
        raise ValueError(
            f"Unexpected alignment table format: {type(data).__name__}"
        )

    normalised_rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        item = dict(row)

        if "telecomts_fault" in item and "fault_type" not in item:
            item["fault_type"] = item["telecomts_fault"]

        if "3gpp_reference" in item and "expected_3gpp_ref" not in item:
            item["expected_3gpp_ref"] = item["3gpp_reference"]

        if "gpp_reference" in item and "expected_3gpp_ref" not in item:
            item["expected_3gpp_ref"] = item["gpp_reference"]

        if "mechanism" in item and "causal_mechanism" not in item:
            item["causal_mechanism"] = item["mechanism"]

        if "o_ran_component" in item and "oran_component" not in item:
            item["oran_component"] = item["o_ran_component"]

        normalised_rows.append(item)

    return normalised_rows


def normalise_c3_records(
    raw_records: list[dict],
) -> tuple[list[dict], list[dict], int]:
    """
    Select C3 records and separate valid explanations from missing ones.

    Uses the actual Track C schema:
    - predicted_fault and true_fault at the record level
    - generated explanation fields under "explanation"
    """
    valid_explanations = []
    missing_explanations = []
    c3_record_count = 0

    for record in raw_records:
        if record.get("condition") != 3:
            continue

        c3_record_count += 1

        condition = record.get("condition")
        window_index = record.get("window_index")
        window_id = f"c{condition}_w{window_index}"

        predicted_fault = (
            record.get("predicted_fault")
            or record.get("fault_type")
            or record.get("anomaly_type")
            or "Unknown"
        )

        inner = record.get("explanation")

        valid = (
            isinstance(inner, dict)
            and isinstance(inner.get("root_cause"), str)
            and bool(inner["root_cause"].strip())
        )

        if not valid:
            missing_explanations.append({
                "window_id": window_id,
                "window_index": window_index,
                "condition": condition,
                "fault_type": predicted_fault,
                "status": "missing_explanation",
                "reason": (
                    "The Track C record has no valid explanation "
                    "with a non-empty root_cause."
                ),
            })
            continue

        valid_explanations.append({
            "window_id": window_id,
            "window_index": window_index,
            "condition": condition,
            "true_fault": record.get("true_fault", "Unknown"),
            "fault_type": predicted_fault,
            "root_cause": inner.get("root_cause", ""),
            "gpp_reference": inner.get("gpp_reference", ""),
            "oran_component": inner.get("oran_component", ""),
            "recommended_action": inner.get("recommended_action", ""),
            "reasoning_trace": inner.get("reasoning_trace", ""),
            # Preserve these for diagnostics, but do not send them to the judge.
            "reference_valid": inner.get("reference_valid"),
            "citation_correct": record.get("citation_correct"),
            "n_tickets": record.get("n_tickets"),
            "rca_injected": record.get("rca_injected"),
            "template_generated": inner.get("template_generated"),
            "fallback_reason": inner.get("fallback_reason", ""),
        })

    return valid_explanations, missing_explanations, c3_record_count


def find_alignment(
    alignment_table: list[dict],
    fault_type: str,
) -> dict:
    """Find an alignment row using a case-insensitive fault-type match."""
    target = str(fault_type or "").strip().casefold()

    for row in alignment_table:
        candidate = str(row.get("fault_type", "")).strip().casefold()

        if candidate == target:
            return row

    return {}


def prompt_value(value) -> str:
    """Render structured evidence as readable text for the judge prompt."""
    if value is None or value == "":
        return "N/A"

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)



# Judge response parsing
def extract_json_object(raw: str) -> dict:
    """Parse a JSON response, tolerating surrounding markdown fences."""
    cleaned = raw.strip()

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the first complete JSON object in the response.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            raise ValueError("Judge response did not contain a JSON object.")

        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object.")

    return result


def parse_dimension(result: dict, key: str) -> DimensionScore:
    """Validate and extract one dimension from the judge response."""
    dimension = result.get(key)

    if not isinstance(dimension, dict):
        raise ValueError(
            f"Judge response is missing the '{key}' object."
        )

    raw_score = dimension.get("score")

    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid score for '{key}': {raw_score!r}"
        ) from exc

    if not math.isfinite(score) or not 1 <= score <= 5:
        raise ValueError(
            f"Score for '{key}' must be between 1 and 5; got {score}."
        )

    justification = dimension.get(
        "justification",
        dimension.get("reason", ""),
    )

    return DimensionScore(
        score=score,
        justification=str(justification),
    )



# Single-window G-Eval
def geval_score_explanation(
    explanation: dict,
    alignment_row: dict,
    client: OpenAI,
    model_name: str,
    temperature: float = 0.0,
) -> GEvalScore:
    """Score one explanation using the configured judge model."""
    prompt = GEVAL_JUDGE_PROMPT.format(
        fault_type=prompt_value(explanation.get("fault_type")),
        expected_3gpp_ref=prompt_value(
            alignment_row.get("expected_3gpp_ref")
        ),
        expected_oran_component=prompt_value(
            alignment_row.get("oran_component")
        ),
        causal_mechanism=prompt_value(
            alignment_row.get("causal_mechanism")
        ),
        root_cause=prompt_value(explanation.get("root_cause")),
        gpp_reference=prompt_value(explanation.get("gpp_reference")),
        generated_oran_component=prompt_value(
            explanation.get("oran_component")
        ),
        recommended_action=prompt_value(
            explanation.get("recommended_action")
        ),
        reasoning_trace=prompt_value(
            explanation.get("reasoning_trace")
        ),
    )

    raw = ""

    # Retry transient empty responses. Rate-limit retries are handled
    # separately by the orchestration function.
    for attempt in range(4):
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict telecommunications evaluator. "
                        "Return only a valid JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw = (response.choices[0].message.content or "").strip()

        if raw:
            break

        if attempt < 3:
            wait = 2 ** attempt
            print(
                f"    [empty response on attempt {attempt + 1}, "
                f"retrying in {wait}s]"
            )
            time.sleep(wait)

    if not raw:
        raise ValueError(
            "The judge returned an empty response after four attempts."
        )

    result = extract_json_object(raw)

    return GEvalScore(
        window_id=explanation["window_id"],
        fault_type=explanation.get("fault_type", "Unknown"),
        citation_validity=parse_dimension(result, "citation_validity"),
        fault_specificity=parse_dimension(result, "fault_specificity"),
        actionability=parse_dimension(result, "actionability"),
        causal_soundness=parse_dimension(result, "causal_soundness"),
    )


# Summary helper
def summarise_values(values: list[float]) -> dict:
    """Return mean, sample standard deviation, minimum and maximum."""
    return {
        "mean": round(statistics.mean(values), 3) if values else None,
        "sd": (
            round(statistics.stdev(values), 3)
            if len(values) > 1
            else (0.0 if values else None)
        ),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarise_composite(values: list[float]) -> dict:
    """Return mean and sample standard deviation for a composite."""
    return {
        "mean": round(statistics.mean(values), 3) if values else None,
        "sd": (
            round(statistics.stdev(values), 3)
            if len(values) > 1
            else (0.0 if values else None)
        ),
    }


# ---------------------------------------------------------------------------
# Track B orchestration
# ---------------------------------------------------------------------------

def run_geval_track_b(
    explanations: list[dict],
    alignment_table: list[dict],
    missing_explanations: list[dict],
    expected_c3_count: int,
    human_scores: Optional[list[dict]] = None,
    output_path: str = "data/processed/track_b_geval_results.json",
    model_name: str = "qwen/qwen3.8-27b",
    base_url: str = "https://api.groq.com/openai/v1",
    api_key: str = "",
    inter_request_delay: float = 7.5,
    timeout: float = 180.0,
    workers: int = 1,
) -> TrackBResults:
    """
    Score valid C3 explanations and preserve missing records separately.

    The progress logger reports one result per window. Rate-limit responses
    are retried with exponential backoff; other errors are recorded.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(timeout, connect=10.0),
    )

    align_map = {
        str(row.get("fault_type", "")).strip().casefold(): row
        for row in alignment_table
    }

    n = len(explanations)

    print(
        f"Scoring {n} valid C3 explanations with {model_name} "
        f"({workers} workers, timeout={timeout:.0f}s per request)..."
    )

    per_sample: list[dict | None] = [None] * n
    completed = 0
    t_start = time.time()

    def score_one(index: int, explanation: dict) -> tuple[int, dict]:
        fault = explanation.get("fault_type", "Unknown")
        align_row = align_map.get(str(fault).strip().casefold(), {})
        start = time.time()
        last_error = None

        for attempt in range(8):
            try:
                score = geval_score_explanation(
                    explanation=explanation,
                    alignment_row=align_row,
                    client=client,
                    model_name=model_name,
                    temperature=0.0,
                )

                elapsed = time.time() - start

                if inter_request_delay > 0:
                    time.sleep(inter_request_delay)

                return index, {
                    "window_id": score.window_id,
                    "window_index": explanation.get("window_index"),
                    "condition": explanation.get("condition"),
                    "fault_type": score.fault_type,
                    "citation_validity": score.citation_validity.score,
                    "fault_specificity": score.fault_specificity.score,
                    "actionability": score.actionability.score,
                    "causal_soundness": score.causal_soundness.score,
                    "weighted_composite": round(
                        score.citation_weighted_composite, 3
                    ),
                    "equal_weight_composite": round(
                        score.equal_weight_composite, 3
                    ),
                    # Keep old key as an equal-weight alias for compatibility.
                    "composite": round(score.equal_weight_composite, 3),
                    "citation_valid": score.citation_valid,
                    "justifications": {
                        "citation_validity":
                            score.citation_validity.justification,
                        "fault_specificity":
                            score.fault_specificity.justification,
                        "actionability":
                            score.actionability.justification,
                        "causal_soundness":
                            score.causal_soundness.justification,
                    },
                    "_elapsed": round(elapsed, 1),
                }

            except Exception as exc:
                last_error = str(exc)

                is_rate_limit = (
                    "429" in last_error
                    or "rate_limit" in last_error.casefold()
                    or "rate limit" in last_error.casefold()
                )

                if is_rate_limit and attempt < 7:
                    match = re.search(
                        r"try again in ([\d.]+)s",
                        last_error,
                        flags=re.IGNORECASE,
                    )

                    wait = (
                        float(match.group(1)) + 1.0
                        if match
                        else min(2 ** (attempt + 1), 60)
                    )

                    print(
                        f"    [rate-limited on attempt {attempt + 1}/8, "
                        f"waiting {wait:.1f}s]"
                    )
                    time.sleep(wait)
                    continue

                # Do not retry ordinary authentication, model-name,
                # malformed-request, or response-validation errors.
                break

        return index, {
            "window_id": explanation.get("window_id"),
            "window_index": explanation.get("window_index"),
            "condition": explanation.get("condition"),
            "fault_type": fault,
            "error": last_error or "Unknown judge error",
            "_elapsed": round(time.time() - start, 1),
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = {
            pool.submit(score_one, index, explanation): index
            for index, explanation in enumerate(explanations)
        }

        for future in concurrent.futures.as_completed(futures):
            index, record = future.result()
            per_sample[index] = record
            completed += 1

            fault = str(record.get("fault_type", "?"))[:35]
            elapsed = record.get("_elapsed", 0)
            wall = time.time() - t_start

            if "error" in record:
                print(
                    f"  [{completed:02d}/{n}] ERROR  {fault}: "
                    f"{record['error']} "
                    f"({elapsed:.0f}s call / {wall:.0f}s wall)"
                )
            else:
                print(
                    f"  [{completed:02d}/{n}] {fault:35s} "
                    f"CV={record['citation_validity']:.1f} "
                    f"FS={record['fault_specificity']:.1f} "
                    f"AC={record['actionability']:.1f} "
                    f"CS={record['causal_soundness']:.1f} "
                    f"→ {'✓' if record['citation_valid'] else '✗'} "
                    f"({elapsed:.0f}s call / {wall:.0f}s wall)"
                )

    # Remove internal timing fields from the saved records.
    per_sample = [row for row in per_sample if row is not None]

    for row in per_sample:
        row.pop("_elapsed", None)

    valid = [row for row in per_sample if "error" not in row]
    judge_errors = [row for row in per_sample if "error" in row]

    dimensions = [
        "citation_validity",
        "fault_specificity",
        "actionability",
        "causal_soundness",
    ]

    n_scored = len(valid)
    n_missing = len(missing_explanations)
    n_judge_errors = len(judge_errors)

    # All selected C3 records count toward the expected total, regardless
    # of whether generation or judging succeeded.
    n_expected = expected_c3_count

    # Generation completeness measures whether Track C produced a valid
    # explanation. Judge/API errors are a separate evaluation-stage outcome.
    n_generated = n_scored + n_judge_errors

    completeness_rate = (
        n_generated / n_expected
        if n_expected > 0
        else 0.0
    )

    summary = {
        "n_expected": n_expected,
        "n_valid_explanations": len(explanations),
        "n_scored": n_scored,
        "n_missing_explanations": n_missing,
        "n_judge_errors": n_judge_errors,
        "explanation_completeness_rate": round(completeness_rate, 3),
        "citation_validity_rate": None,
        "composite_weights": {
            "primary_equal_weight": EQUAL_WEIGHTS,
            "secondary_citation_weighted": CITATION_WEIGHTED_WEIGHTS,
        },
    }

    for dimension in dimensions:
        values = [row[dimension] for row in valid]
        summary[dimension] = summarise_values(values)

    if valid:
        citation_valid_rate = (
            sum(1 for row in valid if row["citation_valid"]) / len(valid)
        )
        summary["citation_validity_rate"] = round(
            citation_valid_rate, 3
        )

    equal_values = [row["equal_weight_composite"] for row in valid]
    weighted_values = [row["weighted_composite"] for row in valid]

    summary["equal_weight_composite"] = summarise_composite(equal_values)
    summary["citation_weighted_composite"] = summarise_composite(
        weighted_values
    )

    # Backward-compatible summary key: equal-weight score is the default.
    summary["composite"] = summary["equal_weight_composite"]

    print("\n=== TRACK B SUMMARY (C3 FULL PIPELINE) ===")
    print(f"  Expected C3 windows:             {n_expected}")
    print(f"  Valid explanations:              {len(explanations)}")
    print(f"  Successfully scored:             {n_scored}")
    print(f"  Missing explanations:            {n_missing}")
    print(f"  Judge/API errors:                {n_judge_errors}")
    print(
        f"  Explanation completeness:        "
        f"{completeness_rate * 100:.1f}%"
    )

    if valid:
        print(
            f"  Citation Validity Rate (>=4):    "
            f"{summary['citation_validity_rate'] * 100:.1f}%"
        )

        for dimension in dimensions:
            item = summary[dimension]
            print(
                f"  {dimension:25s} "
                f"{item['mean']:.2f} ± {item['sd']:.2f} "
                f"[{item['min']}–{item['max']}]"
            )

        equal = summary["equal_weight_composite"]
        weighted = summary["citation_weighted_composite"]

        print(
            f"  {'Equal-weight composite':25s} "
            f"{equal['mean']:.2f} ± {equal['sd']:.2f}"
        )
        print(
            f"  {'Citation-weighted composite':25s} "
            f"{weighted['mean']:.2f} ± {weighted['sd']:.2f}"
        )
    else:
        print(
            "  No judge scores are available. "
            "Review the per-window errors before interpreting results."
        )

    if missing_explanations:
        print("\nMissing C3 explanations:")

        for record in missing_explanations:
            print(
                f"  - {record['window_id']} | "
                f"fault={record['fault_type']}"
            )

    inter_rater = None

    if human_scores:
        inter_rater = compute_inter_rater(
            valid,
            human_scores,
            dimensions,
        )

    results = TrackBResults(
        per_sample=per_sample,
        summary=summary,
        missing_explanations=missing_explanations,
        inter_rater=inter_rater,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            asdict(results),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nResults saved → {output_path}")

    return results


# Command-line interface
def main() -> None:
    parser = argparse.ArgumentParser(
        description="KPI-RAG Track B G-Eval evaluation"
    )

    parser.add_argument(
        "--explanations",
        required=True,
        help="JSONL file containing Track C explanation records.",
    )
    parser.add_argument(
        "--alignment",
        required=True,
        help="JSON standards/fault alignment table.",
    )
    parser.add_argument(
        "--human-scores",
        default=None,
        help="Optional JSON file containing human evaluation scores.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/track_b_geval_results.json",
        help="Path for the Track B result JSON.",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3.8-27b",
        help="Judge model identifier supported by the configured API.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Prefer GROQ_API_KEY in the environment.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds per request.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests. Start with 1 for rate-limited API tiers.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=7.5,
        help="Delay after a successful request, in seconds.",
    )

    args = parser.parse_args()

    # Resolve credentials and endpoint.
    api_key = args.api_key or os.getenv("GROQ_API_KEY", "")
    base_url = args.base_url

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Configure it in the server terminal "
            "or pass --api-key explicitly."
        )

    if base_url is None:
        base_url = "https://api.groq.com/openai/v1"

    print(f"Judge model: {args.model}")
    print(f"API endpoint: {base_url}")

    # Load Track C output and separate C3 generation failures.
    print(f"\nLoading Track C explanations from {args.explanations}")

    raw_records = load_jsonl(args.explanations)

    print(f"Loaded {len(raw_records)} total Track C records.")

    (
        explanations,
        missing_explanations,
        expected_c3_count,
    ) = normalise_c3_records(raw_records)

    print(f"Expected C3 records:              {expected_c3_count}")
    print(f"Valid C3 explanations:             {len(explanations)}")
    print(f"Missing/incomplete C3 explanations:{len(missing_explanations):4d}")

    if expected_c3_count == 0:
        raise ValueError(
            "No condition=3 records were found. "
            "Check the input file and condition field."
        )

    if not explanations:
        raise ValueError(
            "No valid C3 explanations are available for judging. "
            "Missing records will not be sent to the judge."
        )

    # Load alignment data.
    print(f"\nLoading alignment table from {args.alignment}")

    alignment_table = load_alignment_table(args.alignment)

    print(
        f"Alignment table ready: {len(alignment_table)} faults — "
        f"{[row.get('fault_type', 'Unknown') for row in alignment_table]}"
    )

    # Optional human scores.
    human_scores = None

    if args.human_scores:
        with open(args.human_scores, "r", encoding="utf-8") as f:
            human_scores = json.load(f)

        if isinstance(human_scores, dict):
            human_scores = human_scores.get(
                "scores",
                human_scores.get("results", []),
            )

        if not isinstance(human_scores, list):
            raise ValueError(
                "Human scores must be a JSON list or an object containing "
                "'scores' or 'results'."
            )

    run_geval_track_b(
        explanations=explanations,
        alignment_table=alignment_table,
        missing_explanations=missing_explanations,
        expected_c3_count=expected_c3_count,
        human_scores=human_scores,
        output_path=args.output,
        model_name=args.model,
        base_url=base_url,
        api_key=api_key,
        inter_request_delay=args.delay,
        timeout=args.timeout,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()


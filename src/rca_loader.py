"""
src/rca_loader.py

RCA evidence loader for KPI-RAG pipeline.
Provides O(1) lookup of rca_evidence.json records by window_index and
formats a concise prompt context string for LLM injection (C3 condition).
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RCALoader:
    """Load rca_evidence.json and provide fast lookup + prompt formatting.

    Parameters
    ----------
    path : str
        Path to rca_evidence.json (1,235 records, keyed by window_index).
    """

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            records: list[dict] = json.load(f)
        self._index: dict[int, dict] = {
            int(r["window_index"]): r for r in records
        }
        logger.info(
            "RCALoader: loaded %d records from %s", len(self._index), path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, window_index: int) -> Optional[dict]:
        """Return the RCA record for *window_index*, or None if not found."""
        return self._index.get(int(window_index))

    def get_prompt_context(self, window_index: int) -> str:
        """Return a formatted context string (<400 tokens) for C3 prompts.

        Returns empty string "" if the record is not found.
        """
        record = self.get(window_index)
        if record is None:
            return ""
        return self._format_context(record)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_context(self, record: dict) -> str:
        lines: list[str] = []

        # --- Header ---
        fault = record.get("predicted_fault", "Unknown")
        conf = record.get("confidence", 0.0)
        lines.append(
            f"[RCA PIPELINE EVIDENCE — window {record['window_index']}]"
        )
        lines.append(f"Predicted fault: {fault} (confidence: {conf:.1%})")

        # --- Layer B: SHAP attribution (top 3 by |shap_value|) ---
        layer_b: list[dict] = record.get("layer_b_model_attribution", [])
        if layer_b:
            sorted_b = sorted(
                layer_b, key=lambda x: abs(x.get("shap_value", 0)), reverse=True
            )[:3]
            lines.append("Model attribution (SHAP, top 3 features):")
            for entry in sorted_b:
                feature = entry.get("feature", entry.get("channel", "?"))
                shap_val = entry.get("shap_value", 0)
                vs_normal = entry.get("feature_vs_normal", "")
                effect = entry.get("shap_effect", "")
                direction = "above" if "above" in vs_normal else "below"
                lines.append(
                    f"  - {feature}: SHAP={shap_val:+.3f} ({direction} normal,"
                    f" {effect.replace('_', ' ')})"
                )

        # --- KPI evidence: only anomalous/relevant KPIs ---
        kpi_evidence: list[dict] = record.get("kpi_evidence", [])
        anomalous_kpis = [
            k for k in kpi_evidence
            if k.get("evidence_status") in ("unexpected", "missing", "supporting")
            and k.get("shap_supported")
        ]
        if not anomalous_kpis:
            # Fall back to any entry with a non-zero observed mean
            anomalous_kpis = [
                k for k in kpi_evidence
                if k.get("observed_mean") is not None
            ][:4]
        if anomalous_kpis:
            lines.append("Key KPI observations:")
            for k in anomalous_kpis[:4]:
                kpi_name = k.get("kpi", "?")
                obs_mean = k.get("observed_mean")
                status = k.get("evidence_status", "")
                mean_str = f"mean={obs_mean:.4g}" if obs_mean is not None else ""
                lines.append(f"  - {kpi_name}: {status} {mean_str}".rstrip())

        # --- Layer C: domain standards evidence ---
        layer_c: dict = record.get("layer_c_domain_standards", {})
        causal = layer_c.get("causal_mechanism", "") or record.get("causal_mechanism", "")
        ref = layer_c.get("3gpp_reference", "")
        oran = layer_c.get("oran_component", "")

        if causal:
            # Truncate to ~120 chars to stay under token budget
            causal_short = causal[:120].rstrip()
            if len(causal) > 120:
                causal_short += "..."
            lines.append(f"Causal mechanism: {causal_short}")

        if ref and ref != "None - physical RF attack" and ref != "None — physical RF attack":
            lines.append(f"Standards grounding: {ref}")
            if oran:
                lines.append(f"O-RAN component: {oran}")
        elif ref and "physical RF attack" in ref:
            lines.append(
                "Standards grounding: None — physical RF attack (no 3GPP clause applies)"
            )
            fallback = record.get("pipeline_fallback", "")
            if fallback:
                lines.append(f"Note: {fallback}")

        return "\n".join(lines)

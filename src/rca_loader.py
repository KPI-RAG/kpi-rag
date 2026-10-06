"""Lookup of rca_evidence.json records by window_index, plus a short prompt block for C3."""
import json
import logging

logger = logging.getLogger(__name__)


class RCALoader:
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            records: list[dict] = json.load(f)
        self._index: dict[int, dict] = {int(r["window_index"]): r for r in records}
        logger.info("RCALoader: loaded %d records from %s", len(self._index), path)

    def get(self, window_index: int) -> dict | None:
        return self._index.get(int(window_index))

    def find_by_fault(self, fault: str) -> dict | None:
        """First record with this predicted_fault — a representative example, not the window itself."""
        return next((r for r in self._index.values() if r.get("predicted_fault") == fault), None)

    def get_prompt_context(self, window_index: int) -> str:
        """Compact (<400 token) evidence block for the C3 prompt; "" if the window is unknown."""
        record = self.get(window_index)
        if record is None:
            return ""
        return self._format_context(record)

    def _format_context(self, record: dict) -> str:
        lines: list[str] = []
        fault = record.get("predicted_fault", "Unknown")
        conf = record.get("confidence", 0.0)
        lines.append(f"[RCA PIPELINE EVIDENCE — window {record['window_index']}]")
        lines.append(f"Predicted fault: {fault} (confidence: {conf:.1%})")

        # Layer B: top-3 SHAP features by magnitude
        layer_b: list[dict] = record.get("layer_b_model_attribution", [])
        if layer_b:
            sorted_b = sorted(layer_b, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
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

        # Layer A: only the KPIs that carry evidence, max four
        kpi_evidence: list[dict] = record.get("kpi_evidence", [])
        anomalous_kpis = [
            k for k in kpi_evidence
            if k.get("evidence_status") in ("unexpected", "missing", "supporting")
            and k.get("shap_supported")
        ]
        if not anomalous_kpis:
            anomalous_kpis = [k for k in kpi_evidence if k.get("observed_mean") is not None][:4]
        if anomalous_kpis:
            lines.append("Key KPI observations:")
            for k in anomalous_kpis[:4]:
                kpi_name = k.get("kpi", "?")
                obs_mean = k.get("observed_mean")
                status = k.get("evidence_status", "")
                mean_str = f"mean={obs_mean:.4g}" if obs_mean is not None else ""
                lines.append(f"  - {kpi_name}: {status} {mean_str}".rstrip())

        # Layer C: standards
        layer_c: dict = record.get("layer_c_domain_standards", {})
        causal = layer_c.get("causal_mechanism", "") or record.get("causal_mechanism", "")
        ref = layer_c.get("3gpp_reference", "")
        oran = layer_c.get("oran_component", "")

        if causal:
            causal_short = causal[:120].rstrip()
            if len(causal) > 120:
                causal_short += "..."
            lines.append(f"Causal mechanism: {causal_short}")

        if ref and "physical RF attack" not in ref:
            lines.append(f"Standards grounding: {ref}")
            if oran:
                lines.append(f"O-RAN component: {oran}")
        elif ref:
            lines.append("Standards grounding: None — physical RF attack (no 3GPP clause applies)")
            fallback = record.get("pipeline_fallback", "")
            if fallback:
                lines.append(f"Note: {fallback}")

        return "\n".join(lines)

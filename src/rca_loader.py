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
        """Window evidence only. Deliberately left out:
        - the record's standards / O-RAN lines: the alignment table is the single source of
          standards in the prompt (several RCA records still carry v1.1 text, e.g. Handover
          TS 38.133);
        - kpi_evidence["evidence_status"]: derived from class-level expected KPIs that were
          aggregated over labelled windows including the test split.
        """
        lines: list[str] = []
        fault = record.get("predicted_fault", "Unknown")
        conf = record.get("confidence", 0.0)
        lines.append(f"[RCA PIPELINE EVIDENCE — window {record['window_index']}]")
        lines.append(f"Predicted fault: {fault} (confidence: {conf:.1%})")

        # Layer B: top-3 SHAP features; Layer 2 stores |SHAP| and the sign as shap_effect
        top = sorted(record.get("layer_b_model_attribution", []),
                     key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
        if top:
            lines.append("Model attribution (SHAP, top 3 features):")
            for entry in top:
                direction = "above" if "above" in entry.get("feature_vs_normal", "") else "below"
                effect = entry.get("shap_effect", "").replace("_", " ")
                lines.append(
                    f"  - {entry.get('feature', entry.get('channel', '?'))}: "
                    f"|SHAP|={abs(entry.get('shap_value', 0)):.3f} ({direction} normal, {effect})".replace(", )", ")")
                )

        # Layer A: observed means of the KPIs behind those features
        channels = list(dict.fromkeys(e.get("channel") for e in top if e.get("channel")))
        observed = {k.get("kpi"): k.get("observed_mean") for k in record.get("kpi_evidence", [])}
        obs_lines = [f"  - {c}: mean={observed[c]:.4g}" for c in channels if observed.get(c) is not None]
        if obs_lines:
            lines.append("Observed KPI means (window):")
            lines += obs_lines

        causal = record.get("layer_c_domain_standards", {}).get("causal_mechanism", "") or record.get("causal_mechanism", "")
        if causal:
            causal_short = causal[:120].rstrip()
            lines.append(f"Causal mechanism: {causal_short}{'...' if len(causal) > 120 else ''}")

        return "\n".join(lines)

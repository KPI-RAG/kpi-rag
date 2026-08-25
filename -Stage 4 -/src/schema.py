from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class AnomalyType(str, Enum):
    CCI_MILD = "Co-Channel Interference (Mild)"
    BUFFER_OVERFLOW = "Buffer Overflow (Gradual Buildup)"
    CCI_SEVERE = "Co-Channel Interference (Severe)"
    ANTENNA_FAILURE = "Antenna Failure"
    FAULTY_RF_FILTERS = "Faulty RF Filters (Temporal)"
    CONGESTION_GRADUAL = "High Network Congestion (Gradual Buildup)"
    DOPPLER_SHIFT = "Doppler Shift (Severe)"
    FAULTY_HANDOVER = "Faulty Handover Algorithm (Too Frequent)"
    RESOURCE_BUGS = "Resource Allocation Bugs"
    CONGESTION_SUDDEN = "High Network Congestion (Sudden Spike)"
    JAMMING = "Jamming"

class SHAPEntry(BaseModel):
    channel: str
    shap_value: float
    direction: Literal["above_normal", "below_normal"]

class SignalStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float

class ClassifierOutput(BaseModel):
    anomaly_type: AnomalyType
    confidence: float = Field(ge=0.0, le=1.0)
    shap_top3: list[SHAPEntry]
    signal_statistics: dict[str, SignalStats]

    @model_validator(mode="after")
    def validate_shap_len(self):
        if len(self.shap_top3) != 3:
            raise ValueError("shap_top3 must contain exactly 3 items")
        return self

class RetrievedTicket(BaseModel):
    ticket_id: str
    content: str
    anomaly_type: str
    similarity_score: float

class LLMExplanation(BaseModel):
    root_cause: str
    gpp_reference: str
    oran_component: str
    recommended_action: str
    reasoning_trace: str
    reference_valid: bool
    template_generated: bool

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum
import uuid


class JourneyStage(str, Enum):
    ENROLLMENT           = "enrollment"            # newly prescribed, setting up treatment
    ONBOARDING           = "onboarding"            # days 1-3, first experiences
    TREATMENT_INITIATION = "treatment_initiation"  # days 4-14, habit-forming, early side effects
    ADHERENCE            = "adherence"             # ongoing, staying on track
    SIDE_EFFECT_MONITORING = "side_effect_monitoring"  # active symptom tracking


def infer_stage(days_on_treatment: int) -> JourneyStage:
    if days_on_treatment == 0:
        return JourneyStage.ENROLLMENT
    elif days_on_treatment <= 3:
        return JourneyStage.ONBOARDING
    elif days_on_treatment <= 14:
        return JourneyStage.TREATMENT_INITIATION
    else:
        return JourneyStage.ADHERENCE


class PatientContext(BaseModel):
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    drug_name: str
    dose_schedule: str                  # e.g. "10mg once daily"
    days_on_treatment: int
    known_conditions: List[str] = []
    journey_stage: JourneyStage = JourneyStage.ADHERENCE


class ComplianceCitation(BaseModel):
    claim: str                          # what the agent said
    source_section: str                 # e.g. "Section 5.2 - Warnings"
    source_text: str                    # exact passage from FDA label


class EscalationFlag(BaseModel):
    reason: str
    severity: Literal["low", "medium", "high"]
    trigger_phrase: str                 # what the patient said


class CareTeamFlag(BaseModel):
    reason: str
    severity: Literal["low", "medium", "high"]


class AppointmentRequest(BaseModel):
    reason: str
    preferred_timing: Literal["urgent", "this_week", "routine"]


class PostCallSummary(BaseModel):
    session_id: str
    patient_id: str
    drug: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Adherence
    adherence_signal: Literal["on_track", "at_risk", "non_adherent"]
    days_on_treatment: int

    # Conversation
    topics_covered: List[str]
    side_effects_mentioned: List[str]
    questions_asked: List[str]

    # Compliance
    compliance_citations: List[ComplianceCitation]
    guardrail_triggers: int             # how many times guardrails fired

    # Escalation
    escalate_to_human: bool
    escalation_flags: List[EscalationFlag]
    care_team_flags: List[CareTeamFlag] = []
    appointment_requests: List[AppointmentRequest] = []

    # Latency
    avg_response_latency_ms: Optional[float] = None
    time_to_first_audio_ms: Optional[float] = None


class TurnMetrics(BaseModel):
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stt_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    total_latency_ms: float
    guardrail_fired: bool = False

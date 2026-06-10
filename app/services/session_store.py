import logging
from typing import Dict, Optional
from datetime import datetime
from app.models.session import (
    PostCallSummary, PatientContext, TurnMetrics,
    ComplianceCitation, EscalationFlag
)

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self, session_id: str, patient_context: PatientContext):
        self.sessions[session_id] = {
            "patient_context": patient_context,
            "turns": [],
            "citations": [],
            "escalation_flags": [],
            "topics": set(),
            "side_effects": set(),
            "questions": [],
            "guardrail_triggers": 0,
            "escalated": False,
            "started_at": datetime.utcnow()
        }
        logger.info(f"Session created: {session_id} | Drug: {patient_context.drug_name}")

    def add_turn(
        self,
        session_id: str,
        patient_text: str,
        agent_response: str,
        metrics: TurnMetrics,
        citations: list,
        escalation_flags: list,
        guardrail_fired: bool
    ):
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        session["turns"].append({
            "patient": patient_text,
            "agent": agent_response,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
        session["citations"].extend(citations)
        session["escalation_flags"].extend(escalation_flags)

        if guardrail_fired:
            session["guardrail_triggers"] += 1

        if escalation_flags:
            session["escalated"] = True

        # Extract topics from patient text
        self._extract_topics(session_id, patient_text)

    def _extract_topics(self, session_id: str, text: str):
        text_lower = text.lower()
        session = self.sessions[session_id]

        topic_keywords = {
            "missed_dose": ["missed", "forgot", "skip"],
            "side_effects": ["side effect", "feeling", "pain", "nausea", "dizzy"],
            "dosage_question": ["how much", "how many", "dose", "mg"],
            "refill": ["refill", "running out", "prescription"],
            "cost": ["cost", "afford", "insurance", "pay"],
            "effectiveness": ["working", "not working", "feel better", "improvement"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                session["topics"].add(topic)

        # Track side effects mentioned
        side_effects = ["dizziness", "nausea", "headache", "rash", "fatigue",
                       "stomach pain", "vomiting", "diarrhea", "swelling"]
        for effect in side_effects:
            if effect in text_lower:
                session["side_effects"].add(effect)

        # Track questions
        if "?" in text or any(q in text_lower for q in ["what", "how", "when", "why", "can i"]):
            session["questions"].append(text[:100])

    def get_summary(self, session_id: str) -> Optional[PostCallSummary]:
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        ctx: PatientContext = session["patient_context"]
        turns = session["turns"]

        # Calculate avg latencies
        avg_latency = None
        ttfa = None
        if turns:
            latencies = [t["metrics"].total_latency_ms for t in turns]
            avg_latency = sum(latencies) / len(latencies)
            ttfa = turns[0]["metrics"].total_latency_ms if turns else None

        # Adherence signal
        topics = session["topics"]
        flags = session["escalation_flags"]
        if any(f.severity == "high" for f in flags):
            adherence_signal = "non_adherent"
        elif "missed_dose" in topics or len(flags) > 0:
            adherence_signal = "at_risk"
        else:
            adherence_signal = "on_track"

        return PostCallSummary(
            session_id=session_id,
            patient_id=ctx.patient_id,
            drug=ctx.drug_name,
            adherence_signal=adherence_signal,
            days_on_treatment=ctx.days_on_treatment,
            topics_covered=list(topics),
            side_effects_mentioned=list(session["side_effects"]),
            questions_asked=session["questions"],
            compliance_citations=session["citations"],
            guardrail_triggers=session["guardrail_triggers"],
            escalate_to_human=session["escalated"],
            escalation_flags=flags,
            avg_response_latency_ms=avg_latency,
            time_to_first_audio_ms=ttfa
        )

    def end_session(self, session_id: str):
        summary = self.get_summary(session_id)
        self.sessions.pop(session_id, None)
        return summary


session_store = SessionStore()

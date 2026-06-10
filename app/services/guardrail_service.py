import logging
from typing import Tuple, List
from app.models.session import EscalationFlag, ComplianceCitation

logger = logging.getLogger(__name__)

# Side effects / symptoms that trigger escalation.
# Patterns cover natural speech variation: "fainted" not just "fainting",
# "can't breathe" not just "difficulty breathing", etc.
HIGH_SEVERITY_SYMPTOMS = [
    "chest pain", "chest pressure", "chest tightness",
    "difficulty breathing", "shortness of breath", "can't breathe",
    "cannot breathe", "trouble breathing", "hard to breathe",
    "severe rash",
    "swelling of face", "face is swollen", "face is swelling", "face swelling",
    "throat swelling", "throat feels swollen", "throat is swollen", "throat is swelling",
    "severe dizziness", "fainting", "fainted", "passed out", "blacked out",
    "blurred vision", "vision blurred",
    "severe headache", "heart racing", "irregular heartbeat", "beating irregularly",
    "heart beating irregularly", "lost consciousness",
]

MEDIUM_SEVERITY_SYMPTOMS = [
    "dizziness", "dizzy", "lightheaded", "light-headed",
    "nausea", "nauseous", "feeling sick",
    "vomiting", "vomited", "throwing up",
    "rash", "swelling", "swollen",
    "stomach pain", "abdominal pain", "muscle pain", "joint pain", "fever",
    "unusual bleeding", "bruising",
]

# Phrases the agent must NEVER produce — guardrail triggers.
# Covers morphological variants: "taking more" alongside "take more".
DOSE_CHANGE_PATTERNS = [
    "increase your dose", "decrease your dose",
    "take more", "taking more",
    "take less", "taking less",
    "double your dose", "skip your dose",
    "change when you take", "adjust your medication",
]

UNAUTHORIZED_CLAIM_PATTERNS = [
    "will cure", "guaranteed to", "definitely works",
    "no side effects", "completely safe", "better than",
    "instead of seeing your doctor"
]


# Words immediately before a symptom phrase that indicate the patient is
# denying or contextualizing the symptom rather than reporting it.
_NEGATION_PREFIXES = (
    "no ", "not ", "don't ", "dont ", "didn't ", "didnt ",
    "never ", "without ", "denies ", "no more ", "not having ",
)

def _is_negated(text: str, phrase: str) -> bool:
    """Return True if the symptom phrase is preceded by a negation word."""
    idx = text.find(phrase)
    if idx == -1:
        return False
    window = text[max(0, idx - 25): idx]
    return any(neg in window for neg in _NEGATION_PREFIXES)


class GuardrailService:

    def check_patient_input(self, text: str) -> Tuple[bool, List[EscalationFlag]]:
        """
        Scan patient's speech for escalation triggers.
        Returns (should_escalate, list_of_flags)
        """
        text_lower = text.lower()
        flags = []

        for symptom in HIGH_SEVERITY_SYMPTOMS:
            if symptom in text_lower and not _is_negated(text_lower, symptom):
                flags.append(EscalationFlag(
                    reason=f"Patient reported high-severity symptom: {symptom}",
                    severity="high",
                    trigger_phrase=symptom
                ))

        for symptom in MEDIUM_SEVERITY_SYMPTOMS:
            if (symptom in text_lower
                    and not _is_negated(text_lower, symptom)
                    and not any(s in text_lower for s in HIGH_SEVERITY_SYMPTOMS)):
                flags.append(EscalationFlag(
                    reason=f"Patient reported symptom requiring monitoring: {symptom}",
                    severity="medium",
                    trigger_phrase=symptom
                ))

        should_escalate = any(f.severity == "high" for f in flags)
        return should_escalate, flags

    def check_agent_response(self, response: str) -> Tuple[bool, str]:
        """
        Check if agent's response violates compliance rules.
        Returns (is_violation, reason)
        """
        response_lower = response.lower()

        for pattern in DOSE_CHANGE_PATTERNS:
            if pattern in response_lower:
                return True, f"Agent attempted to suggest dose change: '{pattern}'"

        for pattern in UNAUTHORIZED_CLAIM_PATTERNS:
            if pattern in response_lower:
                return True, f"Agent made unauthorized claim: '{pattern}'"

        return False, ""

    def build_safe_response(self, violation_reason: str) -> str:
        """Return a safe fallback response when guardrail fires."""
        if "dose" in violation_reason.lower():
            return (
                "I'm not able to advise on changing your medication dosage. "
                "Please contact your healthcare provider or pharmacist for any "
                "questions about your dose. Is there anything else I can help you with?"
            )
        return (
            "I want to make sure I give you accurate information. "
            "For this question, I'd recommend speaking directly with your "
            "healthcare provider. Is there anything else I can assist you with?"
        )

    def build_escalation_response(self, flags: List[EscalationFlag]) -> str:
        """Return an empathetic escalation response."""
        high_flags = [f for f in flags if f.severity == "high"]
        if high_flags:
            symptom = high_flags[0].trigger_phrase
            return (
                f"I'm concerned about what you've shared regarding {symptom}. "
                "This sounds like something that needs immediate attention from "
                "your healthcare provider. Please contact them now or call emergency "
                "services if you feel this is urgent. I'm flagging this for your care team."
            )
        return (
            "Thank you for sharing that. I'm noting this for your care team "
            "and would recommend mentioning it at your next appointment. "
            "Would you like me to help you prepare what to say to your doctor?"
        )


guardrail_service = GuardrailService()

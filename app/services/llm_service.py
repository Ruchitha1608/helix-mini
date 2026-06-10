import anthropic
import base64
import logging
import time
from typing import Optional, Tuple, List
from app.core.config import settings
from app.services.rag_service import rag_service
from app.services.guardrail_service import guardrail_service
from app.models.session import ComplianceCitation, EscalationFlag, JourneyStage, PatientContext

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

_BASE_SYSTEM_PROMPT = """You are Helix, a compassionate patient support AI for people managing chronic conditions.

YOUR ROLE:
- Help patients understand their treatment, stay adherent, and feel supported
- Answer questions about their medication using ONLY the FDA-approved label information provided
- Provide empathetic, clear, plain-language responses

STRICT RULES — NEVER VIOLATE:
1. NEVER suggest changing, increasing, or decreasing a dose
2. NEVER make claims not supported by the FDA label context provided
3. NEVER claim the medication is guaranteed to work or has no side effects
4. ALWAYS cite which section of the FDA label your answer comes from
5. If a question is outside the FDA label context, say so and direct to their doctor
6. Keep responses concise — under 80 words for voice delivery
7. Always be warm, empathetic, and non-clinical in tone

RESPONSE FORMAT:
- Speak naturally — this will be converted to voice
- End with a brief check-in: "Does that help?" or "Do you have any other questions?"
- If escalating, express genuine care and urgency appropriately

FDA LABEL CONTEXT WILL BE PROVIDED IN EACH MESSAGE.
"""

# Per-stage focus instructions appended to the base prompt.
_STAGE_CONTEXT = {
    JourneyStage.ENROLLMENT: """
PATIENT JOURNEY STAGE: Enrollment
The patient has just been prescribed this medication and is starting their treatment journey.
FOCUS: Welcome them, explain what the medication does, how to take it correctly, and what to expect in the first week.
TONE: Warm, reassuring, confidence-building. They may be anxious or overwhelmed.
PROACTIVELY COVER: drug purpose, dosing instructions, storage, what to expect early on.
""",
    JourneyStage.ONBOARDING: """
PATIENT JOURNEY STAGE: Onboarding (Days 1-3)
The patient has just started taking the medication.
FOCUS: Normalize early experiences, confirm they are taking it correctly, flag anything concerning early.
TONE: Supportive and practical. Answer the "is this normal?" questions.
PROACTIVELY COVER: common first-day experiences, food/timing interactions, when to call their doctor.
""",
    JourneyStage.TREATMENT_INITIATION: """
PATIENT JOURNEY STAGE: Treatment Initiation (Days 4-14)
The patient is building the habit and may be experiencing early side effects.
FOCUS: Reinforce adherence, normalize mild side effects, monitor for anything that needs escalation.
TONE: Encouraging. Help them through the adjustment period.
PROACTIVELY COVER: missed dose guidance, managing mild side effects, importance of staying on schedule.
""",
    JourneyStage.ADHERENCE: """
PATIENT JOURNEY STAGE: Ongoing Adherence
The patient is in long-term treatment.
FOCUS: Sustain adherence, address fatigue or doubts about continuing, support refill planning.
TONE: Steady, motivating. Acknowledge the effort of long-term treatment.
PROACTIVELY COVER: refill timing, lifestyle adjustments, recognizing when treatment is working.
""",
    JourneyStage.SIDE_EFFECT_MONITORING: """
PATIENT JOURNEY STAGE: Side Effect Monitoring
The patient has reported side effects that require active tracking.
FOCUS: Carefully assess reported symptoms against FDA label. Escalate high-severity symptoms immediately.
TONE: Attentive and calm. Do not minimize concerns.
PROACTIVELY COVER: symptom severity, duration, whether to contact their care team.
""",
}

IMAGE_ANALYSIS_PROMPT = """You are Helix, a patient support AI analyzing a patient-submitted image.

RULES:
1. Describe only what you observe — do not diagnose
2. If the image shows a potential adverse reaction (rash, swelling, skin change, unusual appearance), flag it clearly for the care team
3. If the image shows a medication bottle or packaging, confirm the medication name and dosage if visible
4. Reference the FDA label context if it describes relevant visual symptoms
5. NEVER tell the patient what to do medically — only describe and recommend they consult their care team
6. Keep response under 80 words for voice delivery
7. Be warm and calm — patients may be anxious when sharing a photo

FDA LABEL CONTEXT WILL BE PROVIDED IF AVAILABLE.
"""


def _build_system_prompt(stage: JourneyStage) -> str:
    return _BASE_SYSTEM_PROMPT + _STAGE_CONTEXT.get(stage, "")


class LLMService:

    def __init__(self):
        self.conversation_history: dict[str, list] = {}

    def get_response(
        self,
        session_id: str,
        patient_text: str,
        drug_name: str,
        patient_context: dict,
    ) -> Tuple[str, List[ComplianceCitation], float]:
        """
        Get a compliance-grounded response from Claude.
        Returns (response_text, citations, latency_ms)
        """
        start = time.time()

        stage = JourneyStage(patient_context.get("journey_stage", JourneyStage.ADHERENCE))
        system_prompt = _build_system_prompt(stage)

        rag_results = rag_service.retrieve_context(drug_name, patient_text)

        fda_context = ""
        if rag_results:
            fda_context = "\n\nFDA LABEL CONTEXT (answer ONLY from this):\n"
            for i, (passage, section) in enumerate(rag_results):
                fda_context += f"\n[Source {i+1} - {section}]:\n{passage}\n"
        else:
            fda_context = f"\n\nNote: No FDA label indexed for {drug_name}. Respond generally but flag this limitation."

        patient_info = (
            f"Patient context: Taking {drug_name} ({patient_context.get('dose_schedule', 'unknown dose')}), "
            f"Day {patient_context.get('days_on_treatment', '?')} of treatment. "
            f"Journey stage: {stage.value}.\n\n"
            f"Patient says: {patient_text}"
        )

        full_user_message = patient_info + fda_context

        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append({
            "role": "user",
            "content": full_user_message
        })

        # System prompt is identical for every turn in this session (same stage).
        # Marking it with cache_control lets Anthropic reuse the cached KV state
        # on turn 2+, cutting both latency and token cost on the cached portion.
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=self.conversation_history[session_id],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        response_text = response.content[0].text
        latency_ms = (time.time() - start) * 1000

        self.conversation_history[session_id].append({
            "role": "assistant",
            "content": response_text
        })

        citations = [
            ComplianceCitation(
                claim=response_text[:100] + "...",
                source_section=section,
                source_text=passage[:200] + "..."
            )
            for passage, section in rag_results
        ]

        cache_read    = getattr(response.usage, "cache_read_input_tokens", 0)
        cache_created = getattr(response.usage, "cache_creation_input_tokens", 0)
        logger.info(
            f"[{session_id}] LLM latency: {latency_ms:.0f}ms  stage: {stage.value}"
            f"  cache_read: {cache_read}  cache_created: {cache_created}"
        )
        return response_text, citations, latency_ms

    def analyze_image(
        self,
        session_id: str,
        image_b64: str,
        media_type: str,
        drug_name: str,
        patient_context: Optional[PatientContext] = None,
    ) -> Tuple[str, List[ComplianceCitation], float]:
        """
        Analyze a patient-submitted image against FDA label context.
        Returns (analysis_text, citations, latency_ms)
        """
        start = time.time()

        # Pull skin/reaction passages to ground visual assessment
        rag_results = rag_service.retrieve_context(
            drug_name, "skin rash adverse reaction appearance swelling"
        )

        fda_context = ""
        if rag_results:
            fda_context = "\n\nFDA LABEL CONTEXT (reference if image shows relevant symptoms):\n"
            for passage, section in rag_results:
                fda_context += f"\n[{section}]:\n{passage}\n"

        context_str = ""
        if patient_context:
            context_str = (
                f"Patient is taking {drug_name} ({patient_context.dose_schedule}), "
                f"day {patient_context.days_on_treatment} of treatment. "
                f"Journey stage: {patient_context.journey_stage.value}.\n\n"
            )

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            system=[{"type": "text", "text": IMAGE_ANALYSIS_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": context_str + "Please analyze this image." + fda_context,
                    },
                ],
            }],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        response_text = response.content[0].text
        latency_ms = (time.time() - start) * 1000

        citations = [
            ComplianceCitation(
                claim=response_text[:100] + "...",
                source_section=section,
                source_text=passage[:200] + "..."
            )
            for passage, section in rag_results
        ]

        logger.info(f"[{session_id}] Image analysis latency: {latency_ms:.0f}ms")
        return response_text, citations, latency_ms

    def clear_session(self, session_id: str):
        self.conversation_history.pop(session_id, None)


llm_service = LLMService()

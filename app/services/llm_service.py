import anthropic
import base64
import logging
import time
from typing import Optional, Tuple, List
from app.core.config import settings
from app.services.rag_service import rag_service
from app.services.guardrail_service import guardrail_service
from app.models.session import AppointmentRequest, ComplianceCitation, CareTeamFlag, EscalationFlag, JourneyStage, PatientContext

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

_BASE_SYSTEM_PROMPT = """You are Helix, a compassionate patient support AI for people managing chronic conditions.

YOUR ROLE:
- Help patients understand their treatment, stay adherent, and feel supported
- Answer questions about their medication using ONLY information from the FDA-approved label
- Provide empathetic, clear, plain-language responses

TOOLS YOU HAVE:
- search_fda_label: ALWAYS call this before answering any clinical question (dosing, side effects, warnings, interactions, storage, missed doses). Do not answer from memory.
- flag_for_care_team: Call this when the patient reports symptoms needing clinical attention, asks something outside your scope, or needs human follow-up.

STRICT RULES — NEVER VIOLATE:
1. NEVER suggest changing, increasing, or decreasing a dose
2. NEVER make claims not supported by FDA label results from search_fda_label
3. NEVER claim the medication is guaranteed to work or has no side effects
4. ALWAYS cite which section of the FDA label your answer comes from
5. If search_fda_label returns no results, say so and direct to their doctor
6. Keep responses concise — under 80 words for voice delivery
7. Always be warm, empathetic, and non-clinical in tone

RESPONSE FORMAT:
- Speak naturally — this will be converted to voice
- End with a brief check-in: "Does that help?" or "Do you have any other questions?"
- If escalating, express genuine care and urgency appropriately
"""

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

# Tools Claude can call during the agentic loop
AGENT_TOOLS = [
    {
        "name": "search_fda_label",
        "description": (
            "Search the FDA-approved label for a drug. Call this before answering any clinical question "
            "about dosing, side effects, warnings, drug interactions, contraindications, missed doses, or storage. "
            "Always search before answering — do not rely on training data for clinical facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "Name of the drug to search (e.g. 'lisinopril')"
                },
                "query": {
                    "type": "string",
                    "description": "What to look up in the label (e.g. 'missed dose instructions', 'common side effects', 'drug interactions')"
                }
            },
            "required": ["drug_name", "query"]
        }
    },
    {
        "name": "flag_for_care_team",
        "description": (
            "Flag this conversation for the patient's care team. Call this when the patient reports symptoms "
            "that need clinical attention, when their question is outside your scope, or when a human should follow up. "
            "You can still respond to the patient after flagging."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you are flagging this for the care team"
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "high = immediate attention needed, medium = follow up soon, low = informational"
                }
            },
            "required": ["reason", "severity"]
        }
    },
    {
        "name": "check_drug_interactions",
        "description": (
            "Check whether the patient's current medication may interact with another drug they mention. "
            "Searches the FDA label for interaction warnings. Call this whenever the patient mentions "
            "taking another medication, supplement, or asks about combining drugs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "other_drug": {
                    "type": "string",
                    "description": "The other drug or supplement to check for interactions (e.g. 'ibuprofen', 'potassium supplements')"
                }
            },
            "required": ["other_drug"]
        }
    },
    {
        "name": "get_refill_reminder",
        "description": (
            "Check whether the patient needs to refill their prescription soon, based on how long "
            "they have been on treatment. Call this when the patient asks about refills, mentions "
            "running low, or when it seems like they may be near the end of their supply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "schedule_appointment",
        "description": (
            "Record a request to schedule a follow-up appointment with the patient's care team. "
            "Call this when the patient needs to be seen in person, when symptoms need clinical review, "
            "or when ongoing monitoring is warranted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the appointment is needed"
                },
                "preferred_timing": {
                    "type": "string",
                    "enum": ["urgent", "this_week", "routine"],
                    "description": "urgent = within 24h, this_week = within 7 days, routine = next available"
                }
            },
            "required": ["reason", "preferred_timing"]
        }
    }
]


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
    ) -> Tuple[str, List[ComplianceCitation], float, List[CareTeamFlag], List[AppointmentRequest]]:
        """
        Agentic response loop. Claude decides which tools to call.
        Returns (response_text, citations, latency_ms, care_team_flags, appointment_requests).
        """
        start = time.time()

        stage = JourneyStage(patient_context.get("journey_stage", JourneyStage.ADHERENCE))
        system_prompt = _build_system_prompt(stage)

        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        patient_info = (
            f"Patient context: Taking {drug_name} ({patient_context.get('dose_schedule', 'unknown dose')}), "
            f"Day {patient_context.get('days_on_treatment', '?')} of treatment. "
            f"Journey stage: {stage.value}.\n\n"
            f"Patient says: {patient_text}"
        )
        self.conversation_history[session_id].append({
            "role": "user",
            "content": patient_info
        })

        citations: List[ComplianceCitation] = []
        care_team_flags: List[CareTeamFlag] = []
        appointment_requests: List[AppointmentRequest] = []
        response_text = ""

        # Agentic loop: run until Claude stops calling tools
        while True:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                tools=AGENT_TOOLS,
                messages=self.conversation_history[session_id],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )

            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": response.content
            })

            if response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if hasattr(b, "text")]
                response_text = text_blocks[-1].text if text_blocks else ""
                break

            # Claude called one or more tools — execute them and loop
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "search_fda_label":
                    passages = rag_service.retrieve_context(
                        block.input["drug_name"],
                        block.input["query"]
                    )
                    result = (
                        "\n\n".join(f"[{section}]:\n{passage}" for passage, section in passages)
                        or "No relevant passages found in the FDA label for this query."
                    )
                    citations.extend([
                        ComplianceCitation(
                            claim=f"Search: {block.input['query']}",
                            source_section=section,
                            source_text=passage[:200] + "..."
                        )
                        for passage, section in passages
                    ])
                    logger.info(f"[{session_id}] Tool: search_fda_label({block.input['drug_name']!r}, {block.input['query']!r}) → {len(passages)} passages")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

                elif block.name == "flag_for_care_team":
                    care_team_flags.append(CareTeamFlag(
                        reason=block.input["reason"],
                        severity=block.input["severity"]
                    ))
                    logger.info(f"[{session_id}] Tool: flag_for_care_team severity={block.input['severity']!r} reason={block.input['reason']!r}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Care team has been notified."
                    })

                elif block.name == "check_drug_interactions":
                    other_drug = block.input["other_drug"]
                    query = f"drug interactions {other_drug} contraindications concomitant use"
                    passages = rag_service.retrieve_context(drug_name, query)
                    result = (
                        "\n\n".join(f"[{section}]:\n{passage}" for passage, section in passages)
                        or f"No specific interaction information found in the FDA label for {other_drug}. Advise the patient to consult their pharmacist or doctor."
                    )
                    logger.info(f"[{session_id}] Tool: check_drug_interactions({other_drug!r}) → {len(passages)} passages")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

                elif block.name == "get_refill_reminder":
                    days = patient_context.get("days_on_treatment", 0)
                    supply_days = 30
                    days_remaining = max(0, supply_days - days)
                    if days_remaining <= 7:
                        result = f"Refill needed soon — approximately {days_remaining} days of medication remaining based on a standard 30-day supply. Patient should contact their pharmacy now."
                    elif days_remaining <= 14:
                        result = f"About {days_remaining} days of medication remaining. Patient should contact their pharmacy within the next week."
                    else:
                        result = f"Approximately {days_remaining} days of medication remaining. No immediate refill action needed."
                    logger.info(f"[{session_id}] Tool: get_refill_reminder → {days_remaining} days remaining")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

                elif block.name == "schedule_appointment":
                    appointment_requests.append(AppointmentRequest(
                        reason=block.input["reason"],
                        preferred_timing=block.input["preferred_timing"]
                    ))
                    logger.info(f"[{session_id}] Tool: schedule_appointment timing={block.input['preferred_timing']!r} reason={block.input['reason']!r}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Appointment request recorded ({block.input['preferred_timing']}). The care team will reach out to schedule."
                    })

            self.conversation_history[session_id].append({
                "role": "user",
                "content": tool_results
            })

        latency_ms = (time.time() - start) * 1000
        cache_read    = getattr(response.usage, "cache_read_input_tokens", 0)
        cache_created = getattr(response.usage, "cache_creation_input_tokens", 0)
        logger.info(
            f"[{session_id}] Agent latency: {latency_ms:.0f}ms  stage: {stage.value}"
            f"  cache_read: {cache_read}  cache_created: {cache_created}"
        )
        return response_text, citations, latency_ms, care_team_flags, appointment_requests

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
        Single-turn — image analysis doesn't benefit from an agentic loop.
        Returns (analysis_text, citations, latency_ms)
        """
        start = time.time()

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

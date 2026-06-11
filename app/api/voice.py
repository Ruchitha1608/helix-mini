import asyncio
import base64
import json
import logging
import time
from fastapi import APIRouter, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from app.core.config import settings
from app.services.llm_service import llm_service
from app.services.tts_service import tts_service
from app.services.guardrail_service import guardrail_service
from app.services.session_store import session_store
from app.models.session import JourneyStage, PatientContext, TurnMetrics, infer_stage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/{session_id}")
async def voice_ws(websocket: WebSocket, session_id: str):
    """
    Main WebSocket endpoint for real-time voice conversation.

    Client sends:
    - JSON config message first: {"type": "config", "patient": {...}}
    - Then raw audio bytes (PCM 16kHz mono)

    Server sends:
    - JSON status messages: {"type": "transcript", "text": "..."}
    - Raw audio bytes (TTS response)
    - JSON summary on disconnect: {"type": "summary", ...}
    """
    await websocket.accept()
    logger.info(f"[{session_id}] WebSocket connected")

    patient_context = None
    deepgram_client = DeepgramClient(settings.DEEPGRAM_API_KEY)
    dg_connection = None
    transcript_queue: asyncio.Queue = asyncio.Queue()
    transcript_task = None

    try:
        # Step 1: receive config
        config_raw = await websocket.receive_text()
        config = json.loads(config_raw)

        if config.get("type") == "config":
            p = config["patient"]
            days = p.get("days_on_treatment", 0)

            raw_stage = p.get("journey_stage")
            try:
                stage = JourneyStage(raw_stage) if raw_stage else infer_stage(days)
            except ValueError:
                stage = infer_stage(days)

            patient_context = PatientContext(
                drug_name=p.get("drug_name", "unknown"),
                dose_schedule=p.get("dose_schedule", "as prescribed"),
                days_on_treatment=days,
                known_conditions=p.get("known_conditions", []),
                journey_stage=stage,
            )
            session_store.create_session(session_id, patient_context)
            await websocket.send_text(json.dumps({
                "type": "ready",
                "message": f"Helix ready — {patient_context.drug_name} / stage: {stage.value}",
                "journey_stage": stage.value,
            }))

        # Step 2: set up Deepgram async live transcription
        # asynclive is required so start/send/finish are awaitable and callbacks can be async
        dg_connection = deepgram_client.listen.asynclive.v("1")

        async def on_transcript(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if sentence and result.is_final:
                await transcript_queue.put(sentence)

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)

        options = LiveOptions(
            model=settings.DEEPGRAM_MODEL,
            language="en-US",
            smart_format=True,
            interim_results=False,
            utterance_end_ms=1000,
            punctuate=True,
        )
        await dg_connection.start(options)

        # Step 3: process transcripts and respond — runs concurrently with audio streaming
        async def process_transcripts():
            while True:
                patient_text = await transcript_queue.get()
                if patient_text == "__END__":
                    break

                logger.info(f"[{session_id}] Patient: {patient_text}")
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": patient_text,
                }))

                turn_start = time.time()
                llm_latency_ms = 0.0

                should_escalate, escalation_flags = guardrail_service.check_patient_input(patient_text)

                if should_escalate:
                    response_text = guardrail_service.build_escalation_response(escalation_flags)
                    guardrail_fired = True
                    citations = []
                else:
                    response_text, citations, llm_latency_ms, care_team_flags, appointment_requests = llm_service.get_response(
                        session_id=session_id,
                        patient_text=patient_text,
                        drug_name=patient_context.drug_name,
                        patient_context=patient_context.model_dump(),
                    )
                    is_violation, violation_reason = guardrail_service.check_agent_response(response_text)
                    if is_violation:
                        logger.warning(f"[{session_id}] Output guardrail fired: {violation_reason}")
                        response_text = guardrail_service.build_safe_response(violation_reason)
                        guardrail_fired = True
                    else:
                        guardrail_fired = False

                await websocket.send_text(json.dumps({
                    "type": "response",
                    "text": response_text,
                    "escalate": should_escalate,
                }))

                audio_bytes, tts_latency_ms = await tts_service.synthesize(response_text)
                await websocket.send_bytes(audio_bytes)

                total_latency = (time.time() - turn_start) * 1000
                metrics = TurnMetrics(
                    stt_latency_ms=0,
                    llm_latency_ms=llm_latency_ms,
                    tts_latency_ms=tts_latency_ms,
                    total_latency_ms=total_latency,
                    guardrail_fired=guardrail_fired,
                )
                session_store.add_turn(
                    session_id=session_id,
                    patient_text=patient_text,
                    agent_response=response_text,
                    metrics=metrics,
                    citations=citations,
                    escalation_flags=escalation_flags,
                    guardrail_fired=guardrail_fired,
                    care_team_flags=care_team_flags,
                    appointment_requests=appointment_requests,
                )
                logger.info(f"[{session_id}] Turn latency: {total_latency:.0f}ms")

        transcript_task = asyncio.create_task(process_transcripts())

        # Step 4: stream client audio to Deepgram
        while True:
            data = await websocket.receive_bytes()
            await dg_connection.send(data)

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")

    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}", exc_info=True)

    finally:
        # Signal transcript processor to exit and wait for it to drain
        await transcript_queue.put("__END__")
        if transcript_task is not None:
            try:
                await asyncio.wait_for(transcript_task, timeout=5.0)
            except asyncio.TimeoutError:
                transcript_task.cancel()

        if dg_connection:
            await dg_connection.finish()

        summary = session_store.end_session(session_id)
        if summary:
            try:
                await websocket.send_text(json.dumps({
                    "type": "summary",
                    "data": summary.model_dump(mode="json"),
                }))
            except Exception:
                pass

        llm_service.clear_session(session_id)
        logger.info(f"[{session_id}] Session ended")


@router.post("/analyze-image/{session_id}")
async def analyze_image(
    session_id: str,
    file: UploadFile = File(...),
    drug_name: str = Query(default="unknown"),
):
    """
    Analyze a patient-submitted image (rash, medication bottle, etc.).

    If the session_id exists, drug context and patient profile are pulled automatically.
    Otherwise, pass drug_name as a query parameter.

    Returns a guardrail-checked, FDA-grounded visual assessment.
    """
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    media_type = file.content_type or "image/jpeg"

    patient_context = None
    active = session_store.sessions.get(session_id)
    if active:
        patient_context = active["patient_context"]
        drug_name = patient_context.drug_name

    analysis, citations, latency_ms = llm_service.analyze_image(
        session_id=session_id,
        image_b64=image_b64,
        media_type=media_type,
        drug_name=drug_name,
        patient_context=patient_context,
    )

    is_violation, violation_reason = guardrail_service.check_agent_response(analysis)
    if is_violation:
        logger.warning(f"[{session_id}] Image analysis guardrail fired: {violation_reason}")
        analysis = guardrail_service.build_safe_response(violation_reason)

    return {
        "session_id": session_id,
        "drug": drug_name,
        "journey_stage": patient_context.journey_stage.value if patient_context else None,
        "analysis": analysis,
        "citations": [c.model_dump() for c in citations],
        "guardrail_fired": is_violation,
        "latency_ms": round(latency_ms),
    }

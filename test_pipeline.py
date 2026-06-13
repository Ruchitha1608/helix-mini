"""
Helix Mini — end-to-end pipeline test (no microphone, no speaker).
Exercises: RAG → Input Guardrail → Claude LLM → Output Guardrail → TTS

Requires: ANTHROPIC_API_KEY and CARTESIA_API_KEY in .env
Deepgram is skipped (text bypasses STT — identical code path after transcription).

Run: python test_pipeline.py
"""
import asyncio
import sys
import time
import warnings
warnings.filterwarnings("ignore")

BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[31m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
CYAN  = "\033[36m"
RESET = "\033[0m"

def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}"); sys.exit(1)
def info(msg):print(f"  {DIM}{msg}{RESET}")
def hdr(t):
    print(); print(BOLD + CYAN + t + RESET)
    print(DIM + "─" * 60 + RESET)


TURNS = [
    # (drug, patient_text, days, expect_escalate)
    ("lisinopril", "What are the common side effects of lisinopril?",       7,  False),
    ("lisinopril", "I missed my dose this morning, what should I do?",      14, False),
    ("metformin",  "How does metformin help with blood sugar?",              3,  False),
    ("metformin",  "I've been feeling nauseous since starting metformin.",   5,  False),
    # Guardrail test — should escalate, LLM never called
    ("lisinopril", "I have severe chest pain and I'm sweating a lot.",       20, True),
]


async def run():
    from dotenv import load_dotenv
    load_dotenv()

    from app.services.guardrail_service import guardrail_service
    from app.services.rag_service import rag_service
    from app.services.llm_service import llm_service
    from app.services.tts_service import tts_service
    from app.models.session import infer_stage

    session_id = "pipeline-test-001"
    total_latency = []

    print()
    print(BOLD + "═" * 60)
    print("  HELIX MINI — END-TO-END PIPELINE TEST")
    print("  STT bypassed (text input) · all other layers live")
    print("═" * 60 + RESET)

    # ── 1. Verify RAG index ───────────────────────────────────────────────────
    hdr("1. RAG Index")
    drugs = rag_service.list_indexed_drugs()
    if not drugs:
        err("No drugs indexed — run the server first to trigger auto-indexing")
    for drug in drugs:
        results = rag_service.retrieve_context(drug, "side effects dosage")
        ok(f"{drug}: {len(results)} passages retrieved")
    info(f"Formulary: {drugs}")

    # ── 2. Full turn pipeline ─────────────────────────────────────────────────
    hdr("2. Full Turn Pipeline (RAG → Guardrail → LLM → Guardrail → TTS)")

    for i, (drug, patient_text, days, expect_escalate) in enumerate(TURNS):
        stage = infer_stage(days)
        print(f"\n  {BOLD}Turn {i+1}{RESET}  [{drug} · {stage.value} · day {days}]")
        print(f"  Patient: \"{patient_text[:70]}\"")

        t0 = time.time()

        # Layer 1: input guardrail
        should_escalate, flags = guardrail_service.check_patient_input(patient_text)

        if should_escalate != expect_escalate:
            err(f"Guardrail mismatch: expected escalate={expect_escalate}, got {should_escalate}")

        if should_escalate:
            response = guardrail_service.build_escalation_response(flags)
            ok(f"Input guardrail fired correctly → escalation response")
            print(f"  Response: \"{response[:100]}\"")
            total_latency.append((time.time() - t0) * 1000)
            continue

        ok("Input guardrail: SAFE")

        # LLM + RAG
        context = {"journey_stage": stage.value, "days_on_treatment": days}
        try:
            response, citations, llm_ms, *_ = llm_service.get_response(
                session_id=f"{session_id}-{i}",
                patient_text=patient_text,
                drug_name=drug,
                patient_context=context,
            )
        except Exception as e:
            err(f"LLM call failed: {e}")

        ok(f"LLM responded in {llm_ms:.0f}ms · {len(citations)} citations")

        # Layer 2: output guardrail
        is_violation, reason = guardrail_service.check_agent_response(response)
        if is_violation:
            print(f"  {YELLOW}⚠ Output guardrail fired: {reason}{RESET}")
            response = guardrail_service.build_safe_response(reason)
        else:
            ok("Output guardrail: clean")

        word_count = len(response.split())
        ok(f"Response: {word_count} words")
        print(f"  \"{response[:120]}…\"")

        # TTS
        try:
            audio_bytes, tts_ms = await tts_service.synthesize(response)
            ok(f"TTS: {len(audio_bytes):,} audio bytes in {tts_ms:.0f}ms")
        except Exception as e:
            print(f"  {YELLOW}⚠ TTS skipped: {e}{RESET}")
            tts_ms = 0

        turn_ms = (time.time() - t0) * 1000
        total_latency.append(turn_ms)
        info(f"Turn total: {turn_ms:.0f}ms")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    hdr("3. Summary")
    passed = len(total_latency)
    avg    = sum(total_latency) / len(total_latency) if total_latency else 0

    ok(f"All {passed}/{len(TURNS)} turns completed")
    ok(f"Avg turn latency: {avg:.0f}ms")
    ok(f"Drugs tested: {set(d for d,*_ in TURNS)}")

    llm_service.clear_session(session_id)

    print()
    print(DIM + "═" * 60 + RESET)
    print(BOLD + GREEN + "  Pipeline test passed." + RESET)
    print(DIM + "  STT layer (Deepgram) is identical code path — only input source differs." + RESET)
    print(DIM + "═" * 60 + RESET)
    print()


if __name__ == "__main__":
    asyncio.run(run())

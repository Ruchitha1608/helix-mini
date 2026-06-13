"""
Helix Mini — live demo script.
Runs entirely offline: zero API keys, zero network calls.

Shows:
  1. Two-layer compliance guardrails (input + output)
  2. Negation detection
  3. Patient journey state machine
  4. RAG retrieval over the lisinopril FDA label
"""
import sys
import time

# ── formatting helpers ────────────────────────────────────────────────────────

BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[31m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
CYAN  = "\033[36m"
RESET = "\033[0m"

def hr(char="─", width=64):
    print(DIM + char * width + RESET)

def section(title):
    print()
    print(BOLD + CYAN + title + RESET)
    hr()

def pause(seconds=0.6):
    time.sleep(seconds)

# ── 1. Input guardrail ────────────────────────────────────────────────────────

def demo_input_guardrail():
    section("LAYER 1 — Patient Input Guardrail (pre-LLM)")

    from app.services.guardrail_service import guardrail_service

    cases = [
        # (patient says,                              expect escalation)
        ("I have been feeling a bit dizzy today",     False),
        ("Can I take lisinopril with food?",          False),
        ("I missed my dose this morning",             False),
        ("I can't breathe properly",                  True),
        ("I have severe chest pain",                  True),
        ("My face is swollen and my throat is swollen too", True),
        ("I fainted when I stood up",                 True),
        ("My heart has been beating irregularly",     True),
    ]

    for text, expect_escalate in cases:
        pause(0.15)
        should_escalate, flags = guardrail_service.check_patient_input(text)

        if should_escalate:
            badge = f"{RED}{BOLD}▲ ESCALATE{RESET}"
        else:
            badge = f"{GREEN}✓ SAFE    {RESET}"

        marker = "" if should_escalate == expect_escalate else f"  {RED}← BUG{RESET}"
        label  = f'"{text[:55]}"'
        print(f"  {badge}  {label}{marker}")

    print()
    print(f"  {DIM}High-severity match bypasses LLM entirely — response is immediate.{RESET}")

# ── 2. Negation detection ─────────────────────────────────────────────────────

def demo_negation():
    section("NEGATION DETECTION — false positive prevention")

    from app.services.guardrail_service import guardrail_service

    cases = [
        ("I don't have any chest pain",               False),
        ("No shortness of breath to report",          False),
        ("I'm not having trouble breathing",          False),
        ("She denies any chest tightness",            False),
        ("I have chest pain",                         True),   # no negation → escalate
        ("I have shortness of breath",                True),
    ]

    for text, expect_escalate in cases:
        pause(0.15)
        should_escalate, _ = guardrail_service.check_patient_input(text)
        ok = should_escalate == expect_escalate

        if should_escalate:
            badge = f"{RED}{BOLD}▲ ESCALATE{RESET}"
        else:
            badge = f"{GREEN}✓ PASS    {RESET}"

        marker = f"  {GREEN}✓{RESET}" if ok else f"  {RED}← BUG{RESET}"
        print(f"  {badge}  \"{text}\"{marker}")

    print()
    print(f"  {DIM}Window of 25 chars before each match checked for negation prefixes.{RESET}")
    print(f"  {DIM}Without this: 'I don't have chest pain' → false escalation.{RESET}")

# ── 3. Output guardrail ───────────────────────────────────────────────────────

def demo_output_guardrail():
    section("LAYER 2 — Agent Output Guardrail (post-LLM)")

    from app.services.guardrail_service import guardrail_service

    cases = [
        # (agent says,                                               expect violation)
        ("Dizziness is listed in the FDA label as a common side effect.", False),
        ("According to Section 5.2, you should take this with water.",    False),
        ("You should increase your dose if symptoms persist.",             True),
        ("This medication will cure your hypertension completely.",        True),
        ("Lisinopril has no side effects at normal doses.",               True),
        ("You can double your dose if you miss one.",                     True),
    ]

    for text, expect_violation in cases:
        pause(0.15)
        is_violation, reason = guardrail_service.check_agent_response(text)

        if is_violation:
            badge = f"{RED}{BOLD}✗ BLOCKED {RESET}"
            detail = f"  {DIM}({reason}){RESET}"
        else:
            badge = f"{GREEN}✓ ALLOWED {RESET}"
            detail = ""

        ok = is_violation == expect_violation
        marker = "" if ok else f"  {RED}← BUG{RESET}"
        label = f'"{text[:60]}"'
        print(f"  {badge}  {label}{detail}{marker}")

    print()
    print(f"  {DIM}Violations replaced with safe fallback before TTS. Patient never hears the original.{RESET}")

# ── 4. Journey state machine ──────────────────────────────────────────────────

def demo_journey_stages():
    section("PATIENT JOURNEY STATE MACHINE")

    from app.models.session import infer_stage

    stage_desc = {
        "enrollment":            "Welcome, explain drug, build confidence",
        "onboarding":            "Normalize early experiences, 'is this normal?'",
        "treatment_initiation":  "Reinforce habit, monitor early side effects",
        "adherence":             "Sustain long-term adherence, refill planning",
    }

    cases = [
        (0,  "enrollment"),
        (1,  "onboarding"),
        (3,  "onboarding"),
        (4,  "treatment_initiation"),
        (14, "treatment_initiation"),
        (15, "adherence"),
        (90, "adherence"),
    ]

    print(f"  {'Days':>5}   {'Stage':<25}  Focus")
    hr("·")

    for days, expected in cases:
        pause(0.1)
        stage = infer_stage(days)
        ok    = stage.value == expected
        icon  = GREEN + "✓" + RESET if ok else RED + "✗" + RESET
        desc  = stage_desc.get(stage.value, "")
        print(f"  {icon}  {days:>4}d   {BOLD}{stage.value:<25}{RESET}  {DIM}{desc}{RESET}")

    print()
    print(f"  {DIM}Stage is auto-inferred from days_on_treatment or passed explicitly in WebSocket config.{RESET}")
    print(f"  {DIM}Each stage loads a different system prompt block, tone, and escalation sensitivity.{RESET}")

# ── 5. RAG retrieval ──────────────────────────────────────────────────────────

def demo_rag():
    section("RAG — FDA LABEL RETRIEVAL (lisinopril)")

    try:
        from app.services.rag_service import rag_service
    except Exception as e:
        print(f"  {YELLOW}RAG service unavailable: {e}{RESET}")
        return

    queries = [
        ("What are the side effects?",   "side_effects"),
        ("Can I take it with alcohol?",  "interactions"),
        ("What is the normal dose?",     "dosage"),
    ]

    for query, _ in queries:
        pause(0.2)
        t0 = time.time()
        try:
            results = rag_service.retrieve_context("lisinopril", query)
            ms = (time.time() - t0) * 1000
            if results:
                passage, section_name = results[0]
                snippet = passage[:120].replace("\n", " ")
                print(f"  {GREEN}✓{RESET}  {BOLD}\"{query}\"{RESET}")
                print(f"     {DIM}[{section_name}] {snippet}…{RESET}")
                print(f"     {DIM}retrieved {len(results)} passages in {ms:.0f}ms{RESET}")
            else:
                print(f"  {YELLOW}?{RESET}  \"{query}\"  {DIM}(no results){RESET}")
        except Exception as e:
            print(f"  {YELLOW}skip{RESET}  \"{query}\"  {DIM}({e}){RESET}")
        print()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print(BOLD + "═" * 64)
    print("  HELIX MINI — LIVE DEMO")
    print("  Patient Adherence Voice AI  ·  Zero API calls")
    print("═" * 64 + RESET)

    demo_input_guardrail()
    demo_negation()
    demo_output_guardrail()
    demo_journey_stages()
    demo_rag()

    print()
    hr("═")
    print(f"{BOLD}{GREEN}  Demo complete.{RESET}")
    print(f"  {DIM}Run  python -m evals.run_evals        → 33 deterministic tests{RESET}")
    print(f"  {DIM}Run  python -m evals.run_evals --rag  → + RAG retrieval quality{RESET}")
    print(f"  {DIM}Run  python -m evals.run_evals --llm  → + live Claude response tests{RESET}")
    hr("═")
    print()


if __name__ == "__main__":
    main()

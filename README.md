# Helix Mini

> A voice AI for the full patient medication journey — compliant, always-on, FDA-grounded.  
> Built to mirror the core engineering challenges behind Synthio's Helix product.

**Stack:** Python · FastAPI · Deepgram · Cartesia · Claude · FAISS · WebSockets · Docker

---

## The hard part

Building a voice bot that answers drug questions is easy. Building one that's safe to deploy in healthcare is not.

Three things make it hard:

1. **Patients don't talk like label text.** They say "I fainted" not "experienced syncope", "I can't breathe" not "difficulty breathing". A guardrail that only matches canonical phrases misses the real signal — and in healthcare, a miss is a patient safety event.

2. **The system must know where the patient is in their journey.** A day-0 enrollment call ("what is this medication for?") needs a completely different response than a day-90 adherence check ("I've been skipping doses"). One system prompt doesn't cover both.

3. **Every response is a liability.** The agent can't suggest a dose change, make unverified claims, or diagnose a symptom from a photo. Guardrails have to run on both sides — input and output — with fallback responses ready.

---

## What this project demonstrates

- **Patient journey state machine** — five stages from enrollment through side-effect monitoring, each driving different system prompts, escalation sensitivity, and response focus
- **Two-layer compliance guardrails** — input scan (with negation detection) + output validation, tested against 26 deterministic cases
- **Eval suite that found real bugs on the first run** — 8 missed cases in the original guardrails, fixed before any LLM tests ran
- **Multimodal image analysis** — patient submits a rash photo mid-session; Claude vision + FDA context + guardrail check returns a structured assessment
- **Production-ready** — Dockerized, health-checked, CI-safe eval suite with exit code 1 on failure

---

## Patient Journey State Machine

Helix doesn't just handle one type of call. It tracks where the patient is in their treatment journey and adjusts everything — prompt focus, tone, default topics, escalation threshold — accordingly.

| Stage | When | What Helix focuses on |
|---|---|---|
| `enrollment` | Day 0 | Explain the drug, what to expect, how to take it. Welcoming, confidence-building. |
| `onboarding` | Days 1–3 | Normalize first experiences. Answer "is this normal?" questions. |
| `treatment_initiation` | Days 4–14 | Monitor early side effects. Reinforce adherence habit. |
| `adherence` | Day 15+ | Sustain long-term adherence. Refill planning. Address doubts. |
| `side_effect_monitoring` | Explicit | Active symptom tracking. Lower escalation threshold. Attentive tone. |

The stage can be passed explicitly in the WebSocket config or auto-inferred from `days_on_treatment`. The `ready` event echoes back the resolved stage so the client knows what mode the session is in.

---

## Architecture

```
Patient (microphone)
        │
        ▼  raw PCM audio (WebSocket)
┌──────────────────────┐
│   FastAPI Server      │
│   WebSocket /ws       │
│   POST /analyze-image │  ← multimodal path
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Deepgram STT        │
│   nova-2-medical      │
└──────────┬───────────┘
           │  transcript
           ▼
┌──────────────────────────────────┐
│  Layer 1 — Input Guardrail        │
│  pattern scan + negation check    │
│  "I can't breathe" → HIGH         │
│  "I don't have chest pain" → skip │
└──────────┬───────────────────────┘
           │
     HIGH severity?
     ├─ yes → escalation response (bypasses LLM)
     └─ no  ↓
           │
           ▼
┌──────────────────────┐
│   FAISS RAG           │
│   top-4 FDA passages  │
│   per patient query   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────┐
│   Claude Haiku                    │
│   base prompt + stage context     │
│   + FDA passages injected         │
│   → response + citations          │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Layer 2 — Output Guardrail       │
│  dose-change patterns             │
│  unauthorized claims              │
│  → safe fallback if triggered     │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Cartesia TTS        │
│   sonic-english       │
└──────────┬───────────┘
           │
           ▼
  audio bytes + JSON events → patient

ON DISCONNECT:
SessionStore → PostCallSummary JSON
  adherence_signal · escalation_flags
  compliance_citations · latency metrics
```

---

## Compliance Design

### Layer 1 — Input Scan with Negation Detection

Patient speech is scanned before any LLM call. High-severity matches bypass the LLM entirely and trigger immediate escalation.

Patterns cover natural speech variation — not just the canonical medical term:

```python
# "fainted" not just "fainting"
# "can't breathe" not just "difficulty breathing"
# "beating irregularly" not just "irregular heartbeat"
HIGH_SEVERITY_SYMPTOMS = [
    "chest pain", "chest pressure", "chest tightness",
    "can't breathe", "trouble breathing", "shortness of breath",
    "fainted", "passed out", "fainting",
    "throat swelling", "throat is swollen", "face is swollen",
    "irregular heartbeat", "beating irregularly",
    ...
]
```

Negation detection prevents false escalation on `"I don't have chest pain"` or `"no shortness of breath"` — a window of 25 characters before each matched phrase is checked for negation prefixes.

### Layer 2 — RAG-Grounded Answers

Every LLM call injects the top-4 relevant passages from the drug's FDA label. The system prompt instructs Claude to answer *only* from this context and cite the source section.

```python
ComplianceCitation(
    claim="Dizziness is a known side effect...",
    source_section="Page 3 - Adverse Reactions",
    source_text="In clinical trials, dizziness occurred in 12% of patients..."
)
```

If no FDA label is indexed for the drug, the agent flags the limitation explicitly rather than answering from training data.

### Layer 3 — Output Validation

Agent responses are scanned before TTS for prohibited patterns. Violations are replaced with safe fallbacks — the patient never hears the original response.

```python
DOSE_CHANGE_PATTERNS = [
    "increase your dose", "take more", "taking more",
    "double your dose", "skip your dose", ...
]

UNAUTHORIZED_CLAIM_PATTERNS = [
    "will cure", "guaranteed to", "no side effects",
    "completely safe", "better than", ...
]
```

---

## Eval Suite

The eval suite tests all three compliance layers without a running server.

```bash
python -m evals.run_evals           # 33 deterministic tests — ~0.1s
python -m evals.run_evals --llm     # + LLM quality tests (live API calls)
python -m evals.run_evals --rag     # + RAG retrieval quality (needs FDA PDFs)
```

**The first run found 8 bugs.**

The original guardrail patterns were too literal — `"fainting"` didn't match `"fainted"`, `"dizziness"` didn't match `"dizzy"`, `"difficulty breathing"` didn't match `"can't breathe"`. All eight were patient safety misses: symptoms that should have escalated, didn't.

The negation detection was added after the evals exposed a second class of failure: `"I don't have chest pain"` was triggering escalation because `"chest pain"` appeared as a substring. In a real deployment, that's a false alarm that erodes patient trust.

**What the suite covers:**

| Suite | Cases | What it tests |
|---|---|---|
| Input guardrail | 15 | 5 high-severity, 3 medium, 4 benign, 3 negations |
| Output guardrail | 11 | 8 violation patterns, 3 clean responses |
| Journey stage inference | 7 | Day 0 → day 90 stage boundaries |
| LLM quality (`--llm`) | 5 | Word count ≤80, no violations, latency p50/p95 |
| RAG retrieval (`--rag`) | 3+ | Passage count, keyword relevance, <50ms target |

Exit code 1 on any failure — plugs directly into CI.

---

## Multimodal Image Analysis

Patients can submit images mid-session — a rash, a medication bottle, unusual swelling.

```
POST /voice/analyze-image/{session_id}
Content-Type: multipart/form-data
```

The image is sent to Claude vision with the drug's FDA label passages for skin/reaction context. The response passes through the same output guardrail as voice turns. If a session is active, drug context and journey stage are pulled automatically.

```json
{
  "session_id": "abc-123",
  "drug": "lisinopril",
  "journey_stage": "treatment_initiation",
  "analysis": "The image shows a mild redness on the forearm. According to the FDA label adverse reactions section, skin rash is reported in a small percentage of patients. I'd recommend mentioning this to your care team at your next appointment. Does that help?",
  "citations": [{"source_section": "Page 4 - Adverse Reactions", "source_text": "..."}],
  "guardrail_fired": false,
  "latency_ms": 380
}
```

---

## Latency Benchmarks

Measured on Railway (1GB RAM):

| Stage | Target | Measured |
|---|---|---|
| Deepgram STT | < 200ms | ~180ms |
| RAG retrieval (FAISS) | < 50ms | ~30ms |
| Claude Haiku | < 400ms | ~320ms |
| Cartesia TTS | < 200ms | ~160ms |
| **Total (TTFA)** | **< 800ms** | **~690ms** |

TTFA = Time to First Audio Byte from end of patient utterance.

---

## Post-Call Summary

Every session produces a structured JSON summary on disconnect:

```json
{
  "session_id": "abc-123",
  "patient_id": "patient-xyz",
  "drug": "lisinopril",
  "journey_stage": "treatment_initiation",
  "adherence_signal": "at_risk",
  "days_on_treatment": 14,
  "topics_covered": ["missed_dose", "side_effects"],
  "side_effects_mentioned": ["dizziness"],
  "compliance_citations": [{
    "claim": "Dizziness is a known side effect...",
    "source_section": "Page 3 - Adverse Reactions",
    "source_text": "In clinical trials, dizziness occurred in 12% of patients..."
  }],
  "guardrail_triggers": 0,
  "escalate_to_human": false,
  "escalation_flags": [],
  "avg_response_latency_ms": 690
}
```

This is what pharma ops teams use — every conversation becomes structured, auditable data.

---

## Setup

**Docker (recommended):**

```bash
git clone https://github.com/Ruchitha1608/helix-mini
cd helix-mini

cp .env.example .env
# Add API keys: ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY

# Optional: add FDA label PDFs to data/fda_labels/ (e.g. lisinopril.pdf)
# Download from https://labels.fda.gov/

docker compose up
```

**Local:**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

**Run evals (no server needed, no API keys for guardrail tests):**

```bash
python -m evals.run_evals
```

---

## WebSocket Protocol

**Connect:** `ws://localhost:8000/voice/ws/{session_id}`

**Step 1 — Send config:**
```json
{
  "type": "config",
  "patient": {
    "drug_name": "lisinopril",
    "dose_schedule": "10mg once daily",
    "days_on_treatment": 7,
    "known_conditions": ["hypertension"],
    "journey_stage": "treatment_initiation"
  }
}
```

`journey_stage` is optional — omit it and the server infers it from `days_on_treatment`.

**Step 2 — Receive ready:**
```json
{"type": "ready", "message": "Helix ready — lisinopril / stage: treatment_initiation", "journey_stage": "treatment_initiation"}
```

**Step 3 — Stream audio:** Send raw PCM 16kHz mono bytes

**Step 4 — Receive events:**
```json
{"type": "transcript", "text": "I've been feeling dizzy lately"}
{"type": "response", "text": "Dizziness is listed in the FDA label...", "escalate": false}
// followed by raw audio bytes
{"type": "summary", "data": {...}}  // on WebSocket disconnect
```

---

## What I'd Do at Scale

| Concern | Current | Production |
|---|---|---|
| Audio | WebSocket PCM | WebRTC or Twilio Media Streams |
| STT | Deepgram live | Deepgram + fallback (AssemblyAI) |
| LLM | Claude Haiku | Claude Haiku + prompt caching for common Qs |
| Vector store | In-memory FAISS | Pinecone or pgvector (persistent, multi-drug) |
| Sessions | In-memory dict | Redis with TTL |
| Post-call | Sync on disconnect | Async job queue (Celery/Redis) |
| Guardrails | Pattern matching | LLM-as-judge eval pipeline for drift detection |
| Observability | Logging | Datadog APM + latency dashboards |
| Telephony | WebSocket | Twilio Voice + SIP trunking |
| Journey stage | Inferred from days | Pulled from patient CRM/EHR |
| Multi-language | English only | Deepgram + Cartesia multilingual models |

---

Built by [K Ruchitha Reddy](https://github.com/Ruchitha1608)

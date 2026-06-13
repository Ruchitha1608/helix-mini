# Helix Mini

> A voice AI for the full patient medication journey — compliant, always-on, FDA-grounded.  
> Built to mirror the core engineering challenges behind Synthio's Helix product.

**Stack:** Python · FastAPI · Deepgram STT · Claude Haiku · Cartesia TTS · FAISS · WebSockets · Docker

---

## What this is

A production-ready backend for a healthcare voice AI agent. A patient calls in, speaks naturally, and Helix:

1. Transcribes speech in real time (Deepgram `nova-2-medical`)
2. Scans for crisis signals before any LLM call — chest pain, breathing difficulty, fainting
3. Retrieves the top-4 relevant passages from the drug's FDA label (FAISS RAG)
4. Calls Claude Haiku with stage-appropriate context and FDA grounding
5. Validates the response before it reaches TTS — blocks dose advice, unauthorized claims
6. Synthesizes speech (Cartesia `sonic-2`) and streams audio back
7. Produces a structured post-call summary for care team review

Current formulary: **lisinopril** + **metformin**. Any drug with a PDF from [labels.fda.gov](https://labels.fda.gov) can be added by dropping it in `data/fda_labels/`.

---

## The hard problems

**1. Patients don't talk like label text.**  
They say "I fainted" not "experienced syncope", "I can't breathe" not "difficulty breathing". A guardrail that only matches canonical phrases misses the real signal — in healthcare, a miss is a patient safety event.

**2. "I don't have chest pain" should not escalate.**  
Substring matching triggers on the phrase, not the meaning. Negation detection checks a 25-character window before each match for phrases like "don't", "no", "denies", "not having".

**3. The LLM must not improvise.**  
Every response is grounded in FDA-approved text. If the label doesn't say it, the agent says it doesn't know rather than answering from training data. The output guardrail catches hallucinations before TTS.

**4. Journey stage changes everything.**  
Day 0 ("what is this medication?") and day 90 ("I've been skipping doses") need completely different prompts, tone, and escalation sensitivity. Stage is auto-inferred from `days_on_treatment` or passed explicitly.

---

## Architecture

```
Patient (microphone)
        │
        ▼  raw PCM 16kHz mono (WebSocket)
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
│   asynclive + queue   │
└──────────┬───────────┘
           │  transcript
           ▼
┌──────────────────────────────────┐
│  Layer 1 — Input Guardrail        │
│  pattern scan + negation check    │
│  "I can't breathe"  → HIGH        │
│  "I don't have chest pain" → skip │
└──────────┬───────────────────────┘
           │
     HIGH severity?
     ├─ yes → escalation response (LLM bypassed entirely)
     └─ no  ↓
           │
           ▼
┌──────────────────────┐
│   FAISS RAG           │
│   top-4 FDA passages  │
│   sentence-transformers│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────┐
│   Claude Haiku                    │
│   stage-specific system prompt    │
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
│   sonic-2             │
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

## Patient Journey State Machine

| Stage | Days | What Helix focuses on |
|---|---|---|
| `enrollment` | 0 | Explain the drug, what to expect, how to take it |
| `onboarding` | 1–3 | Normalize first experiences, answer "is this normal?" |
| `treatment_initiation` | 4–14 | Monitor early side effects, reinforce habit |
| `adherence` | 15+ | Sustain long-term adherence, refill planning |
| `side_effect_monitoring` | explicit | Active symptom tracking, lower escalation threshold |

Stage is auto-inferred from `days_on_treatment` or can be passed explicitly in the WebSocket config.

---

## Compliance Design

### Layer 1 — Input Guardrail with Negation Detection

Scans patient speech before any LLM call. High-severity matches bypass the LLM entirely.

```python
HIGH_SEVERITY_SYMPTOMS = [
    "chest pain", "chest pressure", "chest tightness",
    "can't breathe", "trouble breathing", "shortness of breath",
    "fainted", "passed out", "fainting",
    "throat swelling", "throat is swollen", "face is swollen",
    "irregular heartbeat", "beating irregularly",
    ...
]
```

Negation detection prevents false escalation: `"I don't have chest pain"` checks the 25 chars before `"chest pain"` for `"don't"`, `"no"`, `"denies"`, `"not having"`, etc.

### Layer 2 — RAG-Grounded Answers

Every LLM call injects top-4 FDA label passages. The agent is instructed to answer only from this context and cite the source section.

```python
ComplianceCitation(
    claim="Dizziness is a known side effect...",
    source_section="Page 3 - Adverse Reactions",
    source_text="In clinical trials, dizziness occurred in 12% of patients..."
)
```

### Layer 3 — Output Guardrail

Agent responses are scanned before TTS. Violations are replaced with safe fallbacks — the patient never hears the original response.

```python
DOSE_CHANGE_PATTERNS    = ["increase your dose", "take more", "double your dose", ...]
UNAUTHORIZED_CLAIMS     = ["will cure", "guaranteed", "no side effects", ...]
```

---

## Eval Suite

```bash
python -m evals.run_evals           # 33 deterministic tests — ~0.1s, no API keys
python -m evals.run_evals --rag     # + RAG retrieval quality (37 total)
python -m evals.run_evals --llm     # + live LLM response quality
```

**37/37 passing** across three layers.

The first run found 8 bugs — original patterns were too literal. `"fainting"` didn't match `"fainted"`, `"dizziness"` didn't match `"dizzy"`, `"difficulty breathing"` didn't match `"can't breathe"`. All eight were patient safety misses: symptoms that should have escalated, didn't. Negation detection was also added after evals exposed false escalations on `"I don't have chest pain"`.

| Suite | Cases | What it tests |
|---|---|---|
| Input guardrail | 15 | 5 high-severity, 3 medium, 4 benign, 3 negations |
| Output guardrail | 11 | 8 violation patterns, 3 clean responses |
| Journey stage | 7 | Day 0 → day 90 boundaries |
| RAG retrieval | 4 | Passage count, keyword relevance, latency |
| LLM quality | 5 | Word count, compliance, citations present |

Exit code 1 on failure — plugs into CI.

---

## Running the Full Pipeline Test

```bash
python test_pipeline.py
```

Exercises all layers without a microphone: text input → RAG → Input Guardrail → Claude → Output Guardrail → TTS. Covers both drugs across 5 turns including a crisis escalation.

```
Turn 1  [lisinopril · treatment_initiation · day 7]
  ✓ Input guardrail: SAFE
  ✓ LLM responded in 4440ms · 4 citations
  ✓ Output guardrail: clean
  ✓ TTS: 7,503,872 audio bytes in 8808ms

Turn 5  [lisinopril · adherence · day 20]
  Patient: "I have severe chest pain and I'm sweating a lot."
  ✓ Input guardrail fired correctly → escalation response
```

---

## Running the Offline Demo

No API keys needed.

```bash
python demo.py
```

Covers: input guardrail cases, negation detection, output guardrail, journey state machine, RAG retrieval — all locally, in ~5 seconds.

---

## Setup

**Local:**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY
uvicorn main:app --reload
```

**Docker:**

```bash
cp .env.example .env
# Fill in API keys
docker compose up
```

**Add a drug:**

Drop any FDA label PDF into `data/fda_labels/<drug_name>.pdf`. The RAG service indexes it on startup.

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
    "known_conditions": ["hypertension"]
  }
}
```

**Step 2 — Receive ready:**
```json
{"type": "ready", "message": "Helix ready — lisinopril / stage: treatment_initiation", "journey_stage": "treatment_initiation"}
```

**Step 3 — Stream audio:** raw PCM 16kHz mono bytes

**Step 4 — Receive events:**
```json
{"type": "transcript", "text": "I've been feeling dizzy lately"}
{"type": "response", "text": "Dizziness is listed in the FDA label...", "escalate": false}
// followed by raw audio bytes (TTS)
{"type": "summary", "data": {...}}   // on disconnect
```

---

## Multimodal Image Analysis

```
POST /voice/analyze-image/{session_id}
Content-Type: multipart/form-data
```

Patient submits a photo (rash, medication bottle) mid-session. Claude Vision analyzes it with FDA label context. Response passes through the same output guardrail.

```json
{
  "drug": "lisinopril",
  "journey_stage": "treatment_initiation",
  "analysis": "The image shows mild redness on the forearm. According to the FDA label adverse reactions section, skin rash is reported in a small percentage of patients...",
  "citations": [{"source_section": "Page 4 - Adverse Reactions", "source_text": "..."}],
  "guardrail_fired": false,
  "latency_ms": 380
}
```

---

## Post-Call Summary

Emitted as JSON on every WebSocket disconnect:

```json
{
  "session_id": "abc-123",
  "drug": "lisinopril",
  "journey_stage": "treatment_initiation",
  "adherence_signal": "at_risk",
  "days_on_treatment": 14,
  "topics_covered": ["missed_dose", "side_effects"],
  "side_effects_mentioned": ["dizziness"],
  "compliance_citations": [{"source_section": "Page 3 - Adverse Reactions", "...": "..."}],
  "guardrail_triggers": 0,
  "escalate_to_human": false,
  "avg_response_latency_ms": 690
}
```

---

## What I'd Do at Scale

| Concern | Current | Production |
|---|---|---|
| Audio transport | WebSocket PCM | WebRTC or Twilio Media Streams |
| Vector store | In-memory FAISS | pgvector or Pinecone (persistent, multi-drug) |
| Sessions | In-memory dict | Redis with TTL |
| Post-call | Sync on disconnect | Async job queue (Celery/Redis) |
| Guardrails | Pattern matching | LLM-as-judge eval pipeline for drift |
| Observability | Logging | Datadog APM + latency dashboards |
| Telephony | WebSocket | Twilio Voice + SIP trunking |
| Journey stage | Inferred from days | Pulled from patient CRM/EHR |
| Multi-language | English only | Deepgram + Cartesia multilingual |
| LLM | Claude Haiku | Haiku + prompt caching for common questions |

---

Built by [K Ruchitha Reddy](https://github.com/Ruchitha1608)

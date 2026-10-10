# Autonomous Maintenance Triage & Dispatch Agent: Implementation Architecture & Master Prompt

## 1. Executive Summary & Product Architecture

This document specifies the technical architecture and phased implementation plan for an **Autonomous Maintenance Triage & Dispatch Agent** built with **LangGraph**, **Google Gemini 1.5 Flash**, **FastAPI**, and **SQLite/Postgres Checkpointing**.

### Core Value Proposition
- **Multimodal Visual Triage:** Classifies property maintenance issues from uploaded photos into granular categories (minor DIY vs. major contractor dispatch).
- **Asynchronous Human-in-the-Loop (HITL):** Employs LangGraph `interrupt()` and `Command(resume=...)` to handle long-lived pauses (technician quotations, property manager approvals) without state loss or hanging connections.
- **Strict Blast-Radius Control:** Property access codes and tenant room details remain protected in state keys and are only hydrated into dispatch packets upon explicit 1-click authorization.

---

## 2. System Topology

```
Tenant Intake (Web/Telegram/Webhook)
                 │
                 ▼
     [ app.api.intake_endpoint ]
                 │
                 ▼
      [ LangGraph StateGraph ]
                 │
                 ├─► (Node: Triage)
                 │     │
                 │     ├── Gemini 1.5 Flash (Vision + Pydantic Schema)
                 │     └── Output: VisualTriageAssessment (Severity, Confidence, Trade)
                 │
                 ├─► [Conditional Router]
                 │     │
                 │     ├── (is_safe_for_diy == True) ──► (Node: Dispatch DIY Guide) ──► (Node: Await Feedback) ──► END
                 │     │
                 │     └── (Severity == "major" / "emergency")
                 │           │
                 │           ▼
                 │        (Node: Request Technician RFQ)
                 │           │
                 │           ▼
                 │        [HITL Gate 1: interrupt("wait_for_quote")] ── (State Persisted to SQLite)
                 │           │
                 │           ▼ (Resumed via Webhook: Command(resume={"quote": 175.0, "eta": "2 PM"}))
                 │        (Node: Prepare PM Approval Card)
                 │           │
                 │           ▼
                 │        [HITL Gate 2: interrupt("wait_for_pm_approval")] ── (Awaits 1-Click Link)
                 │           │
                 │           ├── Rejected ──► (Node: Notify Vendor & Re-route) ──► END
                 │           │
                 │           ▼ Approved (Resumed via Webhook: Command(resume="approved"))
                 │        (Node: Dispatch Packet Assembler)
                 │           │
                 │           ├── Hydrate Unit # & Access Code from Encrypted Vault Key
                 │           └── Transmit Dispatch Token to Technician
                 │
                 ▼
                END
```

---

## 3. Project Directory Structure

```text
proptech-triage-agent/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── README.md
├── app/
│   ├── __init__.py
│   ├── config.py                     # Env vars, API keys, confidence thresholds
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py                # Pydantic models (VisualTriageAssessment, State)
│   │   └── security.py               # Access code hydration / mock vault logic
│   ├── services/
│   │   ├── __init__.py
│   │   └── vision_service.py         # Google GenAI SDK (Gemini 1.5 Flash structured calls)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                  # TicketState TypedDict definition
│   │   ├── nodes.py                  # All functional graph nodes
│   │   ├── edges.py                  # Conditional edge evaluation logic
│   │   └── workflow.py               # StateGraph compilation with checkpointer
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                   # Messaging interface abstraction
│   │   └── mock_notifier.py          # Terminal/Webhook mock adapter
│   └── api/
│       ├── __init__.py
│       ├── main.py                   # FastAPI application entrypoint
│       └── routes.py                 # Intake & resumption webhook endpoints
└── tests/
    ├── conftest.py                   # Fixtures, test images (base64)
    ├── test_phase1_state.py          # Unit tests for state persistence
    ├── test_vision_triage.py         # Unit tests for vision schema parsing
    ├── test_state_machine.py         # Unit tests for LangGraph interrupt & resume
    └── test_end_to_end.py            # Complete ticket lifecycle simulation
```

---

## 4. Master Prompt for Cursor / Claude Code

Copy the prompt below directly into Cursor or Claude Code to start building phase by phase:

```markdown
# Context & Role
You are a Principal AI Systems Architect and Lead Python Engineer. We are building an enterprise-grade, zero-cost-ready Autonomous Maintenance Triage & Dispatch Agent in Python using LangGraph, Google Gemini 1.5 Flash (via `google-genai` SDK), SQLite Checkpointing, and FastAPI.

The core engineering principles are:
1. Strict schema separation: Models handle perception and return Pydantic types; state routing is deterministic.
2. Stateful persistence: Pauses for technician quotes and manager approvals MUST use LangGraph's native `interrupt()` and `Command(resume=...)` pattern backed by a checkpointer.
3. Least privilege data exposure: Property access codes remain encrypted in state keys and are only hydrated in the final dispatch node after explicit approval.

We will develop this systematically across 5 phases. DO NOT build all phases at once. You must implement Phase 1 now, verify it with an automated test, and prompt me to approve moving to Phase 2.

---

## Technical Stack
- Python 3.11+
- `langgraph>=0.2.20`
- `google-genai>=0.1.1`
- `pydantic>=2.7.0`
- `fastapi>=0.111.0`
- `uvicorn>=0.30.0`
- `pytest>=8.0.0`, `pytest-asyncio`

---

## Phase Breakdown Overview
- **Phase 1: Project Foundation, Domain Models & State Management**
  - Dependency setup (`requirements.txt`).
  - Base configuration and TypedDict / Pydantic schemas in `app/core/schemas.py`.
  - Checkpoint persistence setup using `SqliteSaver`.
  - Self-contained verification test confirming state serialization.

- **Phase 2: Multimodal Visual Triage Engine (Gemini 1.5 Flash)**
  - Implement `app/services/vision_service.py` using `google-genai` client.
  - Implement strict JSON output parsing with `VisualTriageAssessment` Pydantic schema.
  - Add confidence-score threshold handling (<0.75 triggers ambiguous flag).
  - Tests with mock base64 images (minor drain vs. active pipe burst).

- **Phase 3: LangGraph State Machine & Asynchronous HITL Gates**
  - Implement `app/graph/nodes.py`, `edges.py`, and `workflow.py`.
  - Triage node -> Conditional fork (DIY vs Contractor).
  - Gate 1: `interrupt({"step": "wait_for_quote", "ticket_id": ...})`.
  - Gate 2: `interrupt({"step": "wait_for_approval", "quote": ...})`.
  - Dispatch packet assembler with access-code hydration.
  - Unit tests verifying graph pauses and programmatic resumption via `Command`.

- **Phase 4: FastAPI Webhook Ingress & Resumption Endpoints**
  - `POST /api/tickets/intake`: accepts image, tenant_id, unit_id, starts thread.
  - `POST /api/webhooks/technician-quote`: resumes Gate 1 with quote and ETA.
  - `GET /api/webhooks/manager-approval`: resumes Gate 2 with 1-click token.
  - Mock messaging adapter logging clear formatted notifications to stdout.

- **Phase 5: End-to-End Integration, Observability & Evaluation Suite**
  - Complete integration test simulating both Happy Path (Minor DIY) and Escalation Path (Major -> Quote -> Approval -> Dispatch).
  - Token and cost tracking utility.

---

## YOUR CURRENT TASK: Execute Phase 1 Now

Please implement **Phase 1**:
1. Create `requirements.txt` with exact compatible versions.
2. Create `app/config.py` loading environment variables (with fallback defaults).
3. Create `app/core/schemas.py` containing:
   - `VisualTriageAssessment` (Pydantic model)
   - `TicketState` (TypedDict for LangGraph)
   - `DispatchPacket` (Pydantic model)
4. Create a test in `tests/test_phase1_state.py` that verifies:
   - Initializing a state dict conforms to `TicketState`.
   - Storing and retrieving state via LangGraph's `SqliteSaver` works correctly.
5. Provide the exact terminal commands to set up the virtualenv, install dependencies, and run the pytest verification.

Once Phase 1 passes, stop and ask for confirmation to proceed to Phase 2.
```

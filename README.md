# Unitriage.AI

Maintenance triage and dispatch agent for property managers. A tenant uploads a photo (and an
optional note); the agent decides how urgent it is and routes it.

- **Minor, DIY:** the tenant gets a self-help guide and is asked for feedback.
- **Minor, not DIY / Major / Emergency:** a technician is asked for a quote, the property manager
  approves with one click, then a dispatch packet is assembled. Unit number and access code are only
  added to the packet after approval.
- **Low confidence:** the ticket goes to manual review instead of being auto-closed.

Built with LangGraph (two human-in-the-loop pauses, state saved in SQLite), Gemini / Groq vision,
and FastAPI.

## How triage works

The vision model only reports what it **sees** (water, damage, hazards, whether a layperson could
fix it). It never judges urgency. `app/core/policy.py` turns that evidence into
minor / major / emergency, so the rules are one small table you can read, test and tune.

A separate screener (`app/services/note_screen.py`) reads the tenant's note for hazards the photo
can't show (gas smell, no water, a vulnerable person in the home) and can raise urgency.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
cp .env.example .env                                           # then fill in your keys
```

Optional: `NOTE_SCREEN=llm|keywords|off` (default `llm`, falls back to keywords without a key).
Set at least one of `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY` in `.env`. OpenRouter is the
last-resort fallback (set `OPENROUTER_MODEL` to any image-capable model). `VISION_PROVIDERS` sets the order
they are tried in. Use upper-case variable names exactly as in `.env.example` (they are
case-sensitive on Linux and macOS).

## Run

```bash
.venv/Scripts/python.exe -m uvicorn app.api.main:app --reload
```

Open `http://localhost:8000/demo` for the demo UI. API docs are at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/tickets/intake` | Photo + tenant/unit; starts a ticket |
| `GET /api/tickets/{id}` | Ticket status |
| `POST /api/tickets/{id}/feedback` | Tenant reply to the DIY guide |
| `POST /api/webhooks/technician-quote` | Resumes the quote pause |
| `GET /api/webhooks/manager-approval` | One-click approve / reject |
| `GET /api/usage` | Token and cost tracking |

On the tenant phone there is no unit picker: the assistant asks which room or unit the photo is for, validates it against the property records, and remembers it in the browser.

Notifications go to a mock notifier (terminal / demo inbox). No real messages are sent.

## Where data is stored

| `DATABASE_URL` | Ticket state | Chat messages | Photos |
| --- | --- | --- | --- |
| set (Neon or any Postgres) | Postgres (LangGraph checkpointer) | `messages` table | `photos` table |
| empty | `checkpoints.sqlite` | memory (lost on restart) | memory + the ticket state |

Use the **direct** connection string (not the pooled `-pooler` one). Tables are created on
startup. With Postgres the photo is kept in the `photos` table and removed from the ticket state
after triage, so checkpoints stay small. Neon's free tier pauses idle databases, so the first
request after a quiet period can take a couple of seconds.

## How work is spread across AI providers

Each job has its own preferred order, so no single free tier carries everything:

| Job | Order (env var) | Why |
| --- | --- | --- |
| Photo triage | groq, gemini, openrouter (`VISION_PROVIDERS`) | fastest and most accurate on the evals first |
| Note screening | groq, openrouter, gemini (`NOTE_LLM_PROVIDERS`) | short text, Groq is fast and exact on the eval |
| Owner summaries | openrouter, groq, gemini (`SUMMARY_PROVIDERS`) | low stakes, keeps quota for the others |

A provider that returns 402/429 (out of credit or quota) is skipped for 5 minutes, and one that
errors for 30 seconds (`app/core/health.py`), so a tenant never waits on a provider that is known
to be down. If everything is cooling down, all are still tried.

## Owner approval summaries

The approval card the owner sees has a short plain-language summary written by an AI
(`app/services/summary.py`). It only sees a small set of facts, never the access code, and is told
not to recommend approving or rejecting. The quote and ETA are printed by the app, never by the AI.
If no AI answers, the card falls back to the fixed template. `SUMMARY_MODE=off` disables it.

## Deploy on Render

1. Push this repo to GitHub, then in Render choose **New > Blueprint** and pick it. `render.yaml`
   defines the service (start command, health check at `/healthz`, Python 3.12).
2. When asked, fill in the secrets: `DATABASE_URL` (Neon, direct connection) and
   `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY`. `VAULT_SECRET` is generated for you.
3. Links in messages use the service's own address automatically (`RENDER_EXTERNAL_URL`); set
   `PUBLIC_BASE_URL` only for a custom domain.

`DEMO_UI=true` serves the phone simulator and a feed of every message, including dispatch
messages with access codes, to anyone with the URL. Keep it for demo data only. Free web services
sleep after about 15 minutes idle, and the first request after that takes around a minute.

## Test and evaluate

```bash
.venv/Scripts/python.exe -m pytest -q                              # unit + end-to-end, no API calls
.venv/Scripts/python.exe evals/run_eval.py --list                  # show photo cases, no API calls
.venv/Scripts/python.exe evals/run_eval.py --provider groq --delay 10
.venv/Scripts/python.exe evals/run_note_eval.py --only keywords    # no API calls
.venv/Scripts/python.exe evals/run_note_eval.py --only llm         # AI check, uses quota
```

Evaluation photos live in `evals/images/<folder>/`; the folder name is the expected answer (see
`evals/images/README.txt`). They are git-ignored because they are third-party images.

## Known limits (demo build)

- Notifications are mocked and the access-code vault is `MockVault`.
- State is SQLite only.
- The keyword note screener misses some real hazards; the AI note check is meant to cover them.
- Vision accuracy varies by provider; see the eval output before relying on a result.

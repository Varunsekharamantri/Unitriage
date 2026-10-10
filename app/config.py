"""Environment-driven settings with safe defaults."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass


# The work is spread by job: each job has its own preferred order (first = does most of it), and a
# provider that runs out of quota is skipped for a few minutes (app/core/health.py), so work moves
# on by itself. Comma-separated; providers without a key are skipped.
# Photos: Groq first (fastest, and the most accurate on evals/run_eval.py), Gemini as the backup.
VISION_PROVIDERS: list[str] = [
    p.strip() for p in os.getenv("VISION_PROVIDERS", "groq,gemini,openrouter").split(",") if p.strip()
]

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# OpenRouter: one key that reaches many models. "openrouter/free" picks whichever free model is
# available, so it needs no credit; set a paid model (it must accept images for photo triage, see
# "input modalities" at openrouter.ai/models) once the account has credit. Free models may log prompts.
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# How the tenant's note is screened for dangers a photo can't show (gas smell, burning smell...):
# "llm" (default: a short text-only AI call, falling back to keywords if no AI answers),
# "keywords" (instant, no AI), or "off". See app/services/note_screen.py for how this was chosen.
NOTE_SCREEN: str = os.getenv("NOTE_SCREEN", "llm")
# Months in which "no heating" counts as urgent (northern-hemisphere cold season by default).
# Outside them it is ignored, unless the note says a baby, elderly or sick person is affected.
HEATING_MONTHS: list[int] = [
    int(m) for m in os.getenv("HEATING_MONTHS", "10,11,12,1,2,3,4").split(",") if m.strip()
]
# Which AI providers the "llm" screener tries, in order (those with a key set).
NOTE_LLM_PROVIDERS: list[str] = [
    p.strip() for p in os.getenv("NOTE_LLM_PROVIDERS", "groq,openrouter,gemini").split(",") if p.strip()
]
# The owner's approval card: "llm" has an AI write a plain-language summary (the quote and ETA
# are always added by the app, never by the AI); "off" uses the fixed template. Any failure
# falls back to the template, so it can never block an approval.
SUMMARY_MODE: str = os.getenv("SUMMARY_MODE", "llm")
SUMMARY_PROVIDERS: list[str] = [
    p.strip() for p in os.getenv("SUMMARY_PROVIDERS", "openrouter,groq,gemini").split(",") if p.strip()
]
# A problem counts as reported when its score is at least this. The AI check answers yes or no
# (1.0 or 0.0), so any value from 0.1 to 0.9 behaves the same; it matters only for other screeners.
# Check it with evals/run_note_eval.py.
NOTE_HAZARD_THRESHOLD: float = float(os.getenv("NOTE_HAZARD_THRESHOLD", "0.5"))

# Only used by evals/fetch_photos.py, to find reference photos for the evaluation set.
UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")

# Postgres (Neon etc.). When set it holds ticket state, chat messages and photos, so they survive
# restarts and redeploys. When empty, SQLite plus in-memory stores are used (fine locally).
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite")
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
# The most seconds photo triage may take in total before the ticket goes to manual review. Each
# provider call also has its own timeout, so a provider that never answers can't hang a ticket.
TRIAGE_BUDGET_SECONDS: float = float(os.getenv("TRIAGE_BUDGET_SECONDS", "30"))
VISION_RETRIES_PER_PROVIDER: int = int(os.getenv("VISION_RETRIES_PER_PROVIDER", "2"))
VAULT_SECRET: str = os.getenv("VAULT_SECRET", "dev-only-secret")
# The address links in messages (approvals, photos) point to. On Render it is filled in
# automatically from RENDER_EXTERNAL_URL; set PUBLIC_BASE_URL yourself for a custom domain.
PUBLIC_BASE_URL: str = (
    os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "http://localhost:8000"
).rstrip("/")
# Chat simulator at "/" plus a feed of every message sent to tenant/technician/manager,
# including the dispatch message that carries the access code. Local demos only:
# set DEMO_UI=false for anything reachable by other people.
DEMO_UI: bool = os.getenv("DEMO_UI", "true").lower() in ("1", "true", "yes")
OWNER_NAME: str = os.getenv("OWNER_NAME", "Alex Morgan")  # the contact name on tenant/technician phones
TECHNICIAN_NAME: str = os.getenv("TECHNICIAN_NAME", "Sam Rivera")  # the contact name on the owner's phone
# Demo tenants by unit, so the UI can send a name with each ticket as a real intake channel
# (Telegram profile, web login) would.
DEMO_TENANTS: dict[str, str] = {"12B": "Priya Sharma", "3A": "Rahul Verma"}
MAX_IMAGE_BYTES: int = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

"""FastAPI entrypoint. Run with: uvicorn app.api.main:app --reload"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from app import config
from app.adapters.mock_notifier import MockNotifier
from app.api import demo
from app.api.routes import router
from app.core.photos import PhotoCache
from app.core.security import MockVault, UnitSecrets
from app.core.usage import UsageTracker
from app.services.summary import build_summarizer
from app.graph.workflow import build_workflow, get_sqlite_checkpointer
from app.services.vision_service import build_default_service

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def demo_vault() -> MockVault:
    """Demo units only. Replace with a real secrets store in production."""
    return MockVault(
        {
            "vault/12B": UnitSecrets("12B", "4821#"),
            "vault/3A": UnitSecrets("3A", "7305#"),
        }
    )


def create_app(
    workflow=None,
    vault: Optional[MockVault] = None,
    usage: Optional[UsageTracker] = None,
    notifier: Optional[MockNotifier] = None,
    background_intake: Optional[bool] = None,
) -> FastAPI:
    """Pass workflow/vault/usage/notifier to inject test doubles; otherwise they are
    built on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = pool = db = None
        # Postgres (Neon) when DATABASE_URL is set and no test workflow was injected: ticket state,
        # chat messages and photos then survive restarts. Otherwise SQLite and memory, as before.
        if workflow is None and config.DATABASE_URL:
            from langgraph.checkpoint.postgres import PostgresSaver

            from app.core.db import Database, open_pool

            pool = open_pool(config.DATABASE_URL)
            db = Database(pool)
            db.ensure_schema()
        # Background assessment is for the real app. An injected workflow (tests) stays synchronous
        # unless asked, so a response already carries the ticket's state.
        app.state.background_intake = (
            background_intake if background_intake is not None else (workflow is None and config.BACKGROUND_INTAKE)
        )
        app.state.vault = vault or demo_vault()
        app.state.usage = usage or UsageTracker()
        app.state.notifier = notifier or MockNotifier(store=db)
        app.state.photos = PhotoCache(db=db)
        if workflow is None:
            if pool is not None:
                checkpointer = PostgresSaver(pool)
                checkpointer.setup()
            else:
                checkpointer = get_sqlite_checkpointer(config.CHECKPOINT_DB_PATH)
                conn = checkpointer.conn
            app.state.workflow = build_workflow(
                build_default_service(tracker=app.state.usage),
                app.state.notifier,
                app.state.vault,
                checkpointer,
                build_summarizer(config.SUMMARY_MODE),
                clear_image=db is not None,
                retry_window=config.TRIAGE_RETRY_SECONDS if app.state.background_intake else 0.0,
            )
        else:
            app.state.workflow = workflow
        yield
        if conn is not None:
            conn.close()
        if pool is not None:
            pool.close()

    app = FastAPI(title="Unitriage.AI", lifespan=lifespan)
    app.include_router(router)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok"}  # for the host's health check: no database or AI call

    if config.DEMO_UI:
        app.include_router(demo.router)

        # "/" and "/demo": all three actors side by side. "/phone?role=tenant|technician|owner":
        # one actor's phone, which is what each frame on that page embeds.
        @app.get("/", include_in_schema=False)
        @app.get("/demo", include_in_schema=False)
        def three_phone_demo():
            return FileResponse(STATIC_DIR / "demo.html")

        @app.get("/phone", include_in_schema=False)
        def one_phone():
            return FileResponse(STATIC_DIR / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/docs")

    return app


app = create_app()

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
) -> FastAPI:
    """Pass workflow/vault/usage/notifier to inject test doubles; otherwise they are
    built on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = None
        app.state.vault = vault or demo_vault()
        app.state.usage = usage or UsageTracker()
        app.state.notifier = notifier or MockNotifier()
        app.state.photos = PhotoCache()
        if workflow is None:
            checkpointer = get_sqlite_checkpointer(config.CHECKPOINT_DB_PATH)
            conn = checkpointer.conn
            app.state.workflow = build_workflow(
                build_default_service(tracker=app.state.usage),
                app.state.notifier,
                app.state.vault,
                checkpointer,
                build_summarizer(config.SUMMARY_MODE),
            )
        else:
            app.state.workflow = workflow
        yield
        if conn is not None:
            conn.close()

    app = FastAPI(title="Unitriage.AI", lifespan=lifespan)
    app.include_router(router)

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

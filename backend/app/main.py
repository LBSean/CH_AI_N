from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Must come after load_dotenv so env vars are available
from app.api import admin, agent, health, ingest  # noqa: E402
from app.auth import router as auth_router         # noqa: E402
from app.core.config import get_settings           # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.rag.pipeline import configure_llamaindex  # noqa: E402

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_llamaindex()
    log.info("startup", version="0.3.0", environment=get_settings().environment)
    yield
    log.info("shutdown")


app = FastAPI(
    title="AI Workspace Backend",
    description="LangGraph + LlamaIndex agent API",
    version="0.3.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth_router.router)
app.include_router(agent.router)
app.include_router(ingest.router)
app.include_router(admin.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

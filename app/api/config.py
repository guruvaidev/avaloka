import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv

from app.core.settings import Settings
from app.services import session_service, storage_service
from app.core.cache import ICache
from app.core.storage import IBlobStore

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("avaloka")

settings = Settings()

DATA_DIR = os.environ.get("DATA_DIR", str(Path.cwd() / "data"))
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

COOKIE_NAME = "avaloka_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", settings.mcp_server_url)
MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", str(settings.mcp_timeout)))

LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://34.174.185.105:2024")
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "15.0"))

MAX_CONTEXT_TURNS = int(os.getenv("MAX_CONTEXT_TURNS", "12"))

TMP_DIR = os.getenv("CHAT_OUTPUT_DIR", "/tmp")
if os.getenv("TMP_ROOT"):
    TMP_ROOT = Path(os.getenv("TMP_ROOT")).resolve()
else:
    TMP_ROOT = Path("/var/tmp/avaloka") if os.name != "nt" else Path(DATA_DIR) / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

base_allowed = ["*"]
extra = [
    o.strip()
    for o in os.getenv("EXTRA_CORS_ORIGINS", "").split(",")
    if o.strip()
]
allow_origins = list(dict.fromkeys(base_allowed + extra))

http_client: httpx.AsyncClient | None = None
cache: ICache | None = None
blob_store: IBlobStore | None = None

@asynccontextmanager
async def lifespan(app):
    from app.services.session_service import _init_cache

    global http_client, cache, blob_store

    session_service.cache = await _init_cache()
    cache = session_service.cache

    blob_store = storage_service.init_blob_store()

    timeout = httpx.Timeout(
        connect=5.0,
        read=UPSTREAM_TIMEOUT,
        write=UPSTREAM_TIMEOUT,
        pool=UPSTREAM_TIMEOUT,
    )
    http_client = httpx.AsyncClient(
        base_url=LANGGRAPH_API_URL, timeout=timeout, headers={"Connection": "keep-alive"}
    )
    try:
        yield
    finally:
        if cache:
            try:
                await cache.close()
            except Exception:
                logger.exception("error closing cache client")
        if http_client:
            try:
                await http_client.aclose()
            except Exception:
                logger.exception("error closing http client")

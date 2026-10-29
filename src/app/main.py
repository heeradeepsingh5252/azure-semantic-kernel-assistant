"""FastAPI REST API for the Azure Semantic Kernel document assistant."""

from __future__ import annotations

import inspect
import threading
import time
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.app.azure_embeddings import AzureOpenAIEmbedder
from src.app.azure_search_store import AzureSearchStore
from src.app.chunker import chunk_pages
from src.app.config import Settings, get_settings
from src.app.loader import DocumentLoadError, extract_document
from src.app.models import ChatRequest, ChatResponse
from src.app.sk_assistant import SemanticKernelAssistant

DEFAULT_RATE_LIMIT = 30
RATE_LIMIT_WINDOW_SECONDS = 60
logger = logging.getLogger(__name__)

@dataclass
class ApplicationServices:
    """The application services, replaceable with fakes in tests."""

    embedder: Any
    search_store: Any
    assistant: Any


async def _call_maybe_async(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a synchronous or asynchronous service method."""
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _build_services() -> ApplicationServices:
    """Construct the real Azure services only when an API route needs them."""
    embedder = AzureOpenAIEmbedder.from_environment()
    search_store = None

    try:
        search_store = AzureSearchStore.from_environment(embedder)
        assistant = SemanticKernelAssistant.from_environment(search_store)
        return ApplicationServices(
            embedder=embedder,
            search_store=search_store,
            assistant=assistant,
        )
    except Exception:
        if search_store is not None:
            close_store = getattr(search_store, "close", None)
            if callable(close_store):
                close_store()

        close_embedder = getattr(embedder, "close", None)
        if callable(close_embedder):
            close_embedder()
        raise


def get_app_settings(request: Request) -> Settings:
    """Return the settings attached to this app instance."""
    return request.app.state.settings


def get_services(request: Request) -> ApplicationServices:
    """Return injected services or lazily initialize configured Azure services."""
    services = request.app.state.services
    if services is not None:
        return services

    settings: Settings = request.app.state.settings
    if settings.app_mode != "azure":
        raise HTTPException(
            status_code=503,
            detail="Services are unavailable. Configure APP_MODE=azure to use them.",
        )

    try:
        services = _build_services()
    except Exception:
        # Do not expose credentials, endpoints, or internal exception details.
        raise HTTPException(
            status_code=503,
            detail="Application services could not be initialized.",
        ) from None

    request.app.state.services = services
    return services


def enforce_rate_limit(request: Request) -> None:
    """Apply a simple in-memory fixed-window limit per client IP."""
    client = request.client
    client_ip = client.host if client is not None else "unknown"
    now = time.monotonic()

    buckets: dict[str, tuple[float, int]] = request.app.state.rate_limit_buckets
    lock: threading.Lock = request.app.state.rate_limit_lock
    limit = request.app.state.rate_limit_max_requests
    window = request.app.state.rate_limit_window_seconds

    with lock:
        bucket = buckets.get(client_ip)
        if bucket is None or now - bucket[0] >= window:
            buckets[client_ip] = (now, 1)
            return

        started_at, count = bucket
        if count >= limit:
            retry_after = max(1, int(window - (now - started_at)))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(retry_after)},
            )

        buckets[client_ip] = (started_at, count + 1)


def _safe_filename(upload_name: str | None) -> str:
    """Keep only a basename and reject empty or control-character filenames."""
    name = Path((upload_name or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required.")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise HTTPException(status_code=400, detail="The filename is invalid.")
    if len(name) > 255:
        raise HTTPException(status_code=400, detail="The filename is too long.")
    return name


async def _read_limited_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read at most the configured upload size plus one detection byte."""
    content = bytearray()

    while True:
        bytes_remaining_before_rejection = max_bytes + 1 - len(content)
        read_size = min(1024 * 1024, bytes_remaining_before_rejection)
        part = await file.read(read_size)
        if not part:
            break

        content.extend(part)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="The uploaded file exceeds the configured size limit.",
            )

    return bytes(content)


def _error_code(status_code: int) -> str:
    """Map HTTP status codes to stable, machine-readable error codes."""
    return {
        400: "bad_request",
        404: "not_found",
        413: "file_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limit_exceeded",
        502: "upstream_service_error",
        503: "service_unavailable",
    }.get(status_code, "http_error")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Release live clients at shutdown if they were initialized."""
    yield

    services: ApplicationServices | None = application.state.services
    if services is None:
        return

    assistant_close = getattr(services.assistant, "aclose", None)
    if callable(assistant_close):
        try:
            await _call_maybe_async(assistant_close)
        except Exception:
            pass

    for service in (services.search_store, services.embedder):
        close_method = getattr(service, "close", None)
        if callable(close_method):
            try:
                await _call_maybe_async(close_method)
            except Exception:
                pass


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app; tests can inject settings and override its dependencies."""
    resolved_settings = settings if settings is not None else get_settings()
    application = FastAPI(
        title="Azure Semantic Kernel Assistant API",
        description="Document ingestion and citation-grounded question answering.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.state.settings = resolved_settings
    application.state.services = None
    application.state.rate_limit_buckets = {}
    application.state.rate_limit_lock = threading.Lock()
    application.state.rate_limit_max_requests = DEFAULT_RATE_LIMIT
    application.state.rate_limit_window_seconds = RATE_LIMIT_WINDOW_SECONDS

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = (
            exc.detail
            if isinstance(exc.detail, str)
            else "The request could not be completed."
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _error_code(exc.status_code),
                    "message": message,
                }
            },
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Avoid echoing submitted values that might contain personal information.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                }
            },
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # The response deliberately excludes stack traces and exception details.
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected server error occurred.",
                }
            },
        )

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Report that the HTTP application is running."""
        return {"status": "ok"}

    @application.post("/ingest", tags=["documents"])
    async def ingest(
        file: Annotated[UploadFile, File(description="A PDF or TXT document")],
        settings: Annotated[Settings, Depends(get_app_settings)],
        services: Annotated[ApplicationServices, Depends(get_services)],
        _rate_limit: Annotated[None, Depends(enforce_rate_limit)],
    ) -> dict[str, Any]:
        """Extract, chunk, and index an uploaded PDF or text file."""
        filename = _safe_filename(file.filename)
        extension = Path(filename).suffix.lower()

        if extension not in {".pdf", ".txt"}:
            await file.close()
            raise HTTPException(
                status_code=415,
                detail="Only PDF and TXT files are supported.",
            )

        try:
            content = await _read_limited_upload(
                file,
                max_bytes=settings.max_upload_mb * 1024 * 1024,
            )
        finally:
            await file.close()

        try:
            pages = extract_document(filename, content)
            if not pages:
                raise HTTPException(
                    status_code=422,
                    detail="No extractable text was found in the document.",
                )
            chunks = chunk_pages(pages)
            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail="No text chunks could be created from the document.",
                )
        except HTTPException:
            raise
        except DocumentLoadError:
            raise HTTPException(
                status_code=400,
                detail="The document could not be read or contains no extractable text.",
            ) from None
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="The uploaded document could not be processed.",
            ) from None

        try:
            # Removing prior chunks first avoids stale chunks after a re-upload.
            await _call_maybe_async(services.search_store.delete_by_source, filename)
            await _call_maybe_async(services.search_store.upsert_chunks, chunks)
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="The document could not be indexed.",
            ) from None

        return {
            "filename": filename,
            "pages": len(pages),
            "chunk_count": len(chunks),
        }

    @application.post(
        "/chat",
        response_model=ChatResponse,
        tags=["chat"],
    )
    async def chat(
        payload: ChatRequest,
        services: Annotated[ApplicationServices, Depends(get_services)],
        _rate_limit: Annotated[None, Depends(enforce_rate_limit)],
    ) -> ChatResponse:
        """Answer a question using the existing Semantic Kernel assistant."""
        try:
            response = await _call_maybe_async(services.assistant.answer, payload)
            if isinstance(response, ChatResponse):
                return response
            return ChatResponse.model_validate(response)

        except Exception:
            logger.exception("Chat request failed")
            raise HTTPException(
                status_code=502,
                detail="The assistant could not answer this request.",
            ) from None


    @application.get("/documents", tags=["documents"])
    async def list_documents(
        services: Annotated[ApplicationServices, Depends(get_services)],
        _rate_limit: Annotated[None, Depends(enforce_rate_limit)],
    ) -> dict[str, list[str]]:
        """List distinct source filenames in the search index."""
        try:
            sources = await _call_maybe_async(services.search_store.list_sources)
            return {"sources": sorted(set(sources))}
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="The document list could not be retrieved.",
            ) from None

    @application.delete("/documents/{source}", tags=["documents"])
    async def delete_document(
        source: str,
        services: Annotated[ApplicationServices, Depends(get_services)],
        _rate_limit: Annotated[None, Depends(enforce_rate_limit)],
    ) -> dict[str, Any]:
        """Delete all indexed chunks belonging to a specific source file."""
        if not source or Path(source).name != source or source in {".", ".."}:
            raise HTTPException(status_code=400, detail="The source name is invalid.")

        try:
            sources = await _call_maybe_async(services.search_store.list_sources)
            if source not in sources:
                raise HTTPException(
                    status_code=404,
                    detail="The requested document was not found.",
                )
            await _call_maybe_async(
                services.search_store.delete_by_source,
                source,
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="The document could not be deleted.",
            ) from None

        return {"deleted": True, "source": source}

    return application


app = create_app()

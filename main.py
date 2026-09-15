"""Single-process ASGI application; startup failure is never healthy."""

from contextlib import asynccontextmanager
import hmac
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from app.config import settings
from app.schema import ChatCompletionRequest
from app.service import ChatService, ClientDisconnected, ServiceError
from app.protocol import OutputError, prepare_prompt

router = APIRouter(tags=["Chat"])


def error_body(message, code, error_type="invalid_request_error"):
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


async def authorize(request: Request):
    key = request.app.state.service.config.API_KEY
    if key is not None:
        expected = "Bearer " + key.get_secret_value()
        if not hmac.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"), expected.encode("utf-8")
        ):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})


@router.get("/v1/models", dependencies=[Depends(authorize)])
async def list_models(request: Request):
    service = request.app.state.service
    return {
        "object": "list",
        "data": [{"id": service.config.MODEL_NAME, "object": "model",
                  "created": 0, "owned_by": "rkllm"}],
    }


def sse(value):
    return f"data: {json.dumps(value, ensure_ascii=False)}\n\n"


def chunk(identifier, created, model, delta, finish_reason=None):
    return {
        "id": identifier, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def stream_response(service, job, http_request, identifier, created):
    try:
        yield sse(chunk(identifier, created, job.request.model, {"role": "assistant"}))
        async for delta in service.events(job, http_request):
            yield sse(chunk(identifier, created, job.request.model, delta))
        yield sse(chunk(
            identifier, created, job.request.model, {}, job.result["finish_reason"]
        ))
        if job.request.stream_options and job.request.stream_options.include_usage:
            yield sse({
                "id": identifier, "object": "chat.completion.chunk", "created": created,
                "model": job.request.model, "choices": [], "usage": job.result["usage"],
            })
        yield "data: [DONE]\n\n"
    except ServiceError as error:
        yield sse(error_body(str(error), error.code, "server_error"))
        yield "data: [DONE]\n\n"
    except (OutputError, UnicodeDecodeError) as error:
        yield sse(error_body(str(error), "generation_error", "server_error"))
        yield "data: [DONE]\n\n"
    finally:
        if not job.done.is_set():
            await service.cancel(job)


@router.post("/v1/chat/completions", dependencies=[Depends(authorize)])
async def chat_completions(body: ChatCompletionRequest, request: Request):
    service = request.app.state.service
    if body.model != service.config.MODEL_NAME:
        return JSONResponse(error_body("Unknown model", "model_not_found"), status_code=404)
    try:
        prepared = prepare_prompt(body, service.config, service.vision is not None)
        job = service.submit(body, prepared)
    except ValueError as error:
        return JSONResponse(error_body(str(error), "invalid_request"), status_code=400)
    except ServiceError as error:
        return JSONResponse(error_body(str(error), error.code), status_code=error.status,
                            headers={"Retry-After": "1"} if error.status == 429 else None)
    identifier = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    if body.stream:
        return StreamingResponse(
            stream_response(service, job, request, identifier, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        async for _ in service.events(job, request):
            pass
    except ClientDisconnected:
        return JSONResponse(error_body("Client disconnected", "cancelled"), status_code=499)
    except ServiceError as error:
        return JSONResponse(error_body(str(error), error.code, "server_error"), status_code=error.status)
    except (OutputError, UnicodeDecodeError) as error:
        return JSONResponse(error_body(str(error), "generation_error", "server_error"), status_code=502)
    return {
        "id": identifier, "object": "chat.completion", "created": created,
        "model": body.model,
        "choices": [{"index": 0, "message": job.result["message"],
                     "finish_reason": job.result["finish_reason"], "logprobs": None}],
        "usage": job.result["usage"],
    }


class BodyLimitMiddleware:
    """Bound body buffering, including chunked requests without Content-Length."""

    def __init__(self, app, maximum):
        self.app = app
        self.maximum = maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                logging.getLogger(__name__).debug("Peer disconnected before request body completed")
                return
            data = message.get("body", b"")
            size += len(data)
            if size > self.maximum:
                response = JSONResponse(
                    error_body("Request exceeds MAX_REQUEST_BYTES", "request_too_large"),
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            chunks.append(data)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)


@asynccontextmanager
async def lifespan(application):
    logging.basicConfig(
        level=application.state.service.config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger("rkllm")
    logger.info("Initializing configured models")
    await application.state.service.start()
    try:
        yield
    finally:
        logger.info("Stopping admission and waiting for native inference to finish")
        await application.state.service.shutdown()


def create_app(service=None):
    application = FastAPI(title="RKLLM OpenAI API", version="0.2.0", lifespan=lifespan)
    application.state.service = service or ChatService(settings)
    application.add_middleware(
        BodyLimitMiddleware, maximum=application.state.service.config.MAX_REQUEST_BYTES
    )
    application.include_router(router)

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request, error):
        # Avoid echoing base64 images or other complete request inputs into errors.
        problems = [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        result = error_body("Request validation failed", "invalid_request")
        result["error"]["details"] = problems
        return JSONResponse(result, status_code=422)

    @application.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse(
            error_body(str(error.detail), "http_error"),
            status_code=error.status_code, headers=error.headers,
        )

    @application.get("/health", tags=["System"])
    async def health():
        current = application.state.service
        return JSONResponse({
            "status": "ready" if current.ready else "not_ready",
            "model": current.config.MODEL_NAME,
            "profile": current.config.MODEL_PROFILE,
            "protocol": current.config.MODEL_PROTOCOL,
            "vision": current.vision is not None,
            "active": current.active is not None,
            "queued": len(current.pending),
        }, status_code=200 if current.ready else 503)

    return application


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, workers=1,
                timeout_graceful_shutdown=0)

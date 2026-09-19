"""One NPU worker with bounded admission and request-scoped cancellation."""

import asyncio
from collections import deque
import codecs
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any

from app.config import Settings
from app.schema import ChatCompletionRequest
from app.protocol import (
    OutputError, OutputParser, PreparedPrompt, StopFilter, continuation_prompt, prepare_prompt,
)

logger = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    def __init__(self, status, message, code):
        super().__init__(message)
        self.status = status
        self.code = code


class ClientDisconnected(asyncio.CancelledError):
    """No response should be sent after the peer disconnects."""


@dataclass(eq=False)
class Job:
    request: ChatCompletionRequest
    prepared: PreparedPrompt
    options: dict
    submitted: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    native: Any = None
    stats: dict | None = None
    result: dict | None = None
    error: Exception | None = None
    cancel_reason: str | None = None


class ChatService:
    """The admission slot is retained until the actual native call returns."""

    def __init__(self, config: Settings, model=None, vision=None):
        self.config = config
        self.model = model
        self.vision = vision
        self.accepting = False
        self.active: Job | None = None
        self.pending: deque[Job] = deque()
        self.worker: asyncio.Task | None = None
        self.tokenizer = None
        self._last_job: Job | None = None

    @property
    def ready(self):
        return self.accepting and self.model is not None and (
            not self.config.VISION_MODEL_PATH or self.vision is not None
        )

    async def start(self):
        try:
            if self.config.TOKENIZER_PATH:
                from tokenizers import Tokenizer  # pylint: disable=import-outside-toplevel
                try:
                    self.tokenizer = Tokenizer.from_file(self.config.TOKENIZER_PATH)
                except Exception as error:  # Tokenizers reports file/format errors as Exception.
                    raise ValueError("Could not load TOKENIZER_PATH") from error
                self.tokenizer.no_truncation()
                self.tokenizer.no_padding()
            prepare_prompt(
                ChatCompletionRequest(
                    model=self.config.MODEL_NAME, messages=[{"role": "user", "content": "Hello"}]
                ),
                self.config, False,
            )
            if not self.config.eos_token_ids:
                logger.warning("EOS_TOKEN_IDS is unset; relying on the model's embedded EOS handling")
            if self.model is None:
                from app._native import LLM  # pylint: disable=import-outside-toplevel
                self.model = await asyncio.to_thread(
                    LLM, self.config.RKLLM_LIB_PATH, self.config.MODEL_PATH,
                    self.config.MAX_CONTEXT_LEN, self.config.MAX_NEW_TOKENS,
                    self.config.IGNORE_EOS_TOKEN, self.config.NATIVE_QUEUE_BYTES,
                    eos_token_ids=self.config.eos_token_ids,
                    skip_special_tokens=self.config.MODEL_PROTOCOL != "qwen35",
                    image_embedding_size=self.config.vision_embedding_size,
                    embed_flash=self.config.EMBED_FLASH,
                )
            if self.config.VISION_MODEL_PATH and self.vision is None:
                from app.vision import VisionEncoder  # pylint: disable=import-outside-toplevel
                self.vision = await asyncio.to_thread(
                    VisionEncoder, self.config.VISION_MODEL_PATH,
                    self.config.RKNN_LIB_PATH, self.config.MAX_IMAGE_PIXELS,
                    embedding_size=self.config.vision_embedding_size,
                )
        except (ImportError, OSError, ValueError, RuntimeError):
            logger.exception("Configured models failed to initialize")
            await self._close_models()
            raise
        self.accepting = True
        logger.info("Ready: model=%s profile=%s vision=%s queue_depth=%d embed_flash=%s kv=%s",
                    self.config.MODEL_NAME, self.config.MODEL_PROFILE,
                    self.vision is not None, self.config.QUEUE_DEPTH,
                    self.config.EMBED_FLASH, self.config.KV_CACHE_MODE)

    def _input_tokens(self, prompt, image=None):
        if self.tokenizer is None:
            return -1
        if image is not None:
            # RKLLM tokenizes multimodal prompts internally; UTF-8 bytes bound BPE text tokens.
            text = prompt.replace("<image>", "<|vision_start|><|vision_end|>")
            return len(text.encode("utf-8")) + self.vision.native.image_tokens
        return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)

    def submit(self, request, prepared):
        if not self.ready:
            raise ServiceError(503, "Configured models are not ready", "not_ready")
        if self.active is not None and len(self.pending) >= self.config.QUEUE_DEPTH:
            raise ServiceError(429, "Inference queue is full", "queue_full")
        limit = request.max_completion_tokens or request.max_tokens or self.config.MAX_NEW_TOKENS
        if limit > self.config.MAX_CONTEXT_LEN:
            raise ServiceError(400, "Token limit exceeds configured context", "invalid_token_limit")
        options = {"max_tokens": limit}
        count = self._input_tokens(prepared.prompt, prepared.image)
        if count >= 0:
            if count + limit > self.config.MAX_CONTEXT_LEN:
                raise ServiceError(400, "Prompt and output allowance exceed configured context",
                                   "context_length_exceeded")
            options["input_token_count"] = count
        for name in (
            "temperature", "top_p", "top_k", "repeat_penalty",
            "frequency_penalty", "presence_penalty",
        ):
            value = getattr(request, name)
            options[name] = getattr(self.config, name.upper()) if value is None else value
        job = Job(request, prepared, options)
        if self.active is None:
            self.active = job
            self.worker = asyncio.create_task(self._run_jobs(job))
        else:
            self.pending.append(job)
        return job

    def _infer(self, job):
        if job.cancelled.is_set():
            return
        options = dict(job.options)
        prompt, image = job.prepared.prompt, job.prepared.image
        previous, self._last_job = self._last_job, None
        mode = self.config.KV_CACHE_MODE
        reusable = (
            mode != "off" and previous is not None and previous.result is not None
            and previous.error is None and not previous.cancelled.is_set()
            and previous.result["finish_reason"] in ("stop", "tool_calls")
            and previous.stats["cache_valid"]
        )
        reuse = False
        if reusable:
            if mode == "prefix":
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
                previous_ids = self.tokenizer.encode(previous.prepared.prompt, add_special_tokens=False).ids
                # RKLLM can return stale logits on zero-prefill Gemma requests.
                reuse = current_ids != previous_ids[:len(current_ids)]
                if not reuse:
                    logger.info("KV reset: identical or shortened prefix")
            else:
                delta = continuation_prompt(previous, previous.result["message"], job, self.config)
                if delta is not None:
                    count = self._input_tokens(delta)
                    if previous.stats["cache_tokens"] + count + options["max_tokens"] <= self.config.MAX_CONTEXT_LEN:
                        prompt, image = delta, None
                        options["input_token_count"] = count
                        reuse = True
                    else:
                        logger.info("KV reset: retained reasoning/history exceeds context budget")
        if mode != "off":
            options.update(keep_history=mode == "stateful", reuse_cache=reuse)
        if self.tokenizer is not None and image is None:
            options["input_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if image is not None:
            encoded = self.vision.encode(image)
            options.update(
                image=encoded["embeddings"],
                width=encoded["image_width"], height=encoded["image_height"],
            )
        if job.cancelled.is_set():
            return
        generation = self.model.request(prompt, **options)
        job.native = generation
        # Cancellation may arrive between encoding, publishing the request, and run.
        if job.cancelled.is_set():
            generation.cancel()
        job.stats = generation.run()
        if mode != "off":
            self._last_job = job
        logger.info(
            "Inference: kv=%s reused=%s prompt_tokens=%d prefill_tokens=%d "
            "completion_tokens=%d prefill_ms=%.1f generate_ms=%.1f cancelled=%s",
            mode, reuse, job.stats["prompt_tokens"],
            job.stats["prefill_tokens"], job.stats["completion_tokens"],
            job.stats["prefill_ms"], job.stats["generate_ms"], job.stats["cancelled"],
        )

    async def _run_jobs(self, job):
        while job is not None:
            try:
                await asyncio.to_thread(self._infer, job)
            except Exception as error:  # A worker error is delivered to its HTTP consumer.
                logger.exception("Inference failed")
                job.error = error
            finally:
                job.done.set()
            job = self.pending.popleft() if self.pending else None
            self.active = job
        self.worker = None

    async def cancel(self, job, reason="client"):
        job.cancel_reason = job.cancel_reason or reason
        job.cancelled.set()
        if job.done.is_set():
            return
        if job in self.pending:
            self.pending.remove(job)
            job.done.set()
            return
        try:
            if job.native is not None:
                await asyncio.to_thread(job.native.cancel)
        finally:
            await job.done.wait()

    async def shutdown(self):
        self.accepting = False
        while self.pending:
            job = self.pending.popleft()
            job.cancel_reason = "shutdown"
            job.cancelled.set()
            job.done.set()
        worker = self.worker
        try:
            if self.active is not None:
                await self.cancel(self.active, "shutdown")
            if worker is not None:
                await worker
        finally:
            await self._close_models()

    async def _close_models(self):
        self._last_job = None
        self.tokenizer = None
        errors = []
        for model in (self.vision, self.model):
            if model is not None:
                try:
                    await asyncio.to_thread(model.close)
                except Exception as error:
                    logger.exception("Native cleanup failed")
                    errors.append(error)
        self.vision = self.model = None
        if errors:
            raise ExceptionGroup("Native resource cleanup failed", errors)

    async def events(self, job, http_request):
        decoder = codecs.getincrementaldecoder("utf-8")()
        stops = StopFilter(job.request.stops)
        parser = OutputParser(
            job.request.thinks, job.prepared.tools, qwen_tags=self.config.MODEL_PROTOCOL == "qwen35"
        )
        output_bytes = 0
        terminated = False
        try:
            while True:
                if await http_request.is_disconnected():
                    raise ClientDisconnected()
                if time.monotonic() - job.submitted > self.config.INFERENCE_TIMEOUT_SECONDS:
                    await self.cancel(job, "timeout")
                    raise ServiceError(504, "Inference deadline exceeded", "inference_timeout")
                chunks = job.native.read() if job.native is not None else []
                for chunk in chunks:
                    output_bytes += len(chunk)
                    if output_bytes > self.config.MAX_OUTPUT_BYTES:
                        raise OutputError("Native output exceeded MAX_OUTPUT_BYTES")
                    text = stops.feed(decoder.decode(chunk))
                    for event in parser.feed(text):
                        yield event
                    if stops.matched or (parser.ended and self.config.KV_CACHE_MODE == "off"):
                        terminated = True
                        await self.cancel(job, "stop")
                        break
                    if parser.ended:
                        terminated = True
                if (terminated and job.done.is_set()) or (job.done.is_set() and not chunks):
                    break
                if not chunks:
                    await asyncio.sleep(0.01)
            if job.error is not None:
                if isinstance(job.error, ValueError):
                    raise ServiceError(400, str(job.error), "invalid_input") from job.error
                raise OutputError(str(job.error)) from job.error
            if job.cancelled.is_set() and job.cancel_reason != "stop":
                raise ServiceError(503, "Inference was cancelled", "cancelled")
            if not terminated:
                for event in parser.feed(stops.feed(decoder.decode(b"", final=True))):
                    yield event
                for event in parser.feed(stops.finish()):
                    yield event
            for event in parser.finish(job.request):
                yield event
            if job.stats is None:
                raise OutputError("Native runtime returned no usage statistics")
            stats = job.stats
            finish_reason = "stop"
            if parser.calls:
                finish_reason = "tool_calls"
            elif not terminated and not stats["saw_eos"] and max(
                stats["generated_tokens"], stats["completion_tokens"] + 1
            ) >= job.options["max_tokens"]:
                finish_reason = "length"
            prompt_tokens, completion_tokens = stats["prompt_tokens"], stats["completion_tokens"]
            if prompt_tokens < 0 or completion_tokens < 0:
                raise OutputError("Native runtime returned invalid usage counters")
            job.result = {
                "message": parser.message(), "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        finally:
            if not job.done.is_set():
                await asyncio.shield(self.cancel(job))

# RKLLM API

OpenAI-compatible API for RKLLM models on RK3588-family devices.

**Exploration project: no support whatsoever.** Provided as-is, with no guarantees of compatibility, stability or correctness. Not intended for production use.

## Build

```sh
docker buildx build --platform linux/arm64 --load -t rkllm-api:local .
```

## Model profiles

Use an RK3588 `.rkllm` export compatible with RKLLM 1.3.0. GGUF and Hugging Face
weights cannot be loaded directly. Select the matching profile and restart the
container when changing models.

| `MODEL_PROFILE` | Default API model name | Features |
| --- | --- | --- |
| `qwen3.5-0.8b` | `Qwen3.5-0.8B-RK3588-W8A8` | Text, tools, reasoning, optional vision |
| `qwen3.5-2b` (default) | `Qwen3.5-2B-RK3588-W8A8` | Text, tools, reasoning, optional vision |
| `qwen3.5-4b` | `Qwen3.5-4B-RK3588-W8A8` | Text, tools, reasoning, optional vision |
| `gemma4-e2b` | `gemma-4-E2B-it` | Text only |
| `generic` | `rkllm-model` | Text with a custom `CHAT_TEMPLATE_PATH` |

Gemma 4 E2B:

```text
MODEL_PROFILE=gemma4-e2b
MODEL_PATH=/models/model.rkllm
```

Qwen4B with vision:

```text
MODEL_PROFILE=qwen3.5-4b
MODEL_PATH=/models/model.rkllm
VISION_MODEL_PATH=/models/vision.rknn
```

Vision requires a matching language model and encoder. Gemma4's
RKLLM export is text-only. Gemma E4B and larger variants have no built-in profile.

For other compatible models, use `MODEL_PROFILE=generic`, set `MODEL_NAME` and
provide the model's `CHAT_TEMPLATE_PATH`, `BOS_TOKEN`, `EOS_TOKEN` and
`EOS_TOKEN_IDS`. Generic mode supports text only. Explicit settings override
profile defaults.

### Model downloads

Download the matching files from their publisher and keep them outside this
repository. Use the publisher's checksums to verify downloads.

| Model | Source |
| --- | --- |
| Qwen3.5-2B and vision encoder | [RK3588 exports](https://huggingface.co/HanzoHuang/Qwen3.5-2B-RKLLM/tree/main/RK3588) |
| Qwen3.5-4B and vision encoder | [RK3588 exports](https://huggingface.co/HanzoHuang/Qwen3.5-4B-RKLLM/tree/main/RK3588) |
| Gemma 4 E2B | [RK3588 export](https://huggingface.co/HanzoHuang/gemma-4-E2B-it-RKLLM/tree/main/RK3588) |

Mount the model directory read-only as `/models`. Use `model.rkllm` and
`vision.rknn` as filenames, or configure their paths explicitly.

## Run

The host needs a working RKNPU driver and NPU device access. Rockchip documents
driver **0.9.8** as the minimum for RKLLM 1.3.0.

Provide NPU and DMA heap access through the container runtime, for example with
an existing CDI allocation. Device permissions must allow read/write access
for UID 10001. Set `NPU_CDI_DEVICE` to the registered CDI device identifier.

```sh
docker run --rm --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --device="${NPU_CDI_DEVICE:?Set your registered CDI device identifier}" \
  --memory="${MEMORY_LIMIT:-4g}" --cpus=4 --pids-limit=128 \
  -e MODEL_PROFILE="${MODEL_PROFILE:-qwen3.5-2b}" \
  --tmpfs=/tmp:size=64m,mode=1777 \
  --mount type=bind,src=/opt/rkllm/models,dst=/models,readonly \
  -p 127.0.0.1:8001:8001 rkllm-api:local
```

For Qwen vision, add `-e VISION_MODEL_PATH=/models/vision.rknn`.

Set `EMBED_FLASH=true` to read the model's embedding table from storage instead
of keeping it in RAM. This defaults to false; use fast local storage and measure
the memory/latency trade-off for your export.

### Optional KV reuse

Mount the matching original model's `tokenizer.json` read-only and set
`TOKENIZER_PATH=/models/tokenizer.json`. No tokenizer or model is downloaded at
runtime. Text requests use explicit tokenizer IDs and exact budget checks.
RKLLM tokenizes vision prompts internally, so admission conservatively budgets
UTF-8 text bytes plus image tokens; reported usage uses RKLLM's actual count.
Large vision prompts can therefore be rejected even when their tokenized form
would fit. This avoids silently overflowing the context.

| `KV_CACHE_MODE` | Behavior |
| --- | --- |
| `off` (default) | Clear native state for each request. |
| `prefix` | Gemma4 only: submit complete prompts and reuse matching prefixes when new input remains. Identical/shortened prompts reset to avoid stale zero-prefill results. |
| `stateful` | Built-in Qwen3.5 template only: append to the last verified conversation, including tools, thinking and same-image follow-ups. |

Clients still send complete message history. Stateful hits require unchanged
prior messages, assistant output (including `reasoning_content` and tool calls),
tool definitions, tool choice, thinking mode and image. A mismatch, failed or
unfinished response, cancellation, output limit or context-budget exhaustion
forces a fresh prefill. Only one conversation is resident; there is no disk
cache or multi-session cache.
Continuations that cannot render without earlier template history also fall
back to the complete submitted prompt.

Stateful reuse retains earlier reasoning in native memory, unlike a fresh
Qwen prompt which can omit older reasoning. This opt-in mode can therefore
change answers and consume more context. If retained state no longer fits,
the server resets and uses the complete submitted history. Custom templates
and `IGNORE_EOS_TOKEN=true` are not supported with stateful reuse.

For Gemma4, start with `EMBED_FLASH=true` and `KV_CACHE_MODE=prefix`.
The tested Gemma export is still limited to 4096 tokens. Qwen 16K operation
requires a 16K-compatible export; neither setting extends a compiled model.

Example for Qwen4B text with a compatible export:

```text
MODEL_PROFILE=qwen3.5-4b
MAX_CONTEXT_LEN=16384
EMBED_FLASH=true
KV_CACHE_MODE=stateful
TOKENIZER_PATH=/models/tokenizer.json
```

## API and configuration

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness |
| `GET /v1/models` | Loaded model name |
| `POST /v1/chat/completions` | Chat, optionally streamed |

```sh
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-2B-RK3588-W8A8","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

Use the model name returned by `/v1/models`.

- All profiles: streaming, sampling, penalties, stop sequences and cancellation.
- Qwen profiles: tools, separate reasoning and one base64 PNG/JPEG/WebP image.
- Tool-call history accepts an optional non-negative integer `index`; this transport metadata is excluded from prompts and cache matching.
- One active request; two waiting by default. A full queue returns 429.
- Reasoning and final content share the token limit; allow 1024-2048 tokens for thinking.
- Unsupported: remote image URLs, animation, video, strict tool decoding,
  logprobs, seed, LoRA and `n > 1`.
  Unknown fields, including top-level `seed`, are rejected.

Qwen defaults: `MAX_CONTEXT_LEN=4096`, `MAX_NEW_TOKENS=256`, `TEMPERATURE=0.6`,
`TOP_P=0.95`, `TOP_K=20`, `QUEUE_DEPTH=2`.
`MAX_CONTEXT_LEN`, `MAX_NEW_TOKENS`, and request `max_tokens` /
`max_completion_tokens` accept values up to 16384. Output limits must not exceed
the configured context; the prompt and generated output share that window.
Contexts above 4096 require a compatible larger-context RKLLM export: increasing
these settings does not extend a model compiled for 4096 tokens.
See [configuration](app/config.py) for all options. Environment variables override
an optional YAML file selected by `CONFIG_FILE`.

Set `API_KEY` to require bearer authentication on `/v1/*`. Keep the service on a
private network; the example binds to loopback only.

## Attribution

| Component | Source / attribution | License |
| --- | --- | --- |
| Qwen template | [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), Copyright 2026 Alibaba Cloud | [Apache-2.0](licenses/Apache-2.0) |
| Gemma template | [Gemma 4](https://huggingface.co/google/gemma-4-E2B-it), Google Gemma Engineering Team | [Apache-2.0](licenses/Apache-2.0) |
| Rockchip runtimes | [RKLLM and RKNN](https://github.com/airockchip/rknn-llm), including ggml/llama.cpp notices | [Redistribution terms](licenses/Rockchip-LICENSE) |

"""Model-selected chat formatting and incremental conversion to OpenAI deltas."""

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
import json
from pathlib import Path
import re
import uuid

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from app.config import Settings
from app.schema import ChatCompletionRequest, NamedToolChoice


class OutputError(RuntimeError):
    """The model returned an invalid or unsupported protocol construct."""


def _template_error(message):
    raise ValueError(message)


@lru_cache(maxsize=1)
def chat_template(template_path=None, protocol="qwen35"):
    if not template_path and protocol not in ("qwen35", "gemma4"):
        raise ValueError("This protocol requires CHAT_TEMPLATE_PATH")
    path = (
        Path(template_path).expanduser() if template_path
        else Path(__file__).parent / "templates" / f"{protocol}.jinja"
    )
    path = path.resolve(strict=True)
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"Chat template must be a non-empty file: {path}")
    environment = Environment(
        loader=FileSystemLoader(path.parent),
        undefined=StrictUndefined,
        autoescape=False,
    )
    environment.globals["raise_exception"] = _template_error
    environment.globals["strftime_now"] = lambda format_string: datetime.now().strftime(format_string)
    environment.filters["tojson"] = partial(json.dumps, ensure_ascii=False)
    try:
        return environment.get_template(path.name)
    except TemplateError as error:
        raise ValueError(f"Invalid chat template {path}: {error}") from error


def _render(config, messages, tools, thinking):
    try:
        prompt = chat_template(config.CHAT_TEMPLATE_PATH, config.MODEL_PROTOCOL).render(
            messages=messages, tools=tools, add_generation_prompt=True,
            enable_thinking=thinking, add_vision_id=False,
            bos_token=config.BOS_TOKEN, eos_token=config.EOS_TOKEN,
        )
    except TemplateError as error:
        raise ValueError(f"Chat template could not render these messages: {error}") from error
    if not prompt.strip() or "\0" in prompt:
        raise ValueError("Chat template must produce a non-empty prompt without NUL characters")
    return prompt


@dataclass
class PreparedPrompt:
    prompt: str
    image: bytes | None
    tools: dict


def prepare_prompt(request: ChatCompletionRequest, config: Settings, vision_enabled):
    """Copy history, associate tool results by ID, and adapt just the image marker."""
    messages = [message.model_dump(mode="json") for message in request.messages]
    if config.MODEL_PROTOCOL != "qwen35":
        if request.tools or request.tool_choice not in ("auto", "none"):
            raise ValueError("Tool calling currently requires the qwen35 protocol")
        if request.thinks:
            raise ValueError("Separate thinking currently requires the qwen35 protocol")
        for message in messages:
            if message["role"] == "tool" or message["tool_calls"] or message["reasoning_content"]:
                raise ValueError("Tool and reasoning history require the qwen35 protocol")
            if isinstance(message["content"], list):
                if any(part["type"] != "text" for part in message["content"]):
                    raise ValueError("Image input requires a Qwen3.5 profile with vision configured")
                message["content"] = "".join(part["text"] for part in message["content"])
            elif message["content"] is None:
                message["content"] = ""
        return PreparedPrompt(_render(config, messages, [], False), None, {})

    system_parts = []
    while messages and messages[0]["role"] in ("system", "developer"):
        content = messages.pop(0)["content"]
        system_parts.append(
            content if isinstance(content, str)
            else "".join(part["text"] for part in content or [])
        )
    if any(message["role"] in ("system", "developer") for message in messages):
        raise ValueError("System/developer messages must precede the conversation")

    history = []
    seen_ids = set()
    image = None
    index = 0
    while index < len(messages):
        message = messages[index]
        if message["role"] == "tool":
            raise ValueError("Tool result has no preceding assistant tool call")
        content = message["content"]
        if isinstance(content, list):
            for part in content:
                if part["type"] != "image_url":
                    continue
                if not vision_enabled:
                    raise ValueError("Image input requires VISION_MODEL_PATH")
                if image is not None:
                    raise ValueError("Only one image is supported across the entire history")
                url = part["image_url"]["url"]
                prefix, comma, payload = url.partition(",")
                if not comma or prefix not in (
                    "data:image/png;base64", "data:image/jpeg;base64",
                    "data:image/webp;base64",
                ):
                    raise ValueError("Use a base64 PNG, JPEG or WebP data URL; remote URLs are disabled")
                if len(payload) > 4 * ((config.MAX_IMAGE_BYTES + 2) // 3):
                    raise ValueError("Encoded image exceeds MAX_IMAGE_BYTES")
                try:
                    image = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError) as error:
                    raise ValueError("Invalid base64 image") from error
                if not image or len(image) > config.MAX_IMAGE_BYTES:
                    raise ValueError("Image is empty or exceeds MAX_IMAGE_BYTES")
        calls = message["tool_calls"] or []
        for call in calls:
            if call["id"] in seen_ids:
                raise ValueError("Duplicate tool_call_id in history")
            seen_ids.add(call["id"])
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        history.append(message)
        index += 1
        if calls:
            responses = {}
            while index < len(messages) and messages[index]["role"] == "tool":
                response = messages[index]
                if response["tool_call_id"] in responses:
                    raise ValueError("Duplicate tool result")
                responses[response["tool_call_id"]] = response
                index += 1
            if set(responses) != {call["id"] for call in calls}:
                raise ValueError("Each assistant tool call requires exactly one matching result")
            for call in calls:
                response = responses[call["id"]]
                if response["name"] not in (None, call["function"]["name"]):
                    raise ValueError("Tool result name does not match its call ID")
                history.append(response)

    tools = request.tools or []
    if request.tool_choice == "none":
        tools = []
    elif isinstance(request.tool_choice, NamedToolChoice):
        name = request.tool_choice.function.name
        tools = [tool for tool in tools if tool.function.name == name]
        system_parts.append(f"You must call the function {name}. Do not answer without calling it.")
    elif request.tool_choice == "required":
        system_parts.append("You must call one or more of the supplied functions. Do not answer without a function call.")
    if system_parts:
        history.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    tool_data = [tool.model_dump(exclude_none=True) for tool in tools]
    prompt = _render(config, history, tool_data, request.thinks)
    prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "<image>")
    if prompt.count("<image>") != (1 if image is not None else 0):
        raise ValueError("Literal <image> markers are reserved for image content parts")
    return PreparedPrompt(
        prompt, image, {tool.function.name: tool.function for tool in tools}
    )


class StopFilter:
    """Hold only the suffix that could become a stop sequence on the next read."""

    def __init__(self, stops):
        self.stops = stops
        self.pending = ""
        self.matched = False

    def feed(self, text):
        if self.matched:
            return ""
        self.pending += text
        positions = [self.pending.find(stop) for stop in self.stops]
        matches = [position for position in positions if position >= 0]
        if matches:
            result = self.pending[:min(matches)]
            self.pending = ""
            self.matched = True
            return result
        keep = 0
        for stop in self.stops:
            for size in range(1, min(len(stop), len(self.pending) + 1)):
                if self.pending.endswith(stop[:size]):
                    keep = max(keep, size)
        end = len(self.pending) - keep
        result, self.pending = self.pending[:end], self.pending[end:]
        return result

    def finish(self):
        result, self.pending = self.pending, ""
        return result


def _argument_value(raw, schema):
    value_text = raw[2:] if raw.startswith("\r\n") else raw.removeprefix("\n")
    value_text = value_text[:-2] if value_text.endswith("\r\n") else value_text.removesuffix("\n")
    expected = schema.get("type")
    allowed = expected if isinstance(expected, list) else [expected]
    # The official template renders scalar Python booleans/null with the string filter.
    literals = {"True": True, "False": False, "None": None}
    literal_type = {"True": "boolean", "False": "boolean", "None": "null"}.get(value_text)
    if value_text in literals and (expected is None or literal_type in allowed):
        value = literals[value_text]
    else:
        try:
            value = json.loads(value_text)
        except json.JSONDecodeError:
            value = value_text
    if expected == "string" and not isinstance(value, str):
        value = value_text
    types = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int or (type(item) is float and item.is_integer()),
        "number": lambda item: type(item) in (int, float),
        "boolean": lambda item: type(item) is bool,
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    if expected is not None and not any(
        name in types and types[name](value) for name in allowed
    ):
        raise OutputError(f"Tool argument does not match its declared type: {expected}")
    return value


class OutputParser:
    """Parse markers at any callback boundary; never expose partial tool syntax."""

    def __init__(self, thinking, tools, qwen_tags=True):
        self.qwen_tags = qwen_tags
        self.state = "reasoning_content" if thinking else "content"
        self.tools = tools
        self.pending = ""
        self.calls = []
        self.content = []
        self.reasoning = []
        self.ended = False

    def _emit(self, text, events):
        if text:
            (self.reasoning if self.state == "reasoning_content" else self.content).append(text)
            events.append({self.state: text})

    def _tool_call(self, body):
        match = re.fullmatch(r"\s*<function=([A-Za-z0-9_-]+)>(.*?)</function>\s*", body, re.S)
        if not match:
            raise OutputError("Malformed Qwen function call")
        name, parameters = match.groups()
        definition = self.tools.get(name)
        if definition is None:
            raise OutputError(f"Model called an unavailable tool: {name}")
        schema = definition.parameters
        properties = schema.get("properties", {})
        arguments = {}
        position = 0
        for parameter in re.finditer(r"<parameter=([^<>\r\n]+)>(.*?)</parameter>", parameters, re.S):
            if parameters[position:parameter.start()].strip():
                raise OutputError("Unexpected text inside a tool call")
            key, raw = parameter.groups()
            if key in arguments:
                raise OutputError(f"Duplicate tool argument: {key}")
            if key not in properties and schema.get("additionalProperties") is False:
                raise OutputError(f"Undeclared tool argument: {key}")
            arguments[key] = _argument_value(raw, properties.get(key, {}))
            position = parameter.end()
        if parameters[position:].strip():
            raise OutputError("Incomplete tool parameter")
        if not set(schema.get("required", [])).issubset(arguments):
            raise OutputError("Model omitted required tool arguments")
        try:
            serialized = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except ValueError as error:
            raise OutputError("Non-finite value in tool arguments") from error
        call = {
            "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": name, "arguments": serialized},
        }
        delta = {"tool_calls": [dict(call, index=len(self.calls))]}
        self.calls.append(call)
        return delta

    def feed(self, text):
        if self.ended:
            return []
        if not self.qwen_tags:
            events = []
            self._emit(text, events)
            return events
        self.pending += text
        events = []
        while self.pending:
            if self.state == "tool":
                position = self.pending.find("</tool_call>")
                if position < 0:
                    break
                events.append(self._tool_call(self.pending[:position]))
                self.pending = self.pending[position + len("</tool_call>"):]
                self.state = "content"
                continue
            markers = (
                ["</think>"] if self.state == "reasoning_content"
                else ["<think>", "<|im_end|>", "<|endoftext|>"] + (["<tool_call>"] if self.tools else [])
            )
            matches = [(self.pending.find(marker), marker) for marker in markers]
            matches = [(position, marker) for position, marker in matches if position >= 0]
            if matches:
                position, marker = min(matches)
                self._emit(self.pending[:position], events)
                self.pending = self.pending[position + len(marker):]
                if marker in ("<|im_end|>", "<|endoftext|>"):
                    self.ended = True
                    self.pending = ""
                elif marker == "<tool_call>":
                    self.state = "tool"
                else:
                    self.state = "reasoning_content" if marker == "<think>" else "content"
                continue
            keep = max(
                (size for marker in markers for size in range(1, len(marker))
                 if self.pending.endswith(marker[:size])),
                default=0,
            )
            end = len(self.pending) - keep
            self._emit(self.pending[:end], events)
            self.pending = self.pending[end:]
            break
        return events

    def finish(self, request):
        if self.state == "tool":
            raise OutputError("Generation ended inside a tool call; increase the token limit")
        events = []
        self._emit(self.pending, events)
        self.pending = ""
        if ((request.tool_choice == "required" or isinstance(request.tool_choice, NamedToolChoice))
                and not self.calls):
            raise OutputError("Model did not satisfy the required tool_choice")
        if not request.parallel_tool_calls and len(self.calls) > 1:
            raise OutputError("Model emitted parallel tool calls when disabled")
        return events

    def message(self):
        message = {"role": "assistant", "content": "".join(self.content)}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            message["tool_calls"] = self.calls
            if not message["content"]:
                message["content"] = None
        return message

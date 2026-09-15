"""The supported OpenAI request surface, validated before NPU admission."""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject unsupported options rather than silently ignoring them."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def invalid_json_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


class TextPart(StrictModel):
    type: Literal["text"]
    text: str


class ImageURL(StrictModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
FunctionName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class FunctionCall(StrictModel):
    name: FunctionName
    arguments: str

    @model_validator(mode="after")
    def check_arguments(self):
        try:
            value = json.loads(self.arguments, parse_constant=invalid_json_constant)
        except ValueError as error:
            raise ValueError("Tool arguments must be a JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return self


class ToolCall(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    reasoning_content: str | None = None
    name: FunctionName | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = Field(None, min_length=1, max_length=16)

    @model_validator(mode="after")
    def check_role_fields(self):
        if self.role != "assistant" and (self.tool_calls or self.reasoning_content):
            raise ValueError("Only assistant messages may contain reasoning/tool calls")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required only for tool messages")
        content = self.content
        if self.role != "user" and isinstance(content, list):
            if any(isinstance(part, ImagePart) for part in content):  # pylint: disable=not-an-iterable
                raise ValueError("Only user messages may contain images")
        if self.name is not None and self.role != "tool":
            raise ValueError("Named non-tool messages are not supported by this template")
        return self


class FunctionDefinition(StrictModel):
    name: FunctionName
    description: str | None = None
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    strict: bool = False

    @model_validator(mode="after")
    def check_schema(self):
        if self.strict:
            raise ValueError("RKLLM does not support strict/constrained tool decoding")
        if self.parameters.get("type", "object") != "object":
            raise ValueError("Function parameters must describe an object")
        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("Invalid function parameter properties/required")
        for name, schema in properties.items():
            if not isinstance(name, str) or not name or any(character in name for character in "<>\r\n\0"):
                raise ValueError("Parameter names cannot contain native marker delimiters")
            if not isinstance(schema, dict):
                raise ValueError("Each parameter schema must be an object")
            expected = schema.get("type")
            types = expected if isinstance(expected, list) else [expected]
            if expected is not None and any(
                not isinstance(value, str) or value not in (
                    "string", "number", "integer", "boolean", "object", "array", "null"
                ) for value in types
            ):
                raise ValueError("Invalid parameter type")
        if any(not isinstance(name, str) or name not in properties for name in required):
            raise ValueError("Required parameters must exist in properties")
        return self


class ToolDefinition(StrictModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedFunction(StrictModel):
    name: FunctionName


class NamedToolChoice(StrictModel):
    type: Literal["function"]
    function: NamedFunction


class ThinkingOptions(StrictModel):
    type: Literal["enabled", "disabled"]


class StreamOptions(StrictModel):
    include_usage: bool = False


class ChatCompletionRequest(StrictModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: Literal[1] = 1
    tools: list[ToolDefinition] | None = Field(None, max_length=32)
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice = "auto"
    parallel_tool_calls: bool = True
    max_tokens: int | None = Field(None, ge=1, le=4096)
    max_completion_tokens: int | None = Field(None, ge=1, le=4096)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=1)
    repeat_penalty: float | None = Field(None, gt=0, le=2)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    stop: str | list[str] | None = None
    enable_thinking: bool | None = None
    thinking: bool | ThinkingOptions | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None

    @property
    def thinks(self):
        flags = []
        if self.enable_thinking is not None:
            flags.append(self.enable_thinking)
        if self.thinking is not None:
            flags.append(
                self.thinking if isinstance(self.thinking, bool)
                else self.thinking.type == "enabled"
            )
        if self.reasoning_effort is not None:
            flags.append(self.reasoning_effort != "none")
        if flags and any(flag != flags[0] for flag in flags):
            raise ValueError("Conflicting thinking options")
        return flags[0] if flags else False

    @property
    def stops(self):
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop or []

    @model_validator(mode="after")
    def check_options(self):
        if (self.max_tokens is not None and self.max_completion_tokens is not None
                and self.max_tokens != self.max_completion_tokens):
            raise ValueError("Conflicting token limits")
        _ = self.thinks
        if len(self.stops) > 4 or any(not stop or len(stop) > 256 for stop in self.stops):
            raise ValueError("Provide at most four non-empty stop strings (max 256 chars)")
        names = [tool.function.name for tool in self.tools or []]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate tool names")
        if self.tool_choice == "required" and not names:
            raise ValueError("tool_choice=required needs tools")
        if isinstance(self.tool_choice, NamedToolChoice):
            if self.tool_choice.function.name not in names:
                raise ValueError("Named tool_choice must refer to a supplied tool")
        if self.messages[-1].role not in ("user", "tool"):
            raise ValueError("The final message must be a user message or tool result")
        return self

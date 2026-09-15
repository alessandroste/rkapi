"""Environment-first configuration for one RK3588 inference worker."""

import os
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import yaml
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProfile(NamedTuple):
    protocol: str
    name: str
    vision_width: int


PROFILES = {
    "qwen3.5-0.8b": ModelProfile("qwen35", "Qwen3.5-0.8B-RK3588-W8A8", 1024),
    "qwen3.5-2b": ModelProfile("qwen35", "Qwen3.5-2B-RK3588-W8A8", 2048),
    "qwen3.5-4b": ModelProfile("qwen35", "Qwen3.5-4B-RK3588-W8A8", 2560),
    "gemma4-e2b": ModelProfile("gemma4", "gemma-4-E2B-it", 0),
    "generic": ModelProfile("generic", "rkllm-model", 0),
}


def yaml_settings():
    """An explicitly configured but missing file is a startup error."""
    path = Path(os.environ.get("CONFIG_FILE", "config.yaml"))
    if not path.exists() and "CONFIG_FILE" not in os.environ:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return data


class Settings(BaseSettings):
    """Paths and limits are configurable without writing inside the image."""

    model_config = SettingsConfigDict(extra="forbid")

    MODEL_PATH: str = "/models/model.rkllm"
    MODEL_NAME: str = "Qwen3.5-2B-RK3588-W8A8"
    MODEL_PROFILE: str = ""
    MODEL_PROTOCOL: Literal["qwen35", "gemma4", "generic"] = "qwen35"
    CHAT_TEMPLATE_PATH: str | None = None
    BOS_TOKEN: str = ""
    EOS_TOKEN: str = ""
    EOS_TOKEN_IDS: list[Annotated[int, Field(strict=True, ge=0, le=2147483647)]] | None = Field(
        None, max_length=32
    )
    RKLLM_LIB_PATH: str = "/runtime/librkllmrt.so"
    VISION_MODEL_PATH: str | None = None
    RKNN_LIB_PATH: str = "/runtime/librknnrt.so"
    MAX_CONTEXT_LEN: int = Field(4096, ge=128, le=4096)
    MAX_NEW_TOKENS: int = Field(256, ge=1, le=4096)
    IGNORE_EOS_TOKEN: bool = False
    TEMPERATURE: float = Field(0.6, ge=0, le=2, allow_inf_nan=False)
    TOP_P: float = Field(0.95, gt=0, le=1, allow_inf_nan=False)
    TOP_K: int = Field(20, ge=1)
    REPEAT_PENALTY: float = Field(1.1, gt=0, le=2, allow_inf_nan=False)
    FREQUENCY_PENALTY: float = Field(0, ge=-2, le=2, allow_inf_nan=False)
    PRESENCE_PENALTY: float = Field(0, ge=-2, le=2, allow_inf_nan=False)
    QUEUE_DEPTH: int = Field(2, ge=0, le=32)
    INFERENCE_TIMEOUT_SECONDS: float = Field(300, gt=0, allow_inf_nan=False)
    MAX_REQUEST_BYTES: int = Field(8 * 1024 * 1024, ge=1024, le=32 * 1024 * 1024)
    MAX_IMAGE_BYTES: int = Field(5 * 1024 * 1024, ge=1, le=16 * 1024 * 1024)
    MAX_IMAGE_PIXELS: int = Field(16_000_000, ge=1, le=32_000_000)
    NATIVE_QUEUE_BYTES: int = Field(65536, ge=1024, le=1024 * 1024)
    MAX_OUTPUT_BYTES: int = Field(256 * 1024, ge=1024, le=4 * 1024 * 1024)
    HOST: str = "0.0.0.0"
    PORT: int = Field(8001, ge=1, le=65535)
    API_KEY: SecretStr | None = None
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def eos_token_ids(self):
        if self.EOS_TOKEN_IDS is not None:
            return self.EOS_TOKEN_IDS
        return {"qwen35": [248044], "gemma4": [1, 106, 50], "generic": []}[self.MODEL_PROTOCOL]

    @property
    def vision_embedding_size(self):
        return PROFILES[self.MODEL_PROFILE].vision_width

    @model_validator(mode="after")
    def check_limits(self):
        explicit = self.model_fields_set.copy()
        selected = self.MODEL_PROFILE or {
            "qwen35": "qwen3.5-2b", "gemma4": "gemma4-e2b", "generic": "generic",
        }[self.MODEL_PROTOCOL]
        if selected not in PROFILES:
            raise ValueError(f"Unknown MODEL_PROFILE; choose from {', '.join(PROFILES)}")
        profile = PROFILES[selected]
        if "MODEL_PROTOCOL" in explicit and self.MODEL_PROTOCOL != profile.protocol:
            raise ValueError("MODEL_PROFILE and MODEL_PROTOCOL conflict")
        self.MODEL_PROFILE = selected
        self.MODEL_PROTOCOL = profile.protocol
        if "MODEL_NAME" not in explicit:
            self.MODEL_NAME = profile.name
        if profile.protocol == "gemma4":
            for name, value in {
                "BOS_TOKEN": "<bos>", "EOS_TOKEN": "<eos>",
                "TEMPERATURE": 1.0, "TOP_K": 64, "REPEAT_PENALTY": 1.0,
            }.items():
                if name not in explicit:
                    setattr(self, name, value)
        if self.MAX_NEW_TOKENS > self.MAX_CONTEXT_LEN:
            raise ValueError("MAX_NEW_TOKENS must not exceed MAX_CONTEXT_LEN")
        if self.VISION_MODEL_PATH == "":
            self.VISION_MODEL_PATH = None
        if self.CHAT_TEMPLATE_PATH == "":
            self.CHAT_TEMPLATE_PATH = None
        if self.MODEL_PROTOCOL == "generic":
            if not self.CHAT_TEMPLATE_PATH:
                raise ValueError("Generic models require CHAT_TEMPLATE_PATH")
        if self.VISION_MODEL_PATH and not self.vision_embedding_size:
            raise ValueError("Vision requires a Qwen3.5 model profile and its matching encoder")
        if self.API_KEY is not None and not self.API_KEY.get_secret_value():
            raise ValueError("API_KEY must not be empty when configured")
        return self

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings,
        file_secret_settings,
    ):
        return init_settings, env_settings, yaml_settings, file_secret_settings


settings = Settings()

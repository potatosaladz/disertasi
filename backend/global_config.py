from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Agent, GlobalLLMConfig


def get_global_llm_config(session: Session) -> GlobalLLMConfig:
    config = session.scalar(select(GlobalLLMConfig).where(GlobalLLMConfig.id == 1))
    if config is None:
        config = GlobalLLMConfig(id=1)
        session.add(config)
        session.flush()
    return config


def apply_global_llm_config(agent: Agent, config: GlobalLLMConfig) -> None:
    if not config.apply_to_all:
        return
    agent.llm_base_url = config.llm_base_url
    agent.llm_api_key = config.llm_api_key
    agent.llm_model = config.llm_model
    agent.temperature = config.temperature
    agent.max_tokens = config.max_tokens


def apply_global_values(values: dict[str, Any], config: GlobalLLMConfig) -> dict[str, Any]:
    if not config.apply_to_all:
        return values
    return {
        **values,
        "llm_base_url": config.llm_base_url,
        "llm_api_key": config.llm_api_key,
        "llm_model": config.llm_model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def global_config_snapshot(config: GlobalLLMConfig) -> dict[str, Any]:
    return {
        "revision": config.revision,
        "apply_to_all": config.apply_to_all,
        "llm_base_url": config.llm_base_url,
        "llm_model": config.llm_model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "has_llm_api_key": bool(config.llm_api_key),
    }

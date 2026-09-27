"""Interfaces for the revised attention-sink inheritance study."""

from importlib import import_module
from typing import Any

from .config import ConfigError, RunSpec, resolve_config

_LAZY_EXPORTS = {
    "AttentionIntervention": (".interventions", "AttentionIntervention"),
    "InterventionError": (".interventions", "InterventionError"),
    "AttentionFeatures": (".models", "AttentionFeatures"),
    "FeatureForward": (".models", "FeatureForward"),
    "GPT2Adapter": (".models", "GPT2Adapter"),
    "GPTNeoXAdapter": (".models", "GPTNeoXAdapter"),
    "ModelAdapterError": (".models", "ModelAdapterError"),
    "ModelShape": (".models", "ModelShape"),
    "adapt_causal_lm": (".models", "adapt_causal_lm"),
}


def __getattr__(name: str) -> Any:
    """Keep foundation-only wheel imports independent of the model stack."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "AttentionFeatures",
    "AttentionIntervention",
    "ConfigError",
    "FeatureForward",
    "GPT2Adapter",
    "GPTNeoXAdapter",
    "InterventionError",
    "ModelAdapterError",
    "ModelShape",
    "RunSpec",
    "adapt_causal_lm",
    "resolve_config",
]

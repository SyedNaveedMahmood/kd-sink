"""Interfaces for the revised attention-sink inheritance study."""

from .config import ConfigError, RunSpec, resolve_config
from .interventions import AttentionIntervention, InterventionError
from .models import (
    AttentionFeatures,
    FeatureForward,
    GPT2Adapter,
    GPTNeoXAdapter,
    ModelAdapterError,
    ModelShape,
    adapt_causal_lm,
)

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

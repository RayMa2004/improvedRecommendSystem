"""Switchable baseline/improved method registry."""

from .registry import ACTIVE_METHODS, BASELINE_METHODS, HISTORY_IMPROVEMENT_METHODS, build_method_registry

__all__ = ["ACTIVE_METHODS", "BASELINE_METHODS", "HISTORY_IMPROVEMENT_METHODS", "build_method_registry"]

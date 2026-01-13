"""System Version: 1.1.11
File Version: 1.0.0"""
from __future__ import annotations

from . import routes as _routes


__all__ = []


def __getattr__(name: str):
    return getattr(_routes, name)

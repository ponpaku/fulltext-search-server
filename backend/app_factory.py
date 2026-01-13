from __future__ import annotations

from fastapi import FastAPI

from .routes import app as routes_app

SYSTEM_VERSION = "1.1.11"
# File Version: 1.0.0


def create_app() -> FastAPI:
    return routes_app

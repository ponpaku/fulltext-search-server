"""ASGI entrypoint."""

# System Version: 1.1.11
# File Version: 1.0.0

from backend.routes import create_app

app = create_app()

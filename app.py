"""System Version: 1.1.11
File Version: 1.0.0"""
from __future__ import annotations

import os
from pathlib import Path

from backend import create_app

app = create_app()


def run() -> None:
    """ローカル開発用エントリポイント."""
    import signal
    import uvicorn

    port = int(os.getenv("PORT", "80"))
    cert_dir = os.getenv("CERT_DIR", "certs")
    base_dir = Path(__file__).resolve().parent
    cert_path = (base_dir / cert_dir).resolve()
    cert_file = cert_path / "lan-cert.pem"
    key_file = cert_path / "lan-key.pem"

    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs = {
            "ssl_certfile": str(cert_file),
            "ssl_keyfile": str(key_file),
        }

    config = uvicorn.Config(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        **ssl_kwargs,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    interrupt_count = {"count": 0}

    def handle_interrupt(signum, frame):
        interrupt_count["count"] += 1
        if interrupt_count["count"] == 1:
            print("Ctrl+Cを検知しました。もう一度Ctrl+Cで強制終了します。")
            server.should_exit = True
            return
        print("Ctrl+Cを再度検知しました。強制終了します。")
        os._exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        signal.signal(signal.SIGTERM, handle_interrupt)
    except Exception:
        pass

    server.run()


if __name__ == "__main__":
    run()

from backend.routes import create_app, run

SYSTEM_VERSION = "1.1.11"
# File Version: 1.0.1

app = create_app()


if __name__ == "__main__":
    run()

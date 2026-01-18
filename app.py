from backend.routes import create_app, run

SYSTEM_VERSION = "1.3.0"

app = create_app()


if __name__ == "__main__":
    run()

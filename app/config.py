"""App-level settings."""

import os

CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data", "results")

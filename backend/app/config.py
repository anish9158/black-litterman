import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    env: str
    cors_origins: list[str]
    groq_api_key: str
    groq_model: str
    max_request_size_bytes: int
    rate_limit_per_minute: int

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


def load_settings() -> Settings:
    env = os.getenv("APP_ENV", "development")
    cors_value = os.getenv("CORS_ORIGINS", "")
    cors_origins = _split_csv(cors_value) if cors_value else ["http://localhost:5173"]
    return Settings(
        env=env,
        cors_origins=cors_origins,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        max_request_size_bytes=int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(1_048_576))),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
    )

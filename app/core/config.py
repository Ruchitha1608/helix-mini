from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    CARTESIA_API_KEY: str = ""

    DEEPGRAM_MODEL: str = "nova-2-medical"
    CARTESIA_VOICE_ID: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    CARTESIA_MODEL: str = "sonic-2"
    CLAUDE_MODEL: str = "claude-haiku-4-5-20251001"

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

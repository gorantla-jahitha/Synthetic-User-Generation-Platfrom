import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"

def reload_config():
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # If .env does not exist, pop cached keys if they were loaded from .env
        pass

class Settings:
    PROJECT_NAME: str = "Synthetic Customer Research Platform"
    VERSION: str = "1.0.0"
    
    @property
    def GEMINI_API_KEY(self) -> str:
        reload_config()
        return os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()

    @property
    def OPENROUTER_API_KEY(self) -> str:
        reload_config()
        return os.getenv("OPENROUTER_API_KEY", "").strip()

    @property
    def MONGODB_URI(self) -> str:
        return os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    @property
    def DATABASE_NAME(self) -> str:
        return os.getenv("DATABASE_NAME", "synthetic_customer_research")

    @property
    def DEFAULT_LLM_PROVIDER(self) -> str:
        return os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

    @property
    def GEMINI_MODEL(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

settings = Settings()

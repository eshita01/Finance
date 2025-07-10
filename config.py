import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@lru_cache()
def get_api_key(env_path: Optional[str] = None) -> str:
    """Load Gemini API key from .env file."""
    env_file = env_path or Path(__file__).resolve().parent / '.env'
    if Path(env_file).exists():
        load_dotenv(env_file)
    from os import getenv
    key = getenv('GEMINI_API_KEY')
    if not key:
        raise ValueError('GEMINI_API_KEY not found in environment')
    return key


@lru_cache()
def get_alpha_vantage_key(env_path: Optional[str] = None) -> str:
    """Load Alpha Vantage API key from .env file."""
    env_file = env_path or Path(__file__).resolve().parent / '.env'
    if Path(env_file).exists():
        load_dotenv(env_file)
    from os import getenv
    key = getenv('ALPHAVANTAGE_API_KEY')
    if not key:
        raise ValueError('ALPHAVANTAGE_API_KEY not found in environment')
    return key

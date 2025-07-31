import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .sentiment_analysis import analyze

logger = logging.getLogger(__name__)


def analyze_with_cache(
    ticker: str,
    feed: List[Dict[str, Any]],
    base_date: datetime,
    cache_dir: str | Path = "results/news_analysis",
) -> Dict[str, Any]:
    """Return sentiment analysis using a cache to avoid recomputation."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{ticker}_{base_date.strftime('%Y%m%d')}.json"

    if file_path.exists():
        try:
            return json.loads(file_path.read_text())
        except Exception:
            logger.exception("Failed reading cached sentiment %s", file_path)

    result = analyze(feed)
    try:
        file_path.write_text(json.dumps(result, indent=2))
    except Exception:
        logger.exception("Failed writing cached sentiment %s", file_path)
    return result


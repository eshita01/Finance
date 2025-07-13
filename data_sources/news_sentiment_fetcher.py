import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NewsSentimentFetcher:
    """Fetch news and sentiment data using the Alpha Vantage API."""

    def __init__(self, tickers: List[str], api_key: str, base_date: Optional[datetime] = None):
        self.tickers = tickers
        self.api_key = api_key
        self.base_date = base_date or datetime.utcnow()

    def fetch(self) -> List[Dict[str, Any]]:
        """Return a list of news articles with sentiment information."""
        try:
            ticker_str = ",".join(self.tickers)
            logger.info("Fetching news sentiment for %s", ticker_str)
            time_from = (self.base_date - timedelta(days=1)).strftime("%Y%m%dT%H%M")
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker_str,
                "sort": "LATEST",
                "limit": 50,
                "time_from": time_from,
                "apikey": self.api_key,
            }
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            feed = data.get("feed", [])
            return feed
        except Exception as e:
            logger.exception("Failed to fetch news sentiment: %s", e)
            raise

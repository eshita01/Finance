import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import finnhub

logger = logging.getLogger(__name__)


class PeerDataFetcher:
    """Fetch peer tickers and related data."""

    def __init__(self, ticker: str, api_key: str, base_date: Optional[datetime] = None, limit: int = 3):
        self.ticker = ticker
        self.api_key = api_key
        self.base_date = base_date or datetime.utcnow()
        self.limit = limit
        self.client = finnhub.Client(api_key=api_key)

    def fetch(self) -> Dict[str, Any]:
        """Return peer list, recent price data and news sentiment."""
        peers: List[str] = []
        try:
            peers = self.client.company_peers(self.ticker)[: self.limit]
        except Exception as e:
            logger.exception("Error fetching peers: %s", e)

        price_data: Dict[str, pd.DataFrame] = {}
        news: Dict[str, Any] = {}
        for peer in peers:
            try:
                start = (self.base_date - timedelta(days=10)).strftime("%Y-%m-%d")
                end = self.base_date.strftime("%Y-%m-%d")
                df = yf.download(peer, start=start, end=end, interval="1d", auto_adjust=True)
                price_data[peer] = df
            except Exception as e:
                logger.exception("Error fetching prices for %s: %s", peer, e)
                price_data[peer] = pd.DataFrame()

            try:
                news[peer] = self.client.news_sentiment(peer)
            except Exception as e:
                logger.exception("Error fetching news sentiment for %s: %s", peer, e)
                news[peer] = {}

        return {"peers": peers, "price_data": price_data, "news": news}

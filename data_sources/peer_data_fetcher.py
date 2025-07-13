import logging
from datetime import datetime, timedelta, timezone

from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import finnhub
import requests

logger = logging.getLogger(__name__)


class PeerDataFetcher:
    """Fetch peer tickers and related data."""

    def __init__(
        self,
        ticker: str,

        finnhub_key: str,
        alpha_key: str,
        base_date: Optional[datetime] = None,
        limit: int = 3,
    ):
        self.ticker = ticker

        self.finnhub_key = finnhub_key
        self.alpha_key = alpha_key
        self.base_date = base_date or datetime.now(timezone.utc)


        self.limit = limit
        self.client = finnhub.Client(api_key=finnhub_key)

    def _fetch_news_score(self, ticker: str) -> Dict[str, Any]:
        """Fetch news sentiment score for a single ticker using Alpha Vantage."""
        try:
            time_from = (self.base_date - timedelta(days=1)).strftime("%Y%m%dT%H%M")
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "sort": "LATEST",
                "limit": 50,
                "time_from": time_from,
                "apikey": self.alpha_key,
            }
            resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            feed = data.get("feed", [])
            scores = []
            for item in feed:
                for tdata in item.get("ticker_sentiment", []):
                    if tdata.get("ticker") == ticker:
                        try:
                            scores.append(float(tdata.get("ticker_sentiment_score")))
                        except (TypeError, ValueError):
                            continue
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return {"companyNewsScore": avg_score}
        except Exception as e:
            logger.exception("Error fetching news sentiment for %s: %s", ticker, e)
            return {}

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
                df = yf.download(
                    peer, start=start, end=end, interval="1d", auto_adjust=True
                )

                price_data[peer] = df
            except Exception as e:
                logger.exception("Error fetching prices for %s: %s", peer, e)
                price_data[peer] = pd.DataFrame()

            news[peer] = self._fetch_news_score(peer)

        return {"peers": peers, "price_data": price_data, "news": news}

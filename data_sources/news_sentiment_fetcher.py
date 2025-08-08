import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NewsSentimentFetcher:
    """Fetch news and sentiment data.

    Tries Alpha Vantage first and falls back to Finnhub + HuggingFace
    sentiment analysis when Alpha Vantage fails or returns no data.
    """

    def __init__(
        self,
        tickers: List[str],
        alpha_key: str,
        finnhub_key: Optional[str] = None,
        hf_key: Optional[str] = None,
        base_date: Optional[datetime] = None,
        cache_dir: Optional[str] = None,
    ):
        self.tickers = tickers
        self.alpha_key = alpha_key
        self.finnhub_key = finnhub_key
        self.hf_key = hf_key
        self.base_date = base_date or datetime.utcnow()
        self.cache_dir = Path(cache_dir or "data/news_sentiment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Return sentiment score and label for text using HuggingFace."""
        if not self.hf_key:
            return {"overall_sentiment_score": 0.0, "overall_sentiment_label": "neutral"}
        try:
            headers = {"Authorization": f"Bearer {self.hf_key}"}
            resp = requests.post(
                "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert",
                headers=headers,
                json={"inputs": text},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            pos = neg = 0.0
            top_label = "neutral"
            top_score = 0.0
            for item in result:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))
                if label == "positive":
                    pos = score
                elif label == "negative":
                    neg = score
                if score > top_score:
                    top_score = score
                    top_label = label
            overall = pos - neg
            return {
                "overall_sentiment_score": overall,
                "overall_sentiment_label": top_label,
                "relevance_score": 1.0,
            }
        except Exception:
            logger.exception("HuggingFace sentiment analysis failed")
            return {"overall_sentiment_score": 0.0, "overall_sentiment_label": "neutral", "relevance_score": 1.0}

    def _fetch_alpha(self, ticker_str: str) -> List[Dict[str, Any]]:
        time_from = (self.base_date - timedelta(days=1)).strftime("%Y%m%dT%H%M")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker_str,
            "sort": "LATEST",
            "limit": 50,
            "time_from": time_from,
            "apikey": self.alpha_key,
        }
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        feed = data.get("feed", [])
        # Alpha Vantage returns "Note" when rate limited
        if not feed:
            raise ValueError("Empty Alpha Vantage feed")
        return feed

    def _fetch_finnhub(self, ticker_str: str) -> List[Dict[str, Any]]:
        if not self.finnhub_key:
            return []
        frm = (self.base_date - timedelta(days=1)).strftime("%Y-%m-%d")
        to = self.base_date.strftime("%Y-%m-%d")
        params = {
            "symbol": ticker_str,
            "from": frm,
            "to": to,
            "token": self.finnhub_key,
        }
        resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()[:50]
        feed: List[Dict[str, Any]] = []
        for art in articles:
            title = art.get("headline")
            if not title:
                continue
            sentiment = self._analyze_text(title)
            feed.append({"title": title, **sentiment})
        return feed

    def fetch(self) -> List[Dict[str, Any]]:
        """Return a list of news articles with sentiment information."""
        ticker_str = ",".join(self.tickers)
        date_str = self.base_date.strftime("%Y%m%d")
        cache_file = self.cache_dir / f"{ticker_str}_{date_str}.json"

        if cache_file.exists():
            logger.info("Using cached news sentiment for %s", ticker_str)
            return json.loads(cache_file.read_text())

        # Try Alpha Vantage first
        try:
            logger.info("Fetching news sentiment for %s via Alpha Vantage", ticker_str)
            feed = self._fetch_alpha(ticker_str)
        except Exception as e:
            logger.warning("Alpha Vantage failed: %s", e)
            feed = self._fetch_finnhub(ticker_str)

        try:
            cache_file.write_text(json.dumps(feed))
        except Exception:
            logger.exception("Failed writing news cache file %s", cache_file)
        return feed

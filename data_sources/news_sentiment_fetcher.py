import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import requests

logger = logging.getLogger(__name__)


class NewsSentimentFetcher:
    """Fetch news and sentiment data.

    Tries Alpha Vantage first and falls back to Finnhub + Gemini sentiment
    analysis when Alpha Vantage fails or returns no data.
    """

    def __init__(
        self,
        tickers: List[str],
        alpha_key: str,
        finnhub_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        base_date: Optional[datetime] = None,
        cache_dir: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash-lite",
    ):
        self.tickers = tickers
        self.alpha_key = alpha_key
        self.finnhub_key = finnhub_key
        self.base_date = base_date or datetime.utcnow()
        self.cache_dir = Path(cache_dir or "data/news_sentiment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.gemini_model_name = gemini_model
        self._gemini: Optional[genai.GenerativeModel] = None
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini = genai.GenerativeModel(gemini_model)

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Return sentiment score and label for text using Gemini."""
        if not self._gemini:
            raise ValueError("Gemini API key not configured")

        prompt = (
            "Classify the sentiment of the following news headline as positive, negative, or "
            "neutral. Respond with JSON containing keys label and score (between -1 and 1).\n"
            f"Headline: {text}"
        )
        try:
            resp = self._gemini.generate_content(prompt)
            logger.debug("Gemini raw response: %s", getattr(resp, "text", resp))
            data = self._parse_gemini_json(resp.text)

            label = str(data.get("label", "neutral")).lower()
            score = float(data.get("score", 0.0))
            return {
                "overall_sentiment_score": score,
                "overall_sentiment_label": label,
                "relevance_score": 1.0,
            }
        except Exception as exc:
            logger.exception("Gemini sentiment analysis failed: %s", exc)
            raise

    @staticmethod
    def _parse_gemini_json(text: str) -> Dict[str, Any]:
        """Extract and parse a JSON object from Gemini response text."""
        if not text:
            raise ValueError("Empty Gemini response")
        # Strip Markdown code fences like ```json ... ```
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove optional language hint e.g. json
            cleaned = cleaned.split(None, 1)[-1] if " " in cleaned else cleaned
        # Find first JSON object in text
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise ValueError("No JSON object found in Gemini response")
        return json.loads(match.group())

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
            raise ValueError("Finnhub API key not configured")
        frm = (self.base_date - timedelta(days=1)).strftime("%Y-%m-%d")
        to = self.base_date.strftime("%Y-%m-%d")
        params = {
            "symbol": ticker_str,
            "from": frm,
            "to": to,
            "token": self.finnhub_key,
        }
        try:
            resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json()[:50]
        except Exception as exc:
            logger.exception("Finnhub request failed: %s", exc)
            raise

        if not articles:
            raise ValueError("Empty Finnhub news feed")

        feed: List[Dict[str, Any]] = []
        for art in articles:
            title = art.get("headline")
            if not title:
                continue
            sentiment = self._analyze_text(title)
            feed.append({"title": title, **sentiment})
        if not feed:
            raise ValueError("No analyzable Finnhub articles")
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
            logger.info("News sentiment fetched from Alpha Vantage")
        except Exception as e:
            logger.warning("Alpha Vantage failed: %s", e)
            logger.info("Fetching news sentiment for %s via Finnhub + Gemini", ticker_str)
            try:
                feed = self._fetch_finnhub(ticker_str)
                logger.info("News sentiment fetched from Finnhub + Gemini")
            except Exception as exc:
                logger.exception("Finnhub + Gemini failed: %s", exc)
                raise

        try:
            cache_file.write_text(json.dumps(feed))
        except Exception:
            logger.exception("Failed writing news cache file %s", cache_file)
        return feed

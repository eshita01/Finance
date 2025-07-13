import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import finnhub

logger = logging.getLogger(__name__)


class InsiderDataFetcher:
    """Fetch insider transactions and sentiment data from Finnhub."""

    def __init__(self, ticker: str, api_key: str, base_date: Optional[datetime] = None, lookback_days: int = 90):
        self.ticker = ticker
        self.api_key = api_key
        self.base_date = base_date or datetime.utcnow()
        self.lookback_days = lookback_days
        self.client = finnhub.Client(api_key=api_key)

    def fetch(self) -> Dict[str, Any]:
        start = (self.base_date - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        end = self.base_date.strftime("%Y-%m-%d")

        try:
            transactions = self.client.stock_insider_transactions(
                self.ticker, start, end
            ).get("data", [])
        except Exception as e:
            logger.exception("Error fetching insider transactions: %s", e)
            transactions = []

        try:
            sentiment = self.client.stock_insider_sentiment(
                self.ticker, start, end
            )
        except Exception as e:
            logger.exception("Error fetching insider sentiment: %s", e)
            sentiment = {}

        return {
            "ticker": self.ticker,
            "insider_transactions": transactions,
            "insider_sentiment": sentiment,
        }

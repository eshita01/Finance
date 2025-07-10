import logging

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch historical OHLCV data for given tickers."""

    def __init__(self, tickers: List[str], period: str = "1mo", interval: str = "1d"):
        self.tickers = tickers
        self.period = period
        self.interval = interval

    def fetch(self) -> pd.DataFrame:
        """Fetch data from yfinance."""
        try:
            logger.info("Fetching data for %s", self.tickers)

            data = yf.download(
                self.tickers,
                period=self.period,
                interval=self.interval,
                group_by="column",
                auto_adjust=True,
            )
            return data
        except Exception as e:
            logger.exception("Failed to fetch data: %s", e)
            raise

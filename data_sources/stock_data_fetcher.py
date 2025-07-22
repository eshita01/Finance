import logging

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch historical OHLCV data for given tickers."""

    def __init__(self, tickers: List[str], lookback_days: int = 30, interval: str = "1d", end_date: Optional[datetime] = None):
        """Initialize the fetcher.

        Parameters
        ----------
        tickers : List[str]
            Tickers to download.
        lookback_days : int, optional
            Number of days of data to retrieve, by default 30.
        interval : str, optional
            Data granularity, by default "1d".
        end_date : datetime, optional
            Final date for the series. Defaults to ``datetime.utcnow()``.
        """

        self.tickers = tickers
        self.lookback_days = lookback_days
        self.interval = interval
        self.end_date = end_date or datetime.utcnow()

    def fetch(self) -> pd.DataFrame:
        """Fetch data from yfinance."""
        try:
            logger.info("Fetching data for %s", self.tickers)

            start = (self.end_date - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            end = self.end_date.strftime("%Y-%m-%d")

            data = yf.download(
                self.tickers,
                start=start,
                end=end,
                interval=self.interval,
                group_by="column",
                auto_adjust=True,
            )

            if data.empty:
                logger.error("No data returned from yfinance")
                raise ValueError("Empty data returned")

            # Ensure expected price columns exist
            cols = data.columns if not isinstance(data.columns, pd.MultiIndex) else data.columns.get_level_values(0)
            if "Close" not in cols and "Adj Close" not in cols:
                logger.error("Price columns missing in fetched data")
                raise ValueError("Price column missing")

            return data
        except Exception as e:
            logger.exception("Failed to fetch data: %s", e)
            raise

import logging
from typing import Any, Dict

import pandas as pd

from .technical_analysis import compute_indicators

logger = logging.getLogger(__name__)


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze peer stocks and return comparison metrics."""
    try:
        peers = data.get("peers", [])
        prices: Dict[str, pd.DataFrame] = data.get("price_data", {})
        news: Dict[str, Any] = data.get("news", {})
        table: Dict[str, Dict[str, Any]] = {}

        for peer in peers:
            df = prices.get(peer)
            price_change_1d = None
            price_change_7d = None
            rsi = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

                if len(df) >= 2:
                    last = float(df[price_col].iloc[-1])
                    prev = float(df[price_col].iloc[-2])
                    price_change_1d = (last - prev) / prev * 100

                if len(df) >= 8:
                    last = float(df[price_col].iloc[-1])
                    prev7 = float(df[price_col].iloc[-8])
                    price_change_7d = (last - prev7) / prev7 * 100

                indicators = compute_indicators(df)
                rsi = float(indicators["RSI_14"].iloc[-1])

            sentiment = news.get(peer, {})
            news_score = 0.0
            if isinstance(sentiment, dict):
                try:
                    news_score = float(sentiment.get("companyNewsScore", 0.0))
                except (TypeError, ValueError):
                    news_score = 0.0

            table[peer] = {
                "sentiment": news_score,
                "change_1d": price_change_1d,
                "change_7d": price_change_7d,
                "rsi": rsi,
            }

        return {"peer_table": table}
    except Exception as e:
        logger.exception("Peer analysis failed: %s", e)
        raise

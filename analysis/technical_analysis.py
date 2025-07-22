import logging
from typing import Dict

import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for a price dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least ``High``, ``Low`` and ``Close`` or
        ``Adj Close`` columns. Can be a regular DataFrame or a DataFrame with a
        ``MultiIndex`` as returned by ``yfinance`` when requesting multiple
        tickers.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with indicator columns appended.
    """
    try:
        logger.info("Computing technical indicators")
        df = data.copy()

        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multiindex by keeping only the price level
            levels = df.columns.get_level_values(0)
            if "Adj Close" in levels or "Close" in levels:
                df.columns = levels
            else:
                levels = df.columns.get_level_values(-1)
                if "Adj Close" in levels or "Close" in levels:
                    df.columns = levels
                else:
                    logger.error("Price columns not found in MultiIndex")
                    raise ValueError("Price column missing")

        if df.empty:
            logger.error("No data received for indicator computation")
            raise ValueError("Empty dataframe")

        if ("Adj Close" in df.columns) or ("Close" in df.columns):
            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        else:
            logger.error("Price column not found in data")
            raise ValueError("Price column missing")

        if "High" not in df.columns or "Low" not in df.columns:
            logger.error("High/Low columns not found in data")
            raise ValueError("High/Low column missing")

        # Core indicators over the full window
        df["SMA_20"] = ta.trend.sma_indicator(df[price_col], window=20)
        df["MACD_12_26_9"] = ta.trend.macd(df[price_col])
        df["MACD_signal"] = ta.trend.macd_signal(df[price_col])
        df["MACD_diff"] = ta.trend.macd_diff(df[price_col])
        bb = ta.volatility.BollingerBands(close=df[price_col], window=20)
        df["BBL_20"] = bb.bollinger_lband()
        df["BBH_20"] = bb.bollinger_hband()

        # Fast reacting indicators using shorter windows
        df["RSI_14"] = ta.momentum.rsi(df[price_col], window=14)
        atr = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df[price_col], window=14
        )
        df["ATR_14"] = atr.average_true_range()
        adx = ta.trend.ADXIndicator(
            high=df["High"], low=df["Low"], close=df[price_col], window=14
        )
        df["ADX_14"] = adx.adx()
        momentum = ta.momentum.ROCIndicator(df[price_col], window=10)
        df["Momentum_10"] = momentum.roc()

        return df
    except Exception as e:
        logger.exception("Error computing indicators: %s", e)
        raise


def analyze(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a dictionary of trading signals from indicator values."""
    try:
        logger.info("Analyzing data for signals")

        if df.empty:
            raise ValueError("Empty dataframe for analysis")

        signal: Dict[str, str] = {}
        latest = df.iloc[-1]

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        # RSI
        if latest["RSI_14"] > 70:
            signal["rsi"] = "overbought"
        elif latest["RSI_14"] < 30:
            signal["rsi"] = "oversold"
        else:
            signal["rsi"] = "neutral"

        # MACD
        if latest["MACD_12_26_9"] > latest["MACD_signal"]:
            signal["macd"] = "bullish"
        else:
            signal["macd"] = "bearish"

        # Bollinger Bands
        if latest[price_col] > latest["BBH_20"]:
            signal["bb"] = "breakout"
        elif latest[price_col] < latest["BBL_20"]:
            signal["bb"] = "breakdown"
        else:
            signal["bb"] = "neutral"

        # Trend strength from ADX
        if latest["ADX_14"] > 25:
            signal["trend_strength"] = "strong"
        else:
            signal["trend_strength"] = "weak"

        # Volatility based on ATR
        atr_avg = df["ATR_14"].mean()
        if latest["ATR_14"] > atr_avg * 1.5:
            signal["volatility"] = "high"
        else:
            signal["volatility"] = "normal"

        # Momentum based on Rate of Change
        if latest["Momentum_10"] > 0:
            signal["momentum"] = "positive"
        else:
            signal["momentum"] = "negative"

        return signal
    except Exception as e:
        logger.exception("Error analyzing data: %s", e)
        raise

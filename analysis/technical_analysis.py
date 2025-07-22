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

        n = len(df)

        # Core indicators over the full window. If the dataset is shorter than
        # the typical periods, fall back to the available length so we still
        # return numeric values for the latest row.
        sma_win = min(20, n)
        df["SMA_20"] = ta.trend.sma_indicator(df[price_col], window=sma_win)

        macd_fast = min(12, n)
        macd_slow = min(26, n)
        macd_signal = min(9, n)
        df["MACD_12_26_9"] = ta.trend.macd(
            df[price_col], window_slow=macd_slow, window_fast=macd_fast
        )
        df["MACD_signal"] = ta.trend.macd_signal(
            df[price_col],
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal,
        )
        df["MACD_diff"] = ta.trend.macd_diff(
            df[price_col],
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal,
        )

        bb_win = min(20, n)
        bb = ta.volatility.BollingerBands(close=df[price_col], window=bb_win)
        df["BBL_20"] = bb.bollinger_lband()
        df["BBH_20"] = bb.bollinger_hband()

        # Fast reacting indicators using shorter windows. When less data is
        # available than the standard lookback, compute the indicators over the
        # shorter period instead of returning NA values.

        rsi_win = min(14, n)
        df["RSI_14"] = ta.momentum.rsi(df[price_col], window=rsi_win)

        atr_win = min(14, n)
        atr = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df[price_col], window=atr_win
        )
        df["ATR_14"] = atr.average_true_range()

        adx_win = min(14, max(1, (n - 1) // 2))
        adx = ta.trend.ADXIndicator(
            high=df["High"], low=df["Low"], close=df[price_col], window=adx_win
        )
        df["ADX_14"] = adx.adx()

        mom_win = min(10, max(1, n - 1))
        momentum = ta.momentum.ROCIndicator(df[price_col], window=mom_win)
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
        if pd.notna(latest["RSI_14"]) and latest["RSI_14"] > 70:
            signal["rsi"] = "overbought"
        elif pd.notna(latest["RSI_14"]) and latest["RSI_14"] < 30:
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
        if pd.notna(latest["ADX_14"]) and latest["ADX_14"] > 25:
            signal["trend_strength"] = "strong"
        else:
            signal["trend_strength"] = "weak"

        # Volatility based on ATR
        atr_avg = df["ATR_14"].dropna().mean()
        if pd.notna(latest["ATR_14"]) and atr_avg and latest["ATR_14"] > atr_avg * 1.5:
            signal["volatility"] = "high"
        else:
            signal["volatility"] = "normal"

        # Momentum based on Rate of Change
        if pd.notna(latest["Momentum_10"]) and latest["Momentum_10"] > 0:

            signal["momentum"] = "positive"
        else:
            signal["momentum"] = "negative"

        return signal
    except Exception as e:
        logger.exception("Error analyzing data: %s", e)
        raise

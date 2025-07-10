import logging
from typing import Dict

import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and append to dataframe."""
    try:
        logger.info("Computing technical indicators")
        df = data.copy()

        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multiindex by keeping only the price level
            if 'Adj Close' in df.columns.get_level_values(0) or 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Adj Close' in df.columns.get_level_values(-1) or 'Close' in df.columns.get_level_values(-1):
                df.columns = df.columns.get_level_values(-1)
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

        df['SMA_20'] = ta.trend.sma_indicator(df[price_col], window=20)
        df['RSI_14'] = ta.momentum.rsi(df[price_col], window=14)
        df['MACD_12_26_9'] = ta.trend.macd(df[price_col])
        df['MACD_signal'] = ta.trend.macd_signal(df[price_col])
        df['MACD_diff'] = ta.trend.macd_diff(df[price_col])
        bb = ta.volatility.BollingerBands(close=df[price_col], window=20)
        df['BBL_20'] = bb.bollinger_lband()
        df['BBH_20'] = bb.bollinger_hband()
        return df
    except Exception as e:
        logger.exception("Error computing indicators: %s", e)
        raise


def analyze(df: pd.DataFrame) -> Dict[str, str]:
    """Analyze indicators to derive simple signals."""
    try:
        logger.info("Analyzing data for signals")
        signal = {}
        latest = df.iloc[-1]

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        if latest['RSI_14'] > 70:
            signal['rsi'] = 'overbought'
        elif latest['RSI_14'] < 30:
            signal['rsi'] = 'oversold'
        else:
            signal['rsi'] = 'neutral'

        if latest['MACD_12_26_9'] > latest['MACD_signal']:
            signal['macd'] = 'bullish'
        else:
            signal['macd'] = 'bearish'

        if latest[price_col] > latest['BBH_20']:
            signal['bb'] = 'breakout'
        elif latest[price_col] < latest['BBL_20']:
            signal['bb'] = 'breakdown'
        else:
            signal['bb'] = 'neutral'
        return signal
    except Exception as e:
        logger.exception("Error analyzing data: %s", e)
        raise

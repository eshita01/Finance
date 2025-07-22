import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "technical_analysis.py"
spec = importlib.util.spec_from_file_location("technical_analysis", MODULE_PATH)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)


def _sample_data():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame({
        "Open": range(1, 31),
        "High": range(2, 32),
        "Low": range(1, 31),
        "Close": range(1, 31),
        "Volume": range(100, 130)
    }, index=dates)
    return data


def test_compute_indicators_adds_columns():
    df = _sample_data()
    result = ta.compute_indicators(df)
    expected = {
        "SMA_20",
        "MACD_12_26_9",
        "MACD_signal",
        "MACD_diff",
        "BBL_20",
        "BBH_20",
        "RSI_14",
        "ATR_14",
        "ADX_14",
        "Momentum_10",
    }
    assert expected.issubset(result.columns)


def test_analyze_returns_all_signals():
    df = ta.compute_indicators(_sample_data())
    signals = ta.analyze(df)
    assert set(signals.keys()) == {
        "rsi",
        "macd",
        "bb",
        "trend_strength",
        "volatility",
        "momentum",
    }

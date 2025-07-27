import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "sentiment_analysis.py"
spec = importlib.util.spec_from_file_location("sentiment_analysis", MODULE_PATH)
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)


def test_analyze_basic_stats():
    feed = [
        {"title": "A", "overall_sentiment_score": 0.5},
        {"title": "B", "overall_sentiment_score": -0.3},
        {"title": "C", "overall_sentiment_score": 0.1},
        {"title": "D", "overall_sentiment_score": 0.9},
    ]
    result = sa.analyze(feed)
    assert result["tone"] == "positive"
    assert result["tone_distribution"] == {"positive": 2, "neutral": 1, "negative": 1}
    assert result["dominant_tone"] == "positive"
    assert result["urgency"] == "normal"
    assert result["hype"] == 4
    assert result["highlight_headline"] == "D"


def test_analyze_no_scores():
    feed = [{"title": "Only title"}]
    result = sa.analyze(feed)
    assert result["average_sentiment"] == 0.0
    assert result["tone"] == "neutral"
    assert result["tone_distribution"] == {"positive": 0, "neutral": 0, "negative": 0}
    assert "highlight_headline" not in result

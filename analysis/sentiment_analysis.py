import logging
from statistics import mean
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def analyze(feed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze news sentiment feed and return summary metrics."""
    try:
        if not feed:
            raise ValueError("Empty news feed")

        scores = []
        headlines = []
        pos = neg = neu = 0
        highlight_headline = None

        for item in feed:
            score = None
            if "overall_sentiment_score" in item:
                try:
                    score = float(item["overall_sentiment_score"])
                    scores.append(score)
                except (TypeError, ValueError):
                    score = None
            if "title" in item:
                headlines.append(item["title"])

            if score is not None:
                if score > 0.2:
                    pos += 1
                elif score < -0.2:
                    neg += 1
                else:
                    neu += 1

                if abs(score) > 0.8 and not highlight_headline:
                    highlight_headline = item.get("title")

        if scores:
            avg_score = mean(scores)
        else:
            avg_score = 0.0

        if avg_score > 0.2:
            tone = "positive"
        elif avg_score < -0.2:
            tone = "negative"
        else:
            tone = "neutral"

        tone_distribution = {"positive": pos, "neutral": neu, "negative": neg}
        dominant_tone = max(tone_distribution, key=tone_distribution.get)

        urgency = "high" if len(feed) > 30 else "normal"
        summary = " | ".join(headlines[:3])

        result = {
            "average_sentiment": avg_score,
            "tone": tone,
            "tone_distribution": tone_distribution,
            "dominant_tone": dominant_tone,
            "headline_summary": summary,
            "urgency": urgency,
            "hype": len(feed),
        }
        if highlight_headline:
            result["highlight_headline"] = highlight_headline

        return result
    except Exception as e:
        logger.exception("Sentiment analysis failed: %s", e)
        raise

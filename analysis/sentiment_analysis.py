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
        for item in feed:
            if 'overall_sentiment_score' in item:
                try:
                    scores.append(float(item['overall_sentiment_score']))
                except (TypeError, ValueError):
                    continue
            if 'title' in item:
                headlines.append(item['title'])

        avg_score = mean(scores) if scores else 0.0
        tone = 'positive' if avg_score > 0.2 else 'negative' if avg_score < -0.2 else 'neutral'

        trend = 'flat'
        if len(scores) >= 2:
            if scores[-1] > scores[0]:
                trend = 'up'
            elif scores[-1] < scores[0]:
                trend = 'down'

        urgency = 'high' if len(feed) > 30 else 'normal'
        summary = ' | '.join(headlines[:3])

        return {
            'average_sentiment': avg_score,
            'tone': tone,
            'headline_summary': summary,
            'urgency': urgency,
            'hype': len(feed),
            'trend': trend,
        }
    except Exception as e:
        logger.exception("Sentiment analysis failed: %s", e)
        raise

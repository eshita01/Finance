import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


TOP_EXEC_ROLES = ["CEO", "CFO", "CTO"]


def _parse_date(value: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(value[:19], fmt)
        except (ValueError, TypeError):
            continue
    return datetime.utcnow()


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze insider transaction and sentiment data."""
    try:
        ticker = data.get("ticker")
        transactions: List[Dict[str, Any]] = data.get("insider_transactions", [])
        sentiment: Dict[str, Any] = data.get("insider_sentiment", {})

        buys = [t for t in transactions if str(t.get("transactionType", t.get("transaction_type", "")).lower()) == "buy"]
        sells = [t for t in transactions if str(t.get("transactionType", t.get("transaction_type", "")).lower()) == "sell"]

        total_buys = len(buys)
        total_sells = len(sells)
        net_activity = "Buy" if total_buys > total_sells else "Sell" if total_sells > total_buys else "Neutral"

        top_execs: List[str] = []
        for t in transactions:
            title = str(t.get("position", t.get("title", ""))).upper()
            for role in TOP_EXEC_ROLES:
                if role in title:
                    top_execs.append(role)
        top_execs = sorted(set(top_execs))

        recent_cluster = False
        now = datetime.utcnow()
        recent = [
            _parse_date(t.get("transactionDate", t.get("date", "")))
            for t in buys
            if t.get("transactionDate") or t.get("date")
        ]
        recent = [d for d in recent if (now - d).days <= 7]
        if len(recent) >= 2:
            recent_cluster = True

        mspr = 0.0
        if isinstance(sentiment, dict):
            if "mspr" in sentiment:
                try:
                    mspr = float(sentiment.get("mspr"))
                except (TypeError, ValueError):
                    mspr = 0.0
            elif isinstance(sentiment.get("data"), list) and sentiment.get("data"):
                try:
                    mspr = float(sentiment["data"][-1].get("mspr", 0.0))
                except (TypeError, ValueError):
                    mspr = 0.0

        if mspr > 0.6:
            mspr_view = "Bullish"
        elif mspr > 0.3:
            mspr_view = "Neutral"
        else:
            mspr_view = "Bearish"

        score = 50
        if net_activity == "Buy" and mspr > 0.6 and top_execs:
            score = 85
        elif net_activity == "Buy" and mspr > 0.6:
            score = 75
        elif net_activity == "Buy" and mspr > 0.3:
            score = 65
        elif net_activity == "Sell" and mspr <= 0.3:
            score = 25

        parts = []
        if top_execs:
            parts.append(f"Executives involved: {', '.join(top_execs)}")
        parts.append(f"Net {net_activity.lower()} activity ({total_buys} buys vs {total_sells} sells)")
        parts.append(f"MSPR {mspr:.2f} -> {mspr_view}")
        if recent_cluster:
            parts.append("recent cluster buying")

        summary = ". ".join(parts)

        return {
            "ticker": ticker,
            "insider_sentiment_score": score,
            "summary": summary,
            "raw_features": {
                "total_buys": total_buys,
                "total_sells": total_sells,
                "net_activity": net_activity,
                "top_execs_involved": top_execs,
                "mspr": mspr,
                "recent_cluster": recent_cluster,
            },
        }
    except Exception as e:
        logger.exception("Failed to analyze insider data: %s", e)
        raise

import logging
from typing import Dict, Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


class DecisionMaker:
    """Use Gemini to turn analysis signals into a buy/sell/hold decision."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def decide(self, signals: Dict[str, Any]) -> str:
        peer_table = signals.get("peer_table", {})
        peer_lines = []
        for sym, info in peer_table.items():
            peer_lines.append(
                f"{sym}: 1d {info.get('change_1d'):.2f}% | 7d {info.get('change_7d'):.2f}% | RSI {info.get('rsi'):.2f} | Sent {info.get('sentiment'):.2f}"
                if info else f"{sym}: data unavailable"
            )
        peer_summary = "\n".join(peer_lines)

        prompt = (
            "You are a trading assistant. Based on the following analysis signals,"
            " provide a single word recommendation (Buy, Sell, or Hold) followed by"
            " a short rationale that references technical, news, insider data, and peer comparisons.\n"
            f"RSI signal: {signals.get('rsi')}\n"
            f"MACD signal: {signals.get('macd')}\n"
            f"Bollinger Bands signal: {signals.get('bb')}\n"
            f"Average sentiment: {signals.get('average_sentiment')}\n"
            f"Tone: {signals.get('tone')}\n"
            f"Urgency: {signals.get('urgency')}\n"
            f"Trend: {signals.get('trend')}\n"
            f"Insider score: {signals.get('insider_sentiment_score')}\n"
            f"Insider summary: {signals.get('summary')}\n"
            f"Peer data:\n{peer_summary}\n"
        )
        try:
            logger.info("Sending prompt to Gemini")
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            logger.info("Gemini response: %s", text)
            return text
        except Exception as e:
            logger.exception("Gemini decision failed: %s", e)
            raise

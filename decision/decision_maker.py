import logging
from typing import Dict

import google.generativeai as genai

logger = logging.getLogger(__name__)


class DecisionMaker:
    """Use Gemini to turn analysis signals into a buy/sell/hold decision."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def decide(self, signals: Dict[str, str]) -> str:
        prompt = (
            "You are a trading assistant. Based on the following technical signals,"
            " provide a single word recommendation: Buy, Sell, or Hold.\n"
            f"RSI signal: {signals.get('rsi')}\n"
            f"MACD signal: {signals.get('macd')}\n"
            f"Bollinger Bands signal: {signals.get('bb')}\n"
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

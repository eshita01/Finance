import argparse
import logging
from datetime import datetime
from typing import TypedDict, Dict, Any, List

import pandas as pd
from langgraph.graph import StateGraph, END

from data_sources.stock_data_fetcher import StockDataFetcher
from data_sources.news_sentiment_fetcher import NewsSentimentFetcher
from data_sources.insider_data_fetcher import InsiderDataFetcher
from analysis.technical_analysis import compute_indicators, analyze
from analysis.sentiment_analysis import analyze as analyze_sentiment
from analysis.insider_analysis import analyze as analyze_insider
from decision.decision_maker import DecisionMaker
from config import get_api_key, get_alpha_vantage_key, get_finnhub_key


class AgentState(TypedDict, total=False):
    data: pd.DataFrame
    news: List[Dict[str, Any]]
    insider: Dict[str, Any]
    indicators: pd.DataFrame
    signals: Dict[str, str]
    sentiment: Dict[str, Any]
    insider_insights: Dict[str, Any]
    decision: str


def build_graph(ticker: str, gemini_key: str, alpha_key: str, finnhub_key: str, base_date: datetime):
    fetcher = StockDataFetcher([ticker], end_date=base_date)
    news_fetcher = NewsSentimentFetcher([ticker], alpha_key, base_date=base_date)
    insider_fetcher = InsiderDataFetcher(ticker, finnhub_key, base_date=base_date)
    decider = DecisionMaker(gemini_key)

    def fetch_node(state: AgentState) -> AgentState:
        data = fetcher.fetch()
        news = news_fetcher.fetch()
        insider = insider_fetcher.fetch()

        print("\n=== Fetch Node ===")
        print(f"Price rows: {len(data)}")
        print(f"News items: {len(news)}")
        print(f"Insider transactions: {len(insider.get('insider_transactions', []))}")
        print()

        return {"data": data, "news": news, "insider": insider}

    def analysis_node(state: AgentState) -> AgentState:
        df = compute_indicators(state["data"])
        signals = analyze(df)
        sentiment = analyze_sentiment(state["news"])
        insider_insights = analyze_insider(state["insider"])

        print("\n=== Analysis Node ===")
        print(f"Signals: {signals}")
        print(f"Sentiment: {sentiment}")
        print(f"Insider: {insider_insights}")
        print()

        return {
            "indicators": df,
            "signals": signals,
            "sentiment": sentiment,
            "insider_insights": insider_insights,
        }

    def decision_node(state: AgentState) -> AgentState:
        combined = {
            **state.get("signals", {}),
            **state.get("sentiment", {}),
            **state.get("insider_insights", {}),
        }
        decision = decider.decide(combined)

        print("\n=== Decision Node ===")
        print(decision)
        print()

        return {"decision": decision}

    graph = StateGraph(AgentState)
    graph.add_node("fetch", fetch_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("decide", decision_node)

    graph.add_edge("fetch", "analyze")
    graph.add_edge("analyze", "decide")
    graph.add_edge("decide", END)

    graph.set_entry_point("fetch")
    graph.set_finish_point("decide")

    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Run trading agent")
    parser.add_argument("ticker", help="Ticker symbol to analyze")
    parser.add_argument("--date", help="Base date YYYY-MM-DD", required=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    gemini_key = get_api_key()
    alpha_key = get_alpha_vantage_key()
    finnhub_key = get_finnhub_key()
    base_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.utcnow()
    graph = build_graph(args.ticker, gemini_key, alpha_key, finnhub_key, base_date)
    result = graph.invoke({})
    print("Decision:", result["decision"])


if __name__ == "__main__":
    main()

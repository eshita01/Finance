import argparse
import logging
from typing import TypedDict, Dict, Any, List

import pandas as pd
from langgraph.graph import StateGraph, END

from data_sources.stock_data_fetcher import StockDataFetcher
from data_sources.news_sentiment_fetcher import NewsSentimentFetcher
from analysis.technical_analysis import compute_indicators, analyze
from analysis.sentiment_analysis import analyze as analyze_sentiment
from decision.decision_maker import DecisionMaker
from config import get_api_key, get_alpha_vantage_key


class AgentState(TypedDict, total=False):
    data: pd.DataFrame
    news: List[Dict[str, Any]]
    indicators: pd.DataFrame
    signals: Dict[str, str]
    sentiment: Dict[str, Any]
    decision: str


def build_graph(ticker: str, gemini_key: str, alpha_key: str):
    fetcher = StockDataFetcher([ticker])
    news_fetcher = NewsSentimentFetcher([ticker], alpha_key)
    decider = DecisionMaker(gemini_key)

    def fetch_node(state: AgentState) -> AgentState:
        data = fetcher.fetch()
        news = news_fetcher.fetch()
        return {"data": data, "news": news}

    def analysis_node(state: AgentState) -> AgentState:
        df = compute_indicators(state["data"])
        signals = analyze(df)
        sentiment = analyze_sentiment(state["news"])
        return {"indicators": df, "signals": signals, "sentiment": sentiment}

    def decision_node(state: AgentState) -> AgentState:
        combined = {**state.get("signals", {}), **state.get("sentiment", {})}
        decision = decider.decide(combined)
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    gemini_key = get_api_key()
    alpha_key = get_alpha_vantage_key()
    graph = build_graph(args.ticker, gemini_key, alpha_key)
    result = graph.invoke({})
    print("Decision:", result["decision"])


if __name__ == "__main__":
    main()

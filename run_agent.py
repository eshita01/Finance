import argparse
import logging
from typing import TypedDict, Dict

import pandas as pd
from langgraph.graph import StateGraph, END

from data_sources.stock_data_fetcher import StockDataFetcher
from analysis.technical_analysis import compute_indicators, analyze
from decision.decision_maker import DecisionMaker
from config import get_api_key


class AgentState(TypedDict, total=False):
    data: pd.DataFrame
    indicators: pd.DataFrame
    signals: Dict[str, str]
    decision: str


def build_graph(ticker: str, api_key: str):
    fetcher = StockDataFetcher([ticker])
    decider = DecisionMaker(api_key)

    def fetch_node(state: AgentState) -> AgentState:
        data = fetcher.fetch()
        return {"data": data}

    def analysis_node(state: AgentState) -> AgentState:
        df = compute_indicators(state["data"])
        signals = analyze(df)
        return {"indicators": df, "signals": signals}

    def decision_node(state: AgentState) -> AgentState:
        decision = decider.decide(state["signals"])
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

    api_key = get_api_key()
    graph = build_graph(args.ticker, api_key)
    result = graph.invoke({})
    print("Decision:", result["decision"])


if __name__ == "__main__":
    main()

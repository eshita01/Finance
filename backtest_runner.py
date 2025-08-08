import argparse
import csv
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import pandas as pd
import yfinance as yf
from langgraph.graph import StateGraph, END

from data_sources.stock_data_fetcher import StockDataFetcher
from data_sources.news_sentiment_fetcher import NewsSentimentFetcher
from data_sources.insider_data_fetcher import InsiderDataFetcher
from data_sources.peer_data_fetcher import PeerDataFetcher
from data_sources.sec_fetcher import SECFetcher
from analysis.technical_analysis import compute_indicators, analyze
from analysis.sentiment_analysis import analyze as analyze_sentiment
from analysis.insider_analysis import analyze as analyze_insider
from analysis.peer_analysis import analyze as analyze_peers
from analysis.sec_risk_analysis import SECRiskAnalyzer
from decision.decision_maker import DecisionMaker
from config import (
    get_api_key,
    get_alpha_vantage_key,
    get_finnhub_key,
    get_huggingface_key,
)


class AgentState(TypedDict, total=False):
    data: pd.DataFrame
    news: List[Dict[str, Any]]
    insider: Dict[str, Any]
    peer_data: Dict[str, Any]
    indicators: pd.DataFrame
    signals: Dict[str, str]
    sentiment: Dict[str, Any]
    insider_insights: Dict[str, Any]
    peer_insights: Dict[str, Any]
    sec_meta: Dict[str, Any]
    sec_analysis: Dict[str, Any]
    decision: str


class SimpleDecisionMaker(DecisionMaker):
    """DecisionMaker variant that only returns a single word and logs prompts."""

    def decide(self, signals: Dict[str, Any], run_date: datetime) -> str:  # type: ignore[override]
        peer_table = signals.get("peer_table", {})
        peer_lines = []
        for sym, info in peer_table.items():
            if not info:
                peer_lines.append(f"{sym}: data unavailable")
                continue

            def fmt(val: Any) -> str:
                return f"{float(val):.2f}" if val is not None else "N/A"

            line = (
                f"{sym}: 1d {fmt(info.get('change_1d'))}% | 7d {fmt(info.get('change_7d'))}% "
                f"| RSI {fmt(info.get('rsi'))} | Sent {fmt(info.get('sentiment'))}"
            )
            peer_lines.append(line)

        peer_summary = "\n".join(peer_lines)
        prompt = (
            "You are a trading assistant. Based on the following analysis signals, "
            "respond with a single word recommendation: Buy, Sell, or Hold. "
            "Do not include any rationale or explanation.\n"
            f"RSI signal: {signals.get('rsi')}\n"
            f"MACD signal: {signals.get('macd')}\n"
            f"Bollinger Bands signal: {signals.get('bb')}\n"
            f"Trend strength: {signals.get('trend_strength')}\n"
            f"Volatility: {signals.get('volatility')}\n"
            f"Momentum: {signals.get('momentum')}\n"
            f"Average sentiment: {signals.get('average_sentiment')}\n"
            f"Tone: {signals.get('tone')}\n"
            f"Dominant tone: {signals.get('dominant_tone')}\n"
            f"Tone distribution: {signals.get('tone_distribution')}\n"
            f"Headline summary: {signals.get('headline_summary')}\n"
            f"Urgency: {signals.get('urgency')}\n"
            f"Hype: {signals.get('hype')}\n"
            f"Insider score: {signals.get('insider_sentiment_score')}\n"
            f"Insider summary: {signals.get('summary')}\n"
            f"Risk summary: {signals.get('risk_summary')}\n"
            f"Risk sentiment: {signals.get('risk_sentiment')}\n"
            f"MD&A summary: {signals.get('mdna_summary')}\n"
            f"MD&A sentiment: {signals.get('mdna_sentiment')}\n"
            f"Filing age (days): {signals.get('sec_filing_age_days')}\n"
            f"Highlight headline: {signals.get('highlight_headline')}\n"
            f"Peer data:\n{peer_summary}\n"
        )
        log_file = Path("results") / "gemini_prompts.log"
        try:
            log_file.parent.mkdir(exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"{run_date.date().isoformat()} | {prompt}\n")
        except Exception:
            logging.getLogger(__name__).exception("Failed writing prompt log")
        try:
            logging.getLogger(__name__).info("Sending prompt to Gemini")
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            logging.getLogger(__name__).info("Gemini response: %s", text)
            # only keep the first word
            word = text.split()[0]
            return word
        except Exception as e:
            logging.getLogger(__name__).exception("Gemini decision failed: %s", e)
            raise


def build_graph(
    ticker: str,
    gemini_key: str,
    alpha_key: str,
    finnhub_key: str,
    hf_key: str,
    base_date: datetime,
):
    fetcher = StockDataFetcher([ticker], end_date=base_date)
    news_fetcher = NewsSentimentFetcher([ticker], alpha_key, finnhub_key, hf_key, base_date=base_date)
    insider_fetcher = InsiderDataFetcher(ticker, finnhub_key, base_date=base_date)
    peer_fetcher = PeerDataFetcher(ticker, finnhub_key, alpha_key, base_date=base_date)
    sec_fetcher = SECFetcher(ticker)
    sec_analyzer = SECRiskAnalyzer(gemini_key)
    decider = SimpleDecisionMaker(gemini_key)

    def fetch_node(state: AgentState) -> AgentState:
        data = fetcher.fetch()
        news = news_fetcher.fetch()
        insider = insider_fetcher.fetch()
        return {"data": data, "news": news, "insider": insider}

    def analysis_node(state: AgentState) -> AgentState:
        df = compute_indicators(state["data"])
        signals = analyze(df)
        sentiment = analyze_sentiment(state["news"])
        insider_insights = analyze_insider(state["insider"])
        return {
            "indicators": df,
            "signals": signals,
            "sentiment": sentiment,
            "insider_insights": insider_insights,
        }

    def sec_fetch_node(state: AgentState) -> AgentState:
        meta = sec_fetcher.fetch(base_date)
        return {"sec_meta": meta}

    def sec_analyze_node(state: AgentState) -> AgentState:
        analysis = sec_analyzer.analyze(state["sec_meta"], base_date)
        return {"sec_analysis": analysis}

    def peer_fetch_node(state: AgentState) -> AgentState:
        peer_data = peer_fetcher.fetch()
        return {"peer_data": peer_data}

    def peer_analysis_node(state: AgentState) -> AgentState:
        insights = analyze_peers(state["peer_data"])
        return {"peer_insights": insights}

    def decision_node(state: AgentState) -> AgentState:
        combined = {
            **state.get("signals", {}),
            **state.get("sentiment", {}),
            **state.get("insider_insights", {}),
            **state.get("peer_insights", {}),
            **{
                "risk_summary": state.get("sec_analysis", {}).get("risk_summary"),
                "mdna_summary": state.get("sec_analysis", {}).get("mdna_summary"),
                "risk_sentiment": state.get("sec_analysis", {}).get("risk_sentiment"),
                "mdna_sentiment": state.get("sec_analysis", {}).get("mdna_sentiment"),
            },
        }

        if "sec_meta" in state:
            try:
                filing_date = datetime.fromisoformat(state["sec_meta"]["filing_date"])
                age_days = (base_date.date() - filing_date.date()).days
                combined["sec_filing_age_days"] = age_days
            except Exception:
                pass
        decision = decider.decide(combined, base_date)
        return {"decision": decision}

    graph = StateGraph(AgentState)
    graph.add_node("fetch", fetch_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("sec_fetch", sec_fetch_node)
    graph.add_node("sec_analyze", sec_analyze_node)
    graph.add_node("peer_fetch", peer_fetch_node)
    graph.add_node("peer_analyze", peer_analysis_node)
    graph.add_node("decide", decision_node)

    graph.add_edge("fetch", "analyze")
    graph.add_edge("analyze", "sec_fetch")
    graph.add_edge("sec_fetch", "sec_analyze")
    graph.add_edge("sec_analyze", "peer_fetch")
    graph.add_edge("peer_fetch", "peer_analyze")
    graph.add_edge("peer_analyze", "decide")
    graph.add_edge("decide", END)

    graph.set_entry_point("fetch")
    graph.set_finish_point("decide")

    return graph.compile()


def get_trading_days(ticker: str, start: datetime, end: datetime) -> pd.DatetimeIndex:
    """Return trading days between start and end inclusive."""
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d", auto_adjust=True)
    return df.index


def run_for_date(ticker: str, day: datetime, keys: Dict[str, str]) -> str:
    graph = build_graph(
        ticker,
        keys["gemini"],
        keys["alpha"],
        keys["finnhub"],
        keys["hf"],
        day,
    )
    result = graph.invoke({})
    decision = str(result["decision"]).strip().split()[0]
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest trading agent")
    parser.add_argument("ticker", help="Ticker symbol to analyze")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    keys = {
        "gemini": get_api_key(),
        "alpha": get_alpha_vantage_key(),
        "finnhub": get_finnhub_key(),
        "hf": get_huggingface_key(),
    }

    trading_days = get_trading_days(args.ticker, start_date, end_date)
    price_end = end_date + timedelta(days=31)
    prices = yf.download(
        args.ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=price_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )

    price_col = "Close"
    if "Adj Close" in prices.columns:
        price_col = "Adj Close"

    dates = list(prices.index)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    for day in trading_days:
        logging.info("Running backtest for %s", day.date())
        decision = run_for_date(
            args.ticker, day.to_pydatetime().replace(tzinfo=timezone.utc), keys
        )
        idx = dates.index(day)


        def future_price(offset: int) -> Any:
            target = idx + offset
            if target < len(dates):
                return float(prices.iloc[target][price_col])
            return None

        row = {
            "date": day.date().isoformat(),
            "close_price": float(prices.iloc[idx][price_col]),
            "prediction": decision,
            "close_price_1d": future_price(1),
            "close_price_5d": future_price(5),
            "close_price_14d": future_price(14),
            "close_price_30d": future_price(30),
        }

        month_file = out_dir / f"backtest_results_{day.strftime('%m_%Y')}.csv"
        write_header = not month_file.exists()
        with month_file.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        time.sleep(20)


if __name__ == "__main__":
    main()

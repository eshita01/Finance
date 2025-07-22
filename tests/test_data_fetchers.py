import argparse
from datetime import datetime, timezone

from data_sources.stock_data_fetcher import StockDataFetcher
from data_sources.news_sentiment_fetcher import NewsSentimentFetcher
from data_sources.insider_data_fetcher import InsiderDataFetcher
from data_sources.peer_data_fetcher import PeerDataFetcher
from config import get_alpha_vantage_key, get_finnhub_key


def main(ticker: str, date_str: str | None = None) -> None:
    """Fetch raw data from each data node and print a short preview."""
    base_date = (
        datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if date_str
        else datetime.now(timezone.utc)
    )

    alpha_key = get_alpha_vantage_key()
    finnhub_key = get_finnhub_key()

    print("=== StockDataFetcher ===")
    stock = StockDataFetcher([ticker], end_date=base_date)
    df = stock.fetch()
    print(df.tail())
    print()

    print("=== NewsSentimentFetcher ===")
    news_fetcher = NewsSentimentFetcher([ticker], alpha_key, base_date=base_date)
    news = news_fetcher.fetch()
    print(news[:2])
    print()

    print("=== InsiderDataFetcher ===")
    insider_fetcher = InsiderDataFetcher(ticker, finnhub_key, base_date=base_date)
    insider = insider_fetcher.fetch()
    print(insider)
    print()

    print("=== PeerDataFetcher ===")
    peer_fetcher = PeerDataFetcher(ticker, finnhub_key, alpha_key, base_date=base_date)
    peer_data = peer_fetcher.fetch()
    print("Peers:", peer_data.get("peers"))
    for peer, pdf in peer_data.get("price_data", {}).items():
        print(f"-- {peer} price data:")
        print(pdf.tail())
    print("News scores:", peer_data.get("news"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data fetchers")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol")
    parser.add_argument("--date", help="Base date YYYY-MM-DD", required=False)
    args = parser.parse_args()
    main(args.ticker, args.date)

# Finance AI Agent

This project implements a simple trading agent using [LangGraph](https://github.com/langchain-ai/langgraph). The agent fetches historical stock data and recent news sentiment, performs technical and sentiment analysis, and uses Google Gemini to make a buy/sell/hold decision.

## Folder structure

- `data_sources/` – data fetching nodes
  - `stock_data_fetcher.py` – fetch OHLCV data with yfinance
  - `news_sentiment_fetcher.py` – fetch recent news and sentiment from Alpha Vantage
  - `insider_data_fetcher.py` – fetch insider transactions and sentiment using the `finnhub-python` client
- `analysis/` – analysis nodes
  - `technical_analysis.py` – compute indicators and signals
  - `sentiment_analysis.py` – summarize news sentiment
  - `insider_analysis.py` – evaluate insider trading activity
- `decision/` – decision nodes
  - `decision_maker.py` – call Gemini for a decision
- `run_agent.py` – script to run the agent
- `config.py` – load configuration such as API keys

## Setup

1. Install dependencies (includes `finnhub-python`)
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your API keys:
   ```ini
   GEMINI_API_KEY=your-key
   ALPHAVANTAGE_API_KEY=your-alpha-key
   FINNHUB_API_KEY=your-finnhub-key
   ```

## Usage

Run the agent by providing a stock ticker. Optionally supply `--date` in `YYYY-MM-DD` format to run the pipeline for a specific day:

```bash
python run_agent.py AAPL --date 2024-05-01
```

The script fetches price history, recent news sentiment and insider trading data
for the selected date. It computes technical, news and insider metrics before
asking Gemini for a final buy/sell/hold decision with a short explanation.

## Node input and output

**insider_data_fetcher**

Input: `ticker` string, optional `base_date` used internally.

Output example:

```python
{
    "ticker": "AAPL",
    "insider_transactions": [...],
    "insider_sentiment": {...}
}
```

**insider_analysis**

Input: dictionary from `insider_data_fetcher`.

Output example:

```python
{
    "ticker": "AAPL",
    "insider_sentiment_score": 82,
    "summary": "Bullish insider activity: CEO bought shares...",
    "raw_features": {
        "total_buys": 3,
        "total_sells": 0,
        "net_activity": "Buy",
        "top_execs_involved": ["CEO"],
        "mspr": 0.65,
        "recent_cluster": True
    }
}
```

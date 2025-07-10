# Finance AI Agent

This project implements a simple trading agent using [LangGraph](https://github.com/langchain-ai/langgraph). The agent fetches historical stock data and recent news sentiment, performs technical and sentiment analysis, and uses Google Gemini to make a buy/sell/hold decision.

## Folder structure

- `data_sources/` – data fetching nodes
  - `stock_data_fetcher.py` – fetch OHLCV data with yfinance
  - `news_sentiment_fetcher.py` – fetch recent news and sentiment from Alpha Vantage
- `analysis/` – analysis nodes
  - `technical_analysis.py` – compute indicators and signals
  - `sentiment_analysis.py` – summarize news sentiment
- `decision/` – decision nodes
  - `decision_maker.py` – call Gemini for a decision
- `run_agent.py` – script to run the agent
- `config.py` – load configuration such as API keys

## Setup

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your Gemini API key:
   ```ini
   GEMINI_API_KEY=your-key
   ALPHAVANTAGE_API_KEY=your-alpha-key
   ```

## Usage

Run the agent by providing a stock ticker:

```bash
python run_agent.py AAPL
```

The script fetches one month of price data along with the last day's news headlines.
It computes technical indicators and sentiment metrics before asking Gemini for a
final buy/sell/hold decision with a short explanation.

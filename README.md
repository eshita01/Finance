# Finance AI Agent

This project implements a simple trading agent using [LangGraph](https://github.com/langchain-ai/langgraph). The agent fetches historical stock data, performs technical analysis, and uses Google Gemini to make a buy/sell/hold decision.

## Folder structure

- `data_sources/` – data fetching nodes
  - `stock_data_fetcher.py` – fetch OHLCV data with yfinance
- `analysis/` – analysis nodes
  - `technical_analysis.py` – compute indicators and signals
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
   ```

## Usage

Run the agent by providing a stock ticker:

```bash
python run_agent.py AAPL
```

The script will fetch one month of historical data for the ticker, compute technical indicators, and ask Gemini for a final decision.

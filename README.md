# Finance AI Agent

An experimental trading assistant built with
[LangGraph](https://github.com/langchain-ai/langgraph).  The agent collects
market data, performs a series of analyses and then asks a language model for a
Buy/Sell/Hold decision.

## Features

### Data collection

- **StockDataFetcher** – OHLCV price history using `yfinance`.
- **NewsSentimentFetcher** – company news via Alpha Vantage with Finnhub and
  Gemini fallback; requests are cached under `data/news_sentiment/`.
- **InsiderDataFetcher** – recent insider transactions and sentiment from
  Finnhub.
- **PeerDataFetcher** – peer tickers, price data and news sentiment.
- **SECFetcher** – downloads the latest 10‑K/10‑Q filing and stores it in
  `data/sec_reports/`.

### Analysis modules

- **technical_analysis.py** – computes indicators (SMA, MACD, RSI, ATR, ADX,
  Bollinger Bands, momentum) and returns trading signals.
- **sentiment_analysis.py** and **cached_sentiment.py** – summarise headline
  sentiment and cache results in `results/news_analysis/`.
- **insider_analysis.py** – scores insider trading activity.
- **peer_analysis.py** – compares peer performance and sentiment.
- **sec_risk_analysis.py** – extracts risk factors and MD&A sections from SEC
  filings and summarises them with Gemini; results are cached in
  `cache/sec_analysis/`.

### Decision

- **decision_maker.py** – sends combined signals to Gemini to obtain the final
  decision.

### Utilities

- **run_agent.py** – build and execute the LangGraph pipeline for a single
  ticker.
- **backtest_runner.py** – run the full pipeline over a date range, logging
  prompts and reusing cached news/analysis.
- **ablation_study.py** – replay logged prompts through ChatGPT with sections
  removed to evaluate the importance of news or peer data.

## Repository structure

```
analysis/             Sentiment, technical, insider, peer and SEC analysis
cache/                Miscellaneous caches
data/                 Downloaded data (news, SEC filings, etc.)
data_sources/         Data-fetching nodes
decision/             Final LLM decision logic
tests/                Unit tests
run_agent.py          Entry point for a single run
backtest_runner.py    Backtesting over multiple days
ablation_study.py     Prompt ablation experiment
config.py             Helper functions to load API keys
requirements.txt      Python dependencies
```

## Installation

1. **Python** – requires Python 3.11+.
2. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment variables** – create a `.env` file with API keys:
   ```ini
   GEMINI_API_KEY=your-gemini-key
   ALPHAVANTAGE_API_KEY=your-alpha-key
   FINNHUB_API_KEY=your-finnhub-key
   ```
   The ablation script also needs `OPENAI_API_KEY`.

## Running the agent

Analyse a single ticker on a given day:

```bash
python run_agent.py AAPL --date 2024-05-01
```

The script fetches price history, news, insider activity, peer data and the
latest SEC filing.  Technical, sentiment, insider, peer and SEC analyses are
performed before querying Gemini for a final decision.

## Backtesting and caching

Run the pipeline across a date range.  All news responses and sentiment
analyses are cached so subsequent runs reuse existing data.

```bash
python backtest_runner.py AAPL --start 2024-05-01 --end 2024-05-31
```

Prompt logs are written to `results/gemini_prompts.log` and SEC analyses are
cached in `cache/sec_analysis/`.

## Ablation study

To evaluate the impact of news or peer information on the model’s output, run:

```bash
python ablation_study.py
```

The script reads `results/gemini_prompts.log`, removes selected sections and
re-sends the prompts to ChatGPT, writing predictions to
`results/ablation_predictions.csv`.

## Testing

Unit tests cover the main analysis components and can be run with:

```bash
pytest
```

## License

This project is provided for educational purposes and carries no warranty.


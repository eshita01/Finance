import pytest
from data_sources.news_sentiment_fetcher import NewsSentimentFetcher


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "```json\n{\"label\": \"positive\", \"score\": 0.8}\n```",
            {"label": "positive", "score": 0.8},
        ),
        (
            "Some text before {\"label\": \"negative\", \"score\": -0.5} and after",
            {"label": "negative", "score": -0.5},
        ),
    ],
)
def test_parse_gemini_json(raw, expected):
    data = NewsSentimentFetcher._parse_gemini_json(raw)
    assert data == expected

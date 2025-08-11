"""Run an ablation study over Gemini-style prompts using ChatGPT.

This script reads a log file containing prompts with various financial
features and sends them to a ChatGPT model while selectively removing
specific sections (news data and peer data) to measure their impact on
the model's prediction. Results and logs are written incrementally so
that partial progress is visible if execution stops.

The OpenAI API key should be supplied via the ``OPENAI_API_KEY``
environment variable. The model can be overridden with
``OPENAI_MODEL``.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration

LOG_FILE = os.path.join("results", "gemini_prompts.log")
CSV_FILE = os.path.join("results", "ablation_predictions.csv")
LOG_OUTPUT = os.path.join("results", "chatgpt_ablation.log")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Lines associated with news data that should be removed when performing
# news ablation.
NEWS_PREFIXES = [
    "Average sentiment:",
    "Tone:",
    "Dominant tone:",
    "Tone distribution:",
    "Headline summary:",
    "Urgency:",
    "Hype:",
    "Highlight headline:",
]


@dataclass
class PromptRecord:
    date: str
    lines: List[str]


def parse_prompt_log(path: str) -> List[PromptRecord]:
    """Parse the prompt log into a list of ``PromptRecord`` objects."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    records: List[PromptRecord] = []

    for chunk in chunks:
        lines = chunk.splitlines()
        header = lines[0]
        if "|" in header:
            date, first_prompt_line = header.split(" | ", 1)
        else:
            date, first_prompt_line = "", header

        prompt_lines = [first_prompt_line] + lines[1:]
        records.append(PromptRecord(date=date.strip(), lines=prompt_lines))

    return records


def ablate_lines(lines: Iterable[str], *, remove_news: bool, remove_peer: bool) -> str:
    """Return a prompt with optional removal of news and peer sections."""
    result: List[str] = []
    skip_peer_block = False

    for line in lines:
        if remove_news and any(line.startswith(prefix) for prefix in NEWS_PREFIXES):
            continue

        if remove_peer:
            if line.startswith("Peer data:"):
                skip_peer_block = True
                continue
            if skip_peer_block:
                if re.match(r"^[A-Z]+:", line.strip()):
                    # Still within the peer block; skip the line.
                    continue
                else:
                    skip_peer_block = False

        result.append(line)

    return "\n".join(result)


def call_model(prompt: str) -> str:
    """Send ``prompt`` to the ChatGPT model and return the response."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def ensure_result_files() -> None:
    """Ensure output directory and CSV header exist."""
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "variant", "prediction"])


def append_result(date: str, variant: str, prompt: str, prediction: str) -> None:
    """Append the prediction and log the interaction."""
    with open(LOG_OUTPUT, "a", encoding="utf-8") as f:
        f.write(f"Date: {date}, Variant: {variant}\n")
        f.write(f"Prompt:\n{prompt}\n")
        f.write(f"Response: {prediction}\n\n")

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([date, variant, prediction])


def main() -> None:
    ensure_result_files()

    records = parse_prompt_log(LOG_FILE)

    variants: List[Tuple[str, bool, bool]] = [
        ("baseline", False, False),
        ("no_news", True, False),
        ("no_peer", False, True),
        ("no_news_no_peer", True, True),
    ]

    for record in records:
        for variant, remove_news, remove_peer in variants:
            prompt = ablate_lines(record.lines, remove_news=remove_news, remove_peer=remove_peer)
            try:
                prediction = call_model(prompt)
            except Exception as exc:  # pragma: no cover - network failure
                prediction = f"ERROR: {exc}"
            append_result(record.date, variant, prompt, prediction)


if __name__ == "__main__":
    main()


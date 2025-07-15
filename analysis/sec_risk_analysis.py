import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional


import google.generativeai as genai
import pdfplumber

logger = logging.getLogger(__name__)


class SECRiskAnalyzer:
    """Parse SEC filing PDF and summarize key risk sections using an LLM."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        """Configure the Google Generative AI model and cache directory."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.analysis_dir = Path("cache/sec_analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def summarize_text_with_llm(self, text: str) -> str:
        """Summarize arbitrary text using the configured LLM."""
        prompt = f"Summarize the following text in a few sentences:\n{text[:4000]}"
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except Exception as exc:
            logger.exception("LLM summarization failed: %s", exc)
            return ""

    def _extract_text(self, pdf_path: Path) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)


    def _parse_sections(self, text: str, form: str) -> Dict[str, str]:
        """Extract risk factors and MD&A sections based on form type."""
        sections: Dict[str, str] = {"risk": "", "mdna": ""}

        if form.upper() == "10-K":
            risk_pat = re.compile(r"item\s*1a[^\n]*risk factors(?P<section>.*?)item\s*1b", re.IGNORECASE | re.DOTALL)
            mdna_pat = re.compile(r"item\s*7[^\n]*management's discussion and analysis(?P<section>.*?)item\s*8", re.IGNORECASE | re.DOTALL)
        elif form.upper() == "10-Q":
            risk_pat = re.compile(r"item\s*1a[^\n]*risk factors(?P<section>.*?)item\s*2", re.IGNORECASE | re.DOTALL)
            mdna_pat = re.compile(r"item\s*2[^\n]*management's discussion and analysis(?P<section>.*?)item\s*3", re.IGNORECASE | re.DOTALL)
        else:
            return sections

        m = risk_pat.search(text)
        if m:
            sections["risk"] = m.group("section").strip()

        m = mdna_pat.search(text)
        if m:
            sections["mdna"] = m.group("section").strip()

        return sections

    def _analyze_section(self, section: str) -> Dict[str, str]:
        """Summarize the section and classify its sentiment."""
        if not section:
            return {"summary": "", "sentiment": "Neutral", "score": 0.0}

        summary = self.summarize_text_with_llm(section)
        sentiment = "Neutral"
        score = 0.0

        prompt = (
            "Classify the sentiment of the following text as Positive, Neutral, or Negative "
            "and provide a numeric score between -1 and 1 as JSON with keys sentiment and score.\n"
            f"Text:\n{summary[:4000]}"
        )
        try:
            resp = self.model.generate_content(prompt)
            data = json.loads(resp.text)
            sentiment = data.get("sentiment", "Neutral")
            score = float(data.get("score", 0.0))
        except Exception as exc:
            logger.warning("Sentiment classification failed: %s", exc)
            if "positive" in summary.lower():
                sentiment = "Positive"
            elif "negative" in summary.lower():
                sentiment = "Negative"
        return {"summary": summary, "sentiment": sentiment, "score": score}


    def analyze(self, meta: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a downloaded SEC filing and cache the results."""
        pdf_file = Path("data/sec_reports") / meta["filename"]
        json_file = self.analysis_dir / f"{meta['ticker']}_{meta['form']}_{meta['filing_date']}.json"
        if json_file.exists():
            logger.info("Using cached SEC analysis %s", json_file)
            return json.loads(json_file.read_text())

        text = self._extract_text(pdf_file)
        sections = self._parse_sections(text, meta["form"])
        risk = self._analyze_section(sections.get("risk", ""))
        mdna = self._analyze_section(sections.get("mdna", ""))

        results = {
            "ticker": meta["ticker"],
            "form_type": meta["form"],
            "report_date": meta["filing_date"],
            "risk_summary": risk["summary"],
            "mdna_summary": mdna["summary"],
            "risk_sentiment": risk["sentiment"],
            "mdna_sentiment": mdna["sentiment"],
        }

        json_file.write_text(json.dumps(results, indent=2))
        return results


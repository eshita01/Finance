import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional


import google.generativeai as genai
import pdfplumber
from bs4 import BeautifulSoup

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

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)

    def _extract_html_text(self, html_path: Path) -> str:
        html = html_path.read_text(errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n")
        return "\n".join(line.strip() for line in text.splitlines())

    def _extract_text(self, pdf_path: Path, html_path: Optional[Path] = None) -> str:
        if html_path and html_path.exists():
            return self._extract_html_text(html_path)
        return self._extract_pdf_text(pdf_path)

    def _clean_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = text.replace("&nbsp;", " ")
        text = re.sub(r"\s+", " ", text)
        return text


    def _parse_sections(self, text: str, form: str) -> Dict[str, str]:
        """Extract risk factors and MD&A sections based on form type."""
        sections: Dict[str, str] = {"risk": "", "mdna": ""}

        text = self._clean_text(text).lower()


        def find_section(src: str, start_pat: str, end_pat: str) -> str:
            start = re.search(start_pat, src, re.IGNORECASE)
            if not start:
                logger.info("Section start not found: %s", start_pat)

                return ""
            remainder = src[start.end():]
            end = re.search(end_pat, remainder, re.IGNORECASE)
            if end:
                return remainder[: end.start()].strip()
            return remainder.strip()

        if form.upper() == "10-K":
            sections["risk"] = find_section(
                text,
                r"item\s*1a\.?\s*risk factors",
                r"item\s*1b",
            )
            sections["mdna"] = find_section(
                text,
                r"item\s*7\.?\s*management'?s discussion",
                r"item\s*7a|item\s*8",
            )
        elif form.upper() == "10-Q":
            sections["risk"] = find_section(
                text,
                r"item\s*1a\.?\s*risk factors",
                r"item\s*2",
            )
            sections["mdna"] = find_section(
                text,
                r"item\s*2\.?\s*management'?s discussion",
                r"item\s*3",
            )

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
        html_file: Optional[Path] = None
        if meta.get("source_filename"):
            potential = Path("data/sec_reports") / meta["source_filename"]
            if potential.exists():
                html_file = potential
        json_file = self.analysis_dir / f"{meta['ticker']}_{meta['form']}_{meta['filing_date']}.json"
        if json_file.exists():
            logger.info("Using cached SEC analysis %s", json_file)
            return json.loads(json_file.read_text())

        text = self._extract_text(pdf_file, html_file)
        sections = self._parse_sections(text, meta["form"])
        risk_text = sections.get("risk", "")
        mdna_text = sections.get("mdna", "")

        print("Retrieved risk section length:", len(risk_text))
        print(risk_text[:500])
        print()
        print("Retrieved MD&A section length:", len(mdna_text))
        print(mdna_text[:500])
        print()

        risk = self._analyze_section(risk_text)
        mdna = self._analyze_section(mdna_text)

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


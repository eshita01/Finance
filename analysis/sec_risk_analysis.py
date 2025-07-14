import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import google.generativeai as genai
import pdfplumber

logger = logging.getLogger(__name__)


class SECRiskAnalyzer:
    """Parse SEC filing PDF and summarize key risk sections using an LLM."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.analysis_dir = Path("data/sec_analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def _extract_text(self, pdf_path: Path) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)

    def _parse_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        patterns = {
            "risk": re.compile(
                r"item\s*1a[^\n]*risk factors(?P<section>.*?)item\s*1b",
                re.IGNORECASE | re.DOTALL,
            ),
            "mdna": re.compile(
                r"item\s*7[^\n]*management's discussion and analysis(?P<section>.*?)item\s*8",
                re.IGNORECASE | re.DOTALL,
            ),
        }
        for key, pat in patterns.items():
            m = pat.search(text)
            if m:
                sections[key] = m.group("section").strip()
            else:
                sections[key] = ""
        return sections

    def _analyze_section(self, section: str) -> Dict[str, str]:
        if not section:
            return {"summary": "", "sentiment": "Neutral", "score": 0.0}
        prompt = (
            "Summarize the following SEC filing section and provide a sentiment "
            "classification (Positive, Neutral, Negative) with a score between -1 and 1.\n"
            f"Section:\n{section[:4000]}"
        )
        try:
            resp = self.model.generate_content(prompt)
            text = resp.text.strip()
        except Exception as e:
            logger.exception("LLM analysis failed: %s", e)
            text = ""
        summary = text
        sentiment = "Neutral"
        score = 0.0
        try:
            if "positive" in text.lower():
                sentiment = "Positive"
            elif "negative" in text.lower():
                sentiment = "Negative"
            match = re.search(r"score[:\s]+([-+]?[0-9]*\.?[0-9]+)", text)
            if match:
                score = float(match.group(1))
        except Exception:
            pass
        return {"summary": summary, "sentiment": sentiment, "score": score}

    def analyze(self, meta: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        pdf_file = Path("data/sec_reports") / meta["filename"]
        json_file = self.analysis_dir / f"{meta['ticker']}_{meta['form']}_{meta['filing_date']}.json"
        if json_file.exists():
            logger.info("Using cached SEC analysis %s", json_file)
            return json.loads(json_file.read_text())

        text = self._extract_text(pdf_file)
        sections = self._parse_sections(text)
        results = {
            "risk_factors": self._analyze_section(sections.get("risk", "")),
            "mdna": self._analyze_section(sections.get("mdna", "")),
        }
        json_file.write_text(json.dumps(results, indent=2))
        return results


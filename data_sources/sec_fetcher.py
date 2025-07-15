import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from bs4 import BeautifulSoup
from fpdf import FPDF
from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)


class SECFetcher:
    """Download the latest 10-K or 10-Q filing as a PDF."""

    def __init__(
        self,
        ticker: str,
        form: str = "10-K",
        company: str = "MyCompany",
        email: str = "email@example.com",
        download_dir: Optional[Path] = None,
    ) -> None:
        self.ticker = ticker.upper()
        self.form = form
        self.download_dir = Path(download_dir or "data/sec_reports")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = Downloader(company, email, str(self.download_dir))

    def _existing_file(self) -> Optional[Path]:
        pattern = f"{self.ticker}_{self.form}_*.pdf"
        files = sorted(self.download_dir.glob(pattern), reverse=True)
        return files[0] if files else None

    def _parse_filing_date(self, text: str) -> str:
        patterns = [
            r"FILED AS OF DATE:\s*(\d{8})",
            r"FILING DATE:\s*(\d{4}-\d{2}-\d{2})",
            r"Filing Date:\s*(\d{4}-\d{2}-\d{2})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                date = m.group(1)
                if len(date) == 8:
                    return f"{date[:4]}-{date[4:6]}-{date[6:]}"
                return date
        return datetime.utcnow().date().isoformat()

    def _extract_text(self, file_path: Path) -> str:
        if file_path.suffix.lower() in {".htm", ".html"}:
            html = file_path.read_text(errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text("\n")
        return file_path.read_text(errors="ignore")

    def _text_to_pdf(self, text: str, pdf_path: Path) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.splitlines():
            pdf.multi_cell(0, 10, line)
        pdf.output(str(pdf_path))

    def fetch(self) -> Dict[str, str]:
        """Fetch the latest filing PDF and return metadata."""
        try:
            logger.info("Checking for existing SEC report")
            existing = self._existing_file()
            if existing:
                filing_date = existing.stem.split("_")[-1]
                logger.info("Using cached SEC filing %s", existing)
                return {
                    "ticker": self.ticker,
                    "form": self.form,
                    "filing_date": filing_date,
                    "filename": existing.name,
                }

            logger.info("Downloading latest %s for %s", self.form, self.ticker)
            self.downloader.get(self.form, self.ticker, limit=1, download_details=True)
        except Exception as e:
            logger.exception("SEC download failed: %s", e)
            raise

        try:
            filings_root = (
                self.download_dir
                / "sec-edgar-filings"
                / self.ticker
                / self.form
            )
            latest_dir = sorted(filings_root.iterdir(), reverse=True)[0]
            pdf_files = list(latest_dir.rglob("*.pdf"))
            text = ""
            if pdf_files:
                pdf_path = pdf_files[0]
                txt_candidates = list(latest_dir.rglob("*.txt"))
                if txt_candidates:
                    text = self._extract_text(txt_candidates[0])
            else:
                candidates = list(latest_dir.rglob("*.htm")) + list(latest_dir.rglob("*.html")) + list(latest_dir.rglob("*.txt"))
                if not candidates:
                    raise FileNotFoundError("No filing document found")
                candidate = candidates[0]
                text = self._extract_text(candidate)
                pdf_path = latest_dir / "converted.pdf"
                self._text_to_pdf(text, pdf_path)

            if not text:
                text = self._extract_text(pdf_path) if pdf_path.suffix.lower() != ".pdf" else ""

            filing_date = self._parse_filing_date(text)
            target_name = f"{self.ticker}_{self.form}_{filing_date}.pdf"
            target_path = self.download_dir / target_name
            pdf_path.replace(target_path)
            return {
                "ticker": self.ticker,
                "form": self.form,
                "filing_date": filing_date,
                "filename": target_name,
            }
        except Exception as e:
            logger.exception("Failed to process downloaded filing: %s", e)
            raise


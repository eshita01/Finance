import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from bs4 import BeautifulSoup
from fpdf import FPDF

from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)


class SECFetcher:
    """Download the latest 10-K or 10-Q filing as a PDF."""

    def __init__(
        self,
        ticker: str,
        company: str = "MyCompany",
        email: str = "email@example.com",
        download_dir: Optional[Path] = None,
    ) -> None:
        self.ticker = ticker.upper()
        self.download_dir = Path(download_dir or "data/sec_reports")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = Downloader(company, email, str(self.download_dir))

    def _existing_file(self) -> Optional[Path]:
        pattern = f"{self.ticker}_*_*.pdf"
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

    def _download_latest_forms(self) -> None:
        """Attempt to download the latest 10-K and 10-Q filings."""
        for form in ("10-K", "10-Q"):
            try:
                logger.info("Downloading latest %s for %s", form, self.ticker)
                self.downloader.get(form, self.ticker, limit=1, download_details=True)
            except Exception as exc:
                logger.warning("Download %s failed: %s", form, exc)

    def _latest_local_filing(self) -> Optional[Tuple[str, Path, str]]:
        """Return (form, path_to_folder, filing_date) for the most recent filing."""
        filings_root = self.download_dir / "sec-edgar-filings" / self.ticker
        latest_info: Optional[Tuple[str, Path, str]] = None
        for form in ("10-K", "10-Q"):
            form_dir = filings_root / form
            if not form_dir.exists():
                continue
            subdirs = sorted(form_dir.iterdir(), reverse=True)
            if not subdirs:
                continue
            folder = subdirs[0]
            candidates = list(folder.rglob("*.txt")) + list(folder.rglob("*.htm")) + list(folder.rglob("*.html"))
            if not candidates:
                continue
            text = self._extract_text(candidates[0])
            filing_date = self._parse_filing_date(text)
            if latest_info is None or filing_date > latest_info[2]:
                latest_info = (form, folder, filing_date)
        return latest_info

    def _text_to_pdf(self, text: str, pdf_path: Path) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.splitlines():
            pdf.multi_cell(0, 10, line)
        pdf.output(str(pdf_path))

    def fetch(self) -> Dict[str, str]:
        """Fetch the latest available 10-K or 10-Q and return metadata."""

        try:
            logger.info("Checking for existing SEC report")
            existing = self._existing_file()
            if existing:
                parts = existing.stem.split("_")
                form = parts[1]
                filing_date = parts[2]
                logger.info("Using cached SEC filing %s", existing)
                return {
                    "ticker": self.ticker,
                    "form": form,
                    "filing_date": filing_date,
                    "filename": existing.name,
                }

            self._download_latest_forms()
        except Exception as e:
            logger.exception("SEC download failed: %s", e)
            raise

        try:
            latest = self._latest_local_filing()
            if not latest:
                raise FileNotFoundError("No filing document found")

            form, folder, filing_date = latest
            pdf_files = list(folder.rglob("*.pdf"))
            text = ""
            if pdf_files:
                pdf_path = pdf_files[0]
                txt_candidates = list(folder.rglob("*.txt"))
                if txt_candidates:
                    text = self._extract_text(txt_candidates[0])
            else:
                candidates = list(folder.rglob("*.htm")) + list(folder.rglob("*.html")) + list(folder.rglob("*.txt"))
                if not candidates:
                    raise FileNotFoundError("No filing document found")
                candidate = candidates[0]
                text = self._extract_text(candidate)
                pdf_path = folder / "converted.pdf"
                self._text_to_pdf(text, pdf_path)

            if not text and pdf_path.suffix.lower() != ".pdf":
                text = self._extract_text(pdf_path)

            target_name = f"{self.ticker}_{form}_{filing_date}.pdf"
            target_path = self.download_dir / target_name
            if target_path.exists():
                logger.info("Using cached SEC filing %s", target_path)
                return {
                    "ticker": self.ticker,
                    "form": form,
                    "filing_date": filing_date,
                    "filename": target_name,
                }

            pdf_path.replace(target_path)
            return {
                "ticker": self.ticker,
                "form": form,

                "filing_date": filing_date,
                "filename": target_name,
            }
        except Exception as e:
            logger.exception("Failed to process downloaded filing: %s", e)
            raise


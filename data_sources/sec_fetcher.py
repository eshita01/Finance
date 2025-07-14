import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
            if not pdf_files:
                raise FileNotFoundError("No PDF in downloaded filing")
            pdf_path = pdf_files[0]
            filing_date = latest_dir.name.split("-")[1]
            target_name = f"{self.ticker}_{self.form}_{filing_date}.pdf"
            target_path = self.download_dir / target_name
            pdf_path.rename(target_path)
            return {
                "ticker": self.ticker,
                "form": self.form,
                "filing_date": filing_date,
                "filename": target_name,
            }
        except Exception as e:
            logger.exception("Failed to process downloaded filing: %s", e)
            raise


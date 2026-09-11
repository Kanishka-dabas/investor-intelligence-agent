from pathlib import Path
import signal
import pymupdf4llm
from loguru import logger


class PDFConversionTimeoutError(Exception):
    """Raised when PDF-to-markdown conversion exceeds the allowed time."""


class PDFToMarkdownConverter:
    """Convert PDF documents to Markdown"""

    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds

    def _timeout_handler(self, signum, frame):
        raise PDFConversionTimeoutError(
            f"PDF conversion exceeded {self.timeout_seconds}s"
        )

    def convert_pdf(self, pdf_path: str, output_dir: str) -> str:
        """
        Convert a PDF document to markdown.

        Args:
           pdf_path : source pdf path
           output_dir : Output markdown Directory
        Returns :
           Generated markdown filepath.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"pdf file not found : {pdf_path}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {pdf_file.name} to markdown...")

        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout_seconds)

        try:
            markdown_content = pymupdf4llm.to_markdown(str(pdf_file))
        finally:
            if has_alarm:
                signal.alarm(0)

        markdown_file = output_path / f"{pdf_file.stem}.md"
        markdown_file.write_text(markdown_content, encoding="utf-8")

        logger.info(f"Saved: {markdown_file}")
        return str(markdown_file)

    def convert_directory(self, input_dir: str, output_dir: str) -> list[str]:
        """
        Converts all pdfs from directory to markdown

        Args:
           input_dir : dir containing pdf files
           output_dir : dir to save markdown files
        Returns :
           List of generated markdown filepaths.
        """
        input_path = Path(input_dir)
        markdown_files = []
        for pdf_file in input_path.glob("*.pdf"):
            try:
                markdown_file = self.convert_pdf(
                    pdf_path=str(pdf_file), output_dir=output_dir
                )
                markdown_files.append(markdown_file)
            except PDFConversionTimeoutError as e:
                logger.error(f"Skipped {pdf_file.name}: {e}")
            except Exception as e:
                logger.error(f"Failed on {pdf_file.name}: {e}")
        return markdown_files

    def extract_layout_text(self, pdf_path: str) -> str:
        """
        Extracts text with spatial column alignment preserved (unlike
        markdown table conversion, which can corrupt multi-column
        financial tables with spanning headers). Used specifically for
        KPI extraction, where accurate column-to-period mapping matters.
        """
        import pdfplumber

        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True) or ""
                full_text.append(text)
        return "\n".join(full_text)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = repo_root / "data" / "raw_pdfs"
    output_dir = repo_root / "data" / "markdown"

    converter = PDFToMarkdownConverter()
    markdown_files = converter.convert_directory(
        input_dir=str(input_dir), output_dir=str(output_dir)
    )
    logger.info(f"Successfully converted {len(markdown_files)} PDFs!")    



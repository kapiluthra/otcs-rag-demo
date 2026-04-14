"""
extractor.py — Extract plain text from document bytes by MIME type

Supported types: PDF, DOCX, plain text
Unsupported types: return empty string (logged, not raised)
"""

import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def extract_text(content: bytes, mime_type: str) -> str:
    """Extract plain text from document bytes.

    Args:
        content: Raw document bytes (from OTCS content download)
        mime_type: MIME type string from Content-Type header

    Returns:
        Extracted plain text, or "" if extraction fails or type unsupported
    """
    if not content:
        return ""

    mime = mime_type.lower()

    try:
        if "pdf" in mime:
            return _extract_pdf(content)
        elif "wordprocessingml" in mime or "docx" in mime:
            return _extract_docx(content)
        elif "text/plain" in mime or "text/" in mime:
            return content.decode("utf-8", errors="replace")
        else:
            logger.debug("Unsupported MIME type for text extraction: %s", mime_type)
            return ""
    except Exception as e:
        logger.warning("Text extraction failed for mime=%s: %s", mime_type, e)
        return ""


def _extract_pdf(content: bytes) -> str:
    try:
        import pdfminer.high_level as pdfminer
        return pdfminer.extract_text(BytesIO(content)) or ""
    except ImportError:
        logger.error("pdfminer.six not installed — run: pip install pdfminer.six")
        return ""


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        logger.error("python-docx not installed — run: pip install python-docx")
        return ""

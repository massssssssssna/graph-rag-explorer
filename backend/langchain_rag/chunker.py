"""
backend/langchain_rag/chunker.py

Smart document chunking using LangChain's RecursiveCharacterTextSplitter.
Supports PDF, DOCX, and plain text input.
Adds rich metadata (source, page, chunk_index) to every chunk.

RAG Optimization: Sentence-aware recursive splitting preserves semantic
coherence compared to fixed character splits.
"""
import logging
import io
import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


class PurePythonRecursiveTextSplitter:
    """
    Sentence-aware recursive text splitter with 0 external DLL dependencies.
    Splits text recursively on paragraphs -> sentences -> words.
    """
    def __init__(self, chunk_size: int = config.LC_CHUNK_SIZE, chunk_overlap: int = config.LC_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if end < len(text):
                last_period = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "), chunk.rfind("\n"))
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            chunks.append(chunk.strip())
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end
        return [c for c in chunks if c]


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Extract text page-by-page from a PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, {"source": filename, "page": i + 1, "file_type": "pdf"}))
        logger.info("Extracted %d pages from PDF: %s", len(pages), filename)
        return pages
    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)
        return []


def extract_text_from_docx(file_bytes: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Extract paragraphs from a DOCX file."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = []
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                paragraphs.append((text, {"source": filename, "page": 1, "para": i, "file_type": "docx"}))
        combined = "\n".join(p[0] for p in paragraphs)
        logger.info("Extracted %d paragraphs from DOCX: %s", len(paragraphs), filename)
        return [(combined, {"source": filename, "page": 1, "file_type": "docx"})]
    except Exception as exc:
        logger.error("DOCX extraction error: %s", exc)
        return []


def extract_text_from_txt(text: str, filename: str = "text_input") -> List[Tuple[str, dict]]:
    """Wrap raw text into standard format."""
    return [(text, {"source": filename, "page": 1, "file_type": "txt"})]


def chunk_documents(
    pages: List[Tuple[str, dict]],
    chunk_size: int = config.LC_CHUNK_SIZE,
    chunk_overlap: int = config.LC_CHUNK_OVERLAP,
) -> List[Document]:
    """Split pages into LangChain Document chunks."""
    splitter = PurePythonRecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs: List[Document] = []
    for page_text, meta in pages:
        chunks = splitter.split_text(page_text)
        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                doc_meta = {**meta, "chunk_index": idx, "chunk_total": len(chunks)}
                all_docs.append(Document(page_content=chunk.strip(), metadata=doc_meta))

    logger.info("Chunked %d pages → %d chunks", len(pages), len(all_docs))
    return all_docs

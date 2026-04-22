"""Document ingestion (10-K, PDFs, plain text) for the investment panel."""

from aegean.investment.documents.provider import (
    DocumentChunk,
    DocumentProvider,
    DocumentResult,
    MarkerDocumentProvider,
    PlainTextDocumentProvider,
    chunk_markdown,
)

__all__ = [
    "DocumentChunk",
    "DocumentProvider",
    "DocumentResult",
    "MarkerDocumentProvider",
    "PlainTextDocumentProvider",
    "chunk_markdown",
]

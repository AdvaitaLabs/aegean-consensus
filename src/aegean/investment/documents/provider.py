"""Document providers that turn PDFs / 10-Ks / plain-text files into
structured evidence the investment panel can consume.

Inspired by FinRobot's use of the ``marker`` library for PDF → markdown
extraction. Marker is a heavy optional dependency, so we import it
lazily; if it isn't installed, :class:`MarkerDocumentProvider` still
imports cleanly and fails with a clear message only when ``ingest`` is
actually called.

The core abstraction is :class:`DocumentProvider.ingest`, which returns
a :class:`DocumentResult` containing ordered :class:`DocumentChunk`
records. Chunks are small enough to embed in prompts without blowing
the context window, and they carry enough metadata (title, section,
source path) that downstream citations can link back.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class DocumentChunk:
    index: int
    text: str
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "section": self.section,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class DocumentResult:
    source: str
    provider: str
    title: str = ""
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def combined_text(self, max_chunks: Optional[int] = None) -> str:
        chunks = self.chunks if max_chunks is None else self.chunks[:max_chunks]
        return "\n\n".join(chunk.text for chunk in chunks)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_markdown(
    markdown_text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Split markdown text into overlapping, section-aware chunks.

    Markdown headings reset the current section label so chunks are
    tagged with the nearest heading above them. Within a section we
    greedy-split on paragraph boundaries, falling back to a hard cut
    only when a single paragraph is larger than ``max_chars``.
    """
    if max_chars < 200:
        raise ValueError("max_chars must be >= 200")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    text = (markdown_text or "").replace("\r", "").strip()
    if not text:
        return []

    # Build (section, body) segments by scanning headings.
    segments: List[tuple] = []
    current_section = ""
    last_end = 0
    for match in _HEADING_RE.finditer(text):
        body = text[last_end:match.start()].strip()
        if body:
            segments.append((current_section, body))
        current_section = match.group(2).strip()
        last_end = match.end()
    tail = text[last_end:].strip()
    if tail:
        segments.append((current_section, tail))
    if not segments:
        segments = [("", text)]

    chunks: List[DocumentChunk] = []
    index = 0
    for section, body in segments:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buffer = ""
        for para in paragraphs:
            if len(para) > max_chars:
                if buffer:
                    chunks.append(DocumentChunk(index=index, text=buffer, section=section))
                    index += 1
                    buffer = ""
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    chunks.append(DocumentChunk(index=index, text=para[start:end], section=section))
                    index += 1
                    if end == len(para):
                        break
                    start = end - overlap
                continue
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                chunks.append(DocumentChunk(index=index, text=buffer, section=section))
                index += 1
                tail_start = max(0, len(buffer) - overlap)
                buffer = (buffer[tail_start:] + "\n\n" + para).strip() if overlap else para
                if len(buffer) > max_chars:
                    buffer = para
        if buffer:
            chunks.append(DocumentChunk(index=index, text=buffer, section=section))
            index += 1
    return chunks


class DocumentProvider(ABC):
    provider_name: str = "document"

    @abstractmethod
    def ingest(self, path: str, **options: Any) -> DocumentResult:
        raise NotImplementedError


class PlainTextDocumentProvider(DocumentProvider):
    """Ingest .txt / .md files directly. Useful for tests and for
    feeding already-extracted filings back into the panel.
    """

    provider_name = "plain_text"

    def __init__(
        self,
        max_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def ingest(self, path: str, **options: Any) -> DocumentResult:
        p = Path(path)
        text = p.read_text(encoding=options.get("encoding", "utf-8"))
        title = options.get("title") or p.stem
        chunks = chunk_markdown(text, max_chars=self.max_chars, overlap=self.overlap)
        for chunk in chunks:
            chunk.metadata["source"] = str(p)
        return DocumentResult(
            source=str(p),
            provider=self.provider_name,
            title=title,
            chunks=chunks,
            metadata={"char_count": len(text), "chunk_count": len(chunks)},
        )


class MarkerDocumentProvider(DocumentProvider):
    """PDF → markdown via the ``marker`` library (lazy-imported).

    We do not import ``marker`` at module load so this provider can be
    referenced even in environments where marker isn't installed. The
    error surfaces only when ``ingest`` is actually called.
    """

    provider_name = "marker_pdf"

    def __init__(
        self,
        max_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        converter_factory: Optional[Any] = None,
    ) -> None:
        self.max_chars = max_chars
        self.overlap = overlap
        self._converter_factory = converter_factory

    def _build_converter(self) -> Any:
        if self._converter_factory is not None:
            return self._converter_factory()
        try:
            from marker.converters.pdf import PdfConverter  # type: ignore
            from marker.models import create_model_dict  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "MarkerDocumentProvider requires the 'marker-pdf' package. "
                "Install it with `pip install marker-pdf` or inject a "
                "converter_factory for testing."
            ) from exc
        return PdfConverter(artifact_dict=create_model_dict())

    def ingest(self, path: str, **options: Any) -> DocumentResult:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        converter = self._build_converter()
        rendered = converter(str(p))
        markdown_text = _extract_markdown(rendered)
        title = options.get("title") or p.stem
        chunks = chunk_markdown(markdown_text, max_chars=self.max_chars, overlap=self.overlap)
        for chunk in chunks:
            chunk.metadata["source"] = str(p)
        return DocumentResult(
            source=str(p),
            provider=self.provider_name,
            title=title,
            chunks=chunks,
            metadata={
                "char_count": len(markdown_text),
                "chunk_count": len(chunks),
            },
        )


def _extract_markdown(rendered: Any) -> str:
    """Pull the markdown payload out of whatever marker returned.

    Marker's return shape has shifted between releases (string, tuple,
    or an object with a ``.markdown`` attribute), so we probe a few
    shapes rather than pinning to one.
    """
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, tuple) and rendered:
        first = rendered[0]
        if isinstance(first, str):
            return first
    for attr in ("markdown", "text", "content"):
        value = getattr(rendered, attr, None)
        if isinstance(value, str):
            return value
    raise TypeError(f"Unrecognized marker output: {type(rendered)!r}")

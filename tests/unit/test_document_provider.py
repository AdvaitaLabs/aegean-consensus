"""Unit tests for document chunking and providers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aegean.investment.documents import (
    DocumentChunk,
    MarkerDocumentProvider,
    PlainTextDocumentProvider,
    chunk_markdown,
)
from aegean.investment.documents.provider import _extract_markdown


def test_chunk_markdown_tags_sections_from_headings():
    text = (
        "# Risk Factors\n\nOur supply chain is concentrated in East Asia.\n\n"
        "## Regulatory\n\nExport controls could restrict our market access.\n\n"
        "# MD&A\n\nRevenue grew 12% year over year."
    )
    chunks = chunk_markdown(text, max_chars=400)
    sections = [c.section for c in chunks]
    assert "Risk Factors" in sections
    assert "Regulatory" in sections
    assert "MD&A" in sections


def test_chunk_markdown_splits_on_size_limit():
    paragraph = "lorem ipsum " * 200  # ~2400 chars, one paragraph
    chunks = chunk_markdown(paragraph, max_chars=500, overlap=50)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert len(chunk.text) <= 500


def test_chunk_markdown_empty_returns_empty():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_chunk_markdown_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_markdown("x", max_chars=100)
    with pytest.raises(ValueError):
        chunk_markdown("x", max_chars=500, overlap=500)


def test_plain_text_provider_reads_and_chunks(tmp_path: Path):
    filing = tmp_path / "10k.md"
    filing.write_text(
        "# Item 1A. Risk Factors\n\nSupply chain risk remains material.\n\n"
        "# Item 7. MD&A\n\nOperating margin expanded to 28%."
    )
    provider = PlainTextDocumentProvider(max_chars=500)
    result = provider.ingest(str(filing))
    assert result.provider == "plain_text"
    assert result.title == "10k"
    assert len(result.chunks) >= 2
    sources = {c.metadata.get("source") for c in result.chunks}
    assert sources == {str(filing)}
    assert result.metadata["chunk_count"] == len(result.chunks)


def test_marker_provider_uses_injected_converter(tmp_path: Path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")

    captured = {}

    def fake_factory():
        def _convert(path: str) -> str:
            captured["path"] = path
            return "# Section A\n\nText one.\n\n# Section B\n\nText two."
        return _convert

    provider = MarkerDocumentProvider(
        max_chars=400,
        converter_factory=fake_factory,
    )
    result = provider.ingest(str(pdf_path), title="FakeFiling")
    assert captured["path"] == str(pdf_path)
    assert result.title == "FakeFiling"
    assert result.provider == "marker_pdf"
    assert any("Section A" == c.section for c in result.chunks)
    assert any("Section B" == c.section for c in result.chunks)


def test_marker_provider_missing_file_raises(tmp_path: Path):
    provider = MarkerDocumentProvider(converter_factory=lambda: (lambda p: ""))
    with pytest.raises(FileNotFoundError):
        provider.ingest(str(tmp_path / "nope.pdf"))


def test_extract_markdown_handles_multiple_shapes():
    class _Obj:
        markdown = "# hi"
    assert _extract_markdown("raw md") == "raw md"
    assert _extract_markdown(("tuple md", {"meta": 1})) == "tuple md"
    assert _extract_markdown(_Obj()) == "# hi"
    with pytest.raises(TypeError):
        _extract_markdown(object())


def test_document_chunk_to_dict_roundtrip():
    chunk = DocumentChunk(index=3, text="body", section="Risk", metadata={"k": "v"})
    d = chunk.to_dict()
    assert d == {"index": 3, "text": "body", "section": "Risk", "metadata": {"k": "v"}}

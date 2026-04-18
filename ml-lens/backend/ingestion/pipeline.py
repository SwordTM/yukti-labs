from __future__ import annotations

from pathlib import Path
from typing import Optional

from schema.models import ComponentManifest

from .arxiv_resolver import resolve_arxiv
from .component_extractor import extract_manifest
from .pdf_parser import parse_pdf


def ingest_paper(
    url_or_id: str, download_dir: Optional[Path] = None
) -> ComponentManifest:
    """End-to-end ingestion: arXiv URL → ComponentManifest.

    Stages:
      1. Resolve arXiv id, download PDF.
      2. Parse PDF with Docling → markdown + equations.
      3. Call Claude with the locked-schema extraction prompt.
      4. Validate response into ComponentManifest and return.
    """
    paper = resolve_arxiv(url_or_id, download_dir=download_dir)
    parsed = parse_pdf(paper.pdf_path)
    return extract_manifest(paper, parsed)

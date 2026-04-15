from __future__ import annotations

import subprocess
from pathlib import Path

from graphify_rag.models import Document
from graphify_rag.utils import slugify


class PdfExtractionError(RuntimeError):
    pass


def extract_pdf_text(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PdfExtractionError("pdftotext is required but not installed.") from exc
    except subprocess.CalledProcessError as exc:
        raise PdfExtractionError(f"Failed to extract text from {path.name}: {exc.stderr}") from exc
    return result.stdout.strip()


def load_documents(input_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(input_dir.glob("*.pdf")):
        content = extract_pdf_text(path)
        title = next((line.strip() for line in content.splitlines() if line.strip()), path.stem)
        doc_id = slugify(path.stem)
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                path=path,
                content=content,
                metadata={"source_type": "pdf"},
            )
        )
    return documents

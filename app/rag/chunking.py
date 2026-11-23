"""Split the markdown knowledge base into one chunk per `##` section."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "kb" / "docs"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    content: str

    @property
    def text_for_embedding(self) -> str:
        return f"{self.title} - {self.section}. {self.content}"

    @property
    def citation(self) -> str:
        return f"{self.title} > {self.section}"


def chunk_markdown(doc_name: str, markdown: str) -> list[Chunk]:
    title_match = re.search(r"^#\s+(.+)$", markdown, re.M)
    title = title_match.group(1).strip() if title_match else doc_name
    chunks: list[Chunk] = []
    parts = re.split(r"^##\s+", markdown, flags=re.M)[1:]
    for i, part in enumerate(parts):
        heading, _, body = part.partition("\n")
        body = body.strip()
        if body:
            chunks.append(Chunk(id=f"{doc_name}#{i}", doc=doc_name, title=title, section=heading.strip(), content=body))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks

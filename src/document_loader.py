from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    raise ValueError(f"Unsupported document type: {path}")


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = end - overlap
    return chunks


def iter_document_paths(docs_dir: Path = DOCS_DIR) -> list[Path]:
    if not docs_dir.exists():
        return []
    supported = {".md", ".txt", ".pdf"}
    return sorted(path for path in docs_dir.rglob("*") if path.suffix.lower() in supported)


def build_chunks(docs_dir: Path = DOCS_DIR, chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_document_paths(docs_dir):
        text = read_document(path)
        rel_source = str(path.relative_to(PROJECT_ROOT))
        for chunk_id, chunk_text in enumerate(split_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(Chunk(source=rel_source, chunk_id=chunk_id, text=chunk_text))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(chunk) for chunk in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path = CHUNKS_PATH, rebuild: bool = False) -> list[Chunk]:
    if rebuild or not path.exists():
        chunks = build_chunks()
        save_chunks(chunks, path)
        return chunks
    for attempt in range(5):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            break
        except (json.JSONDecodeError, OSError):
            if attempt == 4:
                chunks = build_chunks()
                save_chunks(chunks, path)
                return chunks
            time.sleep(0.05)
    return [Chunk(**item) for item in payload]


if __name__ == "__main__":
    built_chunks = build_chunks()
    save_chunks(built_chunks)
    print(f"saved {len(built_chunks)} chunks to {CHUNKS_PATH}")

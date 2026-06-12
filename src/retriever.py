from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import ARTIFACTS_DIR, Chunk, build_chunks, load_chunks, save_chunks


INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "bge_faiss.index"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "bge_embeddings.npy"
DEFAULT_BGE_MODEL = "BAAI/bge-small-zh-v1.5"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        ...


class LocalTfidfRetriever:
    name = "tfidf-char-ngram"

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No document chunks found. Put .md/.txt files under data/docs.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=30000,
        )
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [SearchResult(chunk=self.chunks[index], score=float(scores[index])) for index in ranked]


class BgeFaissRetriever:
    name = "bge-small-faiss"

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_BGE_MODEL, rebuild: bool = False):
        if not chunks:
            raise ValueError("No document chunks found. Put .md/.txt files under data/docs.")
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "BGE+FAISS requires `sentence-transformers` and `faiss-cpu`. "
                "Install requirements or use RAG_RETRIEVER=tfidf."
            ) from exc

        self.faiss = faiss
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        if rebuild or not FAISS_INDEX_PATH.exists() or not EMBEDDINGS_PATH.exists():
            embeddings = self._encode([chunk.text for chunk in chunks])
            self.index = self._build_index(embeddings)
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            np.save(EMBEDDINGS_PATH, embeddings)
        else:
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def _build_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        index = self.faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vec = self._encode([query])
        scores, indices = self.index.search(query_vec, top_k)
        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[int(index)], score=float(score)))
        return results


def write_meta(retriever_name: str, chunks: list[Chunk], extra: dict | None = None) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "retriever": retriever_name,
        "chunk_count": len(chunks),
        "sources": sorted({chunk.source for chunk in chunks}),
    }
    if extra:
        meta.update(extra)
    INDEX_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_retriever(
    rebuild: bool = False,
    backend: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> Retriever:
    backend = (backend or os.getenv("RAG_RETRIEVER") or "tfidf").lower()

    if chunk_size is not None or overlap is not None:
        chunks = build_chunks(
            chunk_size=chunk_size or 700,
            overlap=overlap if overlap is not None else 120,
        )
        save_chunks(chunks)
    else:
        chunks = load_chunks(rebuild=rebuild)

    if backend in {"auto", "bge", "bge-faiss", "faiss"}:
        try:
            model_name = os.getenv("BGE_MODEL_NAME", DEFAULT_BGE_MODEL)
            retriever = BgeFaissRetriever(chunks, model_name=model_name, rebuild=rebuild)
            write_meta(retriever.name, chunks, {"model_name": model_name})
            return retriever
        except RuntimeError as exc:
            if backend != "auto":
                raise
            print(f"[retriever] BGE+FAISS unavailable, fallback to TF-IDF: {exc}")

    retriever = LocalTfidfRetriever(chunks)
    write_meta(retriever.name, chunks)
    return retriever


def format_result(result: SearchResult, rank: int, preview_chars: int = 320) -> str:
    preview = result.chunk.text[:preview_chars].replace("\n", " ")
    return (
        f"[{rank}] score={result.score:.3f}\n"
        f"source={result.chunk.source}#{result.chunk.chunk_id}\n"
        f"excerpt={preview}"
    )


if __name__ == "__main__":
    retriever = build_retriever(rebuild=True)
    for rank, result in enumerate(retriever.search("MiniMind 的 SFT 阶段做了什么？"), start=1):
        print(format_result(result, rank))
        print()

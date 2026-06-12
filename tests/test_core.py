from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import answer, route
from document_loader import Chunk, load_chunks, save_chunks, split_text
from tools import calculator, clear_retriever_cache, get_retriever


class DocumentLoaderTests(unittest.TestCase):
    def test_split_text_uses_overlap(self) -> None:
        chunks = split_text("abcdefghij", chunk_size=6, overlap=2)
        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_split_text_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            split_text("abc", chunk_size=3, overlap=3)

    def test_chunk_cache_round_trip(self) -> None:
        cache_path = PROJECT_ROOT / "artifacts" / "test_chunks.json"
        chunks = [Chunk(source="test.md", chunk_id=0, text="hello")]
        save_chunks(chunks, cache_path)
        self.assertEqual(load_chunks(cache_path), chunks)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clear_retriever_cache()

    def test_routes_supported_tools(self) -> None:
        self.assertEqual(route("计算 12 * 3"), "calculator")
        self.assertEqual(route("总结 MiniMind 的训练流程"), "summarize_doc")
        self.assertEqual(route("MiniMind 的模型结构是什么？"), "search_docs")
        self.assertEqual(route("Qwen2.5 模型有多少参数？"), "search_docs")

    def test_calculator_evaluates_safe_expression(self) -> None:
        self.assertEqual(calculator("计算 12 * (3 + 2)"), "calculator: 12 * (3 + 2) = 60")

    def test_tfidf_retriever_is_cached(self) -> None:
        first = get_retriever(backend="tfidf")
        second = get_retriever(backend="tfidf")
        self.assertIs(first, second)

    def test_local_answer_contains_source(self) -> None:
        response = answer("MiniMind 的 SFT 阶段做了什么？", backend="tfidf", top_k=2)
        self.assertIn("[Agent route: search_docs]", response)
        self.assertIn("minimind_notes.md", response)
        self.assertIn("loss mask", response)


if __name__ == "__main__":
    unittest.main()

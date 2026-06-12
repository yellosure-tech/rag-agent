from __future__ import annotations

import ast
import operator
import re
from collections import Counter

from generator import generate_answer
from retriever import Retriever, SearchResult, build_retriever, format_result


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_RETRIEVER_CACHE: dict[str, Retriever] = {}


def get_retriever(backend: str | None = None, rebuild: bool = False) -> Retriever:
    cache_key = backend or "default"
    if rebuild or cache_key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[cache_key] = build_retriever(rebuild=rebuild, backend=backend)
    return _RETRIEVER_CACHE[cache_key]


def clear_retriever_cache() -> None:
    _RETRIEVER_CACHE.clear()


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("Only numeric expressions with +, -, *, /, //, %, ** are allowed.")


def calculator(expression: str) -> str:
    raw_expression = expression.strip()
    expression = re.sub(r"[^0-9+\-*/().% ]", "", raw_expression).strip()
    if not expression:
        return "calculator: 未识别到可计算表达式。"
    try:
        value = _eval_ast(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError):
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", raw_expression)]
        if len(numbers) >= 2 and ("比例" in raw_expression or "占" in raw_expression or "除以" in raw_expression):
            value = numbers[0] / numbers[1]
            expression = f"{numbers[0]} / {numbers[1]}"
        elif len(numbers) >= 2 and ("乘" in raw_expression or "共" in raw_expression):
            value = numbers[0] * numbers[1]
            expression = f"{numbers[0]} * {numbers[1]}"
        else:
            return "calculator: 只支持基础数值表达式，或包含“除以/比例/乘/共”的简单中文算式。"
    return f"calculator: {expression} = {value:.6g}"


def search_docs(
    query: str,
    top_k: int = 3,
    backend: str | None = None,
    rebuild: bool = False,
) -> tuple[str, list[SearchResult]]:
    retriever = get_retriever(backend=backend, rebuild=rebuild)
    results = retriever.search(query, top_k=top_k)
    lines = ["search_docs: 检索到以下证据："]
    for rank, result in enumerate(results, start=1):
        lines.append(format_result(result, rank))
    return "\n\n".join(lines), results


def summarize_doc(query: str, top_k: int = 4, backend: str | None = None) -> str:
    _, results = search_docs(query, top_k=top_k, backend=backend)
    text = "\n".join(result.chunk.text for result in results)
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    english_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", query.lower())
    chinese_text = "".join(re.findall(r"[\u4e00-\u9fff]", query))
    chinese_terms = [chinese_text[index:index + 2] for index in range(max(0, len(chinese_text) - 1))]
    keywords = set(english_terms + chinese_terms)
    if keywords:
        scored = []
        for index, sentence in enumerate(sentences):
            score = sum(1 for keyword in keywords if keyword in sentence.lower())
            if score > 0:
                scored.append((score, -index, sentence))
        selected = [sentence for _, _, sentence in sorted(scored, reverse=True)[:4]]
    else:
        selected = sentences[:4]
    selected = [sentence.strip() for sentence in selected if sentence.strip()]
    if not selected:
        selected = [result.chunk.text[:180] for result in results]

    sources = Counter(f"{result.chunk.source}#{result.chunk.chunk_id}" for result in results)
    source_line = "；".join(sources.keys())
    summary = "\n".join(f"- {sentence}" for sentence in selected[:4])
    return f"summarize_doc: 基于检索片段的摘要：\n{summary}\n来源：{source_line}"


def synthesize_answer(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> str:
    answer = generate_answer(question, results, history=history)
    sources = "；".join(f"{result.chunk.source}#{result.chunk.chunk_id}" for result in results)
    return f"{answer}\n\n引用来源：{sources}"

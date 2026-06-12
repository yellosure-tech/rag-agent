from __future__ import annotations

import argparse
import re

from tools import calculator, search_docs, summarize_doc, synthesize_answer


def route(question: str) -> str:
    lowered = question.lower()
    if "工具" in question and ("search_docs" in lowered or "summarize_doc" in lowered or "calculator" in lowered):
        return "search_docs"
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    has_expression = bool(re.search(r"\d+(?:\.\d+)?\s*[+\-*/%]\s*\d", question))
    calc_keywords = ["计算", "等于", "tokens/s", "token/s", "多久", "多少", "比例", "占", "除以", "乘以", "增加", "共", "steps", "step"]
    summary_keywords = ["总结", "概括", "归纳", "梳理", "summary", "summarize"]
    asks_calculation = any(keyword in lowered or keyword in question for keyword in calc_keywords)
    if has_expression or ("计算" in question and numbers) or (asks_calculation and len(numbers) >= 2):
        return "calculator"
    if any(keyword in lowered or keyword in question for keyword in summary_keywords):
        return "summarize_doc"
    return "search_docs"


def answer(
    question: str,
    top_k: int = 3,
    history: list[tuple[str, str]] | None = None,
    backend: str | None = None,
    rebuild: bool = False,
) -> str:
    decision = route(question)
    if decision == "calculator":
        return "[Agent route: calculator]\n" + calculator(question)
    if decision == "summarize_doc":
        return "[Agent route: summarize_doc]\n" + summarize_doc(question, top_k=top_k, backend=backend)

    evidence, results = search_docs(question, top_k=top_k, backend=backend, rebuild=rebuild)
    return "[Agent route: search_docs]\n" + evidence + "\n\n" + synthesize_answer(question, results, history=history)


def interactive_loop(top_k: int, backend: str | None = None, rebuild: bool = False) -> None:
    history: list[tuple[str, str]] = []
    print("Local RAG Agent interactive mode. Type `exit` to quit.")
    while True:
        question = input("\nUser> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        response = answer(question, top_k=top_k, history=history, backend=backend, rebuild=rebuild)
        rebuild = False
        print(f"\nAssistant>\n{response}")
        history.append((question, response[:1200]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="用户问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回片段数")
    parser.add_argument("--backend", default=None, choices=["tfidf", "auto", "bge-faiss"], help="检索后端")
    parser.add_argument("--rebuild", action="store_true", help="重新切分文档并构建索引")
    parser.add_argument("--interactive", action="store_true", help="进入多轮问答模式")
    args = parser.parse_args()

    if args.interactive:
        interactive_loop(top_k=args.top_k, backend=args.backend, rebuild=args.rebuild)
        return
    if not args.question:
        print('Usage: python .\\src\\agent.py "your question"')
        return
    question = " ".join(args.question)
    print(answer(question, top_k=args.top_k, backend=args.backend, rebuild=args.rebuild))


if __name__ == "__main__":
    main()

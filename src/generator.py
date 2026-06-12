from __future__ import annotations

import os
import re
from dataclasses import dataclass

from retriever import SearchResult


@dataclass
class GenerationConfig:
    backend: str = "extractive"
    model: str = ""
    max_tokens: int = 512


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for rank, result in enumerate(results, start=1):
        source = f"{result.chunk.source}#{result.chunk.chunk_id}"
        text = result.chunk.text.replace("\n", " ")
        blocks.append(f"[{rank}] source={source}, score={result.score:.3f}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[SearchResult], history: list[tuple[str, str]] | None = None) -> str:
    history = history or []
    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
    context = build_context(results)
    return (
        "你是一个严谨的本地知识库问答助手。请只依据给定检索证据回答；"
        "如果证据不足，请明确说明不足。回答后列出引用来源。\n\n"
        f"历史对话：\n{history_text or '无'}\n\n"
        f"检索证据：\n{context}\n\n"
        f"问题：{question}\n回答："
    )


def generate_with_openai(prompt: str, model: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def generate_with_qwen(prompt: str, model_name_or_path: str, max_tokens: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_extractive(question: str, results: list[SearchResult]) -> str:
    english_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", question.lower())
    chinese_text = "".join(re.findall(r"[\u4e00-\u9fff]", question))
    chinese_terms = [chinese_text[index:index + 2] for index in range(max(0, len(chinese_text) - 1))]
    terms = set(english_terms + chinese_terms)

    candidates: list[tuple[float, str, str]] = []
    for result in results:
        source = f"{result.chunk.source}#{result.chunk.chunk_id}"
        clean_text = re.sub(r"#+\s*", "", result.chunk.text.replace("\n", " "))
        sentences = re.split(r"(?<=[。！？.!?])\s*", clean_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue
            normalized = sentence.lower()
            overlap_score = sum(1 for term in terms if term in normalized)
            score = overlap_score + result.score
            candidates.append((score, sentence, source))

    selected = []
    seen = set()
    for _, sentence, source in sorted(candidates, key=lambda item: item[0], reverse=True):
        if sentence in seen:
            continue
        selected.append(f"- {sentence}（来源：{source}）")
        seen.add(sentence)
        if len(selected) == 3:
            break

    if not selected:
        return "未检索到足够证据，无法基于本地知识库回答。"
    return "基于检索证据的回答：\n" + "\n".join(selected)


def generate_answer(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
    config: GenerationConfig | None = None,
) -> str:
    backend = (config.backend if config else os.getenv("RAG_GENERATOR", "extractive")).lower()
    model = config.model if config and config.model else os.getenv("RAG_LLM_MODEL", "")
    max_tokens = config.max_tokens if config else int(os.getenv("RAG_MAX_TOKENS", "512"))

    if backend == "openai":
        model = model or "gpt-3.5-turbo"
        prompt = build_prompt(question, results, history)
        return generate_with_openai(prompt, model=model, max_tokens=max_tokens)

    if backend == "qwen":
        model = model or os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2-1.5B-Instruct")
        prompt = build_prompt(question, results, history)
        return generate_with_qwen(prompt, model_name_or_path=model, max_tokens=max_tokens)

    return generate_extractive(question, results)

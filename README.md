# Local RAG Agent

一个可在本地直接运行的轻量级 RAG + Tool Use Agent 项目。默认使用 TF-IDF
字符 n-gram 检索和抽取式回答，不需要 GPU、外部 API 或下载模型；也可选用
BGE-small + FAISS 检索、OpenAI API 或本地 Qwen 模型。

## 功能

- 读取 Markdown、TXT 文档，可选支持 PDF
- 支持可配置的 chunk size 与 overlap
- 默认使用 TF-IDF 检索，可选 BGE-small + FAISS
- 回答附带来源文件、chunk 编号和检索分数
- 支持 `search_docs`、`summarize_doc`、`calculator` 工具路由
- 支持多轮交互，并在会话内复用检索器
- 内置 50 条评测集，输出路由准确率、Hit@K、MRR 和平均检索延迟
- 支持 chunk size、overlap、top-k 参数扫描

## 快速开始

```powershell
cd F:\minimind-internship\rag_agent
python -m pip install -r requirements.txt

python .\src\agent.py "MiniMind 的 SFT 阶段做了什么？"
python .\src\agent.py "总结 MiniMind 的训练链路"
python .\src\agent.py "计算 3000000000 / 65536"
python .\src\agent.py --interactive --top-k 3
```

默认检索后端为 `tfidf`。修改 `data/docs/` 中的文档后，使用 `--rebuild` 重新生成
chunk：

```powershell
python .\src\agent.py --rebuild "RAG 的基本流程是什么？"
```

## 评测与参数优化

```powershell
python .\src\eval.py --backend tfidf --top-k 3
python .\src\optimize_retrieval.py --backend tfidf --chunk-sizes 256,512,700 --overlaps 64,120 --top-ks 3,5
python -m unittest discover -s tests -v
```

评测报告保存到 `artifacts/eval_report.json`，包括：

- `route_accuracy`：工具路由准确率
- `source_hit_at_k`：期望来源是否出现在前 K 个结果
- `mean_reciprocal_rank`：正确来源排名质量
- `mean_retrieval_latency_ms`：平均单次检索延迟

参数扫描结果保存到 `artifacts/optimization_report.json`，其中包含全部配置结果和
按 Hit@K、MRR、延迟选择出的最佳配置。

## 可选：BGE-small + FAISS

安装可选依赖，并确保模型可下载或已缓存：

```powershell
python -m pip install -r requirements-optional.txt
python .\src\agent.py --backend bge-faiss --rebuild "RAG 为什么能减少幻觉？"
python .\src\eval.py --backend bge-faiss --top-k 5
```

可通过环境变量指定本地模型路径：

```powershell
$env:BGE_MODEL_NAME="BAAI/bge-small-zh-v1.5"
```

## 可选：LLM 生成

默认 `RAG_GENERATOR=extractive`，无需外部服务。

OpenAI API：

```powershell
$env:RAG_GENERATOR="openai"
$env:RAG_LLM_MODEL="gpt-4o-mini"
$env:OPENAI_API_KEY="..."
python .\src\agent.py "RAG 的基本流程是什么？"
```

本地 Qwen：

```powershell
$env:RAG_GENERATOR="qwen"
$env:QWEN_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
python .\src\agent.py "总结 MiniMind 的训练流程"
```

## 目录结构

```text
rag_agent/
  data/docs/                 本地知识库文档
  data/eval_set.jsonl        50 条功能评测集
  artifacts/                 chunk、索引元信息和评测报告
  src/document_loader.py     文档读取与切分
  src/retriever.py           TF-IDF / BGE+FAISS 检索
  src/generator.py           抽取式 / OpenAI / Qwen 生成
  src/tools.py               检索、摘要、计算器工具
  src/agent.py               意图路由与命令行入口
  src/eval.py                评测脚本
  src/optimize_retrieval.py  参数扫描脚本
  tests/test_core.py         核心功能测试
```

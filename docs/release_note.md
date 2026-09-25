# Release Notes

## v1.0.0 — Initial Release

First internal release of MAQCache: a semantic caching layer for LLM
applications that stores previous prompt/answer pairs and serves them
back when a new request is close enough in meaning to a cached one,
avoiding a repeat call to the model.

### LLM adapters

- MAQ SLM (MAQ Softwares' internal OpenAI-compatible LLM gateway)
- OpenAI (chat completion, including streaming responses, image
  generation, audio transcription, and moderation requests)
- Anthropic
- LangChain (works with any LLM LangChain itself supports)
- Replicate
- Stability AI
- Hugging Face `diffusers`
- MiniGPT4
- Llama.cpp
- Dolly

### Embedding backends

Text: ONNX, Hugging Face Transformers, Sentence-Transformers (SBERT),
OpenAI, Cohere, LangChain, RWKV, PaddleNLP, FastText, and a plain
string passthrough for exact-match use cases.

Multi-modal: UForm (text + image), Timm and ViT (image), Data2Vec
(audio).

### Storage

- **Scalar store** (prompts, answers, metadata): SQLite, MySQL,
  MariaDB, SQL Server, Oracle, PostgreSQL, DuckDB, MongoDB, DynamoDB,
  Redis.
- **Vector store**: FAISS, Milvus, Chroma, hnswlib, Qdrant, Weaviate,
  pgvector, DocArray, USearch, Redis.
- **Object store** (for multi-modal caches): local disk, S3.
- **Eviction**: in-memory LRU/FIFO for single-node deployments, or a
  Redis-backed eviction manager to share cache state across multiple
  nodes.

### Similarity evaluation

Exact match, embedding distance, an ONNX cross-encoder model, k-nearest
reciprocal reranking, Cohere rerank, sequence matching, and a
combinable "join" evaluator for blending multiple strategies.

### Request-processing pipeline

- Pluggable pre-processing per LLM/adapter shape (chat, image, audio,
  moderation, vision-language, etc.), including long-conversation
  compression via summarization or selective context extraction.
- Configurable post-processing to pick which cached candidate(s) are
  returned, including temperature-aware behavior that scales how
  often the cache is bypassed in favor of a fresh model call.
- Multi-level cache chaining (`next_cache`) and per-request overrides
  (`cache_obj`, `cache_context`, `cache_skip`, `session`).

### Server mode

A standalone HTTP server (`maqcache_server`) with `put`/`get`/`flush`
endpoints, an OpenAI-compatible chat-completions proxy, a bundled
Python client, and a Dockerfile for containerized deployment.

"""
RAG pipeline — document ingestion and hybrid retrieval.

Retrieval strategy (Phase 3):
  1. Vector search via pgvector (top-K * 2 candidates)
  2. BM25 re-ranking over those candidates (reduces to top-K)
  3. Optional user_id metadata filter for multi-tenancy

LlamaIndex handles chunking, embedding, and vector storage.
"""

from functools import lru_cache
from urllib.parse import urlparse

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


def configure_llamaindex() -> None:
    """
    Configure LlamaIndex global LLM and embedding settings.
    Call once at application startup (FastAPI lifespan hook) and
    inside Celery tasks before using the pipeline.
    """
    settings = get_settings()

    if settings.litellm_mode == "library":
        from llama_index.llms.litellm import LiteLLM
        Settings.llm = LiteLLM(model=settings.primary_model)
    else:
        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(
            model=settings.primary_model,
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_master_key,
        )

    embed_kwargs: dict = {"model": settings.embedding_model, "embed_batch_size": 100}
    if settings.litellm_mode == "proxy":
        embed_kwargs["base_url"] = settings.litellm_base_url
        embed_kwargs["api_key"] = settings.litellm_master_key

    Settings.embed_model = OpenAIEmbedding(**embed_kwargs)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


@lru_cache
def get_vector_store() -> PGVectorStore:
    db_url = urlparse(get_settings().database_url)
    return PGVectorStore.from_params(
        database=db_url.path.lstrip("/"),
        host=db_url.hostname,
        password=db_url.password,
        port=db_url.port or 5432,
        user=db_url.username,
        table_name="llamaindex_documents",
        embed_dim=1536,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64},
    )


@lru_cache
def get_index() -> VectorStoreIndex:
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def _get_retriever(similarity_top_k: int = 10, user_id: str | None = None):
    """
    Build a retriever with optional user-scoped metadata filter.
    Fetches 2x top_k candidates so BM25 re-ranking has room to work.
    """
    kwargs: dict = {"similarity_top_k": similarity_top_k}

    if user_id:
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
        kwargs["filters"] = MetadataFilters(filters=[
            MetadataFilter(key="user_id", value=user_id),
        ])

    return get_index().as_retriever(**kwargs)


def _bm25_rerank(query: str, nodes: list, top_k: int) -> list:
    """
    Re-rank retrieved nodes by BM25 keyword score and return top_k.
    Gracefully degrades to vector order if rank-bm25 is not installed.
    """
    if len(nodes) <= top_k:
        return nodes
    try:
        from rank_bm25 import BM25Okapi
        corpus = [node.get_content().lower().split() for node in nodes]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(nodes)), key=lambda i: scores[i], reverse=True)
        return [nodes[i] for i in ranked[:top_k]]
    except ImportError:
        log.warning("bm25_unavailable", detail="rank-bm25 not installed; using vector order")
        return nodes[:top_k]


async def retrieve_context(
    query: str,
    user_id: str | None = None,
    top_k: int = 5,
) -> str:
    """
    Hybrid retrieval: vector search → BM25 re-rank → formatted context string.

    Args:
        query:   The user's query or agent reasoning step.
        user_id: If provided, filters results to this user's documents only.
        top_k:   Number of final passages to return after re-ranking.

    Returns:
        Formatted string of passages, or empty string if nothing found.
    """
    retriever = _get_retriever(similarity_top_k=top_k * 2, user_id=user_id)
    nodes = await retriever.aretrieve(query)
    if not nodes:
        return ""
    nodes = _bm25_rerank(query, nodes, top_k=top_k)
    log.debug("retrieval_done", user_id=user_id, returned=len(nodes))
    return "\n\n---\n\n".join(node.get_content() for node in nodes)

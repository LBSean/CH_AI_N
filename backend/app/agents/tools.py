"""
Agent tool definitions.

Tools are registered with the LLM via bind_tools() and executed by
LangGraph's ToolNode. Add new tools here — they are automatically
available to the agent.

Current tools:
  - get_current_datetime: trivial utility tool (demonstrates tool calling)
  - search_knowledge_base: explicit RAG retrieval (agent-driven, complements
                           the automatic retrieve node at graph start)
"""

from datetime import datetime, timezone
from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@tool
async def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Use this when you need additional context beyond what was initially retrieved.

    Args:
        query: A concise search query describing what information you need.

    Returns:
        Relevant text passages from the knowledge base, or a not-found message.
    """
    from app.rag.pipeline import retrieve_context
    context = await retrieve_context(query)
    return context if context else "No relevant information found in the knowledge base."


# Exported list — used by build_research_graph() and nodes.py
ALL_TOOLS = [get_current_datetime, search_knowledge_base]

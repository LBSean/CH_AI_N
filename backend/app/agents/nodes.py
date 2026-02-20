"""
LangGraph node implementations.

Graph topology (Phase 3):

    START → memory_injection → agent → (router) → END
                                  ↑        ↓ tool_calls present
                                  └── tool_node

memory_injection_node:
    - Retrieves RAG context (user-scoped)
    - Queries episodic memory for relevant past sessions
    - Loads active tool state
    - Populates: context, episodic_context

agent_node:
    - Receives full context in state
    - LLM bound with tools — returns AIMessage (text or tool_calls)

should_continue (router):
    - Conditional edge from agent
    - "tools"  → tool_node if agent produced tool calls
    - "__end__" → END otherwise

tool_node:
    - LangGraph built-in ToolNode
    - Executes tool calls, appends ToolMessage to messages
    - Loops back to agent
"""

import json
from langchain_core.messages import SystemMessage

from app.agents.tools import ALL_TOOLS
from app.core.graph_state import AgentState
from app.core.llm import get_chat_model
from app.core.logging import get_logger
from app.memory.episodic import query_episodic_memory
from app.rag.pipeline import retrieve_context

log = get_logger(__name__)


# ── memory_injection_node ──────────────────────────────────────────────────────

async def memory_injection_node(state: AgentState) -> dict:
    """
    Runs once at the start of each agent turn.
    Loads RAG context, episodic memories, and composes them into state.
    Does NOT call the LLM — pure data loading.
    """
    user_id = state.get("user_id", "")
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Parallel retrieval (episodic + RAG)
    import asyncio
    rag_ctx, episodic_ctx = await asyncio.gather(
        retrieve_context(query, user_id=user_id),
        query_episodic_memory(user_id, query, top_k=3),
    )

    log.debug(
        "memory_injection",
        user_id=user_id,
        has_rag=bool(rag_ctx),
        has_episodic=bool(episodic_ctx),
    )

    return {"context": rag_ctx, "episodic_context": episodic_ctx}


# ── agent_node ────────────────────────────────────────────────────────────────

async def agent_node(state: AgentState) -> dict:
    """
    Main reasoning node — LLM with tools bound.
    Returns either:
      - AIMessage with content (final response), or
      - AIMessage with tool_calls (triggers tool_node)
    """
    context = state.get("context", "")
    episodic = state.get("episodic_context", "")

    system_parts = ["You are a helpful AI assistant."]
    if episodic:
        system_parts.append(f"\nContext from past conversations:\n{episodic}")
    if context:
        system_parts.append(f"\nRelevant knowledge base context:\n{context}")

    system_content = "\n".join(system_parts)
    messages_with_system = [SystemMessage(content=system_content)] + list(state["messages"])

    llm = get_chat_model().bind_tools(ALL_TOOLS)
    response = await llm.ainvoke(messages_with_system)

    log.debug(
        "agent_response",
        user_id=state.get("user_id"),
        has_tool_calls=bool(getattr(response, "tool_calls", None)),
        content_length=len(response.content) if response.content else 0,
    )

    return {"messages": [response]}


# ── Router (conditional edge function) ────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    Inspect the last agent message.
    Returns "tools" to route to the tool executor,
    or "__end__" to finish the graph.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"

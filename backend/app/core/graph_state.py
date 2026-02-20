from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""
    messages:         Annotated[list, add_messages]  # full message history (auto-appended)
    user_id:          str
    thread_id:        str
    context:          str   # RAG context from the retrieval node
    episodic_context: str   # relevant past-session summaries (Phase 2 memory injection)
    tool_calls:       list  # pending tool invocations
    tool_results:     dict  # completed tool outputs
    metadata:         dict  # per-run metadata (model, tokens, latency)

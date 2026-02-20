"""
Research agent graph — Phase 3 full topology.

    START → memory_injection → agent → (should_continue) → END
                                  ↑              ↓ tool_calls
                                  └──────── tool_node

memory_injection: loads RAG context + episodic memory into state
agent:            LLM with tools bound — produces text or tool_calls
should_continue:  routes to tool_node or END
tool_node:        executes tool calls (LangGraph built-in ToolNode)
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.nodes import agent_node, memory_injection_node, should_continue
from app.agents.tools import ALL_TOOLS
from app.core.checkpointer import get_checkpointer
from app.core.graph_state import AgentState


def build_research_graph():
    """
    Compile and return the agent graph with Postgres checkpointing.
    The compiled graph is stateless — safe to call once per process and reuse.
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("memory_injection", memory_injection_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    # Edges
    workflow.add_edge(START, "memory_injection")
    workflow.add_edge("memory_injection", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "agent")  # loop: tool results → agent reasoning

    return workflow.compile(checkpointer=get_checkpointer())

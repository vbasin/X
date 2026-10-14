"""Shared agentic substrate (P1) — registry + research_search."""

from __future__ import annotations

from src.agentic.registry import ToolRegistry, obj_schema
from src.agentic.rag import research_search, _substring_internal
from src.research_director.registry import build_default_registry, build_critic_registry
from src.research_director.tools import ResearchTool


def test_tool_registry_add_call_specs():
    reg = ToolRegistry()
    reg.add("echo", "Echo args", obj_schema({"x": {"type": "string"}}, ["x"]),
            lambda x: {"x": x})
    assert "echo" in reg
    assert reg.call("echo", x="hi") == {"x": "hi"}
    specs = reg.specs()
    assert specs[0]["type"] == "function"
    assert specs[0]["function"]["name"] == "echo"


def test_tool_registry_rejects_duplicate():
    reg = ToolRegistry()
    reg.add("t", "d", obj_schema(), lambda: None)
    try:
        reg.add("t", "d2", obj_schema(), lambda: None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_substring_fallback_finds_docs():
    hits = _substring_internal("triple barrier momentum architecture", limit=5)
    # Repo has ADRs / architecture docs mentioning these terms
    assert isinstance(hits, list)
    # May be empty in stripped trees; if non-empty, shape is stable
    for h in hits:
        assert "source" in h or "source_ref" in h


def test_research_search_never_raises():
    # Cold index or missing DB must not raise
    out = research_search("nq momentum triple barrier", k=3)
    assert isinstance(out, list)


def test_research_tool_delegates_to_agentic():
    tool = ResearchTool(queries_per_finding=2)
    out = tool.search("feature importance calibration")
    assert isinstance(out, list)
    assert tool.search("x", used=2) == []  # budget exhausted


def test_build_default_registry_has_rag_tools():
    reg = build_default_registry(
        registry_dir="/tmp/nonexistent_registry",
        program_id="nq-50pt-econ",
        model_name="model",
    )
    names = set(reg.names())
    assert "research_search" in names
    assert "search_research" in names
    assert "db_inventory" in names
    assert "get_skill" in names
    # callable without DB for research_search (RAG/substring)
    hits = reg.call("research_search", query="walk-forward validation")
    assert isinstance(hits, list)


def test_build_critic_registry_tools():
    reg = build_critic_registry(
        registry_dir="/tmp/nonexistent_registry",
        program_id="nq-50pt-econ",
        model_name="model",
    )
    assert "search_research" in reg
    assert "recent_journal" in reg

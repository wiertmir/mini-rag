"""Offline compatibility gate for the dependency experiment (issue #82)."""

import subprocess
import sys
from pathlib import Path


def test_haystack_and_existing_cli_work_offline() -> None:
    # A clean process avoids imports cached by other tests. Disable telemetry before
    # import, remove credentials, and reject network attempts rather than mocking APIs.
    script = r"""
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import socket
import tomllib

for key in list(os.environ):
    if key.endswith("API_KEY"):
        del os.environ[key]
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"
os.environ["DD_TRACE_ENABLED"] = "false"

def deny_network(*args, **kwargs):
    raise AssertionError("Offline smoke test attempted network access")

socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
pins = [d.split("==", 1)[1] for d in project["project"]["dependencies"]
        if d.startswith("haystack-ai==")]
assert len(pins) == 1, "Expected exactly one haystack-ai exact pin"
installed = metadata.version("haystack-ai")
assert installed in (pins[0], pins[0] + ".0"), (installed, pins[0])
print("haystack-ai", installed)

from haystack import Document, Pipeline, component
from haystack.components.joiners import DocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore
assert callable(component)
store = InMemoryDocumentStore()
doc = Document(content="offline compatibility probe", meta={"source": "smoke"})
store.write_documents([doc])
assert store.count_documents() == 1

pipeline = Pipeline()
pipeline.add_component("first", DocumentJoiner())
pipeline.add_component("second", DocumentJoiner())
pipeline.connect("first.documents", "second.documents")
result = pipeline.run({"first": {"documents": [doc]}}, include_outputs_from={"first"})
assert result["first"]["documents"][0].content == doc.content
assert result["second"]["documents"][0].meta == doc.meta
restored = Pipeline.loads(pipeline.dumps())
assert restored.run({"first": {"documents": [doc]}})["second"]["documents"][0] == doc

for name in ("ingest", "embed", "retrieve", "ask", "display"):
    module = importlib.import_module("mini_rag." + name)
    assert callable(module.main)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "haystack-ai" in result.stdout


# Exercise the learner's actual graph rather than substituting a test-only graph.
import pytest


@pytest.fixture
def offline_graph(monkeypatch: pytest.MonkeyPatch):
    import socket

    monkeypatch.setenv("HAYSTACK_TELEMETRY_ENABLED", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def deny_network(*args, **kwargs):
        raise AssertionError("Unexpected network access in the offline graph")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    from haystack import Pipeline

    from mini_rag.haystack_pipeline import components

    graphs = []
    runs = []
    original_run = Pipeline.run

    def capture_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        runs.append(result)
        return result

    def capture_pipeline():
        graph = Pipeline()
        graphs.append(graph)
        return graph

    monkeypatch.setattr(Pipeline, "run", capture_run)
    monkeypatch.setattr(components, "Pipeline", capture_pipeline)
    monkeypatch.setattr(components, "print_obj", lambda value: None)
    components.main([])
    assert len(graphs) == 1
    return graphs[0], runs


def test_actual_graph_returns_leaf_and_requested_intermediate(offline_graph):
    graph, _ = offline_graph
    text = "One   two\nthree"
    inputs = {"normalizer": {"text": text}, "text_stats": {"text": text}}
    default = graph.run(inputs)
    assert "normalizer" not in default
    assert default["text_stats"] == {"word_count": 3, "line_count": 2, "char_count": len(text)}
    assert default["normalized_text_stats"] == {
        "word_count": 3,
        "line_count": 2,
        "char_count": len("One two\nthree"),
    }
    included = graph.run(inputs, include_outputs_from={"normalizer"})
    assert included["normalizer"] == {"normalized_text": "One two\nthree"}
    assert included["normalized_text_stats"] == default["normalized_text_stats"]


def test_actual_graph_rejects_missing_input(offline_graph):
    graph, _ = offline_graph
    with pytest.raises(ValueError):
        graph.run({})


def test_actual_components_reject_incompatible_socket_types():
    from haystack import Pipeline
    from haystack.core.errors import PipelineConnectError

    from mini_rag.haystack_pipeline.components import TextStats, WhitespaceNormalizer

    graph = Pipeline()
    graph.add_component("stats", TextStats())
    graph.add_component("normalizer", WhitespaceNormalizer())
    with pytest.raises(PipelineConnectError):
        graph.connect("stats.word_count", "normalizer.text")


def test_demo_requests_the_first_component_output(offline_graph):
    _, runs = offline_graph
    assert any("normalized_text" in result.get("normalizer", {}) for result in runs), (
        "Issue #83 requires the demo to request its first component's intermediate output; "
        "the current main() runs only the default leaf-output case."
    )


def test_learner_demo_runs_from_a_clean_offline_interpreter():
    script = r"""
import os
import runpy
import socket
for key in list(os.environ):
    if key.endswith("API_KEY"):
        del os.environ[key]
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"
def deny(*args, **kwargs):
    raise AssertionError("Unexpected network access")
socket.socket.connect = deny
socket.create_connection = deny
socket.getaddrinfo = deny
runpy.run_module("mini_rag.haystack_pipeline.components", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Non-modified text:" in result.stdout
    assert "Normalized text:" in result.stdout

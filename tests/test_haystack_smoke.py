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

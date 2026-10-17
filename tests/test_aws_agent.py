"""Offline security regression tests for the optional AWS agent."""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "aws_agent" / "agent.py"
spec = importlib.util.spec_from_file_location("aws_readonly_agent", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": 123, "context": "some text"},
        {"prompt": "a" * 4001, "context": "ok"},
        {"prompt": "ok", "context": "b" * 12001},
        {"prompt": "ok", "context": 1},
        None,
    ],
)
def test_invalid_payload_does_not_trigger_model(payload):
    assert "error" in module.invoke(payload)


def test_empty_context_fails_closed_without_model():
    assert "Insufficient" in module.handle({"prompt": "What happened?"})["answer"]


def test_no_secrets_in_error():
    expected = "Invalid request: prompt/context must be correctly formatted and within size limits"
    assert module.invoke({"prompt": "ok", "context": 12}) == {"error": expected}


def test_no_write_capabilities_exposed():
    source = MODULE_PATH.read_text()
    assert "tools=[]" in source
    assert "subprocess" not in source

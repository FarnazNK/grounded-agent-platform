"""Offline tests for AgentCore transport error handling and no-network mode."""

import io
import json
import sys
import types

import pytest

from rag_agent.aws_agentcore_client import AgentCoreTransportError, invoke_agentcore


class Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_runtime_not_configured_makes_no_aws_call(monkeypatch):
    monkeypatch.delenv("AGENTCORE_RUNTIME_ARN", raising=False)
    with pytest.raises(AgentCoreTransportError):
        invoke_agentcore(prompt="question", context="synthetic")


@pytest.mark.parametrize("prompt, context", [("", "data"), ("x" * 4001, "data"), ("q", "x" * 12001)])
def test_request_bounds_fail_before_sdk_import(monkeypatch, prompt, context):
    monkeypatch.setenv(
        "AGENTCORE_RUNTIME_ARN",
        "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/mock",
    )
    with pytest.raises(AgentCoreTransportError):
        invoke_agentcore(prompt=prompt, context=context)


def test_signed_runtime_invocation_uses_fixed_arn_and_bounded_response(monkeypatch):
    arn = "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/mock"
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", arn)
    client = Client({"statusCode": 200, "response": io.BytesIO(b'{"answer": "ok"}')})
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(client=lambda service: client if service == "bedrock-agentcore" else None),
    )
    assert invoke_agentcore(prompt="question", context="public context") == "ok"
    payload = client.calls[0]
    assert payload["agentRuntimeArn"] == arn
    assert json.loads(payload["payload"]) == {"prompt": "question", "context": "public context"}


@pytest.mark.parametrize(
    "response",
    [
        {"statusCode": 403, "response": io.BytesIO(b"denied")},
        {"statusCode": 200, "response": io.BytesIO(b"not json")},
        {"statusCode": 200, "response": io.BytesIO(b'{"answer": 1}')},
        {"statusCode": 200, "response": io.BytesIO(b"x" * 16001)},
    ],
)
def test_invalid_provider_responses_do_not_leak_details(monkeypatch, response):
    monkeypatch.setenv(
        "AGENTCORE_RUNTIME_ARN",
        "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/mock",
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(client=lambda _: Client(response)),
    )
    with pytest.raises(AgentCoreTransportError, match="AgentCore invocation failed"):
        invoke_agentcore(prompt="question", context="synthetic")

"""Offline-only Strands/Bedrock demonstration without SDK or AWS credentials.

Run: python aws_agent/demo_offline.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


class FakeBedrockModel:
    def __init__(self, **kwargs):
        assert kwargs["model_id"] == "offline-model"
        assert kwargs["guardrail_id"] == "offline-guardrail"
        assert kwargs["guardrail_version"] == "1"


class FakeStrandsAgent:
    def __init__(self, **kwargs):
        assert kwargs["tools"] == []
        assert "untrusted data" in kwargs["system_prompt"]

    def __call__(self, prompt):
        assert "<reference>" in prompt
        return "Offline demo: bounded, untrusted reference context accepted."


def main():
    os.environ["BEDROCK_MODEL_ID"] = "offline-model"
    os.environ["BEDROCK_GUARDRAIL_ID"] = "offline-guardrail"
    os.environ["BEDROCK_GUARDRAIL_VERSION"] = "1"
    sys.modules["strands"] = types.SimpleNamespace(Agent=FakeStrandsAgent)
    sys.modules["strands.models"] = types.SimpleNamespace(BedrockModel=FakeBedrockModel)
    path = Path(__file__).with_name("agent.py")
    spec = importlib.util.spec_from_file_location("offline_aws_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "Insufficient" in module.handle({"prompt": "question", "context": ""})["answer"]
    print("PASS: empty context fails closed")
    response = module.handle({"prompt": "Summarize", "context": "Synthetic public reference."})
    print("PASS:", response["answer"])
    print("No AWS calls made. No account or payment required.")


if __name__ == "__main__":
    main()

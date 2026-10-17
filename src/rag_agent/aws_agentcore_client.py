"""Backend-only AgentCore Runtime transport, with no client-selected ARN."""

from __future__ import annotations

import json
import os
import uuid


class AgentCoreTransportError(RuntimeError):
    pass


def invoke_agentcore(*, prompt: str, context: str) -> str:
    arn = os.getenv("AGENTCORE_RUNTIME_ARN", "")
    if not arn.startswith("arn:aws:bedrock-agentcore:"):
        raise AgentCoreTransportError("AgentCore is not configured")
    if not prompt or len(prompt) > 4000 or len(context) > 12000:
        raise AgentCoreTransportError("Agent request rejected")
    try:
        import boto3

        client = boto3.client("bedrock-agentcore")
        result = client.invoke_agent_runtime(
            agentRuntimeArn=arn,
            runtimeSessionId=str(uuid.uuid4()),
            contentType="application/json",
            accept="application/json",
            payload=json.dumps({"prompt": prompt, "context": context}).encode(),
        )
        if result.get("statusCode", 200) != 200:
            raise AgentCoreTransportError("AgentCore failed")
        stream = result["response"]
        raw = stream.read(16001)
        if len(raw) > 16000:
            raise AgentCoreTransportError("AgentCore response too large")
        decoded = json.loads(raw)
        if not isinstance(decoded, dict) or not isinstance(decoded.get("answer"), str):
            raise AgentCoreTransportError("Invalid AgentCore response")
        return decoded["answer"][:12000]
    except Exception:
        raise AgentCoreTransportError("AgentCore invocation failed") from None

"""AgentCore Runtime entrypoint; enforce AWS-side auth and IAM before public use."""
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent import invoke

app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload, context):
    return invoke(payload)


if __name__ == "__main__":
    app.run()

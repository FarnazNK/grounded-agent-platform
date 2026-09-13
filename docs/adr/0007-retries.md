# ADR 0007: Retry policy boundaries
The service retries transient LLM and embedding failures with exponential backoff. It does not retry malformed uploads, authorization failures, or malformed model output because those failures need explicit correction rather than repetition.

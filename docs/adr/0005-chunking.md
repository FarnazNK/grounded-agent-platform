# ADR 0005: Word-window chunking
The ingestion path uses fixed word windows with overlap because it is deterministic, fast, and easy to compare across retrieval revisions. More advanced semantic chunking is deferred until evaluation data shows a clear win.

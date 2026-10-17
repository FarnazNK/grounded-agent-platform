"""Loopback-only, workspace-scoped gRPC demo with synthetic documents."""

from __future__ import annotations

import hmac
import os
from concurrent import futures

import grpc
from google.protobuf.struct_pb2 import Struct

METHOD = "/portfolio.retrieval.v1.Retrieval/Query"
_DATA = {
    "alpha": ["Alpha: synthetic retrieval source"],
    "beta": ["Beta: separate synthetic retrieval source"],
}


def query(request: Struct, context: grpc.ServicerContext) -> Struct:
    workspace = request.fields.get("workspace")
    prompt = request.fields.get("query")
    if not workspace or not prompt:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing fields")
    wid = workspace.string_value
    question = prompt.string_value
    if not wid or wid not in _DATA:
        context.abort(grpc.StatusCode.PERMISSION_DENIED, "Unknown workspace")
    expected = os.environ.get(f"GRPC_DEMO_TOKEN_{wid.upper()}", "")
    metadata = dict(context.invocation_metadata())
    supplied = metadata.get("authorization", "")
    if not expected or not hmac.compare_digest(supplied, "Bearer " + expected):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")
    if not question or len(question) > 1000:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid query")
    response = Struct()
    response.update({"workspace": wid, "sources": _DATA[wid]})
    return response


def make_server() -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    handler = grpc.unary_unary_rpc_method_handler(
        query,
        request_deserializer=Struct.FromString,
        response_serializer=Struct.SerializeToString,
    )
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "portfolio.retrieval.v1.Retrieval", {"Query": handler}
            ),
        )
    )
    return server


def main() -> None:
    if not os.environ.get("GRPC_DEMO_TOKEN_ALPHA"):
        raise SystemExit("Set GRPC_DEMO_TOKEN_ALPHA for the local-only demo")
    server = make_server()
    port = server.add_insecure_port("127.0.0.1:50059")
    if not port:
        raise SystemExit("Loopback port unavailable")
    server.start()
    print("Offline gRPC demo listening on loopback 127.0.0.1:50059")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=0)


if __name__ == "__main__":
    main()

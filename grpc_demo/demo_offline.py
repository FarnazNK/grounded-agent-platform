"""A zero-cloud local gRPC request: run python -m grpc_demo.demo_offline."""
from __future__ import annotations

import os

import grpc
from google.protobuf.struct_pb2 import Struct

from grpc_demo.service import METHOD, make_server


def main() -> None:
    os.environ["GRPC_DEMO_TOKEN"] = "local-demo-not-for-production"
    server = make_server()
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            method = channel.unary_unary(
                METHOD,
                request_serializer=Struct.SerializeToString,
                response_deserializer=Struct.FromString,
            )
            request = Struct()
            request.update({"workspace": "alpha", "query": "Find sources"})
            response = method(
                request,
                metadata=(("authorization", "Bearer local-demo-not-for-production"),),
                timeout=3,
            )
            assert response["workspace"] == "alpha"
            assert len(response["sources"]) == 1
            print("PASS: authenticated, workspace-scoped gRPC query over loopback")
            print("No cloud services, AWS credentials, or charges required.")
    finally:
        server.stop(grace=0)


if __name__ == "__main__":
    main()

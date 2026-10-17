"""Authenticated local gRPC transport regression tests (no AWS, no network egress)."""
import grpc
import pytest
from google.protobuf.struct_pb2 import Struct

from grpc_demo.service import METHOD, make_server


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("GRPC_DEMO_TOKEN", "test-only-token")
    server = make_server()
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        call = channel.unary_unary(
            METHOD,
            request_serializer=Struct.SerializeToString,
            response_deserializer=Struct.FromString,
        )
        yield call
    server.stop(grace=0)


def invoke(call, workspace="alpha", query="find", token="Bearer test-only-token"):
    request = Struct()
    request.update({"workspace": workspace, "query": query})
    return call(request, metadata=(("authorization", token),), timeout=3)


def test_workspace_isolation(transport):
    a = invoke(transport, workspace="alpha")
    b = invoke(transport, workspace="beta")
    assert a["sources"][0].startswith("Alpha:")
    assert b["sources"][0].startswith("Beta:")
    assert "Beta:" not in str(a)


@pytest.mark.parametrize(
    "workspace, query, token, expected",
    [
        ("alpha", "find", "", grpc.StatusCode.UNAUTHENTICATED),
        ("alpha", "find", "Bearer wrong", grpc.StatusCode.UNAUTHENTICATED),
        ("unknown", "find", "Bearer test-only-token", grpc.StatusCode.PERMISSION_DENIED),
        (
            "alpha",
            "x" * 1001,
            "Bearer test-only-token",
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
    ],
)
def test_transport_rejects_invalid_calls(transport, workspace, query, token, expected):
    with pytest.raises(grpc.RpcError) as error:
        invoke(transport, workspace, query, token)
    assert error.value.code() == expected
